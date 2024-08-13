import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from itertools import product

import pandas as pd
import torch
import torch.nn as nn
import yaml
import yaml_include
from data import OHLCDatasetMmap, Timer
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from model import CryptoLlama, CryptoLlamaModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.plotting import plot_ohlc_candlestick_with_volume_and_prediction
from training.utils import evaluation_metrics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
DEVICE = torch.cuda.current_device()
torch.set_float32_matmul_precision("high")

yaml.add_constructor("!inc", yaml_include.Constructor(), yaml.Loader)


def print_rank(msg):
    logger.info(f"[Rank {RANK}] {msg}")


def print_master(msg):
    if RANK == 0:
        logger.info(f"[Master] {msg}")


def validate(model, all_in_one_config):
    val_batch_size = all_in_one_config["validation"]["batch_size"]
    assert val_batch_size % WORLD_SIZE == 0, "Batch size must be divisible by the number of GPUs"
    val_batch_size_per_rank = val_batch_size // WORLD_SIZE
    valset = OHLCDatasetMmap(
        all_in_one_config["data"]["data_root"],
        window_range=(
            all_in_one_config["data"]["min_seq_len"],
            all_in_one_config["data"]["max_seq_len"],
        ),
        is_train=False,
        sample_n=all_in_one_config["validation"]["sample_n"],
        filter_symbols=all_in_one_config["validation"]["filter_symbols"],
        filter_intervals=all_in_one_config["validation"]["filter_intervals"],
        world_size=WORLD_SIZE,
        rank=RANK,
    )
    val_dataloader = DataLoader(
        valset,
        batch_size=val_batch_size_per_rank,
        shuffle=False,
        collate_fn=valset.collate_fn,
        num_workers=all_in_one_config["data"]["num_workers"],
    )
    model.eval()
    total_loss = torch.zeros((), device=torch.cuda.current_device(), dtype=torch.float32)
    model_dtype = next(model.parameters()).dtype
    val_metrics_list = defaultdict(list)
    with torch.no_grad():
        val_step = 0
        for batch in tqdm(val_dataloader, desc="Validating"):
            inputs = batch["inputs"] = (
                batch["inputs"].to(DEVICE).to(model_dtype)
            )  # (batch_size, seq_len, input_size)
            attention_mask = batch["attention_mask"] = batch["attention_mask"].to(
                DEVICE
            )  # (batch_size, seq_len)
            outputs = model(inputs=inputs, attention_mask=attention_mask)

            # Assume the target is the last value in each sequence
            targets = inputs[:, 1:]  # (batch_size, seq_len-1, input_size)
            # Get the last prediction for each sequence
            predictions = outputs[:, :-1]  # (batch_size, seq_len-1, output_size)
            # Create a mask for valid positions (where attention_mask is 1)
            # Apply the mask to predictions and targets
            valid_positions = attention_mask[:, :-1].bool()
            masked_predictions = predictions[valid_positions]
            masked_targets = targets[valid_positions]

            token_avg_loss = nn.MSELoss()(masked_predictions, masked_targets)
            val_metrics = evaluation_metrics(predictions, targets, valid_positions)
            val_metrics = {f"val/{k}": v for k, v in val_metrics.items()}
            for k, v in val_metrics.items():
                val_metrics_list[k].append(v)
            total_loss.add_(token_avg_loss)
            val_step += 1
            del batch
    model.train()
    stats = {"val/loss": total_loss.item() / val_step}
    for k, v in val_metrics_list.items():
        stats[k] = sum(v) / len(v)
    return stats


def load_config(config_path):
    with open(config_path, "r") as stream:
        config = yaml.load(stream, yaml.Loader)
    config["model"]["max_position_embeddings"] = config["data"]["max_seq_len"]
    return config


@torch.inference_mode()
def generate_visualize(model, valset, symbol, interval, index, type_str, observation_length):
    if len(valset) <= index:
        logging.error(
            f"{symbol} {interval} {type_str} Index {index} out of range. Only has {len(valset)}"
        )
        return
    clip = valset.clip
    item = valset[index]
    inputs = torch.clamp_(item["inputs"], clip[0], clip[1])
    inputs = inputs.unsqueeze(0).to(DEVICE).to(torch.bfloat16)  # (1, seq_len, input_size)
    timestamp_s_iso = datetime.fromtimestamp(item["timestamp_s_start"]).strftime(
        "%Y-%m-%d_%H:%M:%S"
    )

    # Partial observation
    observed_inputs = inputs[:, :observation_length, :]

    # Auto-regressive generation
    generated_outputs = []
    for i in range(observation_length, inputs.shape[1]):
        prediction = model(inputs=observed_inputs)
        last_prediction = prediction[:, -1:, :]  # (1, 1, output_size)
        generated_outputs.append(last_prediction)
        observed_inputs = torch.cat([observed_inputs, last_prediction], dim=1)

    generated_outputs = torch.cat(generated_outputs, dim=1)
    output_file = f"{symbol}_{interval}_{timestamp_s_iso}_{index}_generated.html"
    valset.plot_kline_with_prediction(
        item=item,
        prediction=prediction,
        output_file=output_file,
        observation_length=observation_length,
    )


@torch.inference_mode()
def predict(model, data):
    with torch.no_grad():
        inputs = (
            torch.tensor(data, dtype=torch.bfloat16).unsqueeze(0).to(torch.cuda.current_device())
        )
        inputs_max, input_min, inputs_std = inputs.max(), inputs.min(), inputs.std()
        print(f"inputs_max: {inputs_max}, inputs_min: {input_min}, inputs_std: {inputs_std}")
        # inputs_max: 1.5625, inputs_min: -0.416015625, inputs_std: 0.48046875
        # inputs_max: 1.375, inputs_min: -20.0, inputs_std: 4.75
        outputs = model(inputs=inputs)
    return outputs.squeeze(0).float().cpu().numpy()


def inference_one(model, valset, symbol, interval, index, type_str):
    if len(valset) <= index:
        logging.error(
            f"{symbol} {interval} {type_str} Index {index} out of range. Only has {len(valset)}"
        )
        return
    item = valset[index]
    timestamp_s_iso = datetime.fromtimestamp(item["timestamp_s_start"]).strftime(
        "%Y-%m-%d_%H:%M:%S"
    )
    prediction = predict(model, item["inputs"].numpy())
    output_file = f"{symbol}_{interval}_{timestamp_s_iso}.html"
    valset.plot_kline_with_prediction(item=item, prediction=prediction, output_file=output_file)


def visualize_live_prediction(model, symbol, interval, output_file):
    def prepare_data():
        # load jsonl as dataframe
        with open("training/get_binance_test_data_dumped.jsonl", "r") as f:
            lines = f.readlines()
            lines = [json.loads(line) for line in lines]

        df = pd.DataFrame(lines)
        df = df.sort_values("open_time")
        df["date"] = pd.to_datetime(df["open_time"], unit="s")
        df = df.rename(
            columns={
                "open_price": "open",
                "high_price": "high",
                "low_price": "low",
                "close_price": "close",
            }
        )
        return df

    def normalize_data(data):
        data = data / data[0][0] - 1
        return data

    # Fetch live data
    df = prepare_data()  # [T, 4]
    input_data = torch.tensor(df[["open", "high", "low", "close"]].values)
    # Prepare input data for the model
    normalized_data, denom = OHLCDatasetMmap.normalize_rescale(input_data)
    # df[["open", "high", "low", "close"]] = normalized_data

    predictions = predict(model, normalized_data)
    predictions = OHLCDatasetMmap.unnormalize_rescale(torch.tensor(predictions), denom).numpy()

    # Prepare prediction data
    pred_df = pd.DataFrame(
        {
            "date": df["date"],
            "predicted_price": predictions[
                :, 3
            ],  # Assuming the last column is the close price prediction
        }
    )
    # Visualize
    plot_ohlc_candlestick_with_volume_and_prediction(
        df, pred_df, output_filename=output_file, symbol=symbol, interval=interval
    )
    print(f"Visualization saved to {output_file}")


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--config", type=str, default="training/config.yaml")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--interval", type=str, default=None)
    parser.add_argument("--test_index", type=str, default=None)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--type", type=str, default=None)
    parser.add_argument("--observation_length", type=int, default=0)
    parser.add_argument("--use_live", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    all_in_one_config = load_config(args.config)
    print_master("Config\n" + json.dumps(all_in_one_config, indent=4))

    # Hyperparameters
    # Initialize the model
    model_config = CryptoLlama(**all_in_one_config["model"])
    with Timer("Initialize Model"):
        model = CryptoLlamaModel(model_config).to(torch.bfloat16).to(torch.cuda.current_device())
        ckpt, tag = os.path.split(args.ckpt)
        state = get_fp32_state_dict_from_zero_checkpoint(ckpt, tag=tag)
        model = torch.compile(model)
        model.load_state_dict(state)
        model.eval()
        print_master(f"Number of parameters: {model.num_parameters():,}")
    if args.validate:
        val_metrics = validate(model, all_in_one_config)
        print_master(val_metrics)
    if args.use_live:
        visualize_live_prediction(model, args.symbol, args.interval, "live_prediction.html")

    elif args.test_index is not None:
        assert args.symbol is not None, "Symbol must be provided"
        assert args.interval is not None, "Interval must be provided"
        indexs = args.test_index.split(",")
        symbols = args.symbol.split(",")
        intervals = args.interval.split(",")
        types = args.type.split(",")
        for test_index, symbol, interval, type_str in product(indexs, symbols, intervals, types):
            valset = OHLCDatasetMmap(
                all_in_one_config["data"]["data_root"],
                window_range=(
                    all_in_one_config["data"]["max_seq_len"],
                    all_in_one_config["data"]["max_seq_len"],
                ),
                is_train=False,
                filter_symbols=[symbol],
                filter_intervals=[interval],
                filter_types=[type_str],
            )
            inference_one(model, valset, symbol, interval, int(test_index), type_str)
            if args.observation_length:
                generate_visualize(
                    model,
                    valset,
                    symbol,
                    interval,
                    int(test_index),
                    type_str,
                    observation_length=args.observation_length,
                )


if __name__ == "__main__":
    main()
