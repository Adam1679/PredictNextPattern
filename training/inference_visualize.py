import json
import logging
import os
from collections import defaultdict

import torch
import torch.nn as nn
import yaml
import yaml_include
from data import OHLCDatasetMmap, Timer
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from model import CryptoLlama, CryptoLlamaModel
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def predict_one(model, all_in_one_config):
    OHLCDatasetMmap(
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


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--config", type=str, default="training/config.yaml")
    parser.add_argument("--ckpt", type=str, default="")
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
        # new_state = {k[10:]: v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval()
        print_master(f"Number of parameters: {model.num_parameters():,}")

    val_metrics = validate(model, all_in_one_config)
    print_master(val_metrics)
    return


if __name__ == "__main__":
    main()
