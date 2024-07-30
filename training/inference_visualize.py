import json
import logging
import os
from collections import defaultdict

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
import yaml_include
from data import OHLCDatasetMmap, Timer
from model import CryptoLlama, CryptoLlamaModel
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

deepspeed.init_distributed(dist_backend="nccl")
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


def input_output_distribution(batch, outputs):
    """
    Compute the distribution of inputs and outputs
    """
    inputs = batch["inputs"]
    n_unique_symbols = len(set(batch["symbol"]))
    mean_seq_len = batch["attention_mask"].float().sum(dim=1).mean().item()

    input_mean = inputs.mean().item()
    input_max = inputs.max().item()
    input_min = inputs.min().item()
    input_std = inputs.std().item()
    output_mean = outputs.mean().item()
    output_std = outputs.std().item()
    output_max = outputs.max().item()
    output_min = outputs.min().item()
    metrics = {
        "data/input_mean": round(input_mean, 4),
        "data/input_std": round(input_std, 4),
        "data/output_mean": round(output_mean, 4),
        "data/output_std": round(output_std, 4),
        "data/n_unique_symbols": n_unique_symbols,
        "data/mean_seq_len": round(mean_seq_len, 1),
        "data/input_max": round(input_max, 4),
        "data/input_min": round(input_min, 4),
        "data/output_max": round(output_max, 4),
        "data/output_min": round(output_min, 4),
    }
    return metrics


def evaluation_metrics_single(predictions, labels, mask):
    # predict: [B, T]
    # label: [B, T]
    # Compute mean of valid elements along the time dimension
    dtype = predictions.dtype
    mask = mask.to(dtype)
    valid_predictions = predictions * mask
    valid_labels = labels * mask
    mean_pred = (valid_predictions.sum(dim=1) / mask.sum(dim=1)).unsqueeze(1)
    mean_label = (valid_labels.sum(dim=1) / mask.sum(dim=1)).unsqueeze(1)

    # Center the data by subtracting the mean
    pred_centered = (valid_predictions - mean_pred) * mask
    label_centered = (valid_labels - mean_label) * mask

    # Compute covariance and standard deviations
    covariance = (pred_centered * label_centered).sum(dim=1)
    pred_std = torch.sqrt((pred_centered**2).sum(dim=1))
    label_std = torch.sqrt((label_centered**2).sum(dim=1))

    # Compute Pearson correlation coefficient for each example
    pearson_correlation = covariance / (pred_std * label_std + 1e-12)

    # Average the Pearson correlation over the batch dimension

    mae = torch.abs(valid_predictions - valid_labels).sum(dim=1) / mask.sum(dim=1)
    mse = ((valid_predictions - valid_labels) ** 2).sum(dim=1) / mask.sum(dim=1)

    # calculate the accuracy
    # Apply mask and get binary predictions (threshold at 0.5)
    binary_predictions = (predictions >= 0).to(dtype) * mask
    binary_labels = (labels >= 0).to(dtype) * mask
    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
    TP = (binary_predictions * binary_labels * mask).sum(dim=1)
    FP = (binary_predictions * (1 - binary_labels) * mask).sum(dim=1)
    FN = ((1 - binary_predictions) * binary_labels * mask).sum(dim=1)

    # Calculate precision and recall for each example
    precision = TP / (TP + FP + 1e-8)  # Adding small value to avoid division by zero
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    # Handle cases where TP + FP or TP + FN is zero
    precision[TP + FP == 0] = 0
    recall[TP + FN == 0] = 0
    dist.all_reduce(recall, op=dist.ReduceOp.AVG)
    dist.all_reduce(precision, op=dist.ReduceOp.AVG)
    dist.all_reduce(f1, op=dist.ReduceOp.AVG)
    dist.all_reduce(pearson_correlation, op=dist.ReduceOp.AVG)
    dist.all_reduce(mae, op=dist.ReduceOp.AVG)
    dist.all_reduce(mse, op=dist.ReduceOp.AVG)
    avg_recall = recall.mean()
    avg_precision = precision.mean()
    avg_f1 = f1.mean()
    average_correlation = pearson_correlation.mean()
    avg_mae = mae.mean()
    avg_mse = mse.mean()
    metrics = {
        "correlation/avg": round(average_correlation.item(), 2),
        "mae/avg": round(avg_mae.item(), 2),
        "mse/avg": round(avg_mse.item(), 2),
        "recall/avg": round(avg_recall.item(), 2),
        "precision/avg": round(avg_precision.item(), 2),
        "f1/avg": round(avg_f1.item(), 2),
    }
    return metrics


def evaluation_metrics(predictions, labels, mask):
    all_metrics = {}
    for i, name in enumerate(["open", "high", "low", "close"]):
        metrics = evaluation_metrics_single(predictions[:, :, i], labels[:, :, i], mask)
        metrics = {f"{name}/{k}": v for k, v in metrics.items()}
        all_metrics.update(metrics)
    return all_metrics


# learning rate decay scheduler (cosine with warmup)
def get_lr(config, it):
    import math

    max_lr, min_lr = config["optimizer"]["lr"], config["optimizer"]["min_lr"]
    warmup_iters = config["optimizer"]["warmup_steps"]
    total_steps = config["optimizer"]["total_steps"]
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    lr_decay_iters = total_steps - warmup_iters
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (max_lr - min_lr)


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
    dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
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
    parser.add_argument("--checkpoint_json", type=str, default="training/checkpoint.json")
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
        # state = torch.load(args.ckpt, map_location=torch.cuda.current_device())
        # model.load_state_dict()
        print_master(f"Number of parameters: {model.num_parameters():,}")

    # DeepSpeed configuration
    model = torch.compile(model)
    # Initialize DeepSpeed
    model_engine = deepspeed.init_inference(
        model, tensor_parallel={"tp_size": 1}, dtype=torch.bfloat16, checkpoint=args.ckpt
    )

    val_metrics = validate(model_engine, all_in_one_config)
    print_master(val_metrics)
    return


if __name__ == "__main__":
    main()
