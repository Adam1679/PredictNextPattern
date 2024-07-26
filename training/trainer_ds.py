import json
import logging
import os
import time
from functools import partial

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
import yaml_include
from data import OHLCDatasetMmap, Timer
from deepspeed import DeepSpeedEngine
from model import CryptoLlama, CryptoLlamaModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

import wandb

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

deepspeed.init_distributed(dist_backend="nccl")
RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))


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


def train(
    all_in_one_config,
    model: DeepSpeedEngine,
    train_dataloader,
    val_dataloader,
    optimizer,
    max_steps,
):
    model.train()
    sum_loss = torch.zeros((), device=torch.cuda.current_device(), dtype=torch.float32)
    sum_global_norm = torch.zeros((), device=torch.cuda.current_device(), dtype=torch.float32)
    total_tokens = torch.zeros((), device=torch.cuda.current_device(), dtype=torch.float32)
    model_dtype = next(model.parameters()).dtype
    for step, batch in enumerate(train_dataloader):
        t0 = time.time()
        for param_group in optimizer.param_groups:
            param_group["lr"] = get_lr(all_in_one_config, step)
        inputs = batch["inputs"] = (
            batch["inputs"].to(model_dtype).to(model.device, non_blocking=True)
        )  # (batch_size, seq_len, input_size)
        attention_mask = batch["attention_mask"] = batch["attention_mask"].to(
            model.device, non_blocking=True
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

        loss = nn.MSELoss()(masked_predictions, masked_targets)
        attention_mask.sum().item()
        model.backward(loss)
        global_norm = model.get_global_grad_norm()
        if global_norm is None:
            global_norm = torch.tensor(0.0, device=loss.device)
        model.step()
        t1 = time.time()
        dt = t1 - t0
        total_tokens.add_(attention_mask.sum().detach().data)
        sum_loss.add_(loss.detach().data)
        sum_global_norm.add_(global_norm.detach().data)
        stats = {}
        val_interval = all_in_one_config["validation"]["interval"]
        if val_interval > 0 and (step + 1) % val_interval == 0:
            val_metrics = validate(model, val_dataloader, all_in_one_config["validation"])
            stats.update(val_metrics)
        eta = (max_steps - step) * dt
        if (step + 1) % all_in_one_config["logging"]["log_interval"] == 0:
            dist.all_reduce(sum_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(sum_global_norm, op=dist.ReduceOp.AVG)
            dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
            tokens_per_step = total_tokens.item() / all_in_one_config["logging"]["log_interval"]
            stats = {
                "train/global_step": step,
                "train/loss": round(sum_loss.item(), 4),
                "train/global_grad_norm": round(sum_global_norm.item(), 4),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/time_per_step": round(dt, 4),
                "train/tokens_per_sec": int(tokens_per_step / dt),
                "train/eta(hour)": round(eta / 3600, 2),
            }
            data_stats = input_output_distribution(batch, outputs)
            stats.update(data_stats)
            print_master(stats)
            total_tokens.zero_()
            sum_loss.zero_()
            sum_global_norm.zero_()
            if all_in_one_config["logging"]["wandb_enabled"] and RANK == 0:
                wandb.log(stats, step=step)

        if (step + 1) % all_in_one_config["checkpointing"]["interval"] == 0:
            # Save the model checkpoint
            print_master(f"Saving checkpoint at iteration {step}")
            model.save_checkpoint(f"{all_in_one_config['checkpointing']['dir']}")

    print_master(f"Saving checkpoint at final iteration {step}")
    model.save_checkpoint(f"{all_in_one_config['checkpointing']['dir']}/iter_{step}")
    wandb.finish()


def validate(model, dataloader, validation_config):
    model.eval()
    total_loss = torch.zeros((), device=torch.cuda.current_device(), dtype=torch.float32)
    model_dtype = next(model.parameters()).dtype
    with torch.no_grad():
        val_step = 0
        for batch in tqdm(dataloader, desc="Validating"):
            inputs = batch["inputs"] = (
                batch["inputs"].to(model.device).to(model_dtype)
            )  # (batch_size, seq_len, input_size)
            attention_mask = batch["attention_mask"] = batch["attention_mask"].to(
                model.device
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

            s_loss = nn.MSELoss()(masked_predictions, masked_targets)
            total_loss.add_(s_loss)
            val_step += 1
            del batch
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    model.train()
    return {"val/loss": total_loss.item() / val_step}


def load_config(config_path):
    with open(config_path, "r") as stream:
        config = yaml.load(stream, yaml.Loader)
    config["model"]["max_position_embeddings"] = config["data"]["max_seq_len"]
    return config


def main():

    with Timer("Loading config & Initialize Data"):
        all_in_one_config = load_config("training/config.yaml")
        print_master("Config\n" + json.dumps(all_in_one_config, indent=4))

        # Hyperparameters
        val_batch_size = all_in_one_config["validation"]["batch_size"]
        batch_size = all_in_one_config["optimizer"]["batch_size"]
        assert batch_size % WORLD_SIZE == 0, "Batch size must be divisible by the number of GPUs"
        assert (
            val_batch_size % WORLD_SIZE == 0
        ), "Batch size must be divisible by the number of GPUs"

        total_steps = all_in_one_config["optimizer"]["total_steps"]
        learning_rate = all_in_one_config["optimizer"]["lr"]
        weight_decay = all_in_one_config["optimizer"]["weight_decay"]
        warmup_steps = all_in_one_config["optimizer"]["warmup_steps"]
        seq_len = all_in_one_config["data"]["max_seq_len"]
        min_seq_len = all_in_one_config["data"]["min_seq_len"]
        batch_size_per_rank = batch_size // WORLD_SIZE
        val_batch_size_per_rank = val_batch_size // WORLD_SIZE

        # Create datasets and dataloaders
        dataset = OHLCDatasetMmap(
            data_root=all_in_one_config["data"]["data_root"],
            window_range=(min_seq_len, seq_len),
            is_train=True,
            world_size=WORLD_SIZE,
            rank=RANK,
            filter_symbols=all_in_one_config["data"]["filter_symbols"],
            filter_intervals=all_in_one_config["data"]["filter_intervals"],
        )
        valset = OHLCDatasetMmap(
            all_in_one_config["data"]["data_root"],
            window_range=(min_seq_len, seq_len),
            is_train=False,
            sample_n=all_in_one_config["validation"]["sample_n"],
            filter_symbols=all_in_one_config["validation"]["filter_symbols"],
            filter_intervals=all_in_one_config["validation"]["filter_intervals"],
            world_size=WORLD_SIZE,
            rank=RANK,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size_per_rank,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            num_workers=all_in_one_config["data"]["num_workers"],
            pin_memory=True,
        )
        val_dataloader = DataLoader(
            valset,
            batch_size=val_batch_size_per_rank,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            num_workers=all_in_one_config["data"]["num_workers"],
        )

        # Initialize the model
        model_config = CryptoLlama(**all_in_one_config["model"])
    with Timer("Initialize Model"):
        model = CryptoLlamaModel(model_config)
        print_master(f"Number of parameters: {model.num_parameters():,}")
    # Initialize the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Initialize the learning rate scheduler
    partial(
        get_cosine_schedule_with_warmup,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=1,
    )

    # DeepSpeed configuration
    ds_config = all_in_one_config["distributed"]

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, config=ds_config
    )

    # Initialize wandb if needed
    if all_in_one_config["logging"]["wandb_enabled"] and RANK == 0:
        wandb.init(
            project=all_in_one_config["logging"]["wandb_project"],
            config=all_in_one_config,
            name=all_in_one_config["logging"]["wandb_run_name"],
        )

    # Train the model
    train(
        all_in_one_config,
        model_engine,
        dataloader,
        val_dataloader,
        optimizer,
        total_steps,
    )


if __name__ == "__main__":
    main()
