import logging
import time
from functools import partial

import deepspeed
import torch
import torch.nn as nn
import wandb
import yaml
from data import OHLCDatasetMmap, Timer
from model import CryptoLlamaModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from transformers.models.llama.modeling_llama import LlamaConfig

logger = logging.getLogger(__name__)

deepspeed.init_distributed(dist_backend="nccl")


def train(
    all_in_one_config,
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    max_steps,
    wandb_log=None,
):
    model.train()
    num_steps = 0

    for batch in train_dataloader:
        t0 = time.time()
        inputs = batch["inputs"].to(model.device)
        outputs = model(inputs)

        # Assume the target is the last value in each sequence
        targets = inputs[:, -1, -1]

        # Get the last prediction for each sequence
        predictions = outputs[:, -1, 0]

        loss = nn.MSELoss()(predictions, targets)

        model.backward(loss)
        model.step()
        scheduler.step()
        t1 = time.time()
        dt = t1 - t0
        stats = {
            "train/loss": loss.item(),
            "lr": optimizer.param_groups[0]["lr"],
            "train/time_per_step": dt,
        }
        if (num_steps + 1) % all_in_one_config["validation"]["interval"] == 0:
            val_metrics = validate(model, val_dataloader, all_in_one_config["validation"])
            stats.update(val_metrics)
        if wandb_log:
            wandb.log(stats, step=num_steps)
        num_steps += 1
        if num_steps >= max_steps:
            break


def validate(model, dataloader, validation_config):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            inputs = batch["inputs"]
            outputs = model(inputs)
            targets = inputs[:, -1, -1]
            predictions = outputs[:, -1, 0]
            loss = nn.MSELoss()(predictions, targets)
            total_loss += loss.item()
    model.train()
    return {"val/loss": total_loss / len(dataloader)}


def load_config(config_path):
    with open(config_path, "r") as stream:
        return yaml.safe_load(stream)


def main():
    with Timer("Loading config & Initialize Data"):
        all_in_one_config = load_config("training/config.yaml")

        # Hyperparameters
        batch_size = all_in_one_config["optimizer"]["batch_size"]
        total_steps = all_in_one_config["optimizer"]["total_steps"]
        learning_rate = all_in_one_config["optimizer"]["lr"]
        weight_decay = all_in_one_config["optimizer"]["weight_decay"]
        warmup_steps = all_in_one_config["optimizer"]["warmup_steps"]

        # Create datasets and dataloaders
        dataset = OHLCDatasetMmap("memmap_dataset", window_range=(1600, 4096), is_train=True)
        valset = OHLCDatasetMmap(
            "memmap_dataset",
            window_range=(1600, 4096),
            is_train=False,
            first_n=all_in_one_config["validation"]["first_n"],
            filter_symbols=all_in_one_config["validation"]["filter_symbols"],
            filter_intervals=all_in_one_config["validation"]["filter_intervals"],
        )

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn
        )
        val_dataloader = DataLoader(
            valset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn
        )

        # Initialize the model
        model_config = LlamaConfig(**all_in_one_config["model"])
    with Timer("Initialize Model"):
        model = CryptoLlamaModel(model_config)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Initialize the learning rate scheduler
    lr_scheduler_cls = partial(
        get_cosine_schedule_with_warmup,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # DeepSpeed configuration
    ds_config = all_in_one_config["distributed"]

    # Initialize DeepSpeed
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model, optimizer=optimizer, config=ds_config, lr_scheduler=lr_scheduler_cls
    )

    # Initialize wandb if needed
    wandb_log = all_in_one_config.get("wandb", {}).get("enabled", False)
    if wandb_log:
        wandb.init(project=all_in_one_config["wandb"]["project"], config=all_in_one_config)

    # Train the model
    train(
        all_in_one_config,
        model_engine,
        dataloader,
        val_dataloader,
        optimizer,
        lr_scheduler,
        total_steps,
        wandb_log,
    )


if __name__ == "__main__":
    main()
