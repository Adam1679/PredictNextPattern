import logging
import time

import deepspeed
import torch
import torch.nn as nn
import tqdm
import wandb
from data import OHLCDatasetMmap
from model import CryptoLlamaModel
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from transformers.models.llama.modeling_llama import LlamaConfig

logger = logging.getLogger(__name__)


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
    all_in_one_config.checkpoint.interval
    for batch in train_dataloader:
        t0 = time.time()
        inputs = batch["inputs"]
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
        if (num_steps + 1) % all_in_one_config.validation.interval == 0:
            val_metrics = validate(model, val_dataloader, all_in_one_config.validation)
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
    return total_loss / len(dataloader)


# load yaml config
def load_config(config_path):
    import yaml

    with open(config_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None


def main():
    all_in_one_config = load_config("config.yaml")
    # Hyperparameters
    batch_size = all_in_one_config.optimizer.batch_size
    total_steps = all_in_one_config.optimizer.total_steps
    learning_rate = all_in_one_config.optimizer.learning_rate
    warmup_steps = all_in_one_config.optimizer.warmup_steps

    # Create a sample dataset and dataloader

    dataset = OHLCDatasetMmap("memmap_dataset", window_range=(1600, 4096), is_train=True)
    valset = OHLCDatasetMmap(
        "memmap_dataset",
        window_range=(1600, 4096),
        is_train=False,
        first_n=all_in_one_config.validation.first_n,
        filter_symbols=all_in_one_config.validation.filter_symbols,
        filter_intervals=all_in_one_config.validation.filter_intervals,
    )

    # testset = OHLCDatasetMmap("memmap_dataset", window_range=(1600, 4096), is_train=False,
    #                           filter_symbols=all_in_one_config.test.filter_symbols,
    #                          filter_intervals=all_in_one_config.test.filter_intervals)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn
    )
    val_dataloader = DataLoader(
        valset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn
    )

    # Initialize the model
    model_config = LlamaConfig(**all_in_one_config.model)
    model = CryptoLlamaModel(model_config)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Initialize the learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # DeepSpeed configuration
    ds_config = all_in_one_config.distributed

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, config=ds_config
    )

    # Train the model
    train(
        all_in_one_config,
        model_engine,
        dataloader,
        val_dataloader,
        optimizer,
        scheduler,
    )


if __name__ == "__main__":
    main()
