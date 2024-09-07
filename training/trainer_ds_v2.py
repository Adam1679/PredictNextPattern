import json
import time
from collections import defaultdict

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
from deepspeed import DeepSpeedEngine
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from training.data import Timer
from training.model import CryptoLlama, CryptoLlamaModel, CryptoT5Config, CryptoT5Model
from training.trainer_ds import (
    RANK,
    WORLD_SIZE,
    get_args,
    get_trainset,
    get_valset,
    load_config,
    print_master,
)
from training.utils import get_lr


def input_output_distribution(batch, outputs, names, mask):
    """
    Compute the distribution of inputs and outputs
    mask: (B, T)
    """
    metrics = {}
    mask = mask.bool()
    inputs = batch["inputs"]  # (B, T, D)
    for i, target_name in enumerate(names):
        input_value = inputs[:, :, i][mask].float()
        input_mean = input_value.mean().item()
        input_max = input_value.max().item()
        input_min = input_value.min().item()
        metrics[f"data/input_{target_name}_mean"] = round(input_mean, 4)
        metrics[f"data/input_{target_name}_max"] = round(input_max, 4)
        metrics[f"data/input_{target_name}_min"] = round(input_min, 4)

        output_logits = outputs[i][mask]  # (B, T)
        metrics[f"data/output_{target_name}_mean"] = round(output_logits.mean().item(), 4)
        metrics[f"data/output_{target_name}_max"] = round(output_logits.max().item(), 4)
        metrics[f"data/output_{target_name}_min"] = round(output_logits.min().item(), 4)

    return metrics


def evaluation_metrics(outputs, categorical_labels, mask):
    # categorical_predictions: (B, T, N)
    # categorical_labels: (B, T)
    metrics = []
    for i, categorical_predictions in enumerate(outputs):
        pred = categorical_predictions.argmax(dim=-1)  # Shape: [B, T]
        pred = pred[:, :-1]
        label = categorical_labels[..., i]  # Shape: [B, T]
        scores = {}
        if i in {5, 6, 8}:
            TP = (pred * label * mask).sum().float()
            FP = (pred * (1 - label) * mask).sum().float()
            FN = ((1 - pred) * label * mask).sum().float()
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            scores["precision"] = precision
            scores["recall"] = recall

        correct = (pred == label) & mask
        accuracy = correct.sum().float() / mask.sum()
        scores["acc"] = accuracy
        metrics.append(scores)

    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN) for both up and down
    return metrics


def validate(model, all_in_one_config):
    val_batch_size = all_in_one_config["validation"]["batch_size"]
    assert val_batch_size % WORLD_SIZE == 0, "Batch size must be divisible by the number of GPUs"
    val_batch_size_per_rank = val_batch_size // WORLD_SIZE
    valset = get_valset(all_in_one_config)
    val_dataloader = DataLoader(
        valset,
        batch_size=val_batch_size_per_rank,
        shuffle=False,
        collate_fn=valset.collate_fn,
        num_workers=all_in_one_config["data"]["num_workers"],
    )
    model.eval()
    loss_by_target = {
        target_name: torch.zeros((), device=torch.cuda.current_device(), dtype=torch.float32)
        for target_name in valset.NAMES
    }

    val_metrics_list = defaultdict(list)
    ce_loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        val_step = 0
        for batch in tqdm(val_dataloader, desc="Validating"):
            inputs = batch["inputs"] = batch["inputs"].to(
                model.device
            )  # (batch_size, seq_len, input_size)
            attention_mask = batch["attention_mask"] = batch["attention_mask"].to(
                model.device
            )  # (batch_size, seq_len)
            outputs = model(inputs=inputs, attention_mask=attention_mask)

            # Assume the target is the last value in each sequence
            targets = inputs[:, 1:]  # (batch_size, seq_len-1, input_size)
            valid_positions = attention_mask[:, :-1].bool()
            ce_loss_values = []
            for i in range(len(model.num_categories)):
                ce_loss_values.append(
                    ce_loss(
                        outputs[i][:, :-1][valid_positions].view(-1, model.num_categories[i]),
                        targets[:, :, i][valid_positions].view(-1),
                    )
                )

            for i, target_name in enumerate(valset.NAMES):
                loss_by_target[target_name].add_(ce_loss_values[i])

            val_metrics_raw = evaluation_metrics(outputs, targets, valid_positions)
            val_metrics = {}
            for i, target_name in enumerate(valset.NAMES):
                for k, v in val_metrics_raw[i].items():
                    val_metrics[f"val/{k}_{target_name}"] = v
            for k, v in val_metrics.items():
                val_metrics_list[k].append(v)
            val_step += 1
            del batch
    stats = {}
    for i, target_name in enumerate(valset.NAMES):
        dist.all_reduce(loss_by_target[target_name], op=dist.ReduceOp.AVG)
        stats["val/loss_{}".format(target_name)] = loss_by_target[target_name] / val_step

    for k, v in val_metrics_list.items():
        stats[k] = sum(v) / len(v)

    model.train()
    return stats


def train(
    args,
    all_in_one_config,
    model: DeepSpeedEngine,
    train_dataloader,
    optimizer,
    max_steps,
):
    if args.profile:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=15, warmup=2, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.profile),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()
    else:
        prof = None
    model.train()
    sum_loss = torch.zeros(
        len(model.num_categories), device=torch.cuda.current_device(), dtype=torch.float32
    )
    sum_global_norm = torch.zeros((), device=torch.cuda.current_device(), dtype=torch.float32)
    total_tokens = torch.zeros((), device=torch.cuda.current_device(), dtype=torch.float32)
    global_total_tokens = 0
    ce_loss = nn.CrossEntropyLoss()
    for step, batch in enumerate(train_dataloader):
        t0 = time.time()
        for param_group in optimizer.param_groups:
            param_group["lr"] = get_lr(all_in_one_config, step)
        inputs = batch["inputs"] = batch["inputs"].to(
            model.device, non_blocking=True
        )  # (batch_size, seq_len, input_size)
        attention_mask = batch["attention_mask"] = batch["attention_mask"].to(
            model.device, non_blocking=True
        )  # (batch_size, seq_len)
        outputs = model(inputs=inputs, attention_mask=attention_mask)
        # Assume the target is the last value in each sequence
        targets = inputs[:, 1:]  # (batch_size, seq_len-1, input_size)
        # Create a mask for valid positions (where attention_mask is 1)
        # Apply the mask to predictions and targets
        valid_positions = attention_mask[:, :-1].bool()
        ce_loss_values = []
        for i in range(len(model.num_categories)):
            ce_loss_values.append(
                ce_loss(
                    outputs[i][:, :-1][valid_positions].view(-1, model.num_categories[i]),
                    targets[:, :, i][valid_positions].view(-1),
                )
            )

        loss = sum(ce_loss_values) / len(ce_loss_values)
        target_loss = torch.tensor(ce_loss_values, device=loss.device)
        attention_mask.sum().item()
        model.backward(loss)
        global_norm = model.get_global_grad_norm()
        if global_norm is None:
            global_norm = torch.tensor(0.0, device=loss.device)
        model.step()
        if prof:
            prof.step()
        t1 = time.time()
        dt = t1 - t0
        total_tokens.add_(attention_mask.sum().detach().data)
        sum_loss.add_(target_loss.detach().data)
        sum_global_norm.add_(global_norm.detach().data)
        val_interval = all_in_one_config["validation"]["interval"]
        eta = (max_steps - step) * dt
        if (step + 1) % all_in_one_config["logging"]["log_interval"] == 0:
            stats = {}
            dist.all_reduce(sum_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(sum_global_norm, op=dist.ReduceOp.AVG)
            dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
            tokens_per_step = total_tokens.item() / all_in_one_config["logging"]["log_interval"]
            global_total_tokens += total_tokens.item()
            stats.update(
                {
                    "train/global_step": step,
                    "train/sum_loss": round(sum_loss.sum().item(), 4),
                    "train/global_grad_norm": round(sum_global_norm.item(), 4),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/time_per_step": round(dt, 4),
                    "train/tokens_per_sec": int(tokens_per_step / dt),
                    "train/eta(hour)": round(eta / 3600, 2),
                    "train/consumed_tokens": int(global_total_tokens),
                }
            )
            for i, target_name in enumerate(train_dataloader.dataset.NAMES):
                stats[f"train/loss_{target_name}"] = round(sum_loss[i].item(), 4)

            # eval_metrics = evaluation_metrics(outputs, targets, valid_positions)
            # eval_metrics = {f"train/{k}": v for k, v in eval_metrics.items()}
            data_stats = input_output_distribution(
                batch, outputs, train_dataloader.dataset.NAMES, attention_mask
            )

            if val_interval > 0 and (step + 1) % val_interval == 0:
                val_metrics = validate(model, all_in_one_config)
                stats.update(val_metrics)

            stats.update(data_stats)
            # stats.update(eval_metrics)
            print_master(stats)
            if "bar_pct_change" in batch:
                bar_pct_change = torch.tensor(batch["bar_pct_change"])
                stats["bar_pct_change_max"] = torch.max(bar_pct_change).item()
                stats["bar_pct_change_min"] = torch.min(bar_pct_change).item()
                stats["bar_pct_change_mean"] = torch.mean(bar_pct_change).item()
            if "low_bigger_than_high_error_sum" in batch:
                low_bigger_than_high_error_sum_max = torch.tensor(
                    batch["low_bigger_than_high_error_sum"]
                )
                stats["low_bigger_than_high_error_sum_max"] = torch.max(
                    low_bigger_than_high_error_sum_max
                ).item()
                stats["low_bigger_than_high_error_sum_min"] = torch.min(
                    low_bigger_than_high_error_sum_max
                ).item()
                stats["low_bigger_than_high_error_sum_mean"] = torch.mean(
                    low_bigger_than_high_error_sum_max
                ).item()

            if all_in_one_config["logging"]["wandb_enabled"] and RANK == 0:
                wandb.log(stats, step=step)

        if (step + 1) % all_in_one_config["checkpointing"]["interval"] == 0:
            # Save the model checkpoint
            print_master(f"Saving checkpoint at iteration {step}")
            model.save_checkpoint(
                f"{all_in_one_config['checkpointing']['dir']}/{all_in_one_config['logging']['wandb_run_name']}"
            )

        total_tokens.zero_()
        sum_loss.zero_()
        sum_global_norm.zero_()
        if max_steps is not None and step >= max_steps:
            break

    print_master(f"Saving checkpoint at final iteration {step}")
    model.save_checkpoint(f"{all_in_one_config['checkpointing']['dir']}/iter_{step}")
    wandb.finish()


def main():
    args = get_args()
    with Timer("Loading config & Initialize Data"):
        all_in_one_config = load_config(args.config)
        print_master("Config\n" + json.dumps(all_in_one_config, indent=4))

        # Hyperparameters

        batch_size = all_in_one_config["optimizer"]["batch_size"]
        assert batch_size % WORLD_SIZE == 0, "Batch size must be divisible by the number of GPUs"

        total_steps = all_in_one_config["optimizer"]["total_steps"]
        learning_rate = all_in_one_config["optimizer"]["lr"]
        weight_decay = all_in_one_config["optimizer"]["weight_decay"]
        all_in_one_config["optimizer"]["warmup_steps"]
        batch_size_per_rank = batch_size // WORLD_SIZE

        # Create datasets and dataloaders
        dataset = get_trainset(all_in_one_config)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size_per_rank,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            num_workers=all_in_one_config["data"]["num_workers"],
            pin_memory=False,
        )

        # Initialize the model
        model_config = CryptoLlama(**all_in_one_config["model"])
    with Timer("Initialize Model"):
        model_type = all_in_one_config["model"].get("type", "llama")
        if model_type == "llama":
            model_config = CryptoLlama(**all_in_one_config["model"])
            model = (
                CryptoLlamaModel(model_config).to(torch.bfloat16).to(torch.cuda.current_device())
            )
        elif model_type == "t5":
            model_config = CryptoT5Config(**all_in_one_config["model"])
            model = CryptoT5Model(model_config).to(torch.bfloat16).to(torch.cuda.current_device())
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        print_master(f"Number of parameters: {model.num_parameters():,}")
    # Initialize the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # DeepSpeed configuration
    ds_config = all_in_one_config["distributed"]
    model = torch.compile(model)
    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, config=ds_config
    )
    if args.eval_only:
        model_engine.load_checkpoint(load_dir=args.ckpt, load_module_only=True)
        val_metrics = validate(model_engine, all_in_one_config)
        print_master(val_metrics)
        return

    # Initialize wandb if needed
    if all_in_one_config["logging"]["wandb_enabled"] and RANK == 0:
        wandb.init(
            project=all_in_one_config["logging"]["wandb_project"],
            config=all_in_one_config,
            name=all_in_one_config["logging"]["wandb_run_name"],
        )

    # Train the model
    train(
        args,
        all_in_one_config,
        model_engine,
        dataloader,
        optimizer,
        total_steps,
    )


if __name__ == "__main__":
    main()
