import torch
from torch import distributed as dist


def evaluation_metrics_single(predictions, labels, mask):
    # predict: [B, T]
    # label: [B, T]
    # Compute mean of valid elements along the time dimension
    dtype = predictions.dtype
    mask = mask.to(dtype)
    valid_predictions = predictions * mask
    valid_labels = labels * mask

    mae = torch.abs(valid_predictions - valid_labels).sum(dim=1) / mask.sum(dim=1)
    mse = ((valid_predictions - valid_labels) ** 2).sum(dim=1) / mask.sum(dim=1)

    # Calculate the relative change
    pred_relative_change = (predictions[:, 1:] - predictions[:, :-1]) / (predictions[:, :-1] + 1e-8)
    label_relative_change = (labels[:, 1:] - labels[:, :-1]) / (labels[:, :-1] + 1e-8)

    # Apply mask to relative changes (exclude the first time step)
    mask_relative = mask[:, 1:]
    pred_relative_change = pred_relative_change * mask_relative
    label_relative_change = label_relative_change * mask_relative

    # Calculate binary predictions and labels based on relative change
    pred_up = (pred_relative_change > 2e-4).to(dtype)
    pred_down = (pred_relative_change < -2e-4).to(dtype)
    label_up = (label_relative_change > 2e-4).to(dtype)
    label_down = (label_relative_change < -2e-4).to(dtype)

    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN) for both up and down
    TP_up = (pred_up * label_up * mask_relative).sum(dim=1)
    FP_up = (pred_up * label_down * mask_relative).sum(dim=1)
    FN_up = (pred_down * label_up * mask_relative).sum(dim=1)

    TP_down = (pred_down * label_down * mask_relative).sum(dim=1)
    FP_down = (pred_down * label_up * mask_relative).sum(dim=1)
    FN_down = (pred_up * label_down * mask_relative).sum(dim=1)

    # Calculate precision and recall for each direction
    precision_up = TP_up / (TP_up + FP_up + 1e-8)
    precision_up[TP_up + FP_up == 0] = 0
    precision_up = precision_up.mean()

    recall_up = TP_up / (TP_up + FN_up + 1e-8)
    recall_up[TP_up + FN_up == 0] = 0
    recall_up = recall_up.mean()

    precision_down = TP_down / (TP_down + FP_down + 1e-8)
    recall_down = TP_down / (TP_down + FN_down + 1e-8)
    precision_down[TP_down + FP_down == 0] = 0
    recall_down[TP_down + FN_down == 0] = 0

    # Calculate symmetric precision and recall
    precision = (precision_up + precision_down) / 2
    recall = (recall_up + recall_down) / 2

    # Calculate F1 score
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Handle cases where denominators are zero
    precision[(TP_up + FP_up == 0) & (TP_down + FP_down == 0)] = 0
    recall[(TP_up + FN_up == 0) & (TP_down + FN_down == 0)] = 0

    if dist.is_initialized():
        dist.all_reduce(recall, op=dist.ReduceOp.AVG)
        dist.all_reduce(precision, op=dist.ReduceOp.AVG)
        dist.all_reduce(precision_up, op=dist.ReduceOp.AVG)
        dist.all_reduce(precision_down, op=dist.ReduceOp.AVG)
        dist.all_reduce(recall_up, op=dist.ReduceOp.AVG)
        dist.all_reduce(recall_down, op=dist.ReduceOp.AVG)
        dist.all_reduce(f1, op=dist.ReduceOp.AVG)
        dist.all_reduce(mae, op=dist.ReduceOp.AVG)
        dist.all_reduce(mse, op=dist.ReduceOp.AVG)

    avg_precision = precision.mean()
    avg_recall = recall.mean()
    avg_recall_up = recall_up.mean()
    avg_recall_down = recall_down.mean()
    avg_precision_up = precision_up.mean()
    avg_precision_down = precision_down.mean()
    avg_f1 = f1.mean()
    avg_mae = mae.mean()
    avg_mse = mse.mean()

    metrics = {
        "mae": round(avg_mae.item(), 2),
        "mse": round(avg_mse.item(), 2),
        "recall": round(avg_recall.item(), 2),
        "precision": round(avg_precision.item(), 2),
        "precision_up": round(avg_precision_up.item(), 2),
        "precision_down": round(avg_precision_down.item(), 2),
        "recall_up": round(avg_recall_up.item(), 2),
        "recall_down": round(avg_recall_down.item(), 2),
        "f1": round(avg_f1.item(), 2),
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
