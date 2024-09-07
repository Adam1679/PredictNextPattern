import torch
from torch import distributed as dist


def evaluate_metrics_signal_on_close(predictions, labels, mask):
    close_price_t = labels[:, :, 3].reshape(-1)
    open_price_t = labels[:, :, 0].reshape(-1)
    pred_t = predictions[:, :, 3].reshape(-1)
    mask_t = mask.reshape(-1)
    signal = torch.sign(pred_t - open_price_t) * mask_t
    pnl = signal * (close_price_t - open_price_t)
    win_rate = (pnl > 0).sum() / mask_t.sum()
    avg_pnl = pnl.sum() / mask_t.sum()

    if dist.is_initialized():
        dist.all_reduce(win_rate, op=dist.ReduceOp.AVG)
        dist.all_reduce(avg_pnl, op=dist.ReduceOp.AVG)
    return {
        "win_rate_signal_on_close": win_rate.item(),
        "avg_pnl_signal_on_close": avg_pnl.item(),
    }


def evaluate_metrics_signal_on_high_low(predictions, labels, mask):
    close_price_t = labels[:, :, 3].reshape(-1)
    open_price_t = labels[:, :, 0].reshape(-1)
    pred_low_t = predictions[:, :, 2].reshape(-1)
    pred_high_t = predictions[:, :, 1].reshape(-1)
    mask_t = mask.reshape(-1)
    upside = torch.maximum(pred_high_t - open_price_t, torch.zeros_like(open_price_t))
    downside = torch.maximum(open_price_t - pred_low_t, torch.zeros_like(open_price_t))
    signal = torch.sign(upside - downside) * mask_t
    pnl = signal * (close_price_t - open_price_t)
    win_rate = (pnl > 0).sum() / mask_t.sum()
    avg_pnl = pnl.sum() / mask_t.sum()
    if dist.is_initialized():
        dist.all_reduce(win_rate, op=dist.ReduceOp.AVG)
        dist.all_reduce(avg_pnl, op=dist.ReduceOp.AVG)
    return {
        "win_rate_signal_on_high_low": win_rate.item(),
        "avg_pnl_signal_on_high_low": avg_pnl.item(),
    }


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
    pred_relative_change = ((predictions - labels) / (labels + 1e-8))[:, :-1]
    label_relative_change = (labels[:, 1:] - labels[:, :-1]) / (labels[:, :-1] + 1e-8)

    # pred_relative_change = predictions[:, 1:]
    # label_relative_change = labels[:, 1:]

    # Apply mask to relative changes (exclude the first time step)
    mask_relative = mask[:, 1:]
    pred_relative_change = pred_relative_change * mask_relative
    label_relative_change = label_relative_change * mask_relative

    # Calculate binary predictions and labels based on relative change
    pred_up = (pred_relative_change > 0).to(dtype)
    pred_down = (pred_relative_change < 0).to(dtype)
    label_up = (label_relative_change > 0).to(dtype)
    label_down = (label_relative_change < 0).to(dtype)

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

    precision.mean()
    recall.mean()
    recall_up.mean()
    recall_down.mean()
    precision_up.mean()
    precision_down.mean()
    f1.mean()
    avg_mae = mae.mean()
    avg_mse = mse.mean()

    metrics = {
        "mae": round(avg_mae.item(), 4),
        "mse": round(avg_mse.item(), 8),
        # "recall": round(avg_recall.item(), 2),
        # "precision": round(avg_precision.item(), 2),
        # "precision_up": round(avg_precision_up.item(), 2),
        # "precision_down": round(avg_precision_down.item(), 2),
        # "recall_up": round(avg_recall_up.item(), 2),
        # "recall_down": round(avg_recall_down.item(), 2),
        # "f1": round(avg_f1.item(), 2),
    }
    return metrics


def keep_rightmost_k_ones(mask, k=10):
    # Get the cumulative sum from right to left
    cumsum = torch.fliplr(torch.cumsum(torch.fliplr(mask), dim=1))

    # Create a boolean mask for the rightmost k ones
    rightmost_k_mask = cumsum <= k

    # Apply the mask to the original tensor
    result = mask * rightmost_k_mask.to(mask.dtype)

    return result


def evaluation_metrics(predictions, labels, mask):
    all_metrics = {}
    for i, name in enumerate(["open", "high", "low", "close"]):
        metrics = evaluation_metrics_single(predictions[:, :, i], labels[:, :, i], mask)
        metrics = {f"{name}/{k}": v for k, v in metrics.items()}
        all_metrics.update(metrics)

    metrics1 = evaluate_metrics_signal_on_close(predictions, labels, mask)
    metrics2 = evaluate_metrics_signal_on_high_low(predictions, labels, mask)
    all_metrics.update(metrics1)
    all_metrics.update(metrics2)
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
