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
    if dist.is_initialized():
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
