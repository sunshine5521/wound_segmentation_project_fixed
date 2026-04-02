import torch
import numpy as np
import os

class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = {}

    def update(self, metric_name, val):
        val = val.item() if torch.is_tensor(val) else val
        if metric_name not in self.metrics:
            self.metrics[metric_name] = {"count": 0, "total": 0}
        self.metrics[metric_name]["count"] += 1
        self.metrics[metric_name]["total"] += val

    def get_avg(self, metric_name):
        return self.metrics[metric_name]["total"] / self.metrics[metric_name]["count"]

    def __str__(self):
        return " | ".join(
            [
                "{}: {:.{prec}f}".format(
                    metric_name, self.get_avg(metric_name), prec=self.float_precision
                )
                for metric_name in self.metrics
            ]
        )

def save_checkpoint(model, optimizer, epoch, metrics, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }, path)

def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def compute_dice_score(outputs, masks, threshold=0.5):
    """计算 Dice 系数"""
    outputs = (outputs > threshold).float()
    intersection = (outputs * masks).sum()
    union = outputs.sum() + masks.sum()
    dice = (2. * intersection) / (union + 1e-7)
    return dice.item()

def compute_iou(outputs, masks, threshold=0.5):
    """计算交并比 (IoU)"""
    outputs = (outputs > threshold).float()
    intersection = (outputs * masks).sum()
    union = outputs.sum() + masks.sum() - intersection
    iou = (intersection) / (union + 1e-7)
    return iou.item()
