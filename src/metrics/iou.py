"""
Intersection over Union (IoU) metric for segmentation evaluation.
"""

import torch


def iou_score(predictions, targets, smooth=1e-7):
    """
    Calculate Intersection over Union (IoU) / Jaccard Index.
    
    Args:
        predictions: Predicted masks (B, 1, H, W) or (B, H, W), values in [0, 1]
        targets: Ground truth masks (B, 1, H, W) or (B, H, W), values in {0, 1}
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        iou: IoU score (scalar)
    
    Formula:
        IoU = |X ∩ Y| / |X ∪ Y|
    """
    # Flatten predictions and targets
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # Calculate intersection and union
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    
    # Calculate IoU
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def mean_iou(predictions, targets, smooth=1e-7):
    """
    Calculate mean Intersection over Union (mIoU).
    Same as iou_score but with explicit 'mean' naming for clarity.
    
    Args:
        predictions: Predicted masks (B, 1, H, W), values in [0, 1]
        targets: Ground truth masks (B, 1, H, W), values in {0, 1}
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        miou: mean IoU score (scalar)
    """
    return iou_score(predictions, targets, smooth)


def iou_score_batch(predictions, targets, smooth=1e-7):
    """
    Calculate IoU per sample in batch.
    
    Args:
        predictions: Predicted masks (B, 1, H, W), values in [0, 1]
        targets: Ground truth masks (B, 1, H, W), values in {0, 1}
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        iou_scores: List of IoU scores for each sample
    """
    batch_size = predictions.shape[0]
    iou_scores = []
    
    for i in range(batch_size):
        pred = predictions[i].view(-1)
        target = targets[i].view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou.item())
    
    return iou_scores
