"""
Pixel Accuracy (PA) metric for segmentation evaluation.
"""

import torch


def pixel_accuracy(predictions, targets, threshold=0.5):
    """
    Calculate Pixel Accuracy (PA).
    
    Args:
        predictions: Predicted masks (B, 1, H, W) or (B, H, W), values in [0, 1]
        targets: Ground truth masks (B, 1, H, W) or (B, H, W), values in {0, 1}
        threshold: Threshold to binarize predictions (default: 0.5)
    
    Returns:
        pa: Pixel accuracy (scalar)
    
    Formula:
        PA = (TP + TN) / (TP + TN + FP + FN)
    """
    # Binarize predictions
    predictions_binary = (predictions >= threshold).float()
    
    # Flatten predictions and targets
    predictions_binary = predictions_binary.view(-1)
    targets = targets.view(-1)
    
    # Calculate correct predictions
    correct = (predictions_binary == targets).sum()
    total = targets.numel()
    
    # Calculate pixel accuracy
    pa = correct.float() / total
    
    return pa.item()


def mean_pixel_accuracy(predictions, targets, threshold=0.5):
    """
    Calculate mean Pixel Accuracy (mPA).
    Same as pixel_accuracy but with explicit 'mean' naming for clarity.
    
    Args:
        predictions: Predicted masks (B, 1, H, W), values in [0, 1]
        targets: Ground truth masks (B, 1, H, W), values in {0, 1}
        threshold: Threshold to binarize predictions (default: 0.5)
    
    Returns:
        mpa: mean pixel accuracy (scalar)
    """
    return pixel_accuracy(predictions, targets, threshold)


def pixel_accuracy_batch(predictions, targets, threshold=0.5):
    """
    Calculate Pixel Accuracy per sample in batch.
    
    Args:
        predictions: Predicted masks (B, 1, H, W), values in [0, 1]
        targets: Ground truth masks (B, 1, H, W), values in {0, 1}
        threshold: Threshold to binarize predictions (default: 0.5)
    
    Returns:
        pa_scores: List of pixel accuracy scores for each sample
    """
    batch_size = predictions.shape[0]
    pa_scores = []
    
    # Binarize predictions
    predictions_binary = (predictions >= threshold).float()
    
    for i in range(batch_size):
        pred = predictions_binary[i].view(-1)
        target = targets[i].view(-1)
        
        correct = (pred == target).sum()
        total = target.numel()
        pa = correct.float() / total
        pa_scores.append(pa.item())
    
    return pa_scores
