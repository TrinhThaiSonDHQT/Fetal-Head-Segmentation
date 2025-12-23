"""
Dice Similarity Coefficient (DSC) metric for segmentation evaluation.
"""

import torch


def dice_coefficient(predictions, targets, smooth=1e-7):
    """
    Calculate Dice Similarity Coefficient (DSC).
    
    Args:
        predictions: Predicted masks (B, 1, H, W) or (B, H, W), values in [0, 1]
        targets: Ground truth masks (B, 1, H, W) or (B, H, W), values in {0, 1}
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        dice: Dice coefficient (scalar)
    
    Formula:
        DSC = (2 * |X ∩ Y|) / (|X| + |Y|)
    """
    # Flatten predictions and targets
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # Calculate intersection and union
    intersection = (predictions * targets).sum()
    
    # Calculate Dice coefficient
    dice = (2.0 * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
    
    return dice.item()


def dice_coefficient_batch(predictions, targets, smooth=1e-7):
    """
    Calculate Dice Similarity Coefficient per sample in batch.
    
    Args:
        predictions: Predicted masks (B, 1, H, W), values in [0, 1]
        targets: Ground truth masks (B, 1, H, W), values in {0, 1}
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        dice_scores: List of Dice coefficients for each sample
    """
    batch_size = predictions.shape[0]
    dice_scores = []
    
    for i in range(batch_size):
        pred = predictions[i].view(-1)
        target = targets[i].view(-1)
        
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        dice_scores.append(dice.item())
    
    return dice_scores
