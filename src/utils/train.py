"""
Training Script for Fetal Head Segmentation using Improved U-Net

This script implements the training loop for the improved U-Net model.
"""

import torch
from tqdm import tqdm


def train_one_epoch(loader, model, optimizer, loss_fn, device, epoch):
    """
    Train the model for one epoch.
    
    Args:
        loader: DataLoader for training data
        model: The ImprovedUNet model
        optimizer: Optimizer (SGD or Adam)
        loss_fn: DiceBCELoss function
        device: torch.device (cuda or cpu)
        epoch: Current epoch number (for logging)
    
    Returns:
        avg_loss: Average loss for the epoch
        avg_dice: Average Dice score for the epoch
    """
    # Set model to training mode
    model.train()
    
    # Initialize metrics
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0
    
    # Training loop with progress bar
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for batch_idx, (images, masks) in enumerate(pbar):
        # Move data to device
        images = images.to(device)  # Shape: (B, 1, 256, 256)
        masks = masks.to(device)    # Shape: (B, 1, 256, 256)
        
        # Forward pass
        predictions = model(images)  # Shape: (B, 1, 256, 256) - LOGITS
        # Apply sigmoid for Dice calculation (loss function handles it internally)
        predictions_prob = torch.sigmoid(predictions)
        
        # Calculate loss
        loss = loss_fn(predictions, masks)
        
        # Backward pass
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()         # Compute gradients
        
        # Optional: Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()        # Update weights
        
        # Calculate Dice score for this batch
        with torch.no_grad():
            dice = (2 * (predictions_prob * masks).sum()) / (
                predictions_prob.sum() + masks.sum() + 1e-6
            )
        
        # Accumulate metrics
        total_loss += loss.item()
        total_dice += dice.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice.item():.4f}'
        })
        
        # Log metrics periodically (every 10 batches)
        if (batch_idx + 1) % 10 == 0:
            avg_loss_so_far = total_loss / num_batches
            avg_dice_so_far = total_dice / num_batches
            print(f"  Batch [{batch_idx + 1}/{len(loader)}] - "
                  f"Loss: {avg_loss_so_far:.4f}, Dice: {avg_dice_so_far:.4f}")
    
    # Calculate average metrics for the epoch
    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    
    return avg_loss, avg_dice


def validate_one_epoch(loader, model, loss_fn, device, epoch):
    """
    Validate the model for one epoch.
    
    Args:
        loader: DataLoader for validation data
        model: The ImprovedUNet model
        loss_fn: DiceBCELoss function
        device: torch.device (cuda or cpu)
        epoch: Current epoch number (for logging)
    
    Returns:
        avg_loss: Average loss for the epoch
        avg_dice: Average Dice score for the epoch
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize metrics
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0
    
    # Validation loop with progress bar
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)
    
    with torch.no_grad():
        for images, masks in pbar:
            # Move data to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            predictions = model(images)
            
            # Calculate loss
            loss = loss_fn(predictions, masks)
            
            # Apply sigmoid for metrics
            predictions_prob = torch.sigmoid(predictions)
            
            # Calculate Dice score
            dice = (2 * (predictions_prob * masks).sum()) / (
                predictions_prob.sum() + masks.sum() + 1e-6
            )
            
            # Accumulate metrics
            total_loss += loss.item()
            total_dice += dice.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice.item():.4f}'
            })
    
    # Calculate average metrics
    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    
    return avg_loss, avg_dice


def evaluate_model(loader, model, loss_fn, device):
    """
    Evaluate the model on validation or test data.
    
    Calculates comprehensive metrics as per the research article:
    - Dice Similarity Coefficient (DSC) - Primary metric
    - mean Intersection-over-Union (mIoU) - Secondary metric
    - mean Pixel Accuracy (mPA) - Secondary metric
    
    Expected Performance:
    - DSC: 97.81% ± 1.2
    - mPA: 99.18% ± 0.9
    - mIoU: 97.90% ± 1.7
    
    Args:
        loader: DataLoader for validation/test data
        model: The ImprovedUNet model
        loss_fn: DiceBCELoss function
        device: torch.device (cuda or cpu)
    
    Returns:
        metrics: Dictionary containing average loss, dice, miou, and pixel_accuracy
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize metrics
    total_loss = 0.0
    total_dice = 0.0
    total_miou = 0.0
    total_pa = 0.0
    num_batches = 0
    
    # Evaluation loop with progress bar
    pbar = tqdm(loader, desc="[Evaluation]", leave=False)
    
    # Disable gradient calculations
    with torch.no_grad():
        for images, masks in pbar:
            # Move data to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            predictions = model(images)
            
            # Apply sigmoid and convert to binary
            predictions_prob = torch.sigmoid(predictions)
            preds_binary = (predictions_prob > 0.5).float()
            
            # Calculate loss
            loss = loss_fn(predictions, masks)
            
            # Calculate Dice Similarity Coefficient (DSC)
            dice = (2 * (preds_binary * masks).sum()) / (
                preds_binary.sum() + masks.sum() + 1e-6
            )
            
            # Calculate Intersection-over-Union (IoU)
            # For fetal head (foreground)
            intersection = (preds_binary * masks).sum()
            union = preds_binary.sum() + masks.sum() - intersection
            iou_fh = (intersection + 1e-6) / (union + 1e-6)
            
            # For background
            preds_bg = 1 - preds_binary
            masks_bg = 1 - masks
            intersection_bg = (preds_bg * masks_bg).sum()
            union_bg = preds_bg.sum() + masks_bg.sum() - intersection_bg
            iou_bg = (intersection_bg + 1e-6) / (union_bg + 1e-6)
            
            # Mean IoU
            miou = (iou_fh + iou_bg) / 2
            
            # Calculate Pixel Accuracy (PA)
            correct_pixels = (preds_binary == masks).float().sum()
            total_pixels = masks.numel()
            pixel_accuracy = correct_pixels / total_pixels
            
            # Accumulate metrics
            total_loss += loss.item()
            total_dice += dice.item()
            total_miou += miou.item()
            total_pa += pixel_accuracy.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice.item():.4f}',
                'miou': f'{miou.item():.4f}',
                'mpa': f'{pixel_accuracy.item():.4f}'
            })
    
    # Calculate average metrics
    metrics = {
        'loss': total_loss / num_batches,
        'dice': total_dice / num_batches,
        'miou': total_miou / num_batches,
        'pixel_accuracy': total_pa / num_batches
    }
    
    return metrics
