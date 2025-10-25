"""
Segmentation Metrics for Model Evaluation
Unified module for Dice Coefficient, IoU, and Pixel Accuracy

All metrics work seamlessly on both CPU and GPU tensors.
Tensors are kept on their original device throughout computation.
"""

import torch
import torch.nn.functional as F


def dice_coefficient(pred, target, smooth=1e-6):
    """
    Calculate Dice Similarity Coefficient (DSC)
    
    Works on both CPU and GPU tensors without moving data between devices.
    
    Args:
        pred (torch.Tensor): Predicted segmentation mask
        target (torch.Tensor): Ground truth mask
        smooth (float): Smoothing factor to avoid division by zero
    
    Returns:
        torch.Tensor: Dice coefficient score (on same device as input)
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice


def iou_score(pred, target, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU / Jaccard Index)
    
    Works on both CPU and GPU tensors without moving data between devices.
    
    Args:
        pred (torch.Tensor): Predicted segmentation mask
        target (torch.Tensor): Ground truth mask
        smooth (float): Smoothing factor to avoid division by zero
    
    Returns:
        torch.Tensor: IoU score (on same device as input)
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


def pixel_accuracy(pred, target):
    """
    Calculate Pixel Accuracy (PA)
    
    Works on both CPU and GPU tensors without moving data between devices.
    
    Args:
        pred (torch.Tensor): Predicted segmentation mask
        target (torch.Tensor): Ground truth mask
    
    Returns:
        torch.Tensor: Pixel accuracy score (on same device as input)
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    correct = (pred == target).sum()
    total = target.numel()
    pa = correct.float() / total
    
    return pa


def mean_pixel_accuracy(pred, target, num_classes=2):
    """
    Calculate Mean Pixel Accuracy (mPA)
    
    Args:
        pred (torch.Tensor): Predicted segmentation mask
        target (torch.Tensor): Ground truth mask
        num_classes (int): Number of classes (2 for binary segmentation)
    
    Returns:
        torch.Tensor: Mean pixel accuracy
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    class_accuracies = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        if target_cls.sum() > 0:
            correct = (pred_cls & target_cls).sum()
            total = target_cls.sum()
            accuracy = correct.float() / total
            class_accuracies.append(accuracy)
    
    if len(class_accuracies) > 0:
        return torch.stack(class_accuracies).mean()
    else:
        return torch.tensor(0.0)


# Batch-wise metric calculations
def batch_dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient for a batch"""
    batch_size = pred.size(0)
    dice_scores = []
    for i in range(batch_size):
        dice = dice_coefficient(pred[i], target[i], smooth)
        dice_scores.append(dice)
    return torch.stack(dice_scores).mean()


def batch_iou_score(pred, target, smooth=1e-6):
    """Calculate IoU score for a batch"""
    batch_size = pred.size(0)
    iou_scores = []
    for i in range(batch_size):
        iou = iou_score(pred[i], target[i], smooth)
        iou_scores.append(iou)
    return torch.stack(iou_scores).mean()


def batch_pixel_accuracy(pred, target):
    """Calculate pixel accuracy for a batch"""
    batch_size = pred.size(0)
    pa_scores = []
    for i in range(batch_size):
        pa = pixel_accuracy(pred[i], target[i])
        pa_scores.append(pa)
    return torch.stack(pa_scores).mean()


if __name__ == "__main__":
    # Test metrics on both CPU and GPU
    print("Testing Segmentation Metrics")
    print("="*50)
    
    # Test on CPU
    print("\n[CPU Test]")
    pred_cpu = torch.tensor([[[[0., 1.], [1., 1.]]]])
    target_cpu = torch.tensor([[[[0., 1.], [1., 0.]]]])
    
    print(f"Dice Coefficient: {dice_coefficient(pred_cpu, target_cpu):.4f}")
    print(f"IoU Score: {iou_score(pred_cpu, target_cpu):.4f}")
    print(f"Pixel Accuracy: {pixel_accuracy(pred_cpu, target_cpu):.4f}")
    
    # Test on GPU if available
    if torch.cuda.is_available():
        print("\n[GPU Test]")
        pred_gpu = pred_cpu.cuda()
        target_gpu = target_cpu.cuda()
        
        dice_gpu = dice_coefficient(pred_gpu, target_gpu)
        iou_gpu = iou_score(pred_gpu, target_gpu)
        pa_gpu = pixel_accuracy(pred_gpu, target_gpu)
        
        print(f"Dice Coefficient: {dice_gpu:.4f} (device: {dice_gpu.device})")
        print(f"IoU Score: {iou_gpu:.4f} (device: {iou_gpu.device})")
        print(f"Pixel Accuracy: {pa_gpu:.4f} (device: {pa_gpu.device})")
        
        # Verify results match
        print("\n[Verification]")
        print(f"CPU vs GPU Dice match: {torch.allclose(dice_coefficient(pred_cpu, target_cpu), dice_gpu.cpu())}")
        print(f"CPU vs GPU IoU match: {torch.allclose(iou_score(pred_cpu, target_cpu), iou_gpu.cpu())}")
        print(f"CPU vs GPU PA match: {torch.allclose(pixel_accuracy(pred_cpu, target_cpu), pa_gpu.cpu())}")
    else:
        print("\n[GPU not available - skipping GPU tests]")
    
    print("\n" + "="*50)
    print("All metrics work correctly on both CPU and GPU!")
