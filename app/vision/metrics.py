import torch

def calculate_metrics(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Calculate common segmentation metrics for a batch.
    
    Args:
        logits: Model outputs before sigmoid (B, C, H, W)
        targets: Ground truth masks (B, C, H, W)
        threshold: Probability threshold for binary classification
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        A dictionary containing Dice, IoU, Precision, and Recall.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    
    true_positive = intersection
    false_positive = preds.sum() - true_positive
    false_negative = targets.sum() - true_positive
    
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    iou = (intersection + smooth) / (union + smooth)
    precision = (true_positive + smooth) / (true_positive + false_positive + smooth)
    recall = (true_positive + smooth) / (true_positive + false_negative + smooth)
    
    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item()
    }
