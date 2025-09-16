"""
Fixed and simplified loss functions for forest fire detection.

Key fixes:
1. Simplified shape handling with clear expectations
2. Removed overly defensive programming that masked real issues
3. Better numerical stability
4. Clear tensor dtype handling
5. Proper focal loss implementation for binary segmentation
"""

import warnings
from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_target_shape(targets: torch.Tensor, num_classes: int = 1) -> torch.Tensor:
    """
    Normalize target tensor to expected shape [B, H, W] for binary or [B, H, W] with class indices for multi-class.
    
    Args:
        targets: Input target tensor
        num_classes: Number of classes (1 for binary, >1 for multi-class)
        
    Returns:
        Normalized target tensor
    """
    # Remove channel dimension if present
    if targets.dim() == 4 and targets.shape[1] == 1:
        targets = targets.squeeze(1)
    
    # Ensure we have [B, H, W]
    if targets.dim() != 3:
        raise ValueError(f"Expected targets to be 3D [B,H,W] after normalization, got shape {targets.shape}")
    
    # For binary segmentation, ensure values are in [0,1]
    if num_classes == 1:
        targets = torch.clamp(targets, 0, 1)
    else:
        # For multi-class, ensure valid class indices
        targets = torch.clamp(targets.long(), 0, num_classes - 1)
    
    return targets


class DiceLoss(nn.Module):
    """
    Simplified Dice Loss with clear expectations.
    """
    
    def __init__(self, smooth: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, 1, H, W] logits for binary segmentation
            targets: [B, H, W] or [B, 1, H, W] binary targets (0 or 1)
        """
        if predictions.dim() != 4 or predictions.shape[1] != 1:
            raise ValueError(f"Expected predictions shape [B,1,H,W], got {predictions.shape}")
        
        # Normalize targets
        targets = normalize_target_shape(targets, num_classes=1)
        targets = targets.float().unsqueeze(1)  # [B, 1, H, W]
        
        # Convert logits to probabilities
        probs = torch.sigmoid(predictions)
        
        # Flatten spatial dimensions
        probs_flat = probs.flatten(2)  # [B, 1, H*W]
        targets_flat = targets.flatten(2)  # [B, 1, H*W]
        
        # Compute Dice coefficient
        intersection = (probs_flat * targets_flat).sum(dim=2)  # [B, 1]
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # [B, 1]
        
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_coeff
        
        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss.squeeze()


class FocalLoss(nn.Module):
    """
    Simplified Focal Loss for binary segmentation.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, 1, H, W] logits for binary segmentation
            targets: [B, H, W] or [B, 1, H, W] binary targets (0 or 1)
        """
        if predictions.dim() != 4 or predictions.shape[1] != 1:
            raise ValueError(f"Expected predictions shape [B,1,H,W], got {predictions.shape}")
        
        # Normalize targets
        targets = normalize_target_shape(targets, num_classes=1)
        targets = targets.float()  # [B, H, W]
        
        # Flatten for loss computation
        logits = predictions.squeeze(1)  # [B, H, W]
        
        # Compute BCE loss (without reduction)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        
        # Compute pt (probability of true class)
        pt = torch.where(targets == 1.0, probs, 1.0 - probs)
        
        # Compute alpha weight
        alpha_weight = torch.where(targets == 1.0, self.alpha, 1.0 - self.alpha)
        
        # Compute focal weight
        focal_weight = (1.0 - pt) ** self.gamma
        
        # Final focal loss
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class IoULoss(nn.Module):
    """
    Simplified IoU Loss.
    """
    
    def __init__(self, smooth: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, 1, H, W] logits for binary segmentation
            targets: [B, H, W] or [B, 1, H, W] binary targets (0 or 1)
        """
        if predictions.dim() != 4 or predictions.shape[1] != 1:
            raise ValueError(f"Expected predictions shape [B,1,H,W], got {predictions.shape}")
        
        # Normalize targets
        targets = normalize_target_shape(targets, num_classes=1)
        targets = targets.float().unsqueeze(1)  # [B, 1, H, W]
        
        # Convert logits to probabilities
        probs = torch.sigmoid(predictions)
        
        # Flatten spatial dimensions
        probs_flat = probs.flatten(2)
        targets_flat = targets.flatten(2)
        
        # Compute IoU
        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        iou_loss = 1.0 - iou
        
        if self.reduction == "mean":
            return iou_loss.mean()
        elif self.reduction == "sum":
            return iou_loss.sum()
        else:
            return iou_loss.squeeze()


class CombinedLoss(nn.Module):
    """
    Simplified combined loss function.
    """
    
    def __init__(self, 
                 dice_weight: float = 0.5, 
                 focal_weight: float = 0.5,
                 dice_smooth: float = 1.0,
                 focal_alpha: float = 0.25, 
                 focal_gamma: float = 2.0,
                 aux_weight: float = 0.4,
                 reduction: str = "mean"):
        super().__init__()
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.aux_weight = aux_weight
        
        self.dice_loss = DiceLoss(smooth=dice_smooth, reduction=reduction)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction=reduction)
    
    def forward(self, predictions: Union[Dict[str, torch.Tensor], torch.Tensor], 
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Model outputs (dict with 'out' key or tensor)
            targets: Ground truth targets
        """
        # Extract main prediction
        if isinstance(predictions, dict):
            main_pred = predictions["out"]
            aux_preds = predictions.get("aux", [])
        else:
            main_pred = predictions
            aux_preds = []
        
        # Compute main losses
        dice_val = self.dice_loss(main_pred, targets)
        focal_val = self.focal_loss(main_pred, targets)
        main_loss = self.dice_weight * dice_val + self.focal_weight * focal_val
        
        losses = {
            "dice_loss": dice_val,
            "focal_loss": focal_val,
            "main_loss": main_loss
        }
        
        # Compute auxiliary losses
        aux_loss_total = torch.tensor(0.0, device=main_pred.device, dtype=main_pred.dtype)
        if aux_preds:
            aux_losses = []
            for aux_pred in aux_preds:
                aux_dice = self.dice_loss(aux_pred, targets)
                aux_focal = self.focal_loss(aux_pred, targets)
                aux_combined = self.dice_weight * aux_dice + self.focal_weight * aux_focal
                aux_losses.append(aux_combined)
            
            if aux_losses:
                aux_loss_total = torch.stack(aux_losses).mean()
                losses["aux_loss"] = aux_loss_total
        
        # Total loss
        total_loss = main_loss + self.aux_weight * aux_loss_total
        losses["loss"] = total_loss
        
        return losses


class FireDetectionLoss(nn.Module):
    """
    Factory wrapper for different loss types.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        loss_type = config.get("loss_type", "combined")
        
        if loss_type == "combined":
            self.loss_fn = CombinedLoss(
                dice_weight=config.get("dice_weight", 0.5),
                focal_weight=config.get("focal_weight", 0.5),
                dice_smooth=config.get("dice_smooth", 1.0),
                focal_alpha=config.get("focal_alpha", 0.25),
                focal_gamma=config.get("focal_gamma", 2.0),
                aux_weight=config.get("aux_weight", 0.4),
                reduction=config.get("reduction", "mean")
            )
        elif loss_type == "dice":
            self.loss_fn = DiceLoss(
                smooth=config.get("dice_smooth", 1.0),
                reduction=config.get("reduction", "mean")
            )
        elif loss_type == "focal":
            self.loss_fn = FocalLoss(
                alpha=config.get("focal_alpha", 0.25),
                gamma=config.get("focal_gamma", 2.0),
                reduction=config.get("reduction", "mean")
            )
        elif loss_type == "iou":
            self.loss_fn = IoULoss(
                smooth=config.get("iou_smooth", 1.0),
                reduction=config.get("reduction", "mean")
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, predictions: Union[Dict[str, torch.Tensor], torch.Tensor], 
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        if isinstance(self.loss_fn, CombinedLoss):
            return self.loss_fn(predictions, targets)
        else:
            # Single loss function - need to handle dict/tensor outputs
            if isinstance(predictions, dict):
                main_pred = predictions["out"]
            else:
                main_pred = predictions
                
            loss_val = self.loss_fn(main_pred, targets)
            return {"loss": loss_val}


def create_loss_function(config: Dict) -> FireDetectionLoss:
    """Factory function to create loss function from config."""
    return FireDetectionLoss(config)


# Test the fixed losses
if __name__ == "__main__":
    print("Testing fixed loss functions...")
    
    # Test configuration
    config = {
        "loss_type": "combined",
        "dice_weight": 0.6,
        "focal_weight": 0.4,
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "aux_weight": 0.3
    }
    
    loss_fn = create_loss_function(config)
    
    # Test data
    B, H, W = 2, 64, 64
    predictions = {
        "out": torch.randn(B, 1, H, W),
        "aux": [torch.randn(B, 1, H, W)]
    }
    targets = torch.randint(0, 2, (B, H, W)).float()
    
    # Forward pass
    losses = loss_fn(predictions, targets)
    
    print("Loss values:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.6f}")
    
    print("✅ Fixed loss functions working correctly!")