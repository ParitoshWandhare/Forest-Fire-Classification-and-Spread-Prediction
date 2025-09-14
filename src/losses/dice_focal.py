"""
Combined loss functions for forest fire detection.

This version fixes the binary-target shape mismatch that caused warnings like:
  "Target size must be the same as input size" from binary_cross_entropy_with_logits.

Behavior:
- DiceLoss, FocalLoss, IoULoss handle targets shaped as:
    [B, H, W]          -> common scalar class targets
    [B, 1, H, W]       -> single-channel mask
    [B, C, H, W]       -> one-hot masks (C > 1 or C == 1)
- FocalLoss uses BCEWithLogits for binary (predictions.shape[1] == 1) and
  cross_entropy for multi-class.
- CombinedLoss supports dict input {"out":..., "aux":[...]} or a raw tensor.
"""

import warnings
from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Utility helpers
# ---------------------------
def _ensure_target_2d(targets: torch.Tensor) -> torch.Tensor:
    """
    Normalize targets to shape [B, H, W] if possible.
    Accepts [B, H, W], [B, 1, H, W] or [B, C, H, W] (one-hot) -> returns
    - If integer class indices (no channel): [B, H, W]
    - If single channel masks: [B, H, W]
    - If one-hot (C>1): returns [B, C, H, W] (unchanged) <-- caller should handle
    """
    if targets.dim() == 3:
        # already [B, H, W]
        return targets
    if targets.dim() == 4:
        if targets.shape[1] == 1:
            # squeeze channel
            return targets.squeeze(1)
        else:
            # likely one-hot: return as-is (caller will detect)
            return targets
    raise ValueError(f"Unexpected target shape: {targets.shape}")


def _to_one_hot_if_needed(targets: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    """
    Convert targets to one-hot [B, C, H, W] if targets is [B, H, W].
    If targets is already [B, C, H, W] returns it.
    """
    if targets.dim() == 4 and targets.shape[1] == num_classes:
        return targets.float()
    if targets.dim() == 4 and targets.shape[1] != num_classes:
        # ambiguous; if it's single-channel repeated, try to reduce/expand appropriately
        if targets.shape[1] == 1:
            return targets.repeat(1, num_classes, 1, 1).float()
        else:
            # slice or pad
            c_target = targets.shape[1]
            if c_target >= num_classes:
                return targets[:, :num_classes].float()
            else:
                pad = torch.zeros((targets.shape[0], num_classes - c_target, targets.shape[2], targets.shape[3]),
                                  device=device, dtype=targets.dtype)
                return torch.cat([targets.float(), pad.float()], dim=1)
    # targets is [B, H, W] integer indices
    return F.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2).float()


# ---------------------------
# DiceLoss
# ---------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, reduction: str = "mean", ignore_index: int = -100):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.smooth = float(smooth)
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # NaN protection
        if torch.isnan(predictions).any():
            warnings.warn("NaN detected in predictions input to DiceLoss")
            predictions = torch.where(torch.isnan(predictions), torch.zeros_like(predictions), predictions)
        if torch.isnan(targets).any():
            warnings.warn("NaN detected in targets input to DiceLoss")
            targets = torch.where(torch.isnan(targets), torch.zeros_like(targets), targets)

        # Predictions: [B, C, H, W] or [B, 1, H, W]
        if predictions.dim() != 4:
            raise ValueError(f"Predictions must be 4D tensor [B,C,H,W], got {predictions.shape}")

        num_classes = predictions.shape[1]
        device = predictions.device

        # Normalize targets
        # If targets one-hot [B, C, H, W] and C==num_classes, keep it.
        # If targets [B, H, W] or [B, 1, H, W] convert appropriately.
        if targets.dim() == 4 and targets.shape[1] == num_classes:
            targets_one_hot = targets.float()
        else:
            # If predictions are multiclass, convert labels to one-hot
            if num_classes > 1:
                if targets.dim() == 4 and targets.shape[1] != 1:
                    # unexpected channel count: try argmax to produce indices
                    targets_idx = targets.argmax(dim=1).long()
                else:
                    targets_idx = _ensure_target_2d(targets).long()
                targets_one_hot = _to_one_hot_if_needed(targets_idx, num_classes, device)
            else:
                # binary case: ensure shape [B,1,H,W] float
                if targets.dim() == 4 and targets.shape[1] == 1:
                    targets_one_hot = targets.float()
                else:
                    targets_one_hot = _ensure_target_2d(targets).unsqueeze(1).float()

        # Convert predictions to probabilities
        if num_classes == 1:
            probs = torch.sigmoid(predictions)
        else:
            probs = F.softmax(predictions, dim=1)
        probs = probs.clamp(min=1e-7, max=1.0 - 1e-7)

        # Flatten
        probs_flat = probs.view(probs.shape[0], probs.shape[1], -1)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)

        # Handle ignore_index (only meaningful when targets provided as indices [B,H,W])
        if self.ignore_index != -100 and targets.dim() == 3:
            valid_mask = (targets != self.ignore_index).float().view(targets_flat.shape[0], 1, -1)
            probs_flat = probs_flat * valid_mask
            targets_flat = targets_flat * valid_mask

        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)

        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth + 1e-8)
        dice_loss = 1.0 - dice_coeff  # [B, C]
        dice_loss = dice_loss.clamp(min=0.0, max=2.0)

        if self.reduction == "mean":
            result = dice_loss.mean()
        elif self.reduction == "sum":
            result = dice_loss.sum()
        else:
            result = dice_loss

        if torch.isnan(result).any():
            warnings.warn("NaN in DiceLoss result, falling back to 1.0")
            result = torch.tensor(1.0, device=device, requires_grad=True)

        return result


# ---------------------------
# FocalLoss
# ---------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean", ignore_index: int = -100):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Protective checks
        if torch.isnan(predictions).any():
            warnings.warn("NaN detected in predictions input to FocalLoss")
            predictions = torch.where(torch.isnan(predictions), torch.zeros_like(predictions), predictions)
        if torch.isnan(targets).any():
            warnings.warn("NaN detected in targets input to FocalLoss")
            targets = torch.where(torch.isnan(targets), torch.zeros_like(targets), targets)

        if predictions.dim() != 4:
            raise ValueError(f"Predictions must be 4D [B,C,H,W], got {predictions.shape}")

        device = predictions.device
        num_classes = predictions.shape[1]

        # Normalize targets to indices or squeeze single-channel masks
        # If targets is one-hot [B,C,H,W] and C==num_classes -> convert to indices
        if targets.dim() == 4 and targets.shape[1] == num_classes and num_classes > 1:
            targets_idx = targets.argmax(dim=1).long()
        else:
            # Ensure indices shape [B,H,W] when possible
            if targets.dim() == 4 and targets.shape[1] == 1:
                targets_idx = targets.squeeze(1).long()
            elif targets.dim() == 3:
                targets_idx = targets.long()
            else:
                # if one-hot with mismatched channel count, try argmax
                if targets.dim() == 4:
                    targets_idx = targets.argmax(dim=1).long()
                else:
                    raise ValueError(f"Unsupported target shape for FocalLoss: {targets.shape}")

        # Binary case (single channel logits): use BCEWithLogits
        if num_classes == 1:
            # logits shape: [B, H, W]
            logits = predictions.squeeze(1)
            # Ensure targets_float shape matches logits shape
            targets_float = targets_idx.float()
            if targets_float.shape != logits.shape:
                # try to broadcast or squeeze
                if targets.dim() == 4 and targets.shape[1] == 1:
                    targets_float = targets.squeeze(1).float()
                else:
                    targets_float = targets_float.view_as(logits)

            # compute BCE with logits
            try:
                bce = F.binary_cross_entropy_with_logits(logits, targets_float, reduction="none")
            except Exception as e:
                warnings.warn(f"Error in binary_cross_entropy_with_logits: {e}")
                bce = torch.ones_like(targets_float, device=device)

            probs = torch.sigmoid(logits).clamp(min=1e-7, max=1.0 - 1e-7)
            pt = torch.where(targets_float == 1.0, probs, 1.0 - probs)

            focal_weight = torch.pow(1.0 - pt, self.gamma).clamp(min=1e-8, max=1e6)
            if self.alpha is not None:
                alpha_weight = torch.where(targets_float == 1.0, self.alpha, 1.0 - self.alpha)
            else:
                alpha_weight = 1.0

            loss = alpha_weight * focal_weight * bce  # [B,H,W]

            if self.ignore_index != -100:
                valid_mask = (targets_idx != self.ignore_index).float()
                loss = loss * valid_mask

        else:
            # Multi-class: use cross_entropy as base
            try:
                ce = F.cross_entropy(predictions, targets_idx, ignore_index=self.ignore_index, reduction="none")
            except Exception as e:
                warnings.warn(f"Error in cross_entropy: {e}")
                ce = torch.ones_like(targets_idx, dtype=torch.float32, device=device)

            probs = F.softmax(predictions, dim=1).clamp(min=1e-7, max=1.0 - 1e-7)
            # gather prob of true class: result shape [B,H,W]
            pt = probs.gather(1, targets_idx.unsqueeze(1)).squeeze(1)

            focal_weight = torch.pow(1.0 - pt, self.gamma).clamp(min=1e-8, max=1e6)
            if self.alpha is not None:
                alpha_weight = torch.where(targets_idx > 0, self.alpha, 1.0 - self.alpha).float().to(device)
            else:
                alpha_weight = 1.0

            loss = focal_weight * alpha_weight * ce  # [B,H,W]

            if self.ignore_index != -100:
                valid_mask = (targets_idx != self.ignore_index).float()
                loss = loss * valid_mask

        # Reduction
        if self.reduction == "mean":
            if self.ignore_index != -100:
                valid_count = (targets_idx != self.ignore_index).float().sum()
                if valid_count > 0:
                    result = loss.sum() / valid_count
                else:
                    result = torch.tensor(0.0, device=device)
            else:
                result = loss.mean()
        elif self.reduction == "sum":
            result = loss.sum()
        else:
            result = loss

        if torch.isnan(result).any():
            warnings.warn("NaN in FocalLoss result, falling back to 1.0")
            result = torch.tensor(1.0, device=device, requires_grad=True)

        return result


# ---------------------------
# IoULoss
# ---------------------------
class IoULoss(nn.Module):
    def __init__(self, smooth: float = 1.0, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.smooth = float(smooth)
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if torch.isnan(predictions).any() or torch.isnan(targets).any():
            warnings.warn("NaN detected in IoULoss inputs")
            return torch.tensor(1.0, device=predictions.device, requires_grad=True)

        if predictions.dim() != 4:
            raise ValueError(f"Predictions must be 4D [B,C,H,W], got {predictions.shape}")

        num_classes = predictions.shape[1]
        device = predictions.device

        if num_classes == 1:
            probs = torch.sigmoid(predictions).clamp(min=1e-7, max=1.0 - 1e-7)
            if targets.dim() == 4 and targets.shape[1] == 1:
                targets_proc = targets.float()
            else:
                targets_proc = _ensure_target_2d(targets).unsqueeze(1).float()
        else:
            probs = F.softmax(predictions, dim=1).clamp(min=1e-7, max=1.0 - 1e-7)
            if targets.dim() == 4 and targets.shape[1] == num_classes:
                targets_proc = targets.float()
            else:
                # convert indices to one-hot
                idx = _ensure_target_2d(targets).long()
                targets_proc = _to_one_hot_if_needed(idx, num_classes, device)

        probs_flat = probs.view(probs.shape[0], probs.shape[1], -1)
        targets_flat = targets_proc.view(targets_proc.shape[0], targets_proc.shape[1], -1)

        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth + 1e-8)
        iou_loss = (1.0 - iou).clamp(min=0.0, max=2.0)

        if torch.isnan(iou_loss).any():
            warnings.warn("NaN in IoULoss, using ones")
            iou_loss = torch.ones_like(iou_loss)

        if self.reduction == "mean":
            return iou_loss.mean()
        elif self.reduction == "sum":
            return iou_loss.sum()
        else:
            return iou_loss


# ---------------------------
# CombinedLoss (Dice + Focal + Aux)
# ---------------------------
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, focal_weight: float = 0.5, dice_smooth: float = 1.0,
                 focal_alpha: float = 0.75, focal_gamma: float = 2.0, aux_weight: float = 0.4,
                 reduction: str = "mean"):
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.aux_weight = float(aux_weight)
        self.dice_loss = DiceLoss(smooth=dice_smooth, reduction=reduction)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction=reduction)

    def forward(self, predictions: Union[Dict[str, torch.Tensor], torch.Tensor], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        # supports either dict {"out":..., "aux":[...]} or raw tensor
        if isinstance(predictions, dict):
            main_pred = predictions.get("out", None)
        else:
            main_pred = predictions

        if main_pred is None:
            raise ValueError("No main prediction ('out') provided to CombinedLoss")

        if torch.isnan(main_pred).any():
            warnings.warn("NaN detected in main predictions, replacing with zeros")
            main_pred = torch.where(torch.isnan(main_pred), torch.zeros_like(main_pred), main_pred)

        # compute components
        dice_val = self.dice_weight * self.dice_loss(main_pred, targets)
        focal_val = self.focal_weight * self.focal_loss(main_pred, targets)
        main_loss = dice_val + focal_val

        losses = {
            "dice_loss": (dice_val / (self.dice_weight + 1e-12)) if self.dice_weight != 0 else torch.tensor(0.0, device=main_pred.device),
            "focal_loss": (focal_val / (self.focal_weight + 1e-12)) if self.focal_weight != 0 else torch.tensor(0.0, device=main_pred.device),
            "main_loss": main_loss
        }

        # auxiliary losses (deep supervision)
        aux_loss_val = torch.tensor(0.0, device=main_pred.device)
        if isinstance(predictions, dict) and predictions.get("aux"):
            aux_preds = predictions.get("aux")
            aux_losses = []
            for a in aux_preds:
                if a is None:
                    continue
                if torch.isnan(a).any():
                    warnings.warn("NaN in aux prediction, skipping")
                    continue
                d_aux = self.dice_loss(a, targets)
                f_aux = self.focal_loss(a, targets)
                aux_losses.append(self.dice_weight * d_aux + self.focal_weight * f_aux)
            if aux_losses:
                aux_loss_val = torch.stack(aux_losses).mean()
                losses["aux_loss"] = aux_loss_val

        total = main_loss + (self.aux_weight * aux_loss_val if aux_loss_val > 0 else 0.0)
        total = torch.clamp(total, min=0.0, max=1e6)

        if torch.isnan(total).any():
            warnings.warn("NaN in CombinedLoss total, falling back to 1.0")
            total = torch.tensor(1.0, device=main_pred.device, requires_grad=True)

        losses["loss"] = total
        return losses


# ---------------------------
# FireDetectionLoss wrapper
# ---------------------------
class FireDetectionLoss(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        # accept config style keys used in various places
        loss_type = config.get("loss_type") or config.get("name") or config.get("loss_name") or "combined"
        dice_weight = config.get("dice_weight", 0.6)
        focal_weight = config.get("focal_weight", 0.4)
        focal_alpha = config.get("focal_alpha", 0.8)
        focal_gamma = config.get("focal_gamma", 3.0)
        aux_weight = config.get("aux_weight", 0.3)
        reduction = config.get("reduction", "mean")

        if loss_type in ("combined", "dice+focal", "dice_focal", "dice_focal_combined"):
            self.loss_fn = CombinedLoss(dice_weight=dice_weight, focal_weight=focal_weight,
                                        dice_smooth=config.get("dice_smooth", 1.0),
                                        focal_alpha=focal_alpha, focal_gamma=focal_gamma,
                                        aux_weight=aux_weight, reduction=reduction)
        elif loss_type in ("dice", "dice_loss"):
            self.loss_fn = DiceLoss(smooth=config.get("dice_smooth", 1.0), reduction=reduction)
        elif loss_type in ("focal", "focal_loss"):
            self.loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction=reduction)
        elif loss_type in ("iou", "iou_loss"):
            self.loss_fn = IoULoss(smooth=config.get("iou_smooth", 1.0), reduction=reduction)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, predictions: Union[Dict[str, torch.Tensor], torch.Tensor], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.loss_fn(predictions, targets)


def create_loss_function(config: Dict) -> FireDetectionLoss:
    return FireDetectionLoss(config)


# ---------------------------
# Quick self-test if run standalone
# ---------------------------
if __name__ == "__main__":
    print("Self-test for dice_focal.py")
    cfg = {"loss_type": "combined", "dice_weight": 0.6, "focal_weight": 0.4,
           "focal_alpha": 0.8, "focal_gamma": 2.0, "aux_weight": 0.3}
    loss = create_loss_function(cfg)

    B, C, H, W = 2, 1, 64, 64
    preds = {"out": torch.randn(B, C, H, W), "aux": [torch.randn(B, C, H, W)]}
    targets_idx = torch.randint(0, 2, (B, H, W))

    res = loss(preds, targets_idx)
    for k, v in res.items():
        try:
            print(f"{k}: {float(v):.6f}")
        except Exception:
            print(f"{k}: tensor shape {v.shape}")
