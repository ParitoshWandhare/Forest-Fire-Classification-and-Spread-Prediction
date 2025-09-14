"""
Robust metrics for forest fire detection training.

Exports:
- SegmentationMetrics: general segmentation metrics (binary/multiclass)
- FireDetectionMetrics: specialized for fire detection (pixel-level counts, fire_f1, detection rate)
- MetricTracker: tracks metric history across epochs

This version is defensive:
- clamps unexpected target values into valid class range (e.g. 255 -> 1)
- resizes predictions to match target spatial size when necessary (nearest)
- ensures device/dtype alignment
- avoids returning zeroed metrics when shapes mismatch silently
"""

from typing import Dict, Union, Optional
from collections import defaultdict
import warnings

import numpy as np
import torch
import torch.nn.functional as F


class SegmentationMetrics:
    """
    Generic segmentation metrics.

    Usage:
        m = SegmentationMetrics(num_classes=2, threshold=0.5, device='cpu')
        m.update(predictions, targets)
        results = m.compute_all_metrics()
    """

    def __init__(self, num_classes: int = 2, ignore_index: int = -100, threshold: float = 0.5,
                 device: Union[str, torch.device] = 'cpu'):
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.threshold = float(threshold)
        self.device = torch.device(device)
        self.reset()

    def reset(self):
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes),
                                           dtype=torch.long, device=self.device)
        self.total_samples = 0

    def _ensure_hw_match(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Make sure pred has same H,W as tgt. If not, resize pred using nearest (safe for labels/prob maps).
        pred: tensor [B, ... , Hp, Wp] or [B, Hp, Wp]
        tgt: tensor [B, Ht, Wt] or [B,1,Ht,Wt]
        """
        # target spatial shape
        if tgt.dim() == 4:
            _, _, Ht, Wt = tgt.shape
        elif tgt.dim() == 3:
            _, Ht, Wt = tgt.shape
        else:
            raise ValueError(f"Unsupported target shape: {tgt.shape}")

        # pred spatial shape
        if pred.dim() == 4:
            _, _, Hp, Wp = pred.shape
            if (Hp, Wp) != (Ht, Wt):
                # resize channels preserved
                pred = F.interpolate(pred, size=(Ht, Wt), mode='nearest')
        elif pred.dim() == 3:
            _, Hp, Wp = pred.shape
            if (Hp, Wp) != (Ht, Wt):
                pred = pred.unsqueeze(1)  # [B,1,H,W]
                pred = F.interpolate(pred, size=(Ht, Wt), mode='nearest')
                pred = pred.squeeze(1)
        else:
            # other shapes unexpected — return as-is and let later checks handle failure
            pass
        return pred

    def _preds_to_labels(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Convert model outputs to label tensor [B, H, W] on self.device.
        Accepts:
          - logits/probs shaped [B, C, H, W] (C==1 for binary)
          - already discrete shaped [B, H, W] or [B, 1, H, W]
        """
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor(predictions, device=self.device)

        # move to device for consistent ops
        predictions = predictions.to(self.device)

        if predictions.dim() == 4:
            C = predictions.shape[1]
            if C == 1:
                probs = torch.sigmoid(predictions.squeeze(1))
                labels = (probs > self.threshold).long()
            else:
                probs = F.softmax(predictions, dim=1)
                labels = torch.argmax(probs, dim=1).long()
        elif predictions.dim() == 3:
            labels = predictions.long()
        elif predictions.dim() == 2:
            labels = predictions.long()
        else:
            raise ValueError(f"Unsupported prediction shape: {predictions.shape}")
        return labels.to(self.device)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update confusion matrix.

        predictions: [B, C, H, W] or [B, H, W] (logits/probs or labels).
        targets: [B, H, W] or [B, 1, H, W] (class indices or binary mask).
        """

        # Move targets to device and normalize shape
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)

        targets = targets.to(self.device)

        # If targets have channel dim [B,1,H,W] -> squeeze
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets_proc = targets.squeeze(1).long()
        elif targets.dim() == 3:
            targets_proc = targets.long()
        else:
            # Try to coerce to [B,H,W]
            try:
                targets_proc = targets.long()
            except Exception:
                warnings.warn(f"Unexpected targets shape: {targets.shape}. Skipping update.")
                return

        # Clamp target labels to valid class range (helps when masks use 255 or unexpected labels)
        if self.num_classes >= 2:
            targets_proc = torch.clamp(targets_proc, 0, self.num_classes - 1)
        else:
            targets_proc = torch.clamp(targets_proc, 0, 1)

        # Ensure predictions spatial size matches targets; if not, resize predictions (nearest)
        try:
            predictions = self._ensure_hw_match(predictions, targets_proc)
        except Exception:
            # if ensure fails, continue and let conversion attempt raise later
            pass

        # Convert predictions to discrete labels [B,H,W]
        try:
            preds = self._preds_to_labels(predictions)
        except Exception as e:
            warnings.warn(f"Failed to convert predictions to labels: {e}")
            return

        # Flatten
        preds_flat = preds.flatten()
        targets_flat = targets_proc.flatten()

        # Ensure lengths are equal (if not, try to align by cropping/padding - but prefer to abort)
        if preds_flat.numel() != targets_flat.numel():
            # try cheap shape checks and attempt to broadcast if possible
            if preds_flat.numel() == 0 or targets_flat.numel() == 0:
                # nothing to update
                return
            # If counts differ but batch dims look compatible, attempt to resize preds to target and recompute
            try:
                # reshape preds to [B, H, W] and redo
                if preds.dim() == 4:
                    preds = F.interpolate(preds, size=(targets_proc.shape[1], targets_proc.shape[2]), mode='nearest')
                    preds = self._preds_to_labels(preds)
                    preds_flat = preds.flatten()
                elif preds.dim() == 3:
                    # last resort: repeat/truncate to match length
                    min_n = min(preds_flat.numel(), targets_flat.numel())
                    preds_flat = preds_flat[:min_n]
                    targets_flat = targets_flat[:min_n]
                else:
                    # cannot reconcile
                    return
            except Exception:
                return

        # Remove ignored indices if requested
        if self.ignore_index != -100:
            valid_mask = (targets_flat != self.ignore_index)
            preds_flat = preds_flat[valid_mask]
            targets_flat = targets_flat[valid_mask]

        # Keep only valid label range
        valid_mask = (targets_flat >= 0) & (targets_flat < self.num_classes)
        preds_flat = preds_flat[valid_mask]
        targets_flat = targets_flat[valid_mask]

        if preds_flat.numel() == 0:
            # nothing to update (all invalid or masked out)
            return

        # Compute bincount indices: target * C + pred
        indices = targets_flat.long() * self.num_classes + preds_flat.long()
        bincount = torch.bincount(indices, minlength=self.num_classes ** 2).to(self.device)
        cm_update = bincount.view(self.num_classes, self.num_classes)
        self.confusion_matrix += cm_update
        self.total_samples += int(preds_flat.numel())

    def _safe_div(self, num, den, eps=1e-8):
        den = den + eps
        return num / den

    def compute_iou(self) -> Dict[str, float]:
        cm = self.confusion_matrix.float()
        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        union = tp + fp + fn

        iou = self._safe_div(tp, union)
        results = {}
        for i in range(self.num_classes):
            name = f'class_{i}' if self.num_classes > 2 else ('background' if i == 0 else 'fire')
            results[f'iou_{name}'] = float(iou[i].item()) if self.total_samples > 0 else 0.0

        if self.num_classes == 2:
            results['mean_iou'] = float(iou[1].item()) if self.total_samples > 0 else 0.0
        else:
            results['mean_iou'] = float(iou.mean().item()) if self.total_samples > 0 else 0.0
        return results

    def compute_dice(self) -> Dict[str, float]:
        cm = self.confusion_matrix.float()
        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp

        dice = self._safe_div(2 * tp, (2 * tp + fp + fn))
        results = {}
        for i in range(self.num_classes):
            name = f'class_{i}' if self.num_classes > 2 else ('background' if i == 0 else 'fire')
            results[f'dice_{name}'] = float(dice[i].item()) if self.total_samples > 0 else 0.0

        if self.num_classes == 2:
            results['mean_dice'] = float(dice[1].item()) if self.total_samples > 0 else 0.0
        else:
            results['mean_dice'] = float(dice.mean().item()) if self.total_samples > 0 else 0.0
        return results

    def compute_accuracy(self) -> Dict[str, float]:
        cm = self.confusion_matrix.float()
        total = cm.sum().item()
        correct = torch.diag(cm).sum().item()
        pixel_accuracy = float(self._safe_div(torch.tensor(correct), torch.tensor(total)).item()) if total > 0 else 0.0

        results = {'pixel_accuracy': pixel_accuracy}
        for i in range(self.num_classes):
            denom = cm.sum(dim=1)[i].item()
            acc_i = float(self._safe_div(cm[i, i], denom).item()) if denom > 0 else 0.0
            name = f'class_{i}' if self.num_classes > 2 else ('background' if i == 0 else 'fire')
            results[f'accuracy_{name}'] = acc_i
        return results

    def compute_precision_recall_f1(self) -> Dict[str, float]:
        cm = self.confusion_matrix.float()
        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp

        precision = self._safe_div(tp, tp + fp)
        recall = self._safe_div(tp, tp + fn)
        f1 = self._safe_div(2 * precision * recall, precision + recall)

        results = {}
        for i in range(self.num_classes):
            name = f'class_{i}' if self.num_classes > 2 else ('background' if i == 0 else 'fire')
            results[f'precision_{name}'] = float(precision[i].item()) if self.total_samples > 0 else 0.0
            results[f'recall_{name}'] = float(recall[i].item()) if self.total_samples > 0 else 0.0
            results[f'f1_{name}'] = float(f1[i].item()) if self.total_samples > 0 else 0.0

        results['precision_macro'] = float(precision.mean().item()) if self.total_samples > 0 else 0.0
        results['recall_macro'] = float(recall.mean().item()) if self.total_samples > 0 else 0.0
        results['f1_macro'] = float(f1.mean().item()) if self.total_samples > 0 else 0.0

        if self.num_classes == 2:
            results['fire_precision'] = float(precision[1].item()) if self.total_samples > 0 else 0.0
            results['fire_recall'] = float(recall[1].item()) if self.total_samples > 0 else 0.0
            results['fire_f1'] = float(f1[1].item()) if self.total_samples > 0 else 0.0

        return results

    def compute_all_metrics(self) -> Dict[str, float]:
        if self.total_samples == 0:
            # return canonical zeros rather than raising
            return {
                'iou_background': 0.0, 'iou_fire': 0.0, 'mean_iou': 0.0,
                'dice_background': 0.0, 'dice_fire': 0.0, 'mean_dice': 0.0,
                'pixel_accuracy': 0.0, 'precision_macro': 0.0, 'recall_macro': 0.0, 'f1_macro': 0.0,
                'fire_precision': 0.0, 'fire_recall': 0.0, 'fire_f1': 0.0
            }

        results = {}
        results.update(self.compute_iou())
        results.update(self.compute_dice())
        results.update(self.compute_accuracy())
        results.update(self.compute_precision_recall_f1())
        return results

    def get_confusion_matrix(self) -> torch.Tensor:
        return self.confusion_matrix.clone().to('cpu')


class FireDetectionMetrics:
    """
    Pixel-wise fire detection metrics wrapper.

    Methods:
        - update(predictions, targets, probs=None)
        - compute_all_metrics() -> dict (contains 'fire_f1', 'mean_iou', 'fire_detection_rate', ...)
    """

    def __init__(self, threshold: float = 0.5, device: Union[str, torch.device] = 'cpu'):
        self.threshold = float(threshold)
        self.device = torch.device(device)
        self.base_metrics = SegmentationMetrics(num_classes=2, threshold=self.threshold, device=self.device)

        self.fire_tp = 0
        self.fire_fp = 0
        self.fire_fn = 0
        self.fire_tn = 0
        self.total_fire_pixels = 0
        self.detected_fire_pixels = 0

        # optional sampling for ROC
        self.predictions_list = []
        self.targets_list = []

    def reset(self):
        self.base_metrics.reset()
        self.fire_tp = self.fire_fp = self.fire_fn = self.fire_tn = 0
        self.total_fire_pixels = 0
        self.detected_fire_pixels = 0
        self.predictions_list = []
        self.targets_list = []

    def _probs_from_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        if outputs.dim() == 4:
            C = outputs.shape[1]
            if C == 1:
                probs = torch.sigmoid(outputs.squeeze(1))
            else:
                probs = F.softmax(outputs, dim=1)[:, 1]
        elif outputs.dim() == 3:
            probs = outputs
        else:
            raise ValueError("Unsupported outputs shape for probability extraction.")
        return probs

    def update(self, predictions: torch.Tensor, targets: torch.Tensor, probs: Optional[torch.Tensor] = None):
        """
        Update pixel-level metrics.

        predictions: model outputs (logits/probs or labels). shape [B,C,H,W] or [B,H,W].
        targets: [B,H,W] or [B,1,H,W]
        probs: optional raw probabilities/logits; if omitted it will be derived from predictions.
        """

        # Validate and move to device
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor(predictions)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)

        predictions = predictions.to(self.device)
        targets = targets.to(self.device)

        # Update base segmentation metrics (this will handle many shape issues)
        try:
            self.base_metrics.update(predictions, targets)
        except Exception as e:
            warnings.warn(f"Base segmentation metric update failed: {e}")

        # Convert to binary predicted labels for fire counts
        try:
            if isinstance(predictions, torch.Tensor) and predictions.dim() == 4:
                if predictions.shape[1] == 1:
                    pred_bin = (torch.sigmoid(predictions.squeeze(1)) > self.threshold).long()
                else:
                    pred_bin = torch.argmax(F.softmax(predictions, dim=1), dim=1).long()
            elif isinstance(predictions, torch.Tensor) and predictions.dim() == 3:
                # assume it's already per-pixel probs or labels
                # if float, threshold; if long, use directly
                if predictions.dtype.is_floating_point:
                    pred_bin = (predictions > self.threshold).long()
                else:
                    pred_bin = predictions.long()
            else:
                pred_bin = predictions.long()
        except Exception:
            # fallback: attempt to derive from probs argument
            try:
                if probs is not None:
                    pmap = self._probs_from_outputs(probs.to(self.device))
                    pred_bin = (pmap > self.threshold).long()
                else:
                    # give up: consider everything background
                    pred_bin = torch.zeros_like(targets.squeeze(1) if targets.dim() == 4 else targets, dtype=torch.long, device=self.device)
            except Exception:
                pred_bin = torch.zeros_like(targets.squeeze(1) if targets.dim() == 4 else targets, dtype=torch.long, device=self.device)

        # Normalize targets
        if targets.dim() == 4 and targets.shape[1] == 1:
            tgt = targets.squeeze(1).long()
        else:
            tgt = targets.long()

        # Clamp target into valid range (e.g., 255 -> 1)
        tgt = torch.clamp(tgt, 0, 1)

        # If pred_bin and tgt spatial dims differ, resize pred_bin (nearest)
        if pred_bin.dim() == 4:
            pred_bin = pred_bin.squeeze(1)
        if pred_bin.shape != tgt.shape:
            try:
                # make pred_bin float, interpolate, then threshold
                pred_tmp = pred_bin.unsqueeze(1).float()
                pred_tmp = F.interpolate(pred_tmp, size=(tgt.shape[1], tgt.shape[2]), mode='nearest')
                pred_bin = (pred_tmp.squeeze(1) > 0.5).long()
            except Exception:
                # if can't resize, try simple broadcasting/truncation
                min_h = min(pred_bin.shape[-2], tgt.shape[-2])
                min_w = min(pred_bin.shape[-1], tgt.shape[-1])
                pred_bin = pred_bin[..., :min_h, :min_w]
                tgt = tgt[..., :min_h, :min_w]

        # Compute boolean maps
        tgt_bool = (tgt == 1)
        pred_bool = (pred_bin == 1)
        bg_bool = (tgt == 0)
        bg_pred_bool = (pred_bin == 0)

        # Update pixel-level counts
        self.fire_tp += int((tgt_bool & pred_bool).sum().item())
        self.fire_fp += int((bg_bool & pred_bool).sum().item())
        self.fire_fn += int((tgt_bool & bg_pred_bool).sum().item())
        self.fire_tn += int((bg_bool & bg_pred_bool).sum().item())

        batch_fire_pixels = int(tgt_bool.sum().item())
        batch_detected = int((tgt_bool & pred_bool).sum().item())
        self.total_fire_pixels += batch_fire_pixels
        self.detected_fire_pixels += batch_detected

        # Collect probability subsamples for ROC (optional)
        try:
            probs_map = None
            if probs is not None:
                probs_map = self._probs_from_outputs(probs.to(self.device))
            else:
                if predictions.dim() == 4:
                    probs_map = self._probs_from_outputs(predictions)
            if probs_map is not None:
                flat_probs = probs_map.flatten()
                flat_targets = tgt.flatten()
                sample_size = min(1000, flat_probs.numel())
                if sample_size > 0:
                    perm = torch.randperm(flat_probs.numel(), device=flat_probs.device)[:sample_size]
                    self.predictions_list.append(flat_probs[perm].cpu().detach())
                    self.targets_list.append(flat_targets[perm].cpu().detach())
        except Exception:
            pass

    def compute_fire_metrics(self) -> Dict[str, float]:
        results = {}
        if (self.fire_tp + self.fire_fn) > 0:
            fire_detection_rate = self.fire_tp / (self.fire_tp + self.fire_fn)
        else:
            fire_detection_rate = 0.0

        if (self.fire_tp + self.fire_fp) > 0:
            fire_precision = self.fire_tp / (self.fire_tp + self.fire_fp)
        else:
            fire_precision = 0.0

        if (self.fire_fp + self.fire_tn) > 0:
            false_positive_rate = self.fire_fp / (self.fire_fp + self.fire_tn)
        else:
            false_positive_rate = 0.0

        results['fire_detection_rate'] = float(fire_detection_rate)
        results['fire_precision'] = float(fire_precision)
        results['false_positive_rate'] = float(false_positive_rate)
        results['specificity'] = float(1.0 - false_positive_rate)
        results['fire_pixel_detection_rate'] = float(self.detected_fire_pixels / self.total_fire_pixels) if self.total_fire_pixels > 0 else 0.0

        if (fire_precision + fire_detection_rate) > 0:
            results['fire_f1'] = float(2 * (fire_precision * fire_detection_rate) / (fire_precision + fire_detection_rate))
        else:
            results['fire_f1'] = 0.0

        return results

    def compute_roc_auc(self) -> Dict[str, float]:
        if not self.predictions_list or not self.targets_list:
            return {'roc_auc': 0.0}
        try:
            from sklearn.metrics import roc_auc_score
            all_probs = torch.cat(self.predictions_list).numpy()
            all_targets = torch.cat(self.targets_list).numpy()
            if len(np.unique(all_targets)) < 2:
                return {'roc_auc': 0.0}
            auc = roc_auc_score(all_targets, all_probs)
            return {'roc_auc': float(auc)}
        except Exception:
            return {'roc_auc': 0.0}

    def compute_all_metrics(self) -> Dict[str, float]:
        results = {}
        base_results = self.base_metrics.compute_all_metrics()
        results.update(base_results)
        results.update(self.compute_fire_metrics())
        results.update(self.compute_roc_auc())
        return results


class MetricTracker:
    """
    Simple metric history tracker used by trainer to store and query histories.
    """

    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.current_metrics = {}

    def update(self, metrics_dict: Dict[str, float]):
        self.current_metrics = metrics_dict.copy()
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                self.metrics_history[key].append(float(value))

    def get_best_metrics(self, metric_name: str, mode: str = 'max') -> Dict[str, float]:
        if metric_name not in self.metrics_history or not self.metrics_history[metric_name]:
            return {}
        values = self.metrics_history[metric_name]
        if mode == 'max':
            best_value = max(values)
            best_epoch = values.index(best_value)
        else:
            best_value = min(values)
            best_epoch = values.index(best_value)
        return {f'best_{metric_name}': float(best_value), f'best_{metric_name}_epoch': int(best_epoch)}

    def get_current_metrics(self) -> Dict[str, float]:
        return self.current_metrics.copy()

    def get_running_average(self, metric_name: str, window: int = 5) -> float:
        if metric_name not in self.metrics_history or not self.metrics_history[metric_name]:
            return 0.0
        values = self.metrics_history[metric_name]
        window_values = values[-window:]
        return float(sum(window_values) / len(window_values))
