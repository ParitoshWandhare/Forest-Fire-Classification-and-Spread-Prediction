# diagnostics.py
import torch, numpy as np
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()
model.to(device)

threshold = 0.5
total_pixels = 0
total_fire_pixels_gt = 0
total_fire_pixels_pred = 0
batches = 0

with torch.no_grad():
    for batch_idx, (images, masks) in enumerate(val_loader):
        images = images.to(device)
        # masks may be [B,1,H,W] or [B,H,W] - unify
        if masks.dim() == 4 and masks.shape[1] == 1:
            masks_proc = masks.squeeze(1).long().to(device)
        else:
            masks_proc = masks.long().to(device)

        out = model(images)['out']
        if out.shape[1] == 1:
            probs = torch.sigmoid(out).squeeze(1)
        else:
            probs = torch.softmax(out, dim=1)[:, 1]

        preds_bin = (probs > threshold).long()

        total_pixels += masks_proc.numel()
        total_fire_pixels_gt += (masks_proc == 1).sum().item()
        total_fire_pixels_pred += preds_bin.sum().item()
        batches += 1

        if batch_idx < 2:
            print(f"batch {batch_idx}: GT_fire={int((masks_proc==1).sum())}, PRED_fire={int(preds_bin.sum())}, avg_prob={probs.mean().item():.6f}")

print("SUMMARY")
print("Batches:", batches)
print("GT fire pixels:", total_fire_pixels_gt, "({:.6f}%)".format(total_fire_pixels_gt/total_pixels*100 if total_pixels else 0))
print("PRED fire pixels:", total_fire_pixels_pred, "({:.6f}%)".format(total_fire_pixels_pred/total_pixels*100 if total_pixels else 0))
print("Avg predicted fire fraction per pixel:", total_fire_pixels_pred/total_pixels if total_pixels else 0)
