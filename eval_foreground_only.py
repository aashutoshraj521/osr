"""
Script: eval_foreground_only.py
Purpose: Evaluates OSR models using ONLY the isolated object (foreground) with background removed.
Supports both:
  1. Direct CAM-masking (background zeroed/neutralized: X_fg = M * X)
  2. Scale-normalized foreground (object cropped, enlarged, and placed on neutral canvas)
"""

import os
import argparse
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# Import architecture & metrics from run_osr_experiments
from run_osr_experiments import ResNet18_OSR, StandardEvalDataset, compute_osr_metrics


def extract_foreground_batch(imgs, cam, logits, mask_ratio=0.3, enlarge=False, target_size=32, margin=2):
    """
    Given batch of images (B, C, H, W) and CAM maps (B, num_classes, 8, 8):
    1. Select CAM of the top predicted class y_hat = argmax(logits).
    2. Interpolate CAM to (H, W).
    3. Generate binary mask M by thresholding top mask_ratio (default 30%) pixels.
    4. Return foreground-only batch (background set to 0 neutral).
    """
    B, C, H, W = imgs.shape
    device = imgs.device
    pred_classes = logits.argmax(dim=1)  # (B,)

    # Extract CAM for predicted class: (B, 8, 8)
    cam_pred = cam[torch.arange(B, device=device), pred_classes]

    # Bilinear upsample to (H, W)
    cam_high = F.interpolate(
        cam_pred.unsqueeze(1),
        size=(H, W),
        mode='bilinear',
        align_corners=False
    ).squeeze(1)  # (B, H, W)

    # Top-K threshold per image
    k = max(1, int(mask_ratio * H * W))
    flat = cam_high.view(B, -1)
    thresh = torch.topk(flat, k, dim=1)[0][:, -1].view(B, 1, 1)
    masks = (cam_high >= thresh).float().unsqueeze(1)  # (B, 1, H, W)

    if not enlarge:
        # Standard foreground masking: background zeroed
        return imgs * masks

    # Scale-enlarged foreground (object bounded, resized to canvas)
    out_imgs = torch.zeros_like(imgs)
    target_dim = target_size - margin

    for i in range(B):
        m = masks[i, 0]
        img = imgs[i]
        active = torch.nonzero(m)
        if active.shape[0] < 4:
            out_imgs[i] = img * m.unsqueeze(0)
            continue

        ymin = active[:, 0].min().item()
        ymax = active[:, 0].max().item()
        xmin = active[:, 1].min().item()
        xmax = active[:, 1].max().item()
        h = max(2, ymax - ymin + 1)
        w = max(2, xmax - xmin + 1)

        crop_img = img[:, ymin:ymax+1, xmin:xmax+1].unsqueeze(0)
        crop_m = m[ymin:ymax+1, xmin:xmax+1].unsqueeze(0).unsqueeze(0)

        s = target_dim / max(w, h)
        new_w = max(1, min(target_size, int(round(s * w))))
        new_h = max(1, min(target_size, int(round(s * h))))

        res_img = F.interpolate(crop_img, size=(new_h, new_w), mode='bilinear', align_corners=False)[0]
        res_m = F.interpolate(crop_m, size=(new_h, new_w), mode='bilinear', align_corners=False)[0, 0]
        res_m = (res_m >= 0.5).float().unsqueeze(0)

        py = (target_size - new_h) // 2
        px = (target_size - new_w) // 2
        out_imgs[i, :, py:py+new_h, px:px+new_w] = res_img * res_m

    return out_imgs


def evaluate_foreground_only(model, known_loader, unknown_loader, device, enlarge=False):
    """
    Two-pass evaluation:
    Pass 1: Get CAM and isolate object foreground.
    Pass 2: Classify foreground-only image and compute OSR metrics.
    """
    model.eval()
    known_confs, known_preds, known_targets = [], [], []
    unknown_confs = []

    with torch.no_grad():
        # Known classes
        for imgs, targets, _ in known_loader:
            imgs = imgs.to(device)
            # Pass 1: get CAM
            logits, cam = model(imgs)
            # Isolate foreground
            fg_imgs = extract_foreground_batch(imgs, cam, logits, enlarge=enlarge)
            # Pass 2: predict on foreground only
            fg_logits, _ = model(fg_imgs)
            probs = F.softmax(fg_logits, dim=1)
            conf, pred = probs.max(dim=1)

            known_confs.extend(conf.cpu().numpy())
            known_preds.extend(pred.cpu().numpy())
            known_targets.extend(targets.numpy())

        # Unknown classes
        for imgs, _, _ in unknown_loader:
            imgs = imgs.to(device)
            logits, cam = model(imgs)
            fg_imgs = extract_foreground_batch(imgs, cam, logits, enlarge=enlarge)
            fg_logits, _ = model(fg_imgs)
            probs = F.softmax(fg_logits, dim=1)
            conf, _ = probs.max(dim=1)

            unknown_confs.extend(conf.cpu().numpy())

    return compute_osr_metrics(
        np.array(known_confs),
        np.array(unknown_confs),
        known_preds,
        known_targets
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate OSR Models with Foreground-Only Test Images")
    parser.add_argument("--model-path", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar+10", "svhn"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--enlarge", action="store_true", help="Scale foreground to canvas size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Evaluating {args.model_path} on {args.dataset} (Device: {device}, Enlarge: {args.enlarge})...")

    # Build dataset splits
    if args.dataset == "cifar10":
        test_base = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True)
        known_classes = [0, 1, 2, 3, 4, 5]
        target_map = {c: i for i, c in enumerate(known_classes)}
        k_indices = [i for i, target in enumerate(test_base.targets) if target in known_classes]
        u_indices = [i for i, target in enumerate(test_base.targets) if target not in known_classes]
        num_classes = len(known_classes)
    elif args.dataset == "svhn":
        test_base = torchvision.datasets.SVHN(root=args.data_root, split="test", download=True)
        known_classes = [0, 1, 2, 3, 4, 5]
        target_map = {c: i for i, c in enumerate(known_classes)}
        k_indices = [i for i, target in enumerate(test_base.labels) if target in known_classes]
        u_indices = [i for i, target in enumerate(test_base.labels) if target not in known_classes]
        num_classes = len(known_classes)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented in quick runner")

    known_ds = StandardEvalDataset(test_base, k_indices, target_map=target_map, is_known=True)
    unknown_ds = StandardEvalDataset(test_base, u_indices, target_map=target_map, is_known=False)

    known_loader = DataLoader(known_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    unknown_loader = DataLoader(unknown_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Load model
    model = ResNet18_OSR(num_classes=num_classes)
    ckpt = torch.load(args.model_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)

    t0 = time.time()
    metrics = evaluate_foreground_only(model, known_loader, unknown_loader, device, enlarge=args.enlarge)
    t_elapsed = time.time() - t0

    print(f"\n[Completed in {t_elapsed:.2f}s]")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
