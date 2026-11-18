"""
Script: eval_background_only.py
Purpose: Evaluates OSR models using ONLY the background (object removed/masked out).
Direct counterpart to eval_foreground_only.py, following standard literature (Background Challenge / Only-BG).
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


def extract_background_batch(imgs, cam, logits, mask_ratio=0.3):
    """
    Given batch of images (B, C, H, W) and CAM maps (B, num_classes, 8, 8):
    1. Select CAM of the top predicted class y_hat = argmax(logits).
    2. Interpolate CAM to (H, W).
    3. Generate binary object mask M by thresholding top mask_ratio (default 30%) pixels.
    4. Invert to obtain background mask M_bg = (1 - M).
    5. Return background-only batch: X_bg = M_bg * imgs (object region zeroed out).
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
    fg_mask = (cam_high >= thresh).float().unsqueeze(1)  # (B, 1, H, W)
    bg_mask = 1.0 - fg_mask                              # (B, 1, H, W)

    return imgs * bg_mask


def evaluate_background_only(model, known_loader, unknown_loader, device, mask_ratio=0.3):
    """
    Two-pass evaluation:
    Pass 1: Get CAM and remove object region.
    Pass 2: Classify background-only image and compute OSR metrics.
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
            # Remove object: keep only background
            bg_imgs = extract_background_batch(imgs, cam, logits, mask_ratio=mask_ratio)
            # Pass 2: predict on background only
            bg_logits, _ = model(bg_imgs)
            probs = F.softmax(bg_logits, dim=1)
            conf, pred = probs.max(dim=1)

            known_confs.extend(conf.cpu().numpy())
            known_preds.extend(pred.cpu().numpy())
            known_targets.extend(targets.numpy())

        # Unknown classes
        for imgs, _, _ in unknown_loader:
            imgs = imgs.to(device)
            logits, cam = model(imgs)
            bg_imgs = extract_background_batch(imgs, cam, logits, mask_ratio=mask_ratio)
            bg_logits, _ = model(bg_imgs)
            probs = F.softmax(bg_logits, dim=1)
            conf, _ = probs.max(dim=1)

            unknown_confs.extend(conf.cpu().numpy())

    return compute_osr_metrics(
        np.array(known_confs),
        np.array(unknown_confs),
        known_preds,
        known_targets
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate OSR Models with Background-Only Test Images (Object Removed)")
    parser.add_argument("--model-path", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar+10", "svhn"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mask-ratio", type=float, default=0.3, help="Fraction of top CAM pixels to remove as object")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Evaluating {args.model_path} on {args.dataset} (Device: {device}, Object Removed / BG-Only)...")

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
    metrics = evaluate_background_only(model, known_loader, unknown_loader, device, mask_ratio=args.mask_ratio)
    t_elapsed = time.time() - t0

    print(f"\n[Completed in {t_elapsed:.2f}s]")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
