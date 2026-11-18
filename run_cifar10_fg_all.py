"""
Script: run_cifar10_fg_all.py
Purpose: Evaluates all 6 trained CIFAR-10 models under the Foreground-Only (Object without background) test condition on local CPU.
"""

import os
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import numpy as np

# Optimize CPU parallelism
torch.set_num_threads(4)

from run_osr_experiments import ResNet18_OSR, StandardEvalDataset, compute_osr_metrics
from eval_foreground_only import evaluate_foreground_only

def main():
    device = torch.device("cpu")
    data_root = "/home/aashu/Thesis/osr/data"
    results_dir = Path("/home/aashu/Thesis/osr/osr/backmix_results")
    
    print("=" * 70)
    print("CIFAR-10 FOREGROUND-ONLY (OBJECT WITHOUT BACKGROUND) EVALUATION (CPU)")
    print("=" * 70)
    
    # 1. Load CIFAR-10 Test Dataset once
    print("[1/3] Loading CIFAR-10 test set and creating known/unknown splits...")
    test_base = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True)
    known_classes = [0, 1, 2, 3, 4, 5]
    target_map = {c: i for i, c in enumerate(known_classes)}
    k_indices = [i for i, target in enumerate(test_base.targets) if target in known_classes]
    u_indices = [i for i, target in enumerate(test_base.targets) if target not in known_classes]
    
    known_ds = StandardEvalDataset(test_base, k_indices, target_map=target_map, is_known=True)
    unknown_ds = StandardEvalDataset(test_base, u_indices, target_map=target_map, is_known=False)

    known_loader = DataLoader(known_ds, batch_size=256, shuffle=False, num_workers=2)
    unknown_loader = DataLoader(unknown_ds, batch_size=256, shuffle=False, num_workers=2)
    print(f"      Known test images: {len(known_ds)} | Unknown test images: {len(unknown_ds)}")

    # 2. Define models to evaluate
    model_configs = [
        ("plain", "cifar10_plain_best.pth", False),
        ("backmix", "cifar10_backmix_best.pth", False),
        ("m2_neutral", "cifar10_m2_neutral_best.pth", False),
        ("m3_noise", "cifar10_m3_noise_best.pth", False),
        ("m4_scale", "cifar10_m4_scale_best.pth", False),
        ("m5_proposed", "cifar10_m5_proposed_best.pth", False),
    ]

    summary_results = {}
    
    # 3. Iterate through models
    print("\n[2/3] Evaluating models...")
    for idx, (name, ckpt_name, enlarge) in enumerate(model_configs, 1):
        ckpt_path = results_dir / ckpt_name
        if not ckpt_path.exists():
            print(f"[{idx}/{len(model_configs)}] Warning: {ckpt_path} not found. Skipping.")
            continue
            
        print(f"\n--- [{idx}/{len(model_configs)}] Evaluating {name} ({ckpt_name}) ---")
        t_start = time.time()
        
        model = ResNet18_OSR(num_classes=6)
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        metrics = evaluate_foreground_only(model, known_loader, unknown_loader, device, enlarge=enlarge)
        elapsed = time.time() - t_start
        
        metrics["time_sec"] = round(elapsed, 1)
        summary_results[name] = metrics
        
        # Save individual result
        out_file = results_dir / f"cifar10_{name}_fg_metrics.json"
        with open(out_file, "w") as f:
            json.dump(metrics, f, indent=2)
            
        print(f"      Finished in {elapsed:.1f}s")
        print(f"      Accuracy: {metrics['Accuracy']:.2f}% | AUROC: {metrics['AUROC']:.2f}% | OSCR: {metrics['OSCR']:.2f}% | TNR@95: {metrics['TNR@95']:.2f}%")

    # 4. Save consolidated summary
    print("\n[3/3] Saving summary to backmix_results/cifar10_fg_summary.json...")
    summary_path = results_dir / "cifar10_fg_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_results, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL 6 MODELS COMPLETED UNDER FOREGROUND-ONLY TESTING")
    print("=" * 70)
    print(json.dumps(summary_results, indent=2))

if __name__ == "__main__":
    main()
