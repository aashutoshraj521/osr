"""
Script: eval_all_conditions.py
Purpose: Evaluates an OSR checkpoint across all 3 established testing conditions:
  1. Raw Test Images (Full image with natural background)
  2. Foreground-Only (Salient object isolated via CAM, background removed: X_fg = M * X)
  3. Background-Only (Salient object removed via inverted CAM, background kept: X_bg = (1 - M) * X)
"""

import os
import argparse
import json
import time
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from pathlib import Path

from run_osr_experiments import ResNet18_OSR, StandardEvalDataset, evaluate_model, compute_osr_metrics
from eval_foreground_only import evaluate_foreground_only
from eval_background_only import evaluate_background_only


def evaluate_model_all_conditions(model, known_loader, unknown_loader, device):
    """Run evaluation across all conditions on the given DataLoader pair."""
    results = {}
    
    # Condition 1: Raw Images
    t0 = time.time()
    results["raw"] = evaluate_model(model, known_loader, unknown_loader, device)
    results["raw"]["time_sec"] = round(time.time() - t0, 2)
    
    # Condition 2: Foreground-Only (Unscaled, original position)
    t0 = time.time()
    results["foreground_only"] = evaluate_foreground_only(model, known_loader, unknown_loader, device, enlarge=False)
    results["foreground_only"]["time_sec"] = round(time.time() - t0, 2)
    
    # Condition 3: Foreground-Only (Aspect-ratio enlarged & centered)
    t0 = time.time()
    results["foreground_enlarged"] = evaluate_foreground_only(model, known_loader, unknown_loader, device, enlarge=True)
    results["foreground_enlarged"]["time_sec"] = round(time.time() - t0, 2)
    
    # Condition 4: Background-Only (Object removed)
    t0 = time.time()
    results["background_only"] = evaluate_background_only(model, known_loader, unknown_loader, device, mask_ratio=0.3)
    results["background_only"]["time_sec"] = round(time.time() - t0, 2)
    
    return results


def print_table(results, model_name="Model"):
    print("\n" + "=" * 80)
    print(f"EVALUATION SUMMARY ACROSS ALL TESTING CONDITIONS: {model_name}")
    print("=" * 80)
    print(f"{'Condition':<25} | {'Accuracy':<10} | {'AUROC':<10} | {'OSCR':<10} | {'TNR@95':<10} | {'Time':<6}")
    print("-" * 80)
    for cond, name in [
        ("raw", "1. Raw (Full Image)"),
        ("foreground_only", "2. FG-Only (Unscaled)"),
        ("foreground_enlarged", "3. FG-Only (Enlarged)"),
        ("background_only", "4. Background-Only")
    ]:
        m = results[cond]
        print(f"{name:<25} | {m['Accuracy']:>8.2f}% | {m['AUROC']:>8.2f}% | {m['OSCR']:>8.2f}% | {m['TNR@95']:>8.2f}% | {m['time_sec']:>4.1f}s")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model across all 3 testing conditions")
    parser.add_argument("--model-path", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar+10", "svhn"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--auto-push", action="store_true", help="Automatically commit and push results to GitHub")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading {args.dataset} test set (Device: {device})...")

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

    # Load Model
    model = ResNet18_OSR(num_classes=num_classes)
    ckpt = torch.load(args.model_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    res = evaluate_model_all_conditions(model, known_loader, unknown_loader, device)
    model_name = os.path.basename(args.model_path)
    print_table(res, model_name=model_name)

    # Automatically save results JSON
    out_json = args.output_json
    if not out_json:
        stem = Path(args.model_path).stem
        out_json = os.path.join("backmix_results", f"{stem}_all_conditions.json")
    
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Results saved to {out_json}")

    # Auto push if requested
    if args.auto_push:
        print("[Auto-Push] Pushing evaluation results and checkpoints to GitHub...", flush=True)
        os.system("git add backmix_results/*.json backmix_results/*.jsonl backmix_results/*_best.pth 2>/dev/null")
        msg = f"chore: 4-condition evaluation for {model_name}"
        os.system(f'git commit -m "{msg}" 2>/dev/null')
        os.system("git pull --rebase origin main 2>/dev/null")
        ret = os.system("git push origin main 2>/dev/null")
        if ret == 0:
            print(f"[Auto-Push] Successfully pushed {model_name} evaluation to GitHub!", flush=True)
        else:
            print(f"[Auto-Push Notice] Push returned exit code {ret} (check git remote / token)", flush=True)


if __name__ == "__main__":
    main()
