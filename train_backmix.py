import argparse
import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

# Local imports
from backmix import BMixDataset

# ---------------------------------------------------------------------------
# Utility functions (metrics, dataset split, etc.) – defined inline to keep a
# single‑file script for simplicity. In a real project they'd live in utils.py.
# ---------------------------------------------------------------------------

def get_known_unknown_split(dataset, known_classes, unknown_classes):
    """Return indices for known and unknown subsets.
    `known_classes` and `unknown_classes` are lists of class indices.
    """
    known_idx = [i for i, (_, label) in enumerate(dataset) if label in known_classes]
    unknown_idx = [i for i, (_, label) in enumerate(dataset) if label in unknown_classes]
    return known_idx, unknown_idx

def compute_auroc(scores, labels):
    """Binary AUROC where `labels` == 0 for known, 1 for unknown.
    `scores` is a 1‑D tensor of the maximum softmax confidence for each sample.
    """
    from sklearn.metrics import roc_auc_score
    scores_np = scores.cpu().numpy()
    labels_np = labels.cpu().numpy()
    return roc_auc_score(labels_np, scores_np)

def compute_oscr(confidence, pred_labels, true_labels, known_mask):
    """Open Set Classification Rate (OSCR) as defined in the BackMix paper.
    This implementation follows the common formulation:
      * Accuracy on known samples (where known_mask==True)
      * True Positive Rate on unknown samples vs. rejection threshold.
    For simplicity we compute the area under the OSCR curve by sweeping a
    threshold over the confidence values.
    """
    import numpy as np
    # Sort by confidence descending
    order = np.argsort(-confidence)
    confid_sorted = confidence[order]
    pred_sorted = pred_labels[order]
    true_sorted = true_labels[order]
    known_sorted = known_mask[order]

    tp, fp, tn, fn = 0, 0, 0, 0
    oscr_points = []
    total_known = known_mask.sum()
    total_unknown = (~known_mask).sum()
    for i in range(len(confidence)):
        if known_sorted[i]:
            # known sample
            if pred_sorted[i] == true_sorted[i]:
                tp += 1
            else:
                fn += 1
        else:
            # unknown sample – should be rejected
            if pred_sorted[i] == true_sorted[i]:
                # incorrectly accepted as known class
                fp += 1
            else:
                tn += 1
        # compute rates
        tpr = tp / total_known if total_known > 0 else 0.0
        fpr = fp / total_unknown if total_unknown > 0 else 0.0
        oscr_points.append((fpr, tpr))
    # Approximate area under curve via trapezoidal rule
    oscr = 0.0
    prev_fpr, prev_tpr = 0.0, 0.0
    for fpr, tpr in oscr_points:
        oscr += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr, prev_tpr = fpr, tpr
    return oscr

def evaluate(model, dataloader, device, known_classes, unknown_classes):
    model.eval()
    all_conf = []
    all_pred = []
    all_true = []
    all_known = []
    with torch.no_grad():
        for batch in dataloader:
            # Support both raw datasets (img, label) and BMixDataset (raw, mixed, label, idx)
            if len(batch) == 2:
                imgs, labels = batch
            elif len(batch) == 4:
                imgs, _, labels, _ = batch
            else:
                raise ValueError(f'Unexpected batch size {len(batch)}')
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            max_conf, pred = probs.max(dim=1)
            all_conf.append(max_conf)
            all_pred.append(pred)
            all_true.append(labels.to(device))
            known_mask = torch.tensor([l.item() in known_classes for l in labels], device=device)
            all_known.append(known_mask)
    conf = torch.cat(all_conf)
    pred = torch.cat(all_pred)
    true = torch.cat(all_true)
    known_mask = torch.cat(all_known)
    auroc = compute_auroc(conf, (~known_mask).int())
    oscr = compute_oscr(conf.cpu().numpy(), pred.cpu().numpy(), true.cpu().numpy(), known_mask.cpu().numpy())
    acc = (pred[known_mask] == true[known_mask]).float().mean().item()
    return {"AUROC": auroc, "OSCR": oscr, "Accuracy": acc}

# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train BackMix on multiple OSR datasets")
    parser.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100"],
                        help="List of datasets to run (cifar10, cifar100, imagenet30)")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--output-dir", type=str, default="./backmix_results", help="Where to save logs and models")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # For each dataset
    for ds_name in args.datasets:
        print(f"\n=== Processing dataset: {ds_name} ===")
        if ds_name.lower() == "cifar10":
            num_classes = 10
            train_base = torchvision.datasets.CIFAR10(root="./data", train=True, download=True,
                                                    transform=None)
            test_base = torchvision.datasets.CIFAR10(root="./data", train=False, download=True,
                                                   transform=None)
            known = list(range(0, 6))  # first 6 classes as known (as in the paper)
            unknown = list(range(6, 10))
        elif ds_name.lower() == "cifar100":
            num_classes = 100
            train_base = torchvision.datasets.CIFAR100(root="./data", train=True, download=True,
                                                     transform=None)
            test_base = torchvision.datasets.CIFAR100(root="./data", train=False, download=True,
                                                    transform=T.ToTensor())
            known = list(range(0, 60))  # 60 known, 40 unknown (paper config)
            unknown = list(range(60, 100))
        elif ds_name.lower() == "imagenet30":
            # Placeholder – ImageNet‑30 not bundled with torchvision.
            # Expect user to provide a folder ./imagenet30 with subfolders per class.
            img_root = Path("./imagenet30")
            if not img_root.exists():
                raise RuntimeError("ImageNet‑30 dataset not found at ./imagenet30")
            train_base = torchvision.datasets.ImageFolder(root=str(img_root / "train"), transform=T.ToTensor())
            test_base = torchvision.datasets.ImageFolder(root=str(img_root / "val"), transform=T.ToTensor())
            num_classes = 30
            known = list(range(0, 20))
            unknown = list(range(20, 30))
        else:
            raise ValueError(f"Unsupported dataset {ds_name}")

        # Wrap with BackMix augmentation for training
        train_dataset = BMixDataset(train_base)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        # Validation uses original images (no mixing) – we reuse the base dataset
        val_loader = DataLoader(BMixDataset(test_base), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # Model – ResNet‑18 for all experiments (as per paper for CIFAR, also works for ImageNet‑30)
        model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

        # Training loop
        best_acc = 0.0
        log_path = Path(args.output_dir) / f"{ds_name}_log.jsonl"
        with open(log_path, "w") as log_f:
            for epoch in range(1, args.epochs + 1):
                model.train()
                epoch_loss = 0.0
                for raw, mixed, labels, _ in train_loader:
                    mixed = mixed.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(mixed)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * mixed.size(0)
                epoch_loss /= len(train_loader.dataset)
                scheduler.step()

                # Evaluate on validation set (using original images)
                metrics = evaluate(model, val_loader, device, known, unknown)
                log_entry = {
                    "epoch": epoch,
                    "train_loss": epoch_loss,
                    "val_accuracy": metrics["Accuracy"],
                    "val_auroc": metrics["AUROC"],
                    "val_oscr": metrics["OSCR"],
                    "timestamp": time.time(),
                }
                log_f.write(json.dumps(log_entry) + "\n")
                log_f.flush()
                print(f"Epoch {epoch:03d}: loss={epoch_loss:.4f} acc={metrics['Accuracy']:.4f} AUROC={metrics['AUROC']:.4f} OSCR={metrics['OSCR']:.4f}")
                if metrics["Accuracy"] > best_acc:
                    best_acc = metrics["Accuracy"]
                    torch.save(model.state_dict(), Path(args.output_dir) / f"{ds_name}_best.pth")

        # Final evaluation on test set (mixed images as in training)
        test_metrics = evaluate(model, val_loader, device, known, unknown)
        final_path = Path(args.output_dir) / f"{ds_name}_final_metrics.json"
        with open(final_path, "w") as jf:
            json.dump(test_metrics, jf, indent=2)
        print(f"Completed {ds_name}. Final metrics saved to {final_path}")

if __name__ == "__main__":
    main()
