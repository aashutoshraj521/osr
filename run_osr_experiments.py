import os
import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from backmix import BMixDataset
from fg_stochastic_dataset import ForegroundNoiseDataset

# Set optimal thread count for local multi-core CPU
torch.set_num_threads(4)

# ---------------------------------------------------------------------------
# ResNet-18 adapted for OSR with Conv1x1-GAP-Softmax (as described in Section IV)
# ---------------------------------------------------------------------------
class ResNet18_OSR(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        net = torchvision.models.resnet18(weights=None)
        # Adapt for 32x32 images
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()

        self.features = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4
        )
        # Classification head: Conv1x1 -> GAP -> Softmax
        self.cam_conv = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        feat = self.features(x)
        cam = self.cam_conv(feat)            # (B, num_classes, 4, 4)
        logits = cam.mean(dim=(2, 3))         # (B, num_classes) via GAP
        if cam.shape[-1] != 8:
            cam = F.interpolate(cam, (8, 8), mode='bilinear', align_corners=False)
        return logits, cam

# ---------------------------------------------------------------------------
# Dataset wrapper for testing (pure original images without BackMix)
# ---------------------------------------------------------------------------
class StandardEvalDataset(Dataset):
    def __init__(self, base_dataset, indices, target_map=None, is_known=True):
        self.base = base_dataset
        self.indices = list(indices)
        self.target_map = target_map or {}
        self.is_known = is_known
        self.transform = T.Compose([
            T.Resize(32),
            T.CenterCrop(32),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        orig_idx = self.indices[i]
        item = self.base[orig_idx]
        if isinstance(item, (tuple, list)):
            img, label = item[0], item[1]
        else:
            raise ValueError(f"Unexpected item: {type(item)}")
        
        if not isinstance(img, torch.Tensor):
            img = self.transform(img)
            
        mapped_label = self.target_map.get(label, label if self.is_known else -1)
        return img, mapped_label, self.is_known

# Subset wrapper for training to remap class labels to 0..K-1
class MappedLabelDataset(Dataset):
    def __init__(self, base_dataset, indices, target_map):
        self.base = base_dataset
        self.indices = list(indices)
        self.target_map = target_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        orig_idx = self.indices[i]
        item = self.base[orig_idx]
        if isinstance(item, (tuple, list)):
            img, label = item[0], item[1]
        else:
            raise ValueError(f"Unexpected item: {type(item)}")
        return img, self.target_map[label]

# ---------------------------------------------------------------------------
# Open Set Recognition Metrics
# ---------------------------------------------------------------------------
def compute_osr_metrics(known_confs, unknown_confs, known_preds, known_targets):
    """
    Computes standard OSR evaluation metrics:
    - Closed-set Accuracy (%)
    - AUROC (%)
    - TNR@95 (True Negative Rate at 95% TPR) (%)
    - OSCR (Open Set Classification Rate) (%)
    - AUIN (Area under PR curve, In-distribution positive) (%)
    - AUOUT (Area under PR curve, Out-of-distribution positive) (%)
    """
    # 1. Closed-set Accuracy on known classes
    known_preds_np = np.array(known_preds)
    known_targets_np = np.array(known_targets)
    acc = np.mean(known_preds_np == known_targets_np) * 100.0

    # 2. AUROC
    y_true = np.concatenate([np.ones_like(known_confs), np.zeros_like(unknown_confs)])
    scores = np.concatenate([known_confs, unknown_confs])
    auroc = roc_auc_score(y_true, scores) * 100.0

    # 3. TNR@95 (Threshold where 95% of known samples are retained)
    sorted_known = np.sort(known_confs)
    thresh_idx = int(np.floor(0.05 * len(sorted_known)))
    thresh_95 = sorted_known[thresh_idx]
    tnr95 = np.mean(unknown_confs < thresh_95) * 100.0

    # 4. AUIN and AUOUT
    prec_in, rec_in, _ = precision_recall_curve(y_true, scores)
    auin = auc(rec_in, prec_in) * 100.0

    prec_out, rec_out, _ = precision_recall_curve(1 - y_true, -scores)
    auout = auc(rec_out, prec_out) * 100.0

    # 5. OSCR (Open Set Classification Rate)
    # CCR(t) = Correct Classification Rate on known at threshold t
    # FPR(t) = False Positive Rate on unknown at threshold t
    is_correct = (known_preds_np == known_targets_np)
    all_scores = np.unique(scores)
    # Subsample thresholds if too many points for speed
    if len(all_scores) > 1000:
        thresholds = np.linspace(all_scores.min(), all_scores.max(), 1000)
    else:
        thresholds = all_scores

    ccrs, fprs = [], []
    for t in thresholds:
        ccr = np.sum(is_correct & (known_confs >= t)) / len(known_confs)
        fpr = np.sum(unknown_confs >= t) / len(unknown_confs)
        ccrs.append(ccr)
        fprs.append(fpr)

    # Sort points by FPR ascending
    points = sorted(zip(fprs, ccrs), key=lambda p: p[0])
    fpr_sorted = [p[0] for p in points]
    ccr_sorted = [p[1] for p in points]
    oscr = auc(fpr_sorted, ccr_sorted) * 100.0

    return {
        "Accuracy": round(float(acc), 2),
        "AUROC": round(float(auroc), 2),
        "TNR@95": round(float(tnr95), 2),
        "OSCR": round(float(oscr), 2),
        "AUIN": round(float(auin), 2),
        "AUOUT": round(float(auout), 2),
    }

# ---------------------------------------------------------------------------
# Evaluation on test set
# ---------------------------------------------------------------------------
def evaluate_model(model, known_loader, unknown_loader, device):
    model.eval()
    known_confs, known_preds, known_targets = [], [], []
    unknown_confs = []

    with torch.no_grad():
        for imgs, targets, _ in known_loader:
            imgs = imgs.to(device)
            logits, _ = model(imgs)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            known_confs.extend(conf.cpu().numpy())
            known_preds.extend(pred.cpu().numpy())
            known_targets.extend(targets.numpy())

        for imgs, _, _ in unknown_loader:
            imgs = imgs.to(device)
            logits, _ = model(imgs)
            probs = F.softmax(logits, dim=1)
            conf, _ = probs.max(dim=1)
            unknown_confs.extend(conf.cpu().numpy())

    return compute_osr_metrics(
        np.array(known_confs),
        np.array(unknown_confs),
        known_preds,
        known_targets
    )

def get_device_and_backend():
    # Try TPU via torch_xla first
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"[Device] Using TPU device: {device}", flush=True)
        return device, "tpu", xm
    except Exception:
        pass

    # Try NVIDIA CUDA GPU next
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using CUDA GPU: {torch.cuda.get_device_name(0)}", flush=True)
        return device, "cuda", None

    # Fallback to multi-core CPU
    device = torch.device("cpu")
    print("[Device] Using CPU", flush=True)
    return device, "cpu", None

def save_model_checkpoint(state, path, backend, xm):
    if backend == "tpu" and xm is not None:
        xm.save(state, str(path))
    else:
        torch.save(state, str(path))

def git_push_progress(output_dir, commit_msg):
    try:
        os.system(f"git add {output_dir}/*.json {output_dir}/*.jsonl {output_dir}/*_best.pth {output_dir}/*_latest.pth 2>/dev/null")
        os.system(f'git commit -m "{commit_msg}" 2>/dev/null')
        # Pull before push to handle any concurrent commits
        os.system("git pull --rebase origin main 2>/dev/null")
        ret = os.system("git push origin main 2>/dev/null")
        if ret == 0:
            print(f"[Auto-Push] Successfully pushed to GitHub: {commit_msg}", flush=True)
        else:
            print(f"[Auto-Push Notice] Push returned non-zero code {ret} (check git remote / token)", flush=True)
    except Exception as e:
        print(f"[Auto-Push Warning] Could not push to GitHub: {e}", flush=True)

# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------
def train_single_experiment(dataset_name, method, epochs, batch_size, lr, data_root, output_dir, device, backend="cpu", xm=None, auto_push=False, log_interval=0, noise_type="balanced_uniform", static_noise=True, shared_noise=True):
    print(f"\n========================================================", flush=True)
    print(f" Dataset: {dataset_name} | Method: {method} | Epochs: {epochs}", flush=True)
    print(f"========================================================", flush=True)
    
    # 1. Dataset setup
    if dataset_name.lower() == "cifar10":
        # 6 known classes (0..5), 4 unknown classes (6..9)
        train_full = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True)
        test_full = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True)
        known_classes = list(range(6))
        unknown_classes = list(range(6, 10))
        target_map = {c: i for i, c in enumerate(known_classes)}
        num_classes = len(known_classes)

        train_known_indices = [i for i, (_, y) in enumerate(train_full) if y in known_classes]
        test_known_indices = [i for i, (_, y) in enumerate(test_full) if y in known_classes]
        test_unknown_indices = [i for i, (_, y) in enumerate(test_full) if y in unknown_classes]

        train_base = MappedLabelDataset(train_full, train_known_indices, target_map)
        known_test_ds = StandardEvalDataset(test_full, test_known_indices, target_map, is_known=True)
        unknown_test_ds = StandardEvalDataset(test_full, test_unknown_indices, {}, is_known=False)

    elif dataset_name.lower() == "svhn":
        train_full = torchvision.datasets.SVHN(root=data_root, split='train', download=True)
        test_full = torchvision.datasets.SVHN(root=data_root, split='test', download=True)
        known_classes = list(range(6))
        unknown_classes = list(range(6, 10))
        target_map = {c: i for i, c in enumerate(known_classes)}
        num_classes = len(known_classes)

        train_known_indices = [i for i, (_, y) in enumerate(train_full) if y in known_classes]
        test_known_indices = [i for i, (_, y) in enumerate(test_full) if y in known_classes]
        test_unknown_indices = [i for i, (_, y) in enumerate(test_full) if y in unknown_classes]

        train_base = MappedLabelDataset(train_full, train_known_indices, target_map)
        known_test_ds = StandardEvalDataset(test_full, test_known_indices, target_map, is_known=True)
        unknown_test_ds = StandardEvalDataset(test_full, test_unknown_indices, {}, is_known=False)

    elif dataset_name.lower() == "cifar+10":
        # 4 non-animal classes from CIFAR-10 as known: 0 (airplane), 1 (automobile), 8 (ship), 9 (truck)
        cifar10_train = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True)
        cifar10_test = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True)
        cifar100_test = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True)

        known_classes = [0, 1, 8, 9]
        target_map = {c: i for i, c in enumerate(known_classes)}
        num_classes = len(known_classes)

        train_known_indices = [i for i, (_, y) in enumerate(cifar10_train) if y in known_classes]
        test_known_indices = [i for i, (_, y) in enumerate(cifar10_test) if y in known_classes]
        
        # 10 animal classes from CIFAR-100 (e.g. classes 0..9)
        test_unknown_indices = [i for i, (_, y) in enumerate(cifar100_test) if y in list(range(10))]

        train_base = MappedLabelDataset(cifar10_train, train_known_indices, target_map)
        known_test_ds = StandardEvalDataset(cifar10_test, test_known_indices, target_map, is_known=True)
        unknown_test_ds = StandardEvalDataset(cifar100_test, test_unknown_indices, {}, is_known=False)

    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    # Build Training DataLoader
    m_lower = method.lower()
    if m_lower == "backmix":
        train_ds = BMixDataset(train_base)
    elif m_lower == "m2_neutral":
        train_ds = ForegroundNoiseDataset(train_base, target_size=32, margin=2, enlarge=False, noise_bg=False)
    elif m_lower == "m3_noise":
        train_ds = ForegroundNoiseDataset(train_base, target_size=32, margin=2, enlarge=False, noise_bg=True, noise_type=noise_type, static_noise=static_noise, shared_single_noise=shared_noise)
    elif m_lower == "m4_scale":
        train_ds = ForegroundNoiseDataset(train_base, target_size=32, margin=2, enlarge=True, noise_bg=False)
    elif m_lower in ["m5_proposed", "proposed"]:
        train_ds = ForegroundNoiseDataset(train_base, target_size=32, margin=2, enlarge=True, noise_bg=True, noise_type=noise_type, static_noise=static_noise, shared_single_noise=shared_noise)
    else:
        # Baseline Plain* (standard augmentations without mixing)
        class StandardAugDataset(Dataset):
            def __init__(self, base):
                self.base = base
                self.aug = T.Compose([
                    T.Resize(32),
                    T.CenterCrop(32),
                    T.RandomHorizontalFlip(),
                    T.RandomCrop(size=32, padding=4),
                    T.ToTensor(),
                    T.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
                ])
            def __len__(self): return len(self.base)
            def __getitem__(self, idx):
                img, label = self.base[idx]
                return self.aug(img), label, idx

        train_ds = StandardAugDataset(train_base)

    use_cuda = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=use_cuda)
    known_eval_loader = DataLoader(known_test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=use_cuda)
    unknown_eval_loader = DataLoader(unknown_test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=use_cuda)

    # 2. Model, Optimizer, Loss, Scheduler
    base_model = ResNet18_OSR(num_classes=num_classes)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"[Multi-GPU] Using DataParallel across {torch.cuda.device_count()} GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}", flush=True)
        model = nn.DataParallel(base_model).to(device)
    else:
        model = base_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Helper to access underlying module when using DataParallel
    def get_raw_model(m):
        return m.module if isinstance(m, nn.DataParallel) else m

    # 3. Training Loop with Resume Support
    best_auroc = 0.0
    best_metrics = {}
    start_epoch = 1
    log_file = Path(output_dir) / f"{dataset_name}_{method}_log.jsonl"
    latest_ckpt = Path(output_dir) / f"{dataset_name}_{method}_latest.pth"
    best_ckpt = Path(output_dir) / f"{dataset_name}_{method}_best.pth"

    if latest_ckpt.exists():
        checkpoint = torch.load(latest_ckpt, map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        get_raw_model(model).load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_auroc = checkpoint.get("best_auroc", 0.0)
        best_metrics = checkpoint.get("best_metrics", {})
        print(f"Resuming {dataset_name} [{method}] from epoch {start_epoch} via latest checkpoint (Best AUROC: {best_auroc:.2f}%)", flush=True)
    elif log_file.exists() and best_ckpt.exists():
        try:
            with open(log_file, "r") as f:
                lines = [l.strip() for l in f if l.strip()]
            if lines:
                last_entry = json.loads(lines[-1])
                last_epoch = last_entry.get("epoch", 0)
                if last_epoch < epochs:
                    start_epoch = last_epoch + 1
                    for l in lines:
                        e = json.loads(l)
                        if e.get("AUROC", 0.0) > best_auroc:
                            best_auroc = e["AUROC"]
                            best_metrics = e
                    state = torch.load(best_ckpt, map_location=device)
                    raw_s = state.get("model_state", state)
                    get_raw_model(model).load_state_dict(raw_s)
                    for _ in range(last_epoch):
                        scheduler.step()
                    print(f"Resuming {dataset_name} [{method}] from epoch {start_epoch} via best checkpoint & log (Best AUROC: {best_auroc:.2f}%)", flush=True)
        except Exception as e:
            print(f"[Resume Warning] Could not parse log/best checkpoint: {e}. Starting from epoch 1.", flush=True)

    mode = "a" if (start_epoch > 1 and log_file.exists()) else "w"
    with open(log_file, mode) as lf:
        for epoch in range(start_epoch, epochs + 1):
            t_start = time.time()
            model.train()
            total_loss = 0.0
            n_samples = 0
            total_batches = len(train_loader)

            for batch_idx, batch in enumerate(train_loader):
                if method.lower() == "backmix":
                    raw_img, mix_img, labels, idxs = batch
                    inputs = mix_img.to(device)
                elif method.lower() in ["m2_neutral", "m3_noise", "m4_scale", "m5_proposed", "proposed"]:
                    raw_img, proc_img, labels, idxs = batch
                    inputs = proc_img.to(device)
                else:
                    inputs, labels, idxs = batch
                    inputs = inputs.to(device)

                labels = labels.to(device)
                optimizer.zero_grad()
                logits, cam = model(inputs)
                loss = criterion(logits, labels)
                loss.backward()

                if backend == "tpu" and xm is not None:
                    xm.optimizer_step(optimizer)
                    xm.mark_step()
                else:
                    optimizer.step()

                # For BackMix & Foreground-Noise models: update CAM segmentation bank with network foreground estimation
                if method.lower() in ["backmix", "m2_neutral", "m3_noise", "m4_scale", "m5_proposed", "proposed"]:
                    with torch.no_grad():
                        prob_cam = F.softmax(cam, dim=1) # (B, num_classes, 8, 8)
                        fg_mask = prob_cam[torch.arange(len(labels)), labels].detach().cpu()
                        train_ds.soft_seg(idxs, fg_mask)

                total_loss += loss.item() * inputs.size(0)
                n_samples += inputs.size(0)

                if log_interval > 0 and ((batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches):
                    print(f"  [Epoch {epoch:03d}/{epochs:03d}] Batch {batch_idx+1:03d}/{total_batches:03d} - Loss: {loss.item():.4f}", flush=True)

            scheduler.step()
            train_loss = total_loss / n_samples
            elapsed = time.time() - t_start

            # Evaluate on every epoch
            metrics = evaluate_model(model, known_eval_loader, unknown_eval_loader, device)
            raw_state = get_raw_model(model).state_dict()
            if metrics["AUROC"] > best_auroc:
                best_auroc = metrics["AUROC"]
                best_metrics = metrics.copy()
                save_model_checkpoint(raw_state, Path(output_dir) / f"{dataset_name}_{method}_best.pth", backend, xm)

            # Save checkpoint for resume
            save_model_checkpoint({
                "epoch": epoch,
                "model_state": raw_state,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_auroc": best_auroc,
                "best_metrics": best_metrics
            }, latest_ckpt, backend, xm)

            log_entry = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "epoch_time_sec": round(elapsed, 1),
                **metrics
            }
            lf.write(json.dumps(log_entry) + "\n")
            lf.flush()
            print(f"--> Epoch {epoch:03d}/{epochs:03d} [{elapsed:.1f}s] Loss: {train_loss:.4f} | Acc: {metrics['Accuracy']:.2f}% | AUROC: {metrics['AUROC']:.2f}% | OSCR: {metrics['OSCR']:.2f}% | TNR@95: {metrics['TNR@95']:.2f}% (Best AUROC: {best_auroc:.2f}%)", flush=True)

            if auto_push and (epoch % 10 == 0):
                git_push_progress(output_dir, f"chore: progress {dataset_name} [{method}] epoch {epoch} (AUROC {metrics['AUROC']}%)")

    # Final evaluation with best checkpoint
    best_ckpt = Path(output_dir) / f"{dataset_name}_{method}_best.pth"
    if best_ckpt.exists():
        get_raw_model(model).load_state_dict(torch.load(best_ckpt, map_location=device))
    final_metrics = evaluate_model(model, known_eval_loader, unknown_eval_loader, device)
    
    with open(Path(output_dir) / f"{dataset_name}_{method}_final_metrics.json", "w") as jf:
        json.dump(final_metrics, jf, indent=2)

    print(f"\nFinal Best Results for {dataset_name} [{method}]:", flush=True)
    print(json.dumps(final_metrics, indent=2), flush=True)

    if auto_push:
        git_push_progress(output_dir, f"chore: complete {dataset_name} [{method}] - AUROC {final_metrics['AUROC']}%")

    return final_metrics

# ---------------------------------------------------------------------------
# Main Runner across all datasets and methods
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BackMix Open Set Recognition Benchmark")
    parser.add_argument("--datasets", nargs="+", default=["cifar10", "cifar+10", "svhn"],
                        help="Datasets to evaluate (cifar10, cifar+10, svhn)")
    parser.add_argument("--methods", nargs="+", default=["plain", "backmix", "m5_proposed"],
                        help="Methods to run: plain, backmix, m2_neutral, m3_noise, m4_scale, m5_proposed")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./backmix_results")
    parser.add_argument("--auto-push", action="store_true", help="Auto push progress and results to git")
    parser.add_argument("--log-interval", type=int, default=0, help="Print progress every N batches (default: 0, prints only per-epoch summary)")
    parser.add_argument("--noise-type", type=str, default="balanced_uniform", choices=["balanced_uniform", "gaussian", "uniform"],
                        help="Noise type for M3 and M5 background (default: balanced_uniform)")
    parser.add_argument("--static-noise", action="store_true", default=True,
                        help="Hold background noise static across all epochs (default: True)")
    parser.add_argument("--dynamic-noise", action="store_false", dest="static_noise",
                        help="Regenerate dynamic noise on every epoch")
    parser.add_argument("--shared-noise", action="store_true", default=True,
                        help="Use ONE single noise canvas across all images and epochs (default: True)")
    parser.add_argument("--per-sample-noise", action="store_false", dest="shared_noise",
                        help="Use a separate static noise canvas per sample")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device, backend, xm = get_device_and_backend()
    print(f"Running on device: {device} (backend: {backend}) | Output directory: {args.output_dir}", flush=True)

    summary_results = {}
    for ds in args.datasets:
        summary_results[ds] = {}
        for method in args.methods:
            res = train_single_experiment(
                dataset_name=ds,
                method=method,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                data_root=args.data_root,
                output_dir=args.output_dir,
                device=device,
                backend=backend,
                xm=xm,
                auto_push=args.auto_push,
                log_interval=args.log_interval,
                noise_type=args.noise_type,
                static_noise=args.static_noise,
                shared_noise=args.shared_noise
            )
            summary_results[ds][method] = res

    # Save grand summary
    summary_path = Path(args.output_dir) / "all_experiments_summary.json"
    with open(summary_path, "w") as sf:
        json.dump(summary_results, sf, indent=2)

    if args.auto_push:
        git_push_progress(args.output_dir, "chore: grand summary of all experiments complete")

    print("\n========================================================")
    print("           GRAND SUMMARY OF ALL EXPERIMENTS             ")
    print("========================================================")
    print(json.dumps(summary_results, indent=2))

if __name__ == "__main__":
    main()
