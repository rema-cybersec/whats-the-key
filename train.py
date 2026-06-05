"""
train.py
--------
Training loop for key signature detection.

Runs one or all k folds, logs results, saves the best checkpoint.
Also supports an XGBoost baseline (no PyTorch needed for that path).

Usage:
    # Train fold 0 with default config
    python train.py

    # Train all 5 folds and report mean ± std accuracy
    python train.py --all-folds

    # Override config values on the fly
    python train.py --set model.architecture=cnn_channels training.lr=0.0005

    # Run XGBoost baseline (ignores most of the config)
    python train.py --xgboost

    # Use a different config file
    python train.py --config my_config.yaml
"""

import os
import sys
import copy
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from dataset import build_loaders, ChromaDataset, get_fold_indices
from models  import build_model

# ── Config helpers ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """
    Apply --set key.subkey=value overrides to the config dict.
    Values are auto-cast: "true"/"false" -> bool, numbers -> float/int.

    Example: apply_overrides(cfg, ["training.lr=0.0005", "model.dropout=0.2"])
    """
    for override in overrides:
        key_path, raw_value = override.split("=", 1)
        keys   = key_path.strip().split(".")
        value  = _cast(raw_value.strip())

        node = cfg
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
        print(f"  Config override: {key_path} = {value}")

    return cfg


def _cast(value: str):
    if value.lower() == "true":  return True
    if value.lower() == "false": return False
    if value.lower() == "null":  return None
    try:    return int(value)
    except: pass
    try:    return float(value)
    except: pass
    return value


# ── Optimizer + scheduler factories ──────────────────────────────────────────

def build_optimizer(cfg: dict, model: nn.Module) -> torch.optim.Optimizer:
    lr   = cfg["training"]["lr"]
    name = cfg["training"]["optimizer"].lower()
    return {
        "adam":    lambda: Adam(model.parameters(),  lr=lr),
        "adamw":   lambda: AdamW(model.parameters(), lr=lr, weight_decay=1e-4),
        "sgd":     lambda: SGD(model.parameters(),   lr=lr, momentum=0.9, weight_decay=1e-4),
        "rmsprop": lambda: RMSprop(model.parameters(), lr=lr),
    }[name]()


def build_scheduler(cfg: dict, optimizer):
    name   = (cfg["training"].get("scheduler") or "none").lower()
    epochs = cfg["training"]["epochs"]
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    if name == "step":
        return StepLR(optimizer, step_size=epochs // 3, gamma=0.3)
    return None   # "none"


# ── One epoch of training ─────────────────────────────────────────────────────

def run_epoch(
    model:      nn.Module,
    loader,
    criterion:  nn.Module,
    optimizer,
    device:     torch.device,
    grad_clip,
    is_train:   bool,
) -> tuple[float, float]:
    """
    Run one full pass over loader.
    Returns (mean_loss, accuracy_percent).
    """
    model.train(is_train)
    total_loss = 0.0
    correct    = 0
    total      = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss   = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(dim=1) == y).sum().item()
            total      += len(y)

    return total_loss / total, 100.0 * correct / total


# ── Train one fold ────────────────────────────────────────────────────────────

def train_fold(cfg: dict, fold: int, device: torch.device) -> dict:
    """
    Train for one fold. Returns a dict of results including best val accuracy.
    """
    fold_cfg = copy.deepcopy(cfg)
    fold_cfg["data"]["val_fold"] = fold

    print(f"\n{'─'*55}")
    arch = cfg["model"]["architecture"]
    print(f"  Architecture : {arch}   Fold : {fold+1}/{cfg['data']['n_folds']}")
    print(f"{'─'*55}")

    # ── Data ──
    train_loader, val_loader = build_loaders(fold_cfg)

    # ── Model ──
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters   : {total_params:,}")

    # ── Training setup ──
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    grad_clip = cfg["training"].get("grad_clip")

    epochs      = cfg["training"]["epochs"]
    print_every = cfg["logging"].get("print_every", 5)
    save_best   = cfg["logging"].get("save_best", True)
    log_dir     = cfg["logging"].get("log_dir", "runs/")

    best_val_acc  = 0.0
    best_epoch    = 0
    history       = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip, is_train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, grad_clip, is_train=False
        )

        if scheduler:
            scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss":   val_loss,   "val_acc":   val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            if save_best:
                os.makedirs(log_dir, exist_ok=True)
                ckpt_path = os.path.join(log_dir, f"{arch}_fold{fold}_best.pt")
                torch.save(model.state_dict(), ckpt_path)

        if epoch % print_every == 0 or epoch == epochs:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:3d}/{epochs}"
                f"  train_loss={train_loss:.4f}  train_acc={train_acc:5.1f}%"
                f"  val_loss={val_loss:.4f}  val_acc={val_acc:5.1f}%"
                f"  lr={lr_now:.2e}"
                + (" ←best" if epoch == best_epoch else "")
            )

    print(f"\n  Best val accuracy: {best_val_acc:.1f}% (epoch {best_epoch})")

    return {
        "fold":         fold,
        "best_val_acc": best_val_acc,
        "best_epoch":   best_epoch,
        "history":      history,
    }


# ── XGBoost baseline ──────────────────────────────────────────────────────────

def run_xgboost(cfg: dict):
    """
    Train and evaluate an XGBoost classifier using GroupKFold.
    Uses raw chroma rows (flat mode) directly from the CSV.
    """
    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost not installed. Run: pip install xgboost")
        return

    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GroupKFold

    print("\n" + "="*55)
    print("  XGBoost Baseline")
    print("="*55)

    df = pd.read_csv(cfg["data"]["csv_path"])

    CHROMA_COLS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    X         = df[CHROMA_COLS].values
    y         = df["key"].values
    groups    = df["song_id"].values
    n_folds   = cfg["data"]["n_folds"]

    gkf       = GroupKFold(n_splits=n_folds)
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        # XGBoost requires contiguous labels 0..N-1.
        # Remap based on classes present in the train split for this fold.
        train_classes = np.sort(np.unique(y[train_idx]))
        label_map     = {k: i for i, k in enumerate(train_classes)}
        y_train = np.array([label_map[k] for k in y[train_idx]])
        # Val labels: any class not seen in train gets mapped to -1 (will be wrong, which is correct)
        y_val   = np.array([label_map.get(k, -1) for k in y[val_idx]])

        clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="mlogloss",
            verbosity=0,
            random_state=42,
        )
        clf.fit(X[train_idx], y_train)
        preds    = clf.predict(X[val_idx])
        acc      = 100.0 * accuracy_score(y_val, preds)
        fold_accs.append(acc)
        print(f"  Fold {fold+1}/{n_folds}  val_acc={acc:.1f}%")

    mean = np.mean(fold_accs)
    std  = np.std(fold_accs)
    print(f"\n  XGBoost k-fold: {mean:.1f}% ± {std:.1f}%")
    return {"mean_acc": mean, "std_acc": std, "fold_accs": fold_accs}


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(results: list[dict], cfg: dict):
    arch     = cfg["model"]["architecture"]
    accs     = [r["best_val_acc"] for r in results]
    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)

    print("\n" + "="*55)
    print(f"  {arch}  —  {len(results)}-fold summary")
    print("="*55)
    for r in results:
        print(f"  Fold {r['fold']+1}:  {r['best_val_acc']:.1f}%  (best epoch {r['best_epoch']})")
    print(f"  {'─'*35}")
    print(f"  Mean : {mean_acc:.1f}%  ±  {std_acc:.1f}%")
    print("="*55)

    # Save history CSV for later plotting
    log_dir = cfg["logging"].get("log_dir", "runs/")
    os.makedirs(log_dir, exist_ok=True)
    all_history = []
    for r in results:
        for row in r["history"]:
            row["fold"] = r["fold"]
            all_history.append(row)
    pd.DataFrame(all_history).to_csv(
        os.path.join(log_dir, f"{arch}_history.csv"), index=False
    )
    print(f"\n  Training history saved to {log_dir}{arch}_history.csv")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train key signature detector.")
    parser.add_argument("--config",     default="config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--set",        nargs="*", default=[],
                        metavar="KEY=VALUE",
                        help="Override config values, e.g. --set training.lr=0.001")
    parser.add_argument("--all-folds",  action="store_true",
                        help="Run all k folds and report mean accuracy")
    parser.add_argument("--xgboost",    action="store_true",
                        help="Run XGBoost baseline instead of neural network")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.set:
        print("\nApplying config overrides:")
        cfg = apply_overrides(cfg, args.set)

    if args.xgboost:
        run_xgboost(cfg)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    n_folds = cfg["data"]["n_folds"]
    folds_to_run = list(range(n_folds)) if args.all_folds else [cfg["data"]["val_fold"]]

    results = []
    for fold in folds_to_run:
        result = train_fold(cfg, fold, device)
        results.append(result)

    if len(results) > 1:
        print_summary(results, cfg)
    else:
        r = results[0]
        print(f"\nDone. Best val accuracy: {r['best_val_acc']:.1f}% (epoch {r['best_epoch']})")


if __name__ == "__main__":
    main()
