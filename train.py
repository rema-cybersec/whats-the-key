"""
train.py
--------
Trains and evaluates key signature detection models using GroupKFold.

Works with all three architectures: cnn, random_forest, xgboost.

Usage:
    # Train one fold (default: fold 0)
    python train.py

    # Train all folds and report mean ± std
    python train.py --all-folds

    # Choose architecture
    python train.py --set model.architecture=random_forest
    python train.py --set model.architecture=xgboost
    python train.py --set model.architecture=cnn

    # Run all three architectures and compare
    python train.py --compare

    # Override any config value
    python train.py --set training.lr=0.0005 model.hidden_dim=128
"""

import os
import argparse
import copy
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from models_tf import build_model, ARCHITECTURES

# ── Constants ─────────────────────────────────────────────────────────────────

CHROMA_COLS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
N_CLASSES   = 24


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for override in overrides:
        key_path, raw_value = override.split("=", 1)
        keys  = key_path.strip().split(".")
        value = _cast(raw_value.strip())
        node  = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
        print(f"  Override: {key_path} = {value}")
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


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load features.csv and return:
        X      — (N_songs, 10, 12)  chroma windows per song
        y      — (N_songs,)         key labels (int 0–23)
        groups — (N_songs,)         song_id for GroupKFold

    Windows are grouped by song_id. Songs with fewer than 10 windows are
    zero-padded; songs with more than 10 are truncated (shouldn't happen).
    """
    df = pd.read_csv(csv_path)
    n_windows = 10

    songs_X, songs_y, songs_g = [], [], []

    for song_id, group in df.groupby("song_id"):
        windows = group[CHROMA_COLS].values.astype(np.float32)  # (n, 12)
        key     = int(group["key"].iloc[0])

        # Pad or truncate to exactly n_windows
        if len(windows) < n_windows:
            pad     = np.zeros((n_windows - len(windows), 12), dtype=np.float32)
            windows = np.vstack([windows, pad])
        else:
            windows = windows[:n_windows]

        songs_X.append(windows)
        songs_y.append(key)
        songs_g.append(song_id)

    X      = np.stack(songs_X)            # (N_songs, 10, 12)
    y      = np.array(songs_y)            # (N_songs,)
    groups = np.array(songs_g)            # (N_songs,)

    print(f"  Loaded {len(X)} songs, {len(np.unique(y))} distinct keys")
    print(f"  X shape: {X.shape}  y shape: {y.shape}")
    return X, y, groups


# ── Train one fold ────────────────────────────────────────────────────────────

def train_fold(
    architecture: str,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx:   np.ndarray,
    fold:      int,
    n_folds:   int,
    cfg:       dict,
) -> dict:
    print(f"\n{'─'*55}")
    print(f"  {architecture}   Fold {fold+1}/{n_folds}")
    print(f"  Train: {len(train_idx)} songs   Val: {len(val_idx)} songs")
    print(f"{'─'*55}")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    model = build_model(architecture, N_CLASSES, cfg)
    model.fit(X_train, y_train, X_val, y_val)

    train_acc = model.score(X_train, y_train)
    val_acc   = model.score(X_val,   y_val)

    print(f"  Train acc: {train_acc:.1f}%   Val acc: {val_acc:.1f}%")

    # Save best model checkpoint
    log_dir = cfg["logging"].get("log_dir", "runs/")
    os.makedirs(log_dir, exist_ok=True)
    ext  = "keras" if architecture == "cnn" else "pkl"
    path = os.path.join(log_dir, f"{architecture}_fold{fold}.{ext}")
    model.save(path)
    print(f"  Saved → {path}")

    return {
        "fold":      fold,
        "train_acc": train_acc,
        "val_acc":   val_acc,
        "history":   model.history,
    }


# ── Run all folds for one architecture ───────────────────────────────────────

def run_architecture(
    architecture: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: dict,
    folds_to_run: list[int],
) -> list[dict]:
    n_folds = cfg["data"]["n_folds"]
    gkf     = GroupKFold(n_splits=n_folds)
    all_folds = list(gkf.split(X, y, groups))

    results = []
    for fold in folds_to_run:
        train_idx, val_idx = all_folds[fold]
        result = train_fold(
            architecture, X, y, train_idx, val_idx, fold, n_folds, cfg
        )
        results.append(result)

    return results


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(architecture: str, results: list[dict], cfg: dict):
    accs  = [r["val_acc"] for r in results]
    mean  = np.mean(accs)
    std   = np.std(accs)

    print(f"\n{'='*55}")
    print(f"  {architecture}  —  {len(results)}-fold summary")
    print(f"{'='*55}")
    for r in results:
        print(f"  Fold {r['fold']+1}: val={r['val_acc']:.1f}%  train={r['train_acc']:.1f}%")
    print(f"  {'─'*35}")
    print(f"  Mean val accuracy: {mean:.1f}% ± {std:.1f}%")
    print(f"{'='*55}")

    # Save history to CSV for plotting
    log_dir = cfg["logging"].get("log_dir", "runs/")
    rows = []
    for r in results:
        hist = r["history"]
        if hist:   # CNN has per-epoch history; RF/XGB do not
            for epoch, (tl, vl) in enumerate(zip(
                hist.get("accuracy", []),
                hist.get("val_accuracy", []),
            )):
                rows.append({
                    "fold": r["fold"], "epoch": epoch + 1,
                    "train_acc": tl * 100, "val_acc": vl * 100,
                })
        else:
            rows.append({
                "fold": r["fold"], "epoch": 1,
                "train_acc": r["train_acc"], "val_acc": r["val_acc"],
            })

    hist_path = os.path.join(log_dir, f"{architecture}_history.csv")
    pd.DataFrame(rows).to_csv(hist_path, index=False)
    print(f"  History saved → {hist_path}")

    return mean, std


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train key signature detector.")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--set",        nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--all-folds",  action="store_true",
                        help="Run all k folds (default: fold 0 only)")
    parser.add_argument("--compare",    action="store_true",
                        help="Train all three architectures and print comparison")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.set:
        print("Config overrides:")
        cfg = apply_overrides(cfg, args.set)

    print(f"\n{'='*55}")
    print(f"  Loading data: {cfg['data']['csv_path']}")
    print(f"{'='*55}")
    X, y, groups = load_data(cfg["data"]["csv_path"])

    n_folds      = cfg["data"]["n_folds"]
    folds_to_run = list(range(n_folds)) if args.all_folds else [cfg["data"]["val_fold"]]

    if args.compare:
        archs = list(ARCHITECTURES.keys())
    else:
        archs = [cfg["model"]["architecture"]]

    all_results = {}
    for arch in archs:
        print(f"\n{'#'*55}")
        print(f"  Architecture: {arch.upper()}")
        print(f"{'#'*55}")
        results = run_architecture(arch, X, y, groups, cfg, folds_to_run)
        mean, std = print_summary(arch, results, cfg)
        all_results[arch] = (mean, std)

    if args.compare:
        print(f"\n{'='*55}")
        print("  COMPARISON SUMMARY")
        print(f"{'='*55}")
        ranked = sorted(all_results.items(), key=lambda x: -x[1][0])
        for rank, (arch, (mean, std)) in enumerate(ranked, 1):
            print(f"  #{rank}  {arch:20s}  {mean:.1f}% ± {std:.1f}%")
        print(f"{'='*55}")


if __name__ == "__main__":
    main()
