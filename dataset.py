"""
dataset.py
----------
PyTorch Dataset + DataLoader factory with GroupKFold cross-validation.

The key design decision: GroupKFold splits by song_id, so all 5 windows
of a given song always land in the same fold. This prevents data leakage
where the model could "recognise" a song it partially saw during training.

Usage (from train.py or interactively):

    from dataset import build_loaders
    train_loader, val_loader = build_loaders(cfg)

    for chroma, labels in train_loader:
        # chroma shape depends on cfg.data.grouped_windows:
        #   False → (batch, 12)       flat mode  — for linear / xgboost
        #   True  → (batch, 5, 12)    grouped    — for CNN / GRU
        ...
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupKFold


# ── Dataset ───────────────────────────────────────────────────────────────────

class ChromaDataset(Dataset):
    """
    Loads features.csv and exposes each row as a (chroma_tensor, key_label) pair.

    grouped_windows=False  →  one sample per row  (N_rows samples total)
    grouped_windows=True   →  one sample per song  (N_songs samples total)
                               chroma shape: (5, 12) — all windows stacked

    The grouped_windows=True mode is needed for architectures that process
    the temporal structure of the 5 windows (CNN channels, GRU, etc.)
    """

    CHROMA_COLS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def __init__(self, df: pd.DataFrame, grouped_windows: bool = False):
        self.grouped = grouped_windows

        if not grouped_windows:
            # ── Flat mode: each row is one independent sample ──────────────
            self.chroma = torch.tensor(
                df[self.CHROMA_COLS].values, dtype=torch.float32
            )                                          # (N_rows, 12)
            self.labels = torch.tensor(
                df["key"].values, dtype=torch.long
            )                                          # (N_rows,)
            self.song_ids = df["song_id"].values       # kept for GroupKFold

        else:
            # ── Grouped mode: stack all windows of a song into (5, 12) ─────
            songs = []
            labels = []
            song_ids = []

            for song_id, group in df.groupby("song_id"):
                windows = group[self.CHROMA_COLS].values   # (n_windows, 12)
                key     = group["key"].iloc[0]

                # Pad or truncate to exactly 5 windows for consistent shape
                target_windows = 5
                if len(windows) < target_windows:
                    pad = np.zeros((target_windows - len(windows), 12), dtype=np.float32)
                    windows = np.vstack([windows, pad])
                else:
                    windows = windows[:target_windows]

                songs.append(windows)
                labels.append(key)
                song_ids.append(song_id)

            self.chroma   = torch.tensor(np.stack(songs), dtype=torch.float32)  # (N_songs, 5, 12)
            self.labels   = torch.tensor(labels, dtype=torch.long)              # (N_songs,)
            self.song_ids = np.array(song_ids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.chroma[idx], self.labels[idx]


# ── GroupKFold split ──────────────────────────────────────────────────────────

def get_fold_indices(
    dataset: ChromaDataset,
    n_folds: int,
    val_fold: int,
) -> tuple[list[int], list[int]]:
    """
    Split dataset indices into train / val using GroupKFold on song_id.

    All windows (or grouped entries) belonging to the same song_id are
    guaranteed to be entirely in either train or val — never split across both.

    Returns (train_indices, val_indices).
    """
    song_ids = dataset.song_ids
    indices  = np.arange(len(dataset))

    gkf = GroupKFold(n_splits=n_folds)

    # GroupKFold needs X (ignored), y (ignored), groups
    folds = list(gkf.split(indices, groups=song_ids))

    if val_fold >= len(folds):
        raise ValueError(
            f"val_fold={val_fold} is out of range for n_folds={n_folds}"
        )

    train_idx, val_idx = folds[val_fold]
    return train_idx.tolist(), val_idx.tolist()


# ── Public factory ────────────────────────────────────────────────────────────

def build_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """
    Read config, load CSV, apply GroupKFold, return (train_loader, val_loader).

    cfg is the parsed YAML config dict (or any dict with the same shape).
    """
    dcfg = cfg["data"]
    tcfg = cfg["training"]

    # Load CSV
    df = pd.read_csv(dcfg["csv_path"])
    _validate_csv(df)

    grouped = dcfg.get("grouped_windows", False)
    dataset = ChromaDataset(df, grouped_windows=grouped)

    # GroupKFold split
    train_idx, val_idx = get_fold_indices(
        dataset,
        n_folds=dcfg["n_folds"],
        val_fold=dcfg["val_fold"],
    )

    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=tcfg["batch_size"],
        shuffle=True,                        # shuffle only train
        num_workers=dcfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=tcfg["batch_size"],
        shuffle=False,
        num_workers=dcfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )

    n_train_songs = len(set(dataset.song_ids[train_idx]))
    n_val_songs   = len(set(dataset.song_ids[val_idx]))

    print(f"  Dataset   : {len(dataset)} samples from {len(set(dataset.song_ids))} songs")
    print(f"  Fold      : {dcfg['val_fold']+1}/{dcfg['n_folds']}")
    print(f"  Train     : {len(train_set)} samples ({n_train_songs} songs)")
    print(f"  Val       : {len(val_set)} samples ({n_val_songs} songs)")
    print(f"  Mode      : {'grouped (5×12)' if grouped else 'flat (12,)'}")

    return train_loader, val_loader


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_csv(df: pd.DataFrame):
    required = {"song_id", "C", "C#", "D", "key"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing columns: {missing}\n"
            f"Re-run preprocessor.py to regenerate features.csv"
        )
    if df["song_id"].nunique() < 2:
        raise ValueError("Need at least 2 distinct songs for cross-validation.")


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml, textwrap

    # Build a tiny fake dataset: 10 songs × 5 windows, random chroma, random keys
    rng = np.random.default_rng(42)
    rows = []
    for song_id in range(10):
        key = rng.integers(0, 24)
        for _ in range(5):
            chroma = rng.random(12).tolist()
            rows.append([song_id] + chroma + [key])

    cols = ["song_id", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "key"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv("/tmp/test_features.csv", index=False)

    cfg = {
        "data":     {"csv_path": "/tmp/test_features.csv", "n_folds": 5,
                     "val_fold": 0, "num_workers": 0, "grouped_windows": False},
        "training": {"batch_size": 8},
    }

    print("\n── Flat mode ──")
    train_loader, val_loader = build_loaders(cfg)
    batch_chroma, batch_labels = next(iter(train_loader))
    print(f"  Batch chroma shape : {batch_chroma.shape}")   # expect (8, 12)
    print(f"  Batch labels shape : {batch_labels.shape}")   # expect (8,)

    cfg["data"]["grouped_windows"] = True
    print("\n── Grouped mode ──")
    train_loader, val_loader = build_loaders(cfg)
    batch_chroma, batch_labels = next(iter(train_loader))
    print(f"  Batch chroma shape : {batch_chroma.shape}")   # expect (8, 5, 12)
    print(f"  Batch labels shape : {batch_labels.shape}")

    # Verify no song leaks across the fold boundary
    flat_ds = ChromaDataset(df, grouped_windows=False)
    train_idx, val_idx = get_fold_indices(flat_ds, n_folds=5, val_fold=0)
    train_songs = set(flat_ds.song_ids[train_idx])
    val_songs   = set(flat_ds.song_ids[val_idx])
    overlap     = train_songs & val_songs
    print(f"\n── Leakage check ──")
    print(f"  Train songs : {sorted(train_songs)}")
    print(f"  Val songs   : {sorted(val_songs)}")
    print(f"  Overlap     : {overlap} {'✓ clean' if not overlap else '✗ LEAK!'}")
