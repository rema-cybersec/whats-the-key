"""
evaluate.py
-----------
Loads saved model checkpoints, runs evaluation on each fold, and produces
a set of presentation-ready plots saved to the reports/ directory.

Generated outputs:
    reports/confusion_matrix_<arch>.png   — aggregated confusion matrix per model
    reports/accuracy_per_key_<arch>.png   — per-key accuracy bar chart per model
    reports/learning_curve_cnn.png        — CNN train vs val accuracy over epochs
    reports/model_comparison.png          — side-by-side accuracy comparison
    reports/summary.csv                   — all per-key metrics in one CSV

Usage:
    python evaluate.py                    # evaluate all three models
    python evaluate.py --arch cnn         # evaluate only one
    python evaluate.py --config my.yaml   # use a different config
"""

import os
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, classification_report

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from models_tf import load_model, ARCHITECTURES
from train    import load_data, N_CLASSES

# ── Constants ─────────────────────────────────────────────────────────────────

KEY_LABELS = [
    "C maj",  "C# maj", "D maj",  "D# maj",
    "E maj",  "F maj",  "F# maj", "G maj",
    "G# maj", "A maj",  "A# maj", "B maj",
    "C min",  "C# min", "D min",  "D# min",
    "E min",  "F min",  "F# min", "G min",
    "G# min", "A# min", "A min",  "B min",
]

ARCH_COLORS = {
    "cnn":           "#2563eb",
    "random_forest": "#16a34a",
    "xgboost":       "#d97706",
}

REPORT_DIR = "reports"


# ── Load checkpoint ───────────────────────────────────────────────────────────

def _load_ckpt(arch, fold, cfg):
    log_dir = cfg["logging"].get("log_dir", "runs/")
    ext     = "keras" if arch == "cnn" else "pkl"
    path    = os.path.join(log_dir, f"{arch}_fold{fold}.{ext}")
    if not os.path.exists(path):
        return None
    return load_model(arch, path, N_CLASSES, cfg)


# ── Aggregate out-of-fold predictions ────────────────────────────────────────

def collect_predictions(arch, X, y, groups, cfg):
    n_folds   = cfg["data"]["n_folds"]
    gkf       = GroupKFold(n_splits=n_folds)
    y_true_all, y_pred_all = [], []
    fold_accs = []

    for fold, (_, val_idx) in enumerate(gkf.split(X, y, groups)):
        model = _load_ckpt(arch, fold, cfg)
        if model is None:
            print(f"  ⚠  No checkpoint for fold {fold} — skipping")
            continue
        preds = model.predict(X[val_idx])
        acc   = (preds == y[val_idx]).mean() * 100
        fold_accs.append(acc)
        y_true_all.append(y[val_idx])
        y_pred_all.append(preds)

    if not y_true_all:
        return None, None, []

    return np.concatenate(y_true_all), np.concatenate(y_pred_all), fold_accs


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lighten(hex_color, amount):
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    rgb = [c + (1 - c) * amount for c in rgb]
    return "#{:02x}{:02x}{:02x}".format(*[int(c * 255) for c in rgb])


# ── Plot 1: Confusion matrix ──────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, arch, present_keys):
    labels      = sorted(present_keys)
    label_names = [KEY_LABELS[i] for i in labels]
    cm          = confusion_matrix(y_true, y_pred, labels=labels)

    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(
            cm.sum(axis=1, keepdims=True) > 0,
            cm / cm.sum(axis=1, keepdims=True), 0
        )

    n       = len(labels)
    figsize = max(10, n * 0.55)
    fig, ax = plt.subplots(figsize=(figsize, figsize * 0.85))

    sns.heatmap(
        cm_norm, ax=ax,
        xticklabels=label_names, yticklabels=label_names,
        cmap="Blues", vmin=0, vmax=1,
        annot=(n <= 24), fmt=".2f" if n <= 24 else "",
        linewidths=0.4 if n <= 24 else 0,
        cbar_kws={"label": "Recall"},
    )
    ax.set_title(
        f"{arch.replace('_',' ').title()} — Confusion Matrix\n"
        f"(row-normalised · {len(y_true)} songs)",
        fontsize=14, pad=14
    )
    ax.set_xlabel("Predicted Key", fontsize=11)
    ax.set_ylabel("True Key",      fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, f"confusion_matrix_{arch}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ── Plot 2: Per-key accuracy ──────────────────────────────────────────────────

def plot_per_key_accuracy(y_true, y_pred, arch, present_keys):
    labels      = sorted(present_keys)
    label_names = [KEY_LABELS[i] for i in labels]
    accuracies  = []
    counts      = []

    for k in labels:
        mask = y_true == k
        counts.append(int(mask.sum()))
        accuracies.append((y_pred[mask] == k).mean() * 100 if mask.sum() > 0 else 0.0)

    # Major = indices 0-11, minor = 12-23
    color  = ARCH_COLORS.get(arch, "#6b7280")
    colors = [color if k < 12 else _lighten(color, 0.45) for k in labels]
    chance = 100 / len(labels)

    fig, ax = plt.subplots(figsize=(9, max(5, len(labels) * 0.40)))
    bars = ax.barh(label_names, accuracies, color=colors, edgecolor="white", height=0.72)
    ax.axvline(chance, color="#6b7280", linestyle="--", linewidth=1.2)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Accuracy (%)", fontsize=11)
    ax.set_title(
        f"{arch.replace('_',' ').title()} — Per-Key Accuracy",
        fontsize=13, pad=12
    )

    for bar, val, n in zip(bars, accuracies, counts):
        ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%  (n={n})", va="center", fontsize=8)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=color,              label="Major"),
        Patch(color=_lighten(color, 0.45), label="Minor"),
        plt.Line2D([0], [0], color="#6b7280", linestyle="--",
                   label=f"Chance ({chance:.1f}%)"),
    ], fontsize=9, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, f"accuracy_per_key_{arch}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ── Plot 3: CNN learning curve ────────────────────────────────────────────────

def plot_learning_curve(cfg):
    log_dir   = cfg["logging"].get("log_dir", "runs/")
    hist_path = os.path.join(log_dir, "cnn_history.csv")

    if not os.path.exists(hist_path):
        print(f"  ⚠  No CNN history at {hist_path} — skipping learning curve")
        return

    df      = pd.read_csv(hist_path)
    grouped = df.groupby("epoch")[["train_acc", "val_acc"]]
    mean_df = grouped.mean()
    std_df  = grouped.std().fillna(0)
    epochs  = mean_df.index.values

    fig, ax = plt.subplots(figsize=(9, 5))

    for col, label, color in [
        ("train_acc", "Train",      "#93c5fd"),
        ("val_acc",   "Validation", "#2563eb"),
    ]:
        mu  = mean_df[col].values
        std = std_df[col].values
        ax.plot(epochs, mu, color=color, linewidth=2.2, label=label)
        ax.fill_between(epochs, mu - std, mu + std, color=color, alpha=0.14)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("CNN — Learning Curve (mean ± std across folds)", fontsize=13, pad=12)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "learning_curve_cnn.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ── Plot 4: Model comparison ──────────────────────────────────────────────────

def plot_model_comparison(summary):
    archs  = list(summary.keys())
    means  = [summary[a]["mean"]  for a in archs]
    stds   = [summary[a]["std"]   for a in archs]
    colors = [ARCH_COLORS.get(a, "#6b7280") for a in archs]
    labels = [a.replace("_", "\n") for a in archs]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(archs))

    bars = ax.bar(x, means, color=colors, width=0.5, zorder=2,
                  edgecolor="white", linewidth=1.2)
    ax.errorbar(x, means, yerr=stds, fmt="none",
                color="#1f2937", capsize=7, linewidth=2, zorder=3)

    # Individual fold dots with jitter
    rng = np.random.default_rng(42)
    for i, arch in enumerate(archs):
        jitter = rng.uniform(-0.09, 0.09, len(summary[arch]["folds"]))
        ax.scatter(i + jitter, summary[arch]["folds"],
                   color="white", edgecolors=colors[i],
                   s=45, zorder=4, linewidths=1.8)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                mean + std + 1.5,
                f"{mean:.1f}%", ha="center", fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=11)
    ax.set_title("Model Comparison — 5-Fold Cross-Validation", fontsize=13, pad=12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.set_ylim(0, min(100, max(means) + max(stds) + 15))

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--arch",   default=None,
                        help="Evaluate only one architecture (default: all)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    os.makedirs(REPORT_DIR, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Loading data")
    print(f"{'='*55}")
    X, y, groups = load_data(cfg["data"]["csv_path"])
    present_keys = sorted(np.unique(y).tolist())

    archs        = [args.arch] if args.arch else list(ARCHITECTURES.keys())
    summary      = {}
    summary_rows = []

    for arch in archs:
        print(f"\n{'─'*55}")
        print(f"  Evaluating: {arch.upper()}")
        print(f"{'─'*55}")

        y_true, y_pred, fold_accs = collect_predictions(arch, X, y, groups, cfg)

        if y_true is None:
            print("  No checkpoints found — run train.py --all-folds first")
            continue

        overall = (y_true == y_pred).mean() * 100
        print(f"\n  Overall accuracy : {overall:.1f}%")
        print(f"  Per-fold         : {[f'{a:.1f}%' for a in fold_accs]}")
        print(f"\n  Classification report:")
        print(classification_report(
            y_true, y_pred,
            labels=present_keys,
            target_names=[KEY_LABELS[i] for i in present_keys],
            zero_division=0,
        ))

        summary[arch] = {
            "mean":  np.mean(fold_accs),
            "std":   np.std(fold_accs),
            "folds": fold_accs,
        }

        print("  Generating plots...")
        plot_confusion_matrix(y_true, y_pred, arch, present_keys)
        plot_per_key_accuracy( y_true, y_pred, arch, present_keys)

        for k in present_keys:
            mask = y_true == k
            if mask.sum() == 0:
                continue
            summary_rows.append({
                "architecture": arch,
                "key":          KEY_LABELS[k],
                "n_samples":    int(mask.sum()),
                "accuracy_pct": round((y_pred[mask] == k).mean() * 100, 2),
                "overall_acc_pct": round(overall, 2),
            })

    # Cross-model plots
    if len(summary) > 1:
        print(f"\n{'─'*55}")
        print("  Generating cross-model comparison chart...")
        plot_model_comparison(summary)

    plot_learning_curve(cfg)

    if summary_rows:
        csv_path = os.path.join(REPORT_DIR, "summary.csv")
        pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
        print(f"  Saved → {csv_path}")

    if summary:
        print(f"\n{'='*55}")
        print("  FINAL RANKING")
        print(f"{'='*55}")
        for rank, (arch, s) in enumerate(
            sorted(summary.items(), key=lambda x: -x[1]["mean"]), 1
        ):
            print(f"  #{rank}  {arch:20s}  {s['mean']:.1f}% ± {s['std']:.1f}%")
        print(f"{'='*55}")

    print(f"\n✓ All plots saved to ./{REPORT_DIR}/\n")


if __name__ == "__main__":
    main()
