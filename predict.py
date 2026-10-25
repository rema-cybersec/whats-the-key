"""
predict.py
----------
Interactive CLI for predicting the key signature of a song from a YouTube URL.
Handles downloading, preprocessing, and inference in one shot.

Usage:
    python predict.py                                      # interactive prompt loop
    python predict.py --url "https://youtu.be/..."        # single prediction
    python predict.py --url "https://youtu.be/..." --model runs/cnn_fold0.keras
    python predict.py --url "https://youtu.be/..." --start 12 --end 240

Requirements:
    - A trained model checkpoint in runs/  (from train.py)
    - ffmpeg installed
    - All project dependencies installed
"""

import os
import sys
import argparse
import tempfile
import pickle

import numpy as np
import yaml

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from preprocessor import download_audio, process_song, KEY_LABELS

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH = "config.yaml"
RUNS_DIR    = "runs"

# ── Model loading ─────────────────────────────────────────────────────────────

def find_best_model() -> tuple[str, str]:
    """
    Auto-detect a saved model checkpoint in runs/.
    Preference order: cnn > random_forest > xgboost, fold 0.
    Returns (path, architecture).
    """
    for arch in ["cnn", "random_forest", "xgboost"]:
        ext  = "keras" if arch == "cnn" else "pkl"
        path = os.path.join(RUNS_DIR, f"{arch}_fold0.{ext}")
        if os.path.exists(path):
            return path, arch

    raise FileNotFoundError(
        f"No model checkpoints found in {RUNS_DIR}/\n"
        f"Run: python train.py --all-folds"
    )


def load_model(model_path: str, arch: str, cfg: dict):
    """Load a saved model from disk and return a wrapped model object."""
    if arch == "cnn":
        import tensorflow as tf
        return tf.keras.models.load_model(model_path), "keras"

    with open(model_path, "rb") as f:
        data = pickle.load(f)

    if arch == "xgboost":
        from models_tf import XGBoostModel
        m = XGBoostModel.__new__(XGBoostModel)
        m.clf, m._label_map, m._label_inv = data
        return m, "sklearn"
    else:
        from models_tf import RandomForestModel
        m = RandomForestModel.__new__(RandomForestModel)
        m.clf = data
        return m, "sklearn"


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_from_audio(audio_path: str, model, model_type: str,
                       arch: str, cfg: dict) -> list[dict]:
    """
    Run inference on a local audio file.
    Returns a list of top-3 predictions: [{ key, confidence }, ...]
    """
    windows = process_song(audio_path)          # list of 10 × (12,) arrays
    arr     = np.stack(windows, axis=0)         # (10, 12)

    if model_type == "keras":
        x     = arr[np.newaxis, ...]            # (1, 10, 12)
        probs = model.predict(x, verbose=0)[0]  # (24,)
        top3i = probs.argsort()[::-1][:3]
        return [
            {"key": KEY_LABELS[i], "confidence": float(probs[i])}
            for i in top3i
        ]
    else:
        x     = arr.flatten()[np.newaxis, ...]  # (1, 120)
        proba = model.predict_proba(x)[0]

        if arch == "xgboost":
            inv   = model._label_inv
            top3i = proba.argsort()[::-1][:3]
            return [
                {"key": KEY_LABELS[inv[i]], "confidence": float(proba[i])}
                for i in top3i
            ]
        else:
            top3i = proba.argsort()[::-1][:3]
            return [
                {"key": KEY_LABELS[i], "confidence": float(proba[i])}
                for i in top3i
            ]


def predict_url(url: str, model, model_type: str, arch: str, cfg: dict,
                start_time=None, end_time=None) -> list[dict]:
    """Download audio from URL and run inference. Returns top-3 predictions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        print("  Downloading audio...")
        audio_path = download_audio(url, tmpdir)
        print("  Extracting features...")
        return predict_from_audio(audio_path, model, model_type, arch, cfg)


# ── Display ───────────────────────────────────────────────────────────────────

def display_result(predictions: list[dict]):
    top      = predictions[0]
    key      = top["key"]
    conf_pct = top["confidence"] * 100
    bar_len  = int(conf_pct / 5)           # max 20 chars
    bar      = "█" * bar_len + "░" * (20 - bar_len)

    print()
    print(f"  ┌─────────────────────────────────────┐")
    print(f"  │  Detected key:  {key:<20s} │")
    print(f"  │  Confidence:    {bar}  {conf_pct:4.1f}%  │")
    print(f"  └─────────────────────────────────────┘")
    print()
    print("  Top 3:")
    for i, p in enumerate(predictions, 1):
        print(f"    {i}. {p['key']:<12s}  {p['confidence']*100:5.1f}%")
    print()


# ── Entry points ──────────────────────────────────────────────────────────────

def run_once(url: str, model, model_type: str, arch: str, cfg: dict,
             start_time=None, end_time=None):
    print(f"\n  URL : {url}")
    try:
        predictions = predict_url(url, model, model_type, arch, cfg,
                                  start_time, end_time)
        display_result(predictions)
    except Exception as e:
        print(f"\n  Error: {e}\n")


def run_interactive(model, model_type: str, arch: str, cfg: dict):
    print(f"\n  Key Signature Detector  [{arch}]")
    print("  Paste a YouTube URL and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            url = input("  YouTube URL › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            break

        if not url:
            continue
        if url.lower() in ("quit", "exit", "q"):
            print("  Bye!")
            break

        run_once(url, model, model_type, arch, cfg)


def main():
    parser = argparse.ArgumentParser(description="Predict key signature from a YouTube URL.")
    parser.add_argument("--url",    default=None,  help="YouTube URL to predict")
    parser.add_argument("--model",  default=None,  help="Path to model checkpoint (auto-detected if omitted)")
    parser.add_argument("--arch",   default=None,  help="Architecture: cnn | random_forest | xgboost (auto-detected)")
    parser.add_argument("--start",  type=float, default=None, help="Music start time in seconds")
    parser.add_argument("--end",    type=float, default=None, help="Music end time in seconds")
    parser.add_argument("--config", default=CONFIG_PATH, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    # Resolve model path and architecture
    if args.model:
        model_path = args.model
        if args.arch:
            arch = args.arch
        elif model_path.endswith(".keras"):
            arch = "cnn"
        elif "random_forest" in model_path:
            arch = "random_forest"
        elif "xgboost" in model_path:
            arch = "xgboost"
        else:
            print("Could not auto-detect architecture from filename. Use --arch.")
            sys.exit(1)
    else:
        model_path, arch = find_best_model()

    print(f"\n  Loading model: {model_path}  [{arch}]")
    model, model_type = load_model(model_path, arch, cfg)
    print("  Model ready.")

    if args.url:
        run_once(args.url, model, model_type, arch, cfg, args.start, args.end)
    else:
        run_interactive(model, model_type, arch, cfg)


if __name__ == "__main__":
    main()
