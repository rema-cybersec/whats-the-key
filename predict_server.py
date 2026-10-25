"""
predict_server.py
-----------------
FastAPI server exposing the trained key-detection model over HTTP.
Spring Boot calls this internally; it is NOT exposed to the browser.

Start with:
    uvicorn predict_server:app --host 127.0.0.1 --port 8000

Or point to a specific model checkpoint:
    MODEL_PATH=runs/cnn_fold0.keras uvicorn predict_server:app --host 127.0.0.1 --port 8000

Endpoints:
    POST /predict   { "url": "https://youtu.be/..." }
                 -> { "key": "C major", "confidence": 0.87, "top3": [...] }

    GET  /health  -> { "status": "ok", "model": "<arch>" }
"""

import os
import tempfile
import traceback
from contextlib import asynccontextmanager

import numpy as np
import yaml
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from preprocessor import download_audio, process_song, KEY_LABELS

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH  = os.getenv("MODEL_PATH",  "runs/cnn_fold0.keras")
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")

# ── Global model state (loaded once at startup) ───────────────────────────────

_model  = None
_cfg    = None


def load_model_on_startup():
    global _model, _cfg

    with open(CONFIG_PATH) as f:
        _cfg = yaml.safe_load(f)

    arch = _cfg["model"]["architecture"]
    ext  = os.path.splitext(MODEL_PATH)[1]

    if ext == ".keras":
        # TensorFlow / Keras CNN
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[predict_server] Loaded Keras model: {MODEL_PATH}")

    elif ext == ".pkl":
        # Random Forest or XGBoost
        import pickle
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)

        if arch == "xgboost":
            import xgboost as xgb
            from models_tf import XGBoostModel
            m = XGBoostModel.__new__(XGBoostModel)
            m.clf, m._label_map, m._label_inv = data
            _model = m
        else:
            from sklearn.ensemble import RandomForestClassifier
            from models_tf import RandomForestModel
            m = RandomForestModel.__new__(RandomForestModel)
            m.clf = data
            _model = m

        print(f"[predict_server] Loaded {arch} model: {MODEL_PATH}")
    else:
        raise ValueError(f"Unrecognised model file extension: {ext}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_on_startup()
    yield


app = FastAPI(title="Key Signature Predictor", lifespan=lifespan)


# ── Request / response schemas ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    url:        str
    start_time: float | None = None
    end_time:   float | None = None


class PredictResponse(BaseModel):
    key:        str
    confidence: float
    top3:       list[dict]


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(audio_path: str, start_time, end_time) -> PredictResponse:
    arch    = _cfg["model"]["architecture"]
    grouped = _cfg["data"].get("grouped_windows", True)

    # Extract 10 chroma windows
    windows = process_song(audio_path, start_time, end_time)   # list of (12,)
    arr     = np.stack(windows, axis=0)                        # (10, 12)

    if arch == "cnn":
        # Keras expects (batch, 10, 12)
        x      = arr[np.newaxis, ...]                          # (1, 10, 12)
        probs  = _model.predict(x, verbose=0)[0]               # (24,)
        top3_i = probs.argsort()[::-1][:3].tolist()
        top3   = [{"key": KEY_LABELS[i], "confidence": round(float(probs[i]), 4)}
                  for i in top3_i]

    else:
        # RF / XGBoost expect (batch, 120) flat
        x      = arr.flatten()[np.newaxis, ...]                # (1, 120)
        proba  = _model.predict_proba(x)[0]                    # (n_classes,)

        if arch == "xgboost":
            # Map local class indices back to global KEY_LABELS indices
            inv = _model._label_inv
            top3_local = proba.argsort()[::-1][:3].tolist()
            top3 = [{"key": KEY_LABELS[inv[i]], "confidence": round(float(proba[i]), 4)}
                    for i in top3_local]
        else:
            top3_i = proba.argsort()[::-1][:3].tolist()
            top3   = [{"key": KEY_LABELS[i], "confidence": round(float(proba[i]), 4)}
                      for i in top3_i]

    return PredictResponse(
        key=top3[0]["key"],
        confidence=top3[0]["confidence"],
        top3=top3,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":     "ok",
        "model":      _cfg["model"]["architecture"] if _cfg else "unknown",
        "model_path": MODEL_PATH,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = download_audio(req.url, tmpdir)
            return run_inference(audio_path, req.start_time, req.end_time)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
