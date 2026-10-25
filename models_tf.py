"""
models_tf.py
------------
Three models for key signature detection, all sharing the same interface:

    model = build_model(architecture, n_classes, cfg)
    model.fit(X_train, y_train)
    acc = model.score(X_val, y_val)
    preds = model.predict(X_val)
    model.save(path)
    model = load_model(architecture, path)

Architectures:
    cnn          — 1D CNN in Keras/TensorFlow  (input: (N, 10, 12))
    random_forest — sklearn RandomForestClassifier (input: (N, 120) flat)
    xgboost      — XGBoost XGBClassifier         (input: (N, 120) flat)

Input conventions:
    CNN expects (N, 10, 12) — 10 windows × 12 chroma bins, grouped per song.
    RF and XGBoost expect (N, 120)  — the same data flattened.
    The build_model factory returns a wrapper that handles reshaping internally,
    so the training loop always passes (N, 10, 12) regardless of architecture.
"""

import os
import numpy as np
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF info/warnings


# ── CNN (Keras / TensorFlow) ──────────────────────────────────────────────────

def _build_cnn(n_classes: int, cfg: dict):
    """
    1D CNN that treats the 10 chroma windows as a sequence of length 10,
    each with 12 features (pitch classes).

    Architecture:
        Input (10, 12)
        → Conv1D × 2  (learn local patterns across adjacent windows)
        → GlobalAvgPool
        → Dense + Dropout × num_layers
        → Dense(n_classes, softmax)
    """
    import tensorflow as tf
    from tensorflow import keras

    mcfg    = cfg["model"]
    hidden  = mcfg["hidden_dim"]
    dropout = mcfg["dropout"]
    layers  = mcfg["num_layers"]

    inputs = keras.Input(shape=(10, 12), name="chroma_windows")

    x = keras.layers.Conv1D(hidden, kernel_size=3, padding="same", activation="relu")(inputs)
    if mcfg["use_batchnorm"]:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout)(x)

    x = keras.layers.Conv1D(hidden, kernel_size=3, padding="same", activation="relu")(x)
    if mcfg["use_batchnorm"]:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout)(x)

    x = keras.layers.GlobalAveragePooling1D()(x)

    for _ in range(layers):
        x = keras.layers.Dense(hidden, activation="relu")(x)
        if mcfg["use_batchnorm"]:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(dropout)(x)

    outputs = keras.layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="key_cnn")

    opt_name = cfg["training"]["optimizer"].lower()
    lr       = cfg["training"]["lr"]
    optimizer = {
        "adam":    keras.optimizers.Adam(lr),
        "adamw":   keras.optimizers.AdamW(lr, weight_decay=1e-4),
        "sgd":     keras.optimizers.SGD(lr, momentum=0.9),
        "rmsprop": keras.optimizers.RMSprop(lr),
    }[opt_name]

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class CNNModel:
    """Keras CNN wrapped in the unified interface."""

    def __init__(self, n_classes: int, cfg: dict):
        self.n_classes = n_classes
        self.cfg       = cfg
        self.model     = _build_cnn(n_classes, cfg)
        self._history  = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val:   np.ndarray, y_val:   np.ndarray):
        """
        X shape: (N, 10, 12)
        y shape: (N,)  integer class labels
        """
        import tensorflow as tf
        from tensorflow import keras

        tcfg    = self.cfg["training"]
        epochs  = tcfg["epochs"]
        batch   = tcfg["batch_size"]

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=15,
                restore_best_weights=True, verbose=0
            ),
        ]

        sched = (self.cfg["training"].get("scheduler") or "none").lower()
        if sched == "cosine":
            # Cosine decay schedule via LearningRateScheduler
            base_lr = tcfg["lr"]
            def cosine_decay(epoch):
                import math
                return base_lr * 0.5 * (1 + math.cos(math.pi * epoch / epochs))
            callbacks.append(keras.callbacks.LearningRateScheduler(cosine_decay, verbose=0))

        self._history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch,
            callbacks=callbacks,
            verbose=0,
        )
        return self

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy as a percentage."""
        _, acc = self.model.evaluate(X, y, verbose=0)
        return acc * 100.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, verbose=0).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, verbose=0)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    @property
    def history(self):
        return self._history.history if self._history else {}


# ── Random Forest (sklearn) ───────────────────────────────────────────────────

class RandomForestModel:
    """
    sklearn RandomForestClassifier wrapped in the unified interface.
    Expects (N, 10, 12) input but flattens to (N, 120) internally.
    """

    def __init__(self, n_classes: int, cfg: dict):
        from sklearn.ensemble import RandomForestClassifier
        self.n_classes = n_classes
        self.cfg       = cfg
        # n_estimators and max_depth from config's rf section, with defaults
        rfcfg = cfg.get("random_forest", {})
        self.clf = RandomForestClassifier(
            n_estimators = rfcfg.get("n_estimators", 500),
            max_depth    = rfcfg.get("max_depth", None),   # None = full trees
            min_samples_leaf = rfcfg.get("min_samples_leaf", 2),
            n_jobs       = -1,
            random_state = 42,
            class_weight = "balanced",   # handles any key imbalance automatically
        )
        self._history = {}  # RF has no epoch history

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val:   np.ndarray, y_val:   np.ndarray):
        self.clf.fit(self._flat(X_train), y_train)
        return self

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return 100.0 * self.clf.score(self._flat(X), y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(self._flat(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self._flat(X))

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.clf, f)

    @property
    def history(self):
        return self._history

    @staticmethod
    def _flat(X: np.ndarray) -> np.ndarray:
        return X.reshape(len(X), -1)   # (N, 10, 12) → (N, 120)


# ── XGBoost ───────────────────────────────────────────────────────────────────

class XGBoostModel:
    """
    XGBoost classifier wrapped in the unified interface.
    Expects (N, 10, 12) input but flattens to (N, 120) internally.
    """

    def __init__(self, n_classes: int, cfg: dict):
        import xgboost as xgb
        self.n_classes = n_classes
        self.cfg       = cfg
        xgcfg = cfg.get("xgboost", {})
        self.clf = xgb.XGBClassifier(
            n_estimators  = xgcfg.get("n_estimators", 500),
            max_depth     = xgcfg.get("max_depth", 5),
            learning_rate = xgcfg.get("learning_rate", 0.05),
            subsample     = xgcfg.get("subsample", 0.8),
            colsample_bytree = xgcfg.get("colsample_bytree", 0.8),
            eval_metric   = "mlogloss",
            verbosity     = 0,
            random_state  = 42,
            n_jobs        = -1,
        )
        self._history = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val:   np.ndarray, y_val:   np.ndarray):
        # Remap labels to contiguous 0..N-1 (handles missing keys in a fold)
        classes = np.sort(np.unique(y_train))
        self._label_map    = {c: i for i, c in enumerate(classes)}
        self._label_inv    = {i: c for c, i in self._label_map.items()}

        y_tr = np.array([self._label_map[y] for y in y_train])
        self.clf.set_params(num_class=len(classes))
        self.clf.fit(self._flat(X_train), y_tr)
        return self

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self._predict_remapped(X)
        return 100.0 * np.mean(preds == y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._predict_remapped(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self._flat(X))

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump((self.clf, self._label_map, self._label_inv), f)

    @property
    def history(self):
        return self._history

    def _predict_remapped(self, X: np.ndarray) -> np.ndarray:
        raw = self.clf.predict(self._flat(X))
        return np.array([self._label_inv.get(r, -1) for r in raw])

    @staticmethod
    def _flat(X: np.ndarray) -> np.ndarray:
        return X.reshape(len(X), -1)


# ── Registry + factory ────────────────────────────────────────────────────────

ARCHITECTURES = {
    "cnn":           CNNModel,
    "random_forest": RandomForestModel,
    "xgboost":       XGBoostModel,
}


def build_model(architecture: str, n_classes: int, cfg: dict):
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Choose from: {list(ARCHITECTURES.keys())}"
        )
    return ARCHITECTURES[architecture](n_classes, cfg)


def load_model(architecture: str, path: str, n_classes: int, cfg: dict):
    """Load a saved model from disk."""
    if architecture == "cnn":
        import tensorflow as tf
        m = CNNModel(n_classes, cfg)
        m.model = tf.keras.models.load_model(path)
        return m
    else:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if architecture == "xgboost":
            m = XGBoostModel(n_classes, cfg)
            m.clf, m._label_map, m._label_inv = data
        else:
            m = RandomForestModel(n_classes, cfg)
            m.clf = data
        return m
