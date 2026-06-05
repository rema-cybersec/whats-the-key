"""
models.py
---------
All model architectures for key signature detection.

Each architecture corresponds to a step in the experiment ladder:

  linear_flat    (2.1)  whole chroma matrix -> single Linear
  linear_windows (2.2)  one Linear per window -> concat -> Linear
  cnn_windows    (4.1)  one CNN per window -> concat -> Linear
  cnn_channels   (4.2)  5 windows stacked as channels -> CNN -> Linear
  cnn_gru        (5.1)  CNN per window -> GRU -> Linear

All models accept a config dict and expose the same interface:
    model = build_model(cfg)
    logits = model(x)          # x shape depends on architecture

Input shapes:
  flat mode    (batch, 12)       — linear_flat
  grouped mode (batch, 5, 12)    — everything else

Output: (batch, 24)  — logits over 24 key classes
"""

import torch
import torch.nn as nn


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_mlp(
    in_dim:      int,
    hidden_dim:  int,
    out_dim:     int,
    num_layers:  int,
    dropout:     float,
    batchnorm:   bool,
    activation:  str = "relu",
) -> nn.Sequential:
    """
    Build a stack of Linear -> [BN] -> Activation -> Dropout layers.
    num_layers controls the number of hidden layers (minimum 1).
    The final Linear maps to out_dim with no activation — raw logits.
    """
    act_fn = {
        "relu":  nn.ReLU,
        "gelu":  nn.GELU,
        "tanh":  nn.Tanh,
        "silu":  nn.SiLU,
    }[activation]

    layers = []
    dim = in_dim
    for _ in range(max(num_layers, 1)):
        layers.append(nn.Linear(dim, hidden_dim))
        if batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(act_fn())
        layers.append(nn.Dropout(dropout))
        dim = hidden_dim

    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


def _make_conv_block(
    in_channels: int,
    out_channels: int,
    dropout: float,
    batchnorm: bool,
) -> nn.Sequential:
    """
    1D conv block: Conv1d -> [BN] -> ReLU -> Dropout
    kernel_size=3, padding=1 keeps the time dimension unchanged.
    """
    layers = [nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)]
    if batchnorm:
        layers.append(nn.BatchNorm1d(out_channels))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ── (2.1) LinearFlat ──────────────────────────────────────────────────────────

class LinearFlat(nn.Module):
    """
    Simplest possible baseline.
    Flattens the entire input and passes through an MLP.

    Input:  (batch, 12)  — mean chroma across all windows combined
    Output: (batch, 24)
    """
    def __init__(self, cfg: dict):
        super().__init__()
        mcfg = cfg["model"]
        self.net = _make_mlp(
            in_dim=12,
            hidden_dim=mcfg["hidden_dim"],
            out_dim=24,
            num_layers=mcfg["num_layers"],
            dropout=mcfg["dropout"],
            batchnorm=mcfg["use_batchnorm"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 12)  — already flat in this mode
        return self.net(x)


# ── (2.2) LinearWindows ───────────────────────────────────────────────────────

class LinearWindows(nn.Module):
    """
    One small Linear encoder per window, then concat all representations
    and pass through a shared classifier.

    The per-window encoders are shared weights (same Linear for all 5 windows)
    which acts as a mild regularizer and keeps parameter count low.

    Input:  (batch, 5, 12)
    Output: (batch, 24)
    """
    def __init__(self, cfg: dict):
        super().__init__()
        mcfg = cfg["model"]
        hidden = mcfg["hidden_dim"]

        # Shared encoder applied independently to each window
        self.window_encoder = _make_mlp(
            in_dim=12,
            hidden_dim=hidden,
            out_dim=hidden,
            num_layers=1,
            dropout=mcfg["dropout"],
            batchnorm=mcfg["use_batchnorm"],
        )

        # Classifier takes all 5 window representations concatenated
        self.classifier = _make_mlp(
            in_dim=hidden * 5,
            hidden_dim=hidden,
            out_dim=24,
            num_layers=mcfg["num_layers"],
            dropout=mcfg["dropout"],
            batchnorm=mcfg["use_batchnorm"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 5, 12)
        batch_size = x.size(0)

        # Encode each window independently: reshape to (batch*5, 12), encode,
        # then reshape back to (batch, 5, hidden)
        x_flat = x.view(batch_size * 5, 12)
        encoded = self.window_encoder(x_flat)            # (batch*5, hidden)
        encoded = encoded.view(batch_size, 5, -1)        # (batch, 5, hidden)

        # Concatenate all window representations along the feature axis
        concat = encoded.view(batch_size, -1)            # (batch, 5*hidden)
        return self.classifier(concat)


# ── (4.1) CNNWindows ──────────────────────────────────────────────────────────

class CNNWindows(nn.Module):
    """
    One 1D CNN encoder per window (shared weights), concat, classify.
    The CNN treats the 12 chroma bins as a 1D sequence of length 12.

    This lets it learn local pitch patterns (e.g. adjacent semitones
    activating together) that a flat Linear misses.

    Input:  (batch, 5, 12)
    Output: (batch, 24)
    """
    def __init__(self, cfg: dict):
        super().__init__()
        mcfg = cfg["model"]
        hidden = mcfg["hidden_dim"]

        # Each window: (1, 12) -> conv -> pool -> flat
        self.window_encoder = nn.Sequential(
            # in_channels=1, treat 12 chroma bins as a length-12 signal
            _make_conv_block(1, hidden, mcfg["dropout"], mcfg["use_batchnorm"]),
            _make_conv_block(hidden, hidden, mcfg["dropout"], mcfg["use_batchnorm"]),
            nn.AdaptiveAvgPool1d(1),    # collapse the 12-bin dimension -> (batch, hidden, 1)
            nn.Flatten(),               # -> (batch, hidden)
        )

        self.classifier = _make_mlp(
            in_dim=hidden * 5,
            hidden_dim=hidden,
            out_dim=24,
            num_layers=mcfg["num_layers"],
            dropout=mcfg["dropout"],
            batchnorm=mcfg["use_batchnorm"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 5, 12)
        batch_size = x.size(0)

        # Encode each window: reshape to (batch*5, 1, 12) for Conv1d
        x_win = x.view(batch_size * 5, 1, 12)
        encoded = self.window_encoder(x_win)             # (batch*5, hidden)
        encoded = encoded.view(batch_size, -1)           # (batch, 5*hidden)
        return self.classifier(encoded)


# ── (4.2) CNNChannels ─────────────────────────────────────────────────────────

class CNNChannels(nn.Module):
    """
    Stack all 5 windows as separate channels, then apply a CNN that can
    mix information across channels (windows) and across the 12 pitch bins.

    This is the most expressive CNN variant: the network can learn patterns
    like "beginning and end agree on C while middle transitions through G"
    — relationships that the per-window models can't see.

    Input:  (batch, 5, 12)   treated as (batch, channels=5, length=12)
    Output: (batch, 24)
    """
    def __init__(self, cfg: dict):
        super().__init__()
        mcfg = cfg["model"]
        hidden = mcfg["hidden_dim"]

        self.encoder = nn.Sequential(
            # in_channels=5 (one per window), mixes across windows from layer 1
            _make_conv_block(5,      hidden, mcfg["dropout"], mcfg["use_batchnorm"]),
            _make_conv_block(hidden, hidden, mcfg["dropout"], mcfg["use_batchnorm"]),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.classifier = _make_mlp(
            in_dim=hidden,
            hidden_dim=hidden,
            out_dim=24,
            num_layers=mcfg["num_layers"],
            dropout=mcfg["dropout"],
            batchnorm=mcfg["use_batchnorm"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 5, 12) — already in (batch, channels, length) form
        encoded = self.encoder(x)       # (batch, hidden)
        return self.classifier(encoded)


# ── (5.1) CNNWithGRU ──────────────────────────────────────────────────────────

class CNNWithGRU(nn.Module):
    """
    CNN encodes each window into a representation, then a GRU processes
    the sequence of 5 representations in temporal order (start -> end).

    The GRU can learn transitions like "if early windows suggest G major
    and later ones suggest C major, it might be a modulation in C."

    Caveat: with only 5 time steps the GRU has limited sequence to work
    with, so gains over CNNChannels may be modest on a small dataset.

    Input:  (batch, 5, 12)
    Output: (batch, 24)
    """
    def __init__(self, cfg: dict):
        super().__init__()
        mcfg = cfg["model"]
        hidden = mcfg["hidden_dim"]

        # Same per-window CNN encoder as CNNWindows
        self.window_encoder = nn.Sequential(
            _make_conv_block(1, hidden, mcfg["dropout"], mcfg["use_batchnorm"]),
            _make_conv_block(hidden, hidden, mcfg["dropout"], mcfg["use_batchnorm"]),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # GRU processes the sequence of 5 window encodings
        # batch_first=True: input shape is (batch, seq_len, features)
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,    # GRU internal dropout only meaningful for num_layers > 1
        )

        self.classifier = _make_mlp(
            in_dim=hidden,
            hidden_dim=hidden,
            out_dim=24,
            num_layers=mcfg["num_layers"],
            dropout=mcfg["dropout"],
            batchnorm=mcfg["use_batchnorm"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 5, 12)
        batch_size = x.size(0)

        # Encode each window: (batch*5, 1, 12) -> (batch*5, hidden)
        x_win = x.view(batch_size * 5, 1, 12)
        encoded = self.window_encoder(x_win)             # (batch*5, hidden)
        encoded = encoded.view(batch_size, 5, -1)        # (batch, 5, hidden)

        # GRU over the temporal sequence; take the final hidden state
        _, h_n = self.gru(encoded)                       # h_n: (1, batch, hidden)
        h_last = h_n.squeeze(0)                          # (batch, hidden)

        return self.classifier(h_last)


# ── Registry + factory ────────────────────────────────────────────────────────

ARCHITECTURES = {
    "linear_flat":    LinearFlat,
    "linear_windows": LinearWindows,
    "cnn_windows":    CNNWindows,
    "cnn_channels":   CNNChannels,
    "cnn_gru":        CNNWithGRU,
}


def build_model(cfg: dict) -> nn.Module:
    arch = cfg["model"]["architecture"]
    if arch not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            f"Choose from: {list(ARCHITECTURES.keys())}"
        )
    return ARCHITECTURES[arch](cfg)
