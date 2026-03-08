"""
activity_model.py — Stage 3: Activity Analysis (The Context Engine)
====================================================================
Architecture:
    Frame sequence (N frames)
        → ResNet-18 CNN (spatial feature extraction per frame)
        → LSTM (temporal sequence analysis)
        → FC + Sigmoid → anomaly_score ∈ [0, 1]

A score above the configured threshold indicates significant /
anomalous activity that should be captured as an event clip.

Training target : UCF-Crime dataset (future milestone).
Until trained weights are available the main pipeline automatically
falls back to heuristic-based event triggering (new-track detection).

Usage from main.py
------------------
    engine = ActivityEngine(weights_path, device, seq_length, threshold)
    ...
    engine.push_frame(frame)
    verdict = engine.is_anomalous()   # True / False / None (no model)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


# ──────────────────────────────────────────────────────────────────────
#  Spatial Feature Extractor — CNN (ResNet-18)
# ──────────────────────────────────────────────────────────────────────

class SpatialCNN(nn.Module):
    """
    Uses a pre-trained ResNet-18 backbone with the final classification
    layer removed.  Each input frame is mapped to a 512-dim feature
    vector that encodes *what* is happening spatially.
    """

    def __init__(self, feature_dim: int = 512):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Strip the final FC — keep everything up to the avg-pool
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) batch of normalised frames.
        Returns:
            (B, 512) spatial feature vectors.
        """
        feat = self.features(x)           # (B, 512, 1, 1)
        return feat.flatten(start_dim=1)  # (B, 512)


# ──────────────────────────────────────────────────────────────────────
#  Temporal Sequence Analyser — LSTM
# ──────────────────────────────────────────────────────────────────────

class TemporalLSTM(nn.Module):
    """
    Consumes a sequence of spatial feature vectors over time and
    outputs an anomaly probability.  The final hidden state captures
    the temporal dynamics of the scene.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 512) temporal sequence of spatial features.
        Returns:
            (B,) anomaly scores in [0, 1].
        """
        lstm_out, _ = self.lstm(x)             # (B, T, hidden)
        last_hidden = lstm_out[:, -1, :]       # (B, hidden)
        return self.classifier(last_hidden).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────
#  Combined Model — CNN + LSTM end-to-end
# ──────────────────────────────────────────────────────────────────────

class ActivityAnalyzer(nn.Module):
    """
    Full spatial-temporal model.

        Input : (B, T, 3, 224, 224) — batch of frame sequences
        Output: (B,) anomaly scores
    """

    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        seq_length: int = 16,
    ):
        super().__init__()
        self.cnn  = SpatialCNN(feature_dim)
        self.lstm = TemporalLSTM(feature_dim, hidden_dim, num_layers)
        self.seq_length = seq_length

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        features = self.cnn(frames.reshape(B * T, C, H, W))  # (B*T, 512)
        features = features.view(B, T, -1)                    # (B, T, 512)
        return self.lstm(features)                             # (B,)


# ──────────────────────────────────────────────────────────────────────
#  High-level Engine  (used by main.py)
# ──────────────────────────────────────────────────────────────────────

class ActivityEngine:
    """
    Maintains a sliding window of recent frames.  When enough frames
    have been collected it runs the CNN + LSTM and returns an anomaly
    score / boolean verdict.

    If no trained weights file is found the engine signals the pipeline
    to fall back to the heuristic event trigger.
    """

    # ImageNet normalisation (required by ResNet-18 backbone)
    _transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    def __init__(
        self,
        weights_path: str,
        device: str = "cuda",
        seq_length: int = 16,
        threshold: float = 0.5,
    ):
        self.device     = device
        self.seq_length = seq_length
        self.threshold  = threshold
        self._buffer: list[torch.Tensor] = []

        self.model = ActivityAnalyzer(seq_length=seq_length)
        self.model_ready = False

        weights = Path(weights_path)
        if weights.exists():
            print(f"[Stage-3] Loading activity model weights: {weights}")
            state = torch.load(weights, map_location=device, weights_only=True)
            self.model.load_state_dict(state)
            self.model.to(device)
            self.model.eval()
            self.model_ready = True
            print("[Stage-3] CNN + LSTM activity model loaded ✓")
        else:
            print(f"[Stage-3] Weights not found → {weights}")
            print("[Stage-3] Falling back to heuristic event trigger")
            print("[Stage-3] Train on UCF-Crime to enable full activity analysis")

    # ── public API ────────────────────────────────────────────────────

    def push_frame(self, frame: np.ndarray) -> None:
        """Add an OpenCV BGR frame to the sliding window."""
        if not self.model_ready:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb)
        self._buffer.append(tensor)
        if len(self._buffer) > self.seq_length:
            self._buffer.pop(0)

    @torch.no_grad()
    def get_anomaly_score(self) -> float:
        """
        Returns:
            Anomaly score ∈ [0, 1], or -1.0 when the model is
            unavailable or the buffer has not filled yet.
        """
        if not self.model_ready:
            return -1.0
        if len(self._buffer) < self.seq_length:
            return -1.0

        seq = torch.stack(self._buffer[-self.seq_length:])   # (T, 3, 224, 224)
        seq = seq.unsqueeze(0).to(self.device)                # (1, T, 3, 224, 224)
        score = self.model(seq)                               # (1,)
        return float(score.item())

    def is_anomalous(self) -> bool | None:
        """
        Returns:
            True  — model says this activity is anomalous
            False — model says this is normal / static
            None  — model not available → caller should use heuristic
        """
        if not self.model_ready:
            return None
        score = self.get_anomaly_score()
        if score < 0:
            return None
        return score >= self.threshold

    def reset(self) -> None:
        """Clear the frame buffer (e.g. between events)."""
        self._buffer.clear()
