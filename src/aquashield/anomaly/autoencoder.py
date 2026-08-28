"""Unsupervised anomaly branch — a convolutional autoencoder over normal seabed.

Why this exists (PS 26057 asks for anomaly detection; the thesis lists its absence
as a gap): a supervised detector can only find classes it was trained on. An
autoencoder trained ONLY on normal seabed learns to reconstruct ordinary ripples,
sand and rock, and reconstructs *poorly* anywhere an unfamiliar structure appears.
The per-patch reconstruction error is an "unlike-normal-seabed" score that is
independent of the detector's class list.

Deliberately small: this is an MVP, not a generative model. It runs on the M5 in
minutes and adds a score the operator reads SEPARATELY from detection confidence
(PS 26057 §16/§27 — confidence, anomaly and priority are three different things).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class SeabedAutoencoder(nn.Module):
    """Tiny conv AE for 1-channel PATCH reconstruction (default 64x64)."""

    def __init__(self, ch: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1), nn.BatchNorm2d(ch), nn.LeakyReLU(0.2, True),      # 32
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.BatchNorm2d(ch * 2), nn.LeakyReLU(0.2, True),  # 16
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.BatchNorm2d(ch * 4), nn.LeakyReLU(0.2, True),  # 8
            nn.Conv2d(ch * 4, ch * 4, 4, 2, 1), nn.BatchNorm2d(ch * 4), nn.LeakyReLU(0.2, True),  # 4
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ch * 4, ch * 4, 4, 2, 1), nn.BatchNorm2d(ch * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1), nn.BatchNorm2d(ch * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1), nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.ConvTranspose2d(ch, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


@dataclass
class AnomalyScorer:
    """Wraps a trained AE and turns a frame into per-patch reconstruction errors."""
    model: SeabedAutoencoder
    device: str
    patch: int = 64
    stride: int = 32
    # calibration: errors are z-scored against the normal-seabed error distribution
    norm_mean: float = 0.0
    norm_std: float = 1.0

    @torch.inference_mode()
    def _errors(self, gray01: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
        h, w = gray01.shape
        ph = self.patch
        xs = list(range(0, max(1, w - ph + 1), self.stride)) or [0]
        ys = list(range(0, max(1, h - ph + 1), self.stride)) or [0]
        patches, coords = [], []
        for y in ys:
            for x in xs:
                p = gray01[y:y + ph, x:x + ph]
                if p.shape != (ph, ph):
                    p = np.pad(p, ((0, ph - p.shape[0]), (0, ph - p.shape[1])), mode="edge")
                patches.append(p)
                coords.append((x, y))
        t = torch.from_numpy(np.stack(patches)[:, None].astype(np.float32)).to(self.device)
        out = self.model(t)
        err = ((out - t) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
        return err, coords

    def frame_score(self, gray01: np.ndarray, reduce: str = "p99") -> float:
        """One anomaly score per frame. p99 of patch errors is robust to a single
        odd patch while still reacting to a localised anomaly."""
        err, _ = self._errors(gray01)
        raw = float(np.percentile(err, 99) if reduce == "p99" else err.max()
                    if reduce == "max" else err.mean())
        return float((raw - self.norm_mean) / (self.norm_std + 1e-9))

    def region_score(self, gray01: np.ndarray, box_xyxy) -> float:
        """Anomaly score for one detection's neighbourhood (z-scored)."""
        h, w = gray01.shape
        x0, y0, x1, y1 = (int(round(v)) for v in box_xyxy)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        ph = self.patch
        x = int(np.clip(cx - ph // 2, 0, max(0, w - ph)))
        y = int(np.clip(cy - ph // 2, 0, max(0, h - ph)))
        p = gray01[y:y + ph, x:x + ph]
        if p.shape != (ph, ph):
            p = np.pad(p, ((0, ph - p.shape[0]), (0, ph - p.shape[1])), mode="edge")
        t = torch.from_numpy(p[None, None].astype(np.float32)).to(self.device)
        with torch.inference_mode():
            out = self.model(t)
        raw = float(((out - t) ** 2).mean().item())
        return float((raw - self.norm_mean) / (self.norm_std + 1e-9))


def build_scorer(weights: str, device: str, ch: int = 32, patch: int = 64,
                 stride: int = 32, norm_mean: float = 0.0, norm_std: float = 1.0):
    m = SeabedAutoencoder(ch).to(device)
    state = torch.load(weights, map_location=device)
    m.load_state_dict(state["model"] if "model" in state else state)
    m.eval()
    return AnomalyScorer(m, device, patch, stride, norm_mean, norm_std)
