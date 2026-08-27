"""Physically-motivated features for separating man-made targets from seabed clutter.

Every feature here is computed from the actual pixels of the detection's
neighbourhood. None of them are model outputs, so they give the verification
stage evidence that is genuinely independent of the detector's own opinion.

The dominant discriminator in manual side-scan interpretation
------------------------------------------------------------
An object standing PROUD of the seabed blocks the acoustic beam and casts a
shadow on the far-range side of its highlight. Sand ripples, gravel and flat
scour marks produce bright returns with NO isolated shadow. So "bright blob with
a coherent shadow immediately down-range" is the single strongest evidence that a
return is a real 3-D object rather than a texture artefact.

Range geometry: the sonar is at the nadir column. Range increases with distance
from nadir, so for a target left of nadir the shadow lies further LEFT, and for a
target right of nadir it lies further RIGHT.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

FEATURE_NAMES = [
    "target_contrast",
    "shadow_ratio",
    "shadow_side_consistent",
    "highlight_compactness",
    "aspect_ratio",
    "edge_straightness",
    "texture_homogeneity",
    "background_roughness",
    "local_snr",
    "size_rank",
]


@dataclass
class PatchFeatures:
    target_contrast: float          # (target mean - bg mean) / bg std
    shadow_ratio: float             # how much darker the down-range strip is than bg
    shadow_side_consistent: float   # 1.0 if the darker side is the far-range side
    highlight_compactness: float    # 4*pi*A/P^2 of the thresholded highlight (1 = circle)
    aspect_ratio: float             # long/short side of the box
    edge_straightness: float        # fraction of highlight perimeter explained by line segments
    texture_homogeneity: float      # GLCM-free proxy: 1 - normalised local gradient energy
    background_roughness: float     # std of the surrounding annulus (clutter level)
    local_snr: float                # target peak over background noise sigma
    size_rank: float                # box area as a fraction of frame area (log-scaled)

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in FEATURE_NAMES], dtype=np.float32)

    def as_dict(self) -> dict:
        return {k: (round(float(v), 4) if np.isfinite(v) else 0.0)
                for k, v in asdict(self).items()}


def _safe(v: float, default: float = 0.0) -> float:
    return float(v) if np.isfinite(v) else default


def _crop(gray: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    h, w = gray.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((1, 1), np.float32)
    return gray[y0:y1, x0:x1]


def extract(gray01: np.ndarray, box_xyxy, nadir_col: float | None = None) -> PatchFeatures:
    """Compute features for one detection.

    gray01    : full frame, float32 in [0,1]
    box_xyxy  : detection box in whole-image pixel coordinates
    nadir_col : column of the sonar nadir. If None we assume the frame centre and
                mark shadow-side consistency as unknown (0.5) rather than
                pretending we know the range direction.
    """
    h, w = gray01.shape
    x0, y0, x1, y1 = (int(round(v)) for v in box_xyxy)
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)

    target = _crop(gray01, x0, y0, x1, y1)
    if target.size < 4:
        return PatchFeatures(*([0.0] * len(FEATURE_NAMES)))

    # --- background annulus: a ring around the box, excluding the box itself ---
    pad = int(max(bw, bh) * 1.5) + 4
    ring = _crop(gray01, x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    mask = np.ones(ring.shape, bool)
    ry0 = min(max(y0 - max(0, y0 - pad), 0), ring.shape[0] - 1)
    rx0 = min(max(x0 - max(0, x0 - pad), 0), ring.shape[1] - 1)
    mask[ry0 : ry0 + bh, rx0 : rx0 + bw] = False
    bg = ring[mask] if mask.any() else ring.ravel()
    bg_mean, bg_std = float(bg.mean()), float(bg.std())

    t_mean, t_max = float(target.mean()), float(target.max())
    target_contrast = _safe((t_mean - bg_mean) / (bg_std + 1e-6))
    local_snr = _safe((t_max - bg_mean) / (bg_std + 1e-6))

    # --- shadow evidence -----------------------------------------------------
    # Sample a strip of the same height immediately beyond the box on each side.
    strip_w = max(4, int(bw * 1.2))
    left = _crop(gray01, x0 - strip_w, y0, x0, y1)
    right = _crop(gray01, x1, y0, x1 + strip_w, y1)
    l_mean = float(left.mean()) if left.size > 1 else bg_mean
    r_mean = float(right.mean()) if right.size > 1 else bg_mean

    # A shadow is a strip significantly darker than the surrounding background.
    l_ratio = _safe((bg_mean - l_mean) / (bg_mean + 1e-6))
    r_ratio = _safe((bg_mean - r_mean) / (bg_mean + 1e-6))
    shadow_ratio = float(np.clip(max(l_ratio, r_ratio), 0.0, 1.0))

    cx = (x0 + x1) / 2.0
    if nadir_col is None:
        side_consistent = 0.5           # unknown range direction -- do not pretend
    else:
        far_is_left = cx < nadir_col
        darker_is_left = l_ratio >= r_ratio
        side_consistent = 1.0 if (far_is_left == darker_is_left) else 0.0

    # --- highlight shape -----------------------------------------------------
    t8 = (np.clip(target, 0, 1) * 255).astype(np.uint8)
    thr = max(int(t8.mean()) + 1, 1)
    _, binm = cv2.threshold(t8, thr, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    compactness, straightness = 0.0, 0.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        per = float(cv2.arcLength(c, True))
        if per > 1e-6:
            compactness = float(np.clip(4 * np.pi * area / (per * per), 0.0, 1.0))
            # Douglas-Peucker: few vertices for the same perimeter => straight edges
            approx = cv2.approxPolyDP(c, 0.02 * per, True)
            straightness = float(np.clip(1.0 - len(approx) / max(len(c) / 4.0, 1.0), 0.0, 1.0))

    aspect_ratio = float(max(bw, bh) / max(min(bw, bh), 1))

    # --- texture -------------------------------------------------------------
    # Smoothness of the target RELATIVE to its own surroundings. An absolute
    # gradient threshold is meaningless because sonar gain varies per survey.
    def _grad_energy(a: np.ndarray) -> float:
        if a.size < 9:
            return float("nan")
        gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.sqrt(gx * gx + gy * gy).mean())

    g_t = _grad_energy(target)
    g_b = _grad_energy(ring)
    if np.isfinite(g_t) and np.isfinite(g_b) and (g_t + g_b) > 1e-9:
        # 1 => target far smoother than background (man-made highlight)
        # 0 => target far rougher than background (rock/gravel cluster)
        texture_homogeneity = float(np.clip(g_b / (g_t + g_b), 0.0, 1.0))
    else:
        texture_homogeneity = 0.5

    background_roughness = _safe(bg_std)
    # log-scaled box area as a fraction of the frame: 1e-6 -> 0, 1.0 -> 1
    area_frac = (bw * bh) / float(h * w)
    size_rank = float(np.clip((np.log10(max(area_frac, 1e-6)) + 6.0) / 6.0, 0.0, 1.0))

    return PatchFeatures(
        target_contrast=float(np.clip(target_contrast, -10, 10)),
        shadow_ratio=shadow_ratio,
        shadow_side_consistent=float(side_consistent),
        highlight_compactness=compactness,
        aspect_ratio=float(np.clip(aspect_ratio, 1.0, 20.0)),
        edge_straightness=straightness,
        texture_homogeneity=texture_homogeneity,
        background_roughness=float(np.clip(background_roughness, 0, 1)),
        local_snr=float(np.clip(local_snr, -10, 20)),
        size_rank=size_rank,
    )
