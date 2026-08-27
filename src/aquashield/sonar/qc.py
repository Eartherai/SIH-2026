"""Quality control for side-scan sonar frames.

Every number returned here is MEASURED from the pixel data. Nothing is assumed
or filled in with a plausible-looking constant. Where a quantity cannot be
determined from the image alone (e.g. true slant range, altitude), the field is
returned as None rather than guessed.

Waterfall convention used throughout AQUA-SHIELD
------------------------------------------------
    rows    (axis 0) = along-track  -- successive pings
    columns (axis 1) = across-track -- range bins, nadir near the centre

If a source image does not follow this convention the caller must transpose it
before QC; we do not silently guess the orientation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class QCReport:
    quality_score: float                 # 0..1 composite, see _composite()
    dynamic_range: float                 # p99 - p1, in 0..1 intensity units
    unique_levels: int                   # distinct 8-bit levels present
    speckle_index: float                 # sigma/mu in flat regions (multiplicative noise proxy)
    dropout_ratio: float                 # fraction of ping rows that are dead/constant
    saturated_ratio: float               # fraction of pixels at 0 or 255
    water_column_detected: bool
    water_column_bounds: tuple[int, int] | None   # (col_start, col_end) or None
    usable_region_ratio: float           # fraction of pixels considered analysable
    blur_score: float                    # variance of Laplacian (higher = sharper)
    notes: list[str]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["water_column_bounds"] = list(self.water_column_bounds) if self.water_column_bounds else None
        return d


def _to_gray01(img: np.ndarray) -> np.ndarray:
    a = img
    if a.ndim == 3:
        a = a.mean(axis=2)
    a = a.astype(np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def detect_water_column(gray: np.ndarray, dark_frac: float = 0.55,
                        min_frac: float = 0.01, max_frac: float = 0.35) -> tuple[int, int] | None:
    """Locate the nadir / water-column band.

    Physics: directly beneath the towfish there is no seabed return until the
    first bottom echo, so the swath centre shows a DARK band. Crucially that
    dark band is split by a BRIGHT spike -- the first bottom return at nadir.
    A naive "find the darkest contiguous run" therefore finds only half the band.

    Algorithm:
      1. smooth the along-track column-mean profile (suppresses speckle spikes),
      2. mark columns darker than `dark_frac` x the frame's median column mean,
      3. bridge narrow bright gaps so the nadir echo does not split the band,
      4. keep the run containing (or nearest) the swath centre,
      5. sanity-check its width.

    Returns (start_col, end_col_exclusive), or None when no band is present --
    the normal case for nadir-removed or georectified products.
    """
    h, w = gray.shape
    if w < 16:
        return None
    col_mean = gray.mean(axis=0)
    k = max(3, (w // 128) | 1)
    smooth = np.convolve(col_mean, np.ones(k, dtype=np.float32) / k, mode="same")

    med = float(np.median(smooth))
    if med <= 1e-6:
        return None
    dark = smooth < (dark_frac * med)
    if not dark.any():
        return None

    # Bridge the bright nadir echo (and any narrow specular return) so the two
    # halves of the water column are recognised as one band.
    bridge = max(2, int(0.03 * w))
    filled = dark.copy()
    idx = np.flatnonzero(dark)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < (b - a) <= bridge:
            filled[a:b] = True

    # Contiguous runs of `filled`
    runs: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(filled):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(filled)))
    if not runs:
        return None

    centre = w / 2.0
    # Prefer a run that straddles the centre; otherwise the closest one.
    containing = [r for r in runs if r[0] <= centre < r[1]]
    best = containing[0] if containing else min(
        runs, key=lambda r: abs((r[0] + r[1]) / 2.0 - centre))

    width = best[1] - best[0]
    if width < max(3, int(min_frac * w)) or width > max_frac * w:
        return None
    # The band must actually be near the centre; a dark patch at the swath edge
    # is outer-range fall-off, not a water column.
    if abs((best[0] + best[1]) / 2.0 - centre) > 0.25 * w:
        return None
    return int(best[0]), int(best[1])


def dropout_ratio(gray: np.ndarray, tol: float = 1e-3, mad_k: float = 4.0) -> float:
    """Fraction of ping rows carrying no usable information.

    Vehicle heave/pitch/roll and telemetry gaps produce two distinct artefacts:
      * fully CONSTANT rows (usually all-zero), and
      * rows that are present but anomalously DARK and low-variance -- a partial
        dropout that still contains noise.
    Both appear as horizontal stripes and both generate strong artificial edges
    that a detector reports as long thin objects, so we count both.
    """
    row_ptp = gray.max(axis=1) - gray.min(axis=1)
    constant = row_ptp < tol

    rm = gray.mean(axis=1)
    med = float(np.median(rm))
    mad = float(np.median(np.abs(rm - med)))
    if mad > 1e-6:
        degraded = rm < (med - mad_k * 1.4826 * mad)
    else:
        degraded = np.zeros_like(constant)
    return float((constant | degraded).mean())


def dropout_rows(gray: np.ndarray, tol: float = 1e-3, mad_k: float = 4.0) -> np.ndarray:
    """Indices of the rows counted by `dropout_ratio` (used by the repair step)."""
    row_ptp = gray.max(axis=1) - gray.min(axis=1)
    constant = row_ptp < tol
    rm = gray.mean(axis=1)
    med = float(np.median(rm))
    mad = float(np.median(np.abs(rm - med)))
    degraded = (rm < (med - mad_k * 1.4826 * mad)) if mad > 1e-6 else np.zeros_like(constant)
    return np.flatnonzero(constant | degraded)


def speckle_index(gray: np.ndarray, patch: int = 16) -> float:
    """Estimate sigma/mu over locally flat patches.

    Side-scan speckle is multiplicative, so the coefficient of variation in a
    homogeneous patch is the natural noise descriptor. We take the median CV
    over the flattest quartile of patches so that real targets and shadows do
    not dominate the estimate.
    """
    h, w = gray.shape
    ph, pw = min(patch, h), min(patch, w)
    if ph < 4 or pw < 4:
        return float("nan")
    nh, nw = h // ph, w // pw
    if nh == 0 or nw == 0:
        return float("nan")
    blocks = (gray[: nh * ph, : nw * pw]
              .reshape(nh, ph, nw, pw)
              .transpose(0, 2, 1, 3)
              .reshape(-1, ph * pw))
    mu = blocks.mean(axis=1)
    sd = blocks.std(axis=1)
    valid = mu > 1e-3
    if not valid.any():
        return float("nan")
    cv = sd[valid] / mu[valid]
    # flattest quartile == lowest CV; median of those is a stable speckle proxy
    q = np.quantile(cv, 0.25)
    flat = cv[cv <= q]
    return float(np.median(flat)) if flat.size else float(np.median(cv))


def blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian. Higher = sharper. Used to flag defocused tiles."""
    lap = (-4.0 * gray
           + np.roll(gray, 1, 0) + np.roll(gray, -1, 0)
           + np.roll(gray, 1, 1) + np.roll(gray, -1, 1))
    return float(lap[1:-1, 1:-1].var())


def _composite(dr: float, drop: float, sat: float, spk: float, usable: float) -> float:
    """Composite 0..1 quality score.

    This is an ENGINEERING HEURISTIC, not a calibrated physical measure. It is
    used only to (a) warn the operator and (b) act as one soft input to the
    confidence engine. It is deliberately simple and fully inspectable.
    """
    s_dr = np.clip(dr / 0.6, 0, 1)                     # want a wide dynamic range
    s_drop = 1.0 - np.clip(drop / 0.20, 0, 1)          # want few dead pings
    s_sat = 1.0 - np.clip(sat / 0.30, 0, 1)            # want little clipping
    s_spk = 1.0 - np.clip((spk - 0.15) / 0.55, 0, 1) if np.isfinite(spk) else 0.5
    s_use = np.clip(usable, 0, 1)
    w = np.array([0.25, 0.25, 0.15, 0.20, 0.15])
    v = np.array([s_dr, s_drop, s_sat, s_spk, s_use], dtype=float)
    return float(np.clip((w * v).sum(), 0.0, 1.0))


def assess(img: np.ndarray) -> QCReport:
    """Run all QC measurements on a single sonar frame."""
    gray = _to_gray01(img)
    h, w = gray.shape
    notes: list[str] = []

    p1, p99 = np.percentile(gray, [1, 99])
    dr = float(p99 - p1)
    levels = int(np.unique((gray * 255).astype(np.uint8)).size)
    sat = float(((gray <= 1e-6) | (gray >= 1 - 1e-6)).mean())
    drop = dropout_ratio(gray)
    spk = speckle_index(gray)
    blur = blur_score(gray)

    wc = detect_water_column(gray)
    wc_px = (wc[1] - wc[0]) * h if wc else 0
    dead_px = drop * h * w
    usable = float(max(0.0, 1.0 - (wc_px + dead_px) / (h * w)))

    if dr < 0.15:
        notes.append("Very low dynamic range - contrast normalisation strongly recommended.")
    if drop > 0.05:
        notes.append(f"{drop:.1%} of ping rows are dead (probable data dropout).")
    if sat > 0.25:
        notes.append(f"{sat:.1%} of pixels are clipped at black/white.")
    if np.isfinite(spk) and spk > 0.5:
        notes.append("High speckle index - denoising recommended before detection.")
    if wc:
        notes.append(f"Water column detected at columns {wc[0]}-{wc[1]} "
                     f"({(wc[1]-wc[0])/w:.1%} of swath width).")
    else:
        notes.append("No water-column band detected (image may already be nadir-removed "
                     "or georectified).")
    if levels < 32:
        notes.append(f"Only {levels} distinct intensity levels - possible over-quantisation.")

    return QCReport(
        quality_score=_composite(dr, drop, sat, spk, usable),
        dynamic_range=round(dr, 4),
        unique_levels=levels,
        speckle_index=round(spk, 4) if np.isfinite(spk) else float("nan"),
        dropout_ratio=round(drop, 4),
        saturated_ratio=round(sat, 4),
        water_column_detected=wc is not None,
        water_column_bounds=wc,
        usable_region_ratio=round(usable, 4),
        blur_score=round(blur, 6),
        notes=notes,
    )
