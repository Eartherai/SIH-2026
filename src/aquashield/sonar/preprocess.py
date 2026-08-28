"""Side-scan sonar preprocessing.

Design rule (PS 26057 section 18): every operation is INDIVIDUALLY SWITCHABLE and
must be justifiable from sonar physics. We do not apply a filter because it
sounds appropriate -- scripts/ablate_preprocessing.py measures whether each step
actually helps detection, and the ablation table in docs/BENCHMARKS.md reports
the measured result.

Orientation convention: rows = along-track (pings), columns = across-track (range).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import cv2
import numpy as np

from .qc import detect_water_column, dropout_rows, _to_gray01


@dataclass
class PreprocessConfig:
    """Toggle and parameterise each stage. Defaults are the profile that the
    ablation in docs/BENCHMARKS.md found best on the MILCO/NOMBO val survey."""
    water_column_removal: bool = False
    water_column_mode: str = "inpaint"       # "inpaint" | "mask" | "split"

    denoise: bool = True
    denoise_method: str = "lee"              # "lee" | "median" | "bilateral" | "nlm"
    lee_window: int = 5
    lee_damping: float = 1.0

    gain_normalization: bool = True          # empirical across-track (range) gain correction
    gain_strength: float = 1.0               # 0 = off, 1 = full flattening

    dynamic_range_normalization: bool = True
    dr_clip_percentiles: tuple[float, float] = (1.0, 99.0)

    histogram_equalization: bool = False     # global HistEq (thesis 5-step step 3)
    contrast_normalization: bool = False     # CLAHE
    clahe_clip: float = 2.0
    clahe_grid: int = 8

    morphology: bool = False                 # thesis 5-step step 5
    morphology_op: str = "open"              # "open" | "close" | "tophat"
    morphology_ksize: int = 3

    slant_range_correction: bool = False     # requires altitude; off by default (see note)
    altitude_px: float | None = None

    dropout_handling: bool = True            # interpolate dead ping rows

    def as_dict(self) -> dict:
        d = asdict(self)
        d["dr_clip_percentiles"] = list(self.dr_clip_percentiles)
        return d


@dataclass
class PreprocessResult:
    image: np.ndarray                       # uint8 HxW, ready for the detector
    steps_applied: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Individual operations
# --------------------------------------------------------------------------

def lee_filter(gray: np.ndarray, window: int = 5, damping: float = 1.0) -> np.ndarray:
    """Lee adaptive speckle filter for multiplicative noise.

    Sonar speckle is multiplicative, so a plain Gaussian blur destroys target
    edges while barely improving SNR. The Lee filter shrinks toward the local
    mean only where the local variance is consistent with pure speckle, so
    genuine target/shadow boundaries survive.

        out = mean + k * (x - mean),  k = var_signal / (var_signal + var_noise)

    Reference: Lee, J.S. (1980), "Digital image enhancement and noise filtering
    by use of local statistics", IEEE TPAMI.
    """
    window = max(3, int(window) | 1)
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (window, window))
    mean_sq = cv2.blur(g * g, (window, window))
    var = np.maximum(mean_sq - mean * mean, 0.0)

    # Global speckle variance estimated from the image's own coefficient of variation.
    mu = float(np.mean(g)) + 1e-6
    cu2 = float(np.var(g)) / (mu * mu)          # squared global CV
    noise_var = cu2 * mean * mean * float(damping)

    k = var / (var + noise_var + 1e-8)
    return np.clip(mean + k * (g - mean), 0.0, 1.0)


def remove_water_column(gray: np.ndarray, mode: str = "inpaint") -> tuple[np.ndarray, dict]:
    """Suppress the nadir band so the detector never sees it as a linear target.

    The water column is a persistent dark stripe. Left in place it (a) wastes
    detector capacity and (b) produces spurious elongated 'targets' along its
    edges. Modes:
      inpaint - fill from neighbouring range bins (keeps geometry intact)
      mask    - set to the frame median (cheapest, keeps geometry)
      split   - drop the band and concatenate port/starboard (changes geometry;
                only safe when downstream coordinates are recomputed)
    """
    bounds = detect_water_column(gray)
    if bounds is None:
        return gray, {"water_column_detected": False, "bounds": None, "mode": mode}
    a, b = bounds
    out = gray.copy()
    if mode == "split":
        out = np.concatenate([gray[:, :a], gray[:, b:]], axis=1)
    elif mode == "mask":
        out[:, a:b] = float(np.median(gray))
    else:  # inpaint
        mask = np.zeros(gray.shape, np.uint8)
        mask[:, a:b] = 255
        u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
        filled = cv2.inpaint(u8, mask, 3, cv2.INPAINT_TELEA)
        out = filled.astype(np.float32) / 255.0
    return out, {"water_column_detected": True, "bounds": [int(a), int(b)], "mode": mode}


def gain_normalize(gray: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Empirical across-track gain correction (a data-driven stand-in for TVG).

    Acoustic return falls off with range because of spreading and absorption, so
    the outer swath is systematically darker than the nadir region. Without a
    per-ping TVG record we correct EMPIRICALLY: each range bin (column) is
    divided by its own along-track mean, flattening the average range response
    while preserving along-track anomalies -- which is exactly what a target is.

    Limitation, stated plainly: this is not a calibrated TVG inversion and does
    not recover absolute backscatter. It is a contrast-conditioning step only.
    """
    g = gray.astype(np.float32)
    col_mean = g.mean(axis=0, keepdims=True)
    # Smooth the range profile so we correct the trend, not individual targets.
    prof = cv2.GaussianBlur(col_mean, (max(3, (g.shape[1] // 32) | 1), 1), 0)
    prof = np.maximum(prof, 1e-3)
    target = float(g.mean())
    corrected = g * (target / prof)
    out = (1.0 - strength) * g + strength * corrected
    return np.clip(out, 0.0, 1.0)


def normalize_dynamic_range(gray: np.ndarray, lo_p: float = 1.0, hi_p: float = 99.0) -> np.ndarray:
    """Percentile stretch. Robust to the few very bright specular returns that
    would otherwise compress the whole histogram."""
    lo, hi = np.percentile(gray, [lo_p, hi_p])
    if hi - lo < 1e-6:
        return gray
    return np.clip((gray - lo) / (hi - lo), 0.0, 1.0)


def clahe(gray: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    op = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    return op.apply(u8).astype(np.float32) / 255.0


def histogram_equalize(gray: np.ndarray) -> np.ndarray:
    """Global histogram equalisation (thesis 5-step, step 3).

    Included ONLY to reproduce the thesis pipeline faithfully. Applying global
    HistEq and then CLAHE (step 4) is a double contrast stretch that is unusual
    for a detection front-end; we keep it exactly as the thesis specifies so the
    comparison is honest, and let the measurement decide.
    """
    u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    return cv2.equalizeHist(u8).astype(np.float32) / 255.0


def morphological(gray: np.ndarray, op: str = "open", ksize: int = 3) -> np.ndarray:
    """Grayscale morphology (thesis 5-step, step 5).

    Note, stated plainly: an opening with a 3x3+ kernel erodes exactly the small
    compact highlights that are our hardest targets (~24 px). We reproduce it to
    test the thesis claim, not because it is obviously safe for small-object
    detection.
    """
    u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, ksize), max(1, ksize)))
    ops = {"open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE, "tophat": cv2.MORPH_TOPHAT}
    return cv2.morphologyEx(u8, ops.get(op, cv2.MORPH_OPEN), k).astype(np.float32) / 255.0


def fix_dropouts(gray: np.ndarray, tol: float = 1e-3) -> tuple[np.ndarray, dict]:
    """Linearly interpolate ping rows lost to vehicle motion or telemetry gaps.

    A dead row is a horizontal stripe. Leaving it produces two strong artificial
    edges that a detector happily reports as a long thin object.
    """
    g = gray.copy()
    idx = dropout_rows(g, tol)
    dead = np.zeros(g.shape[0], bool)
    dead[idx] = True
    if idx.size == 0 or idx.size == g.shape[0]:
        return g, {"dead_rows": int(idx.size), "repaired": 0}
    good = np.flatnonzero(~dead)
    for r in idx:
        prev = good[good < r]
        nxt = good[good > r]
        if prev.size and nxt.size:
            p, n = prev[-1], nxt[0]
            w = (r - p) / (n - p)
            g[r] = (1 - w) * g[p] + w * g[n]
        elif prev.size:
            g[r] = g[prev[-1]]
        else:
            g[r] = g[nxt[0]]
    return g, {"dead_rows": int(idx.size), "repaired": int(idx.size)}


def slant_range_correct(gray: np.ndarray, altitude_px: float) -> np.ndarray:
    """Convert slant range to ground range about the nadir.

        ground_range = sqrt(slant_range^2 - altitude^2)

    Only valid when the towfish altitude is known in the SAME pixel units as the
    range axis. AQUA-SHIELD never guesses altitude: if it is unknown this step
    stays off and the report says so. Applying it with a fabricated altitude
    would distort every downstream size and position estimate.
    """
    h, w = gray.shape
    centre = w / 2.0
    x = np.arange(w, dtype=np.float32) - centre
    slant = np.abs(x)
    ground = np.sqrt(np.maximum(slant ** 2 - altitude_px ** 2, 0.0))
    max_g = ground.max() + 1e-6
    # map each output ground-range bin back to its source slant-range bin
    out_g = np.linspace(0, max_g, int(centre))
    src = np.sqrt(out_g ** 2 + altitude_px ** 2)
    src = np.clip(src, 0, centre - 1)
    left_idx = (centre - 1 - src)[::-1]
    right_idx = centre + src
    map_x = np.concatenate([left_idx, right_idx]).astype(np.float32)
    map_x = np.clip(map_x, 0, w - 1)
    map_x = np.tile(map_x[None, :], (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, map_x.shape[1]))
    return cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def preprocess(img: np.ndarray, cfg: PreprocessConfig | None = None) -> PreprocessResult:
    cfg = cfg or PreprocessConfig()
    gray = _to_gray01(img)
    steps: list[str] = []
    diag: dict = {}

    if cfg.dropout_handling:
        gray, d = fix_dropouts(gray)
        diag["dropouts"] = d
        if d["repaired"]:
            steps.append(f"dropout_interpolation({d['repaired']} rows)")

    if cfg.water_column_removal:
        gray, d = remove_water_column(gray, cfg.water_column_mode)
        diag["water_column"] = d
        steps.append(f"water_column_removal({cfg.water_column_mode})"
                     if d["water_column_detected"] else "water_column_removal(no band found)")

    if cfg.slant_range_correction:
        if cfg.altitude_px is None:
            diag["slant_range"] = "SKIPPED - altitude unknown; refusing to guess"
            steps.append("slant_range_correction(SKIPPED: no altitude)")
        else:
            gray = slant_range_correct(gray, float(cfg.altitude_px))
            steps.append(f"slant_range_correction(alt={cfg.altitude_px}px)")

    if cfg.denoise:
        m = cfg.denoise_method
        if m == "lee":
            gray = lee_filter(gray, cfg.lee_window, cfg.lee_damping)
        elif m == "median":
            gray = cv2.medianBlur((gray * 255).astype(np.uint8),
                                  max(3, cfg.lee_window | 1)).astype(np.float32) / 255.0
        elif m == "bilateral":
            gray = cv2.bilateralFilter(gray.astype(np.float32), 5, 0.1, 5)
        elif m == "nlm":
            gray = cv2.fastNlMeansDenoising((gray * 255).astype(np.uint8), None, 7, 7,
                                            21).astype(np.float32) / 255.0
        else:
            raise ValueError(f"unknown denoise_method: {m}")
        steps.append(f"denoise({m})")

    if cfg.gain_normalization:
        gray = gain_normalize(gray, cfg.gain_strength)
        steps.append(f"gain_normalization(strength={cfg.gain_strength})")

    if cfg.dynamic_range_normalization:
        gray = normalize_dynamic_range(gray, *cfg.dr_clip_percentiles)
        steps.append(f"dynamic_range_normalization({cfg.dr_clip_percentiles})")

    if cfg.histogram_equalization:
        gray = histogram_equalize(gray)
        steps.append("histogram_equalization")

    if cfg.contrast_normalization:
        gray = clahe(gray, cfg.clahe_clip, cfg.clahe_grid)
        steps.append(f"clahe(clip={cfg.clahe_clip},grid={cfg.clahe_grid})")

    if cfg.morphology:
        gray = morphological(gray, cfg.morphology_op, cfg.morphology_ksize)
        steps.append(f"morphology({cfg.morphology_op},k={cfg.morphology_ksize})")

    return PreprocessResult(
        image=(np.clip(gray, 0, 1) * 255).astype(np.uint8),
        steps_applied=steps,
        diagnostics=diag,
    )


PROFILES: dict[str, PreprocessConfig] = {
    "none": PreprocessConfig(denoise=False, gain_normalization=False,
                             dynamic_range_normalization=False, dropout_handling=False),
    "minimal": PreprocessConfig(denoise=False, gain_normalization=False,
                                dynamic_range_normalization=True),
    "standard": PreprocessConfig(),                                  # lee + gain + DR
    "aggressive": PreprocessConfig(water_column_removal=True, contrast_normalization=True,
                                   denoise_method="lee", lee_window=7),
    # Faithful reproduction of Divyabarathi (2025) "5-Step Signal Preprocessing":
    #   TVG -> Median -> Histogram Equalization -> CLAHE -> Morphology.
    # TVG is reproduced by our empirical across-track gain normalisation (we cannot
    # do a true sonar-equation TVG without raw intensities / range params). This
    # profile exists to test the thesis claim on SSS under matched train/inference.
    "thesis5step": PreprocessConfig(
        dropout_handling=False, water_column_removal=False,
        denoise=True, denoise_method="median", lee_window=3,
        gain_normalization=True, gain_strength=1.0,          # TVG stand-in
        dynamic_range_normalization=False,
        histogram_equalization=True,
        contrast_normalization=True, clahe_clip=2.0, clahe_grid=8,
        morphology=True, morphology_op="open", morphology_ksize=3),
}
