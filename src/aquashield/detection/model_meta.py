"""Which preprocessing profile was a checkpoint trained with?

Preprocessing is not a free-standing inference option: a detector only performs
as intended on the distribution it was trained on. We measured a 12x F1
degradation from applying the `standard` profile at inference to a model trained
on raw frames. So the profile is a property OF THE CHECKPOINT, and it travels
with it in a sidecar file.

    models/best.pt  ->  models/best.meta.json  {"preprocess_profile": "standard"}

When no sidecar exists we assume "none" (raw), because that is what
`scripts/train.py` produces unless it was pointed at a preprocessed dataset. The
assumption is reported so it is never silent.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_PROFILE = "none"


def meta_path(weights: str | Path) -> Path:
    p = Path(weights)
    return p.with_suffix(".meta.json")


def write_meta(weights: str | Path, *, preprocess_profile: str, **extra) -> Path:
    mp = meta_path(weights)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps({"preprocess_profile": preprocess_profile, **extra}, indent=2))
    return mp


def read_meta(weights: str | Path) -> dict:
    mp = meta_path(weights)
    if mp.exists():
        try:
            d = json.loads(mp.read_text())
            d["_source"] = str(mp)
            return d
        except json.JSONDecodeError:
            pass
    return {"preprocess_profile": DEFAULT_PROFILE, "_source": None,
            "_assumed": True,
            "_note": (f"No sidecar metadata for {Path(weights).name}; assuming the "
                      f"checkpoint was trained on '{DEFAULT_PROFILE}' (raw) frames. "
                      "If it was trained on preprocessed imagery, write a "
                      ".meta.json or detections will be degraded by a "
                      "train/inference mismatch.")}


def preprocess_profile_for_model(weights: str | Path) -> str:
    return read_meta(weights).get("preprocess_profile", DEFAULT_PROFILE)
