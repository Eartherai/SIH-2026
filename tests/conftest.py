import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def synthetic_waterfall():
    """A side-scan-like frame with a known water column, a known target and a
    known dropout row, so QC/preprocessing can be checked against ground truth."""
    rng = np.random.default_rng(42)
    h, w = 400, 400
    img = (rng.gamma(4.0, 0.04, (h, w))).astype(np.float32)      # speckled seabed
    # range-dependent fall-off: brighter near nadir
    x = np.abs(np.arange(w) - w / 2) / (w / 2)
    img *= (1.0 - 0.5 * x)[None, :]
    # water column: dark band with a bright nadir echo at its centre
    img[:, 185:215] *= 0.05
    img[:, 199:201] = 0.9
    # a bright compact target with a shadow immediately down-range of it
    img[150:165, 300:318] = 0.85
    img[150:165, 318:340] = 0.02
    # one dead ping row
    img[300, :] = 0.0
    return np.clip(img, 0, 1)


@pytest.fixture
def repo_root():
    return ROOT
