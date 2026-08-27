"""GeoTIFF ingestion: pixels plus the affine map that georeferences them."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .image import read as read_raster
from ..geolocation.reference import GeoTIFFReference


def can_read(path: str | Path) -> bool:
    return Path(path).suffix.lower() in {".tif", ".tiff"}


def is_georeferenced(path: str | Path) -> bool:
    """True only when the file genuinely carries a usable affine transform."""
    try:
        GeoTIFFReference(path)
        return True
    except Exception:                                          # noqa: BLE001
        return False


def read(path: str | Path) -> tuple[np.ndarray, GeoTIFFReference | None]:
    """Return (frame, georeference). The georeference is None when the TIFF has
    no spatial tags -- we do not fabricate one."""
    img = read_raster(path)
    try:
        return img, GeoTIFFReference(path)
    except Exception:                                          # noqa: BLE001
        return img, None
