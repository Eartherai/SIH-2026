"""Input adapters. Each declares what it can read; none guesses a format."""
from . import geotiff, image, jsonl_bbox
from ..geolocation.nav import load_nav_csv

__all__ = ["image", "geotiff", "jsonl_bbox", "load_nav_csv", "load_frames"]


def load_frames(paths):
    """Load an arbitrary mix of supported rasters into (name, array) pairs."""
    return image.read_many(paths)
