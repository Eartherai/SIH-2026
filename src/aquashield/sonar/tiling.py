"""Resolution-aware tiling for large sonar mosaics.

Why this exists
---------------
A survey waterfall can be tens of thousands of pings long, while a ghost-net or
crab-pot signature may span only 20-40 px. Feeding the whole strip to a detector
at a fixed network resolution destroys exactly the small targets we care about.
We therefore cut the frame into overlapping tiles at (approximately) native
resolution, detect in each, and stitch detections back into whole-image
coordinates.

The overlap is what makes this correct: a target sitting on a tile seam would
otherwise be cut in half and missed, or reported twice. `merge_tiled_detections`
in detection/boxes.py resolves the duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tile:
    index: int
    x0: int
    y0: int
    x1: int
    y1: int
    image: np.ndarray

    @property
    def offset(self) -> tuple[int, int]:
        return self.x0, self.y0


@dataclass(frozen=True)
class TilingPlan:
    tile_size: int
    overlap: int
    n_tiles: int
    grid: tuple[int, int]      # (n_rows, n_cols)
    image_size: tuple[int, int]  # (h, w)
    tiled: bool                # False when the image was small enough to pass through


def plan_tiles(h: int, w: int, tile_size: int = 640, overlap: int = 128) -> TilingPlan:
    if tile_size <= overlap:
        raise ValueError("tile_size must exceed overlap")
    if h <= tile_size and w <= tile_size:
        return TilingPlan(tile_size, overlap, 1, (1, 1), (h, w), tiled=False)
    stride = tile_size - overlap
    n_rows = max(1, int(np.ceil(max(h - overlap, 1) / stride)))
    n_cols = max(1, int(np.ceil(max(w - overlap, 1) / stride)))
    return TilingPlan(tile_size, overlap, n_rows * n_cols, (n_rows, n_cols), (h, w), tiled=True)


def tile_image(img: np.ndarray, tile_size: int = 640, overlap: int = 128) -> tuple[list[Tile], TilingPlan]:
    """Split into overlapping tiles.

    Edge tiles are shifted inward rather than zero-padded: padding introduces a
    hard synthetic edge that detectors reliably mistake for a man-made linear
    feature.
    """
    h, w = img.shape[:2]
    plan = plan_tiles(h, w, tile_size, overlap)
    if not plan.tiled:
        return [Tile(0, 0, 0, w, h, img)], plan

    stride = tile_size - overlap
    tiles: list[Tile] = []
    idx = 0
    for r in range(plan.grid[0]):
        y0 = min(r * stride, max(0, h - tile_size))
        y1 = min(y0 + tile_size, h)
        for c in range(plan.grid[1]):
            x0 = min(c * stride, max(0, w - tile_size))
            x1 = min(x0 + tile_size, w)
            tiles.append(Tile(idx, int(x0), int(y0), int(x1), int(y1), img[y0:y1, x0:x1]))
            idx += 1
    return tiles, plan


def to_global(boxes_xyxy: np.ndarray, offset: tuple[int, int]) -> np.ndarray:
    """Shift tile-local xyxy boxes into whole-image coordinates."""
    if boxes_xyxy.size == 0:
        return boxes_xyxy.reshape(0, 4)
    dx, dy = offset
    out = np.asarray(boxes_xyxy, dtype=np.float32).copy()
    out[:, [0, 2]] += dx
    out[:, [1, 3]] += dy
    return out
