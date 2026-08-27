"""Temporal / spatial deduplication of detections into unique hazards.

The problem
-----------
A towfish pings continuously, so one physical crab pot or one length of net is
imaged in many consecutive frames, and again on an overlapping return line. A
raw detection count is therefore NOT a hazard count -- it can overstate the
number of objects on the seabed several-fold. Every operational output
(hazard tables, cleanup tasking, "objects per km") must be built on unique
physical objects, not on detections.

Two association modes
---------------------
geographic : when detections carry latitude/longitude, cluster by true ground
             distance in metres. This is the correct mode and the only one that
             can merge observations across different survey lines.
sequence   : when there is no navigation data, fall back to associating
             detections in NEARBY FRAMES whose image positions overlap. This
             only works for a contiguous ping sequence and we say so - it is
             never presented as a geographic result.

Implementation is single-link agglomerative clustering via union-find, which is
the simplest algorithm that gives stable, order-independent clusters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np

from ..detection.boxes import iou_matrix

EARTH_R = 6_371_008.8   # mean Earth radius, metres (IUGG)


@dataclass
class Observation:
    """One detection of (possibly) one physical object, in one frame."""
    obs_id: str
    frame_id: str
    frame_index: int
    box_xyxy: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float
    latitude: float | None = None
    longitude: float | None = None
    geoloc_uncertainty_m: float | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class Hazard:
    hazard_id: str
    class_id: int
    class_name: str
    observation_count: int
    observation_ids: list[str]
    frame_ids: list[str]
    best_confidence: float
    mean_confidence: float
    latitude: float | None
    longitude: float | None
    position_spread_m: float | None      # spread of member observations
    geoloc_uncertainty_m: float | None
    association_mode: str
    bbox_span_px: tuple[float, float]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["bbox_span_px"] = list(self.bbox_span_px)
        return d


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(min(1.0, math.sqrt(a)))


def deduplicate(observations: list[Observation], *, radius_m: float = 12.0,
                max_frame_gap: int = 3, min_iou: float = 0.20,
                class_aware: bool = True, prefix: str = "AQS") -> list[Hazard]:
    """Group observations into unique hazards.

    radius_m       : ground distance within which two geolocated observations are
                     taken to be the same object. Should be >= the geolocation
                     uncertainty; the default suits a towed SSS survey.
    max_frame_gap  : sequence mode only - how many frames apart two detections may
                     be and still be associated.
    min_iou        : sequence mode only - required image overlap.
    class_aware    : only merge detections that agree on class.
    """
    n = len(observations)
    if n == 0:
        return []

    geo = [o for o in observations if o.latitude is not None and o.longitude is not None]
    mode = "geographic" if len(geo) == n else ("sequence" if not geo else "mixed")

    uf = _UnionFind(n)

    if mode == "geographic":
        for i in range(n):
            for j in range(i + 1, n):
                if class_aware and observations[i].class_id != observations[j].class_id:
                    continue
                a, b = observations[i], observations[j]
                # Allow the radius to grow with the observations' own stated
                # uncertainty: we must not split one object into two just because
                # the fix was poor.
                r = radius_m + 0.5 * ((a.geoloc_uncertainty_m or 0.0) +
                                      (b.geoloc_uncertainty_m or 0.0))
                if haversine_m(a.latitude, a.longitude, b.latitude, b.longitude) <= r:
                    uf.union(i, j)
    else:
        boxes = np.array([o.box_xyxy for o in observations], np.float32)
        idx_by_frame: dict[int, list[int]] = {}
        for i, o in enumerate(observations):
            idx_by_frame.setdefault(o.frame_index, []).append(i)
        frames = sorted(idx_by_frame)
        for fi_pos, fi in enumerate(frames):
            for fj in frames[fi_pos:]:
                if fj - fi > max_frame_gap:
                    break
                for i in idx_by_frame[fi]:
                    for j in idx_by_frame[fj]:
                        if i >= j:
                            continue
                        if class_aware and observations[i].class_id != observations[j].class_id:
                            continue
                        if iou_matrix(boxes[i:i + 1], boxes[j:j + 1])[0, 0] >= min_iou:
                            uf.union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(uf.find(i), []).append(i)

    hazards: list[Hazard] = []
    # Deterministic ordering so hazard IDs are stable across runs -- essential
    # for a live demo and for re-running a survey.
    for k, (_, members) in enumerate(
            sorted(clusters.items(),
                   key=lambda kv: (observations[min(kv[1])].frame_index,
                                   observations[min(kv[1])].box_xyxy[0])), start=1):
        obs = [observations[i] for i in members]
        confs = [o.confidence for o in obs]
        lats = [o.latitude for o in obs if o.latitude is not None]
        lons = [o.longitude for o in obs if o.longitude is not None]

        lat = float(np.mean(lats)) if lats else None
        lon = float(np.mean(lons)) if lons else None
        spread = None
        if lat is not None and len(lats) > 1:
            spread = float(max(haversine_m(lat, lon, a, b) for a, b in zip(lats, lons)))

        uncs = [o.geoloc_uncertainty_m for o in obs if o.geoloc_uncertainty_m is not None]
        # Averaging N independent fixes reduces random error by ~sqrt(N); we keep
        # the systematic floor by never going below unc/2.
        unc = None
        if uncs:
            base = float(np.mean(uncs))
            unc = float(max(base / math.sqrt(len(uncs)), base * 0.5))

        best = obs[int(np.argmax(confs))]
        w = float(np.mean([o.box_xyxy[2] - o.box_xyxy[0] for o in obs]))
        h = float(np.mean([o.box_xyxy[3] - o.box_xyxy[1] for o in obs]))

        hazards.append(Hazard(
            hazard_id=f"{prefix}-{k:05d}",
            class_id=best.class_id,
            class_name=best.class_name,
            observation_count=len(obs),
            observation_ids=[o.obs_id for o in obs],
            frame_ids=sorted({o.frame_id for o in obs}),
            best_confidence=float(max(confs)),
            mean_confidence=float(np.mean(confs)),
            latitude=lat, longitude=lon,
            position_spread_m=spread,
            geoloc_uncertainty_m=unc,
            association_mode=mode,
            bbox_span_px=(round(w, 2), round(h, 2)),
        ))
    return hazards
