"""Turn pixel coordinates into geographic coordinates -- or refuse to.

Three cases are supported, exactly as PS 26057 describes them:

  Case A  GeoTIFF / georectified mosaic
          The affine map from pixel to projected CRS is carried in the file.
          This is the most accurate case.

  Case B  Raw waterfall + per-ping navigation
          Row -> ping -> vessel/towfish fix; column -> across-track slant range,
          converted to ground range using altitude, then projected along the
          heading-perpendicular bearing with a geodesic forward solution.

  Case C  Image only, no metadata
          NO COORDINATES ARE PRODUCED. `NoGeoReference` returns None and states
          why. Inventing a plausible latitude would be the single most damaging
          thing this system could do, because a cleanup vessel would be sent to
          it. The dashboard shows "Geolocation unavailable" instead.

Every successful fix carries an uncertainty in metres, combined in quadrature
from the individual error sources. A coordinate without an uncertainty is not
a measurement, it is a guess with extra decimal places.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:                      # avoids a circular import at runtime
    from .nav import NavTable


@dataclass
class GeoFix:
    latitude: float
    longitude: float
    uncertainty_m: float
    method: str
    confidence: str                      # HIGH | MEDIUM | LOW
    detail: dict

    def as_dict(self) -> dict:
        d = asdict(self)
        d["latitude"] = round(self.latitude, 7)
        d["longitude"] = round(self.longitude, 7)
        d["uncertainty_m"] = round(self.uncertainty_m, 2)
        return d


def _band(unc_m: float) -> str:
    if unc_m <= 10:
        return "HIGH"
    if unc_m <= 40:
        return "MEDIUM"
    return "LOW"


class NoGeoReference:
    """Case C. Present so the pipeline has a uniform interface and never crashes."""
    available = False
    method = "none"

    def __init__(self, reason: str = "No navigation metadata or georeferencing found."):
        self.reason = reason

    def locate(self, x: float, y: float) -> None:
        return None

    def describe(self) -> dict:
        return {"available": False, "method": self.method, "reason": self.reason}


# ---------------------------------------------------------------------------
# Case A -- georectified raster
# ---------------------------------------------------------------------------
class GeoTIFFReference:
    """Affine pixel -> CRS mapping read from GeoTIFF tags, reprojected to WGS84."""
    available = True
    method = "geotiff_affine"

    def __init__(self, path: str | Path, pixel_size_uncertainty: float = 1.0):
        import tifffile
        from pyproj import CRS, Transformer

        self.path = str(path)
        with tifffile.TiffFile(self.path) as tf:
            page = tf.pages[0]
            tags = page.tags
            scale = tags.get("ModelPixelScaleTag")
            tie = tags.get("ModelTiepointTag")
            trans = tags.get("ModelTransformationTag")
            gk = tags.get("GeoKeyDirectoryTag")
            self.shape = (page.imagelength, page.imagewidth)

            if trans is not None:
                m = np.array(trans.value, np.float64).reshape(4, 4)
                self._a = (m[0, 0], m[0, 1], m[0, 3], m[1, 0], m[1, 1], m[1, 3])
            elif scale is not None and tie is not None:
                sx, sy = float(scale.value[0]), float(scale.value[1])
                i, j, _, X, Y, _ = [float(v) for v in tie.value[:6]]
                # x = X + (col - i)*sx ;  y = Y - (row - j)*sy
                self._a = (sx, 0.0, X - i * sx, 0.0, -sy, Y + j * sy)
            else:
                raise ValueError(f"{path} carries no GeoTIFF affine tags "
                                 "(ModelTransformation or PixelScale+Tiepoint)")

            epsg = self._epsg_from_geokeys(gk.value if gk is not None else None)

        self.src_crs = CRS.from_epsg(epsg) if epsg else CRS.from_epsg(4326)
        self.epsg = epsg or 4326
        self._to_wgs84 = Transformer.from_crs(self.src_crs, CRS.from_epsg(4326),
                                              always_xy=True)
        self.pixel_size_m = self._estimate_pixel_size_m()
        self.pixel_size_uncertainty = float(pixel_size_uncertainty)

    @staticmethod
    def _epsg_from_geokeys(vals) -> int | None:
        if not vals:
            return None
        v = list(vals)
        n = v[3] if len(v) > 3 else 0
        for k in range(n):
            off = 4 + k * 4
            if off + 3 >= len(v):
                break
            key, loc, _, val = v[off], v[off + 1], v[off + 2], v[off + 3]
            # 3072 = ProjectedCSTypeGeoKey, 2048 = GeographicTypeGeoKey
            if key in (3072, 2048) and loc == 0 and 1024 <= val <= 32767:
                return int(val)
        return None

    def _estimate_pixel_size_m(self) -> float:
        a = self._a
        if self.src_crs.is_projected:
            return float(math.hypot(a[0], a[3]))
        # geographic CRS: convert degrees to metres at the raster centre
        h, w = self.shape
        _, lat0, _ = self._pixel_to_crs(w / 2, h / 2)
        return float(abs(a[0]) * 111_320.0 * max(math.cos(math.radians(lat0)), 1e-6))

    def _pixel_to_crs(self, x: float, y: float):
        a = self._a
        X = a[0] * x + a[1] * y + a[2]
        Y = a[3] * x + a[4] * y + a[5]
        return X, Y, Y if not self.src_crs.is_projected else 0.0

    def locate(self, x: float, y: float) -> GeoFix:
        X, Y, _ = self._pixel_to_crs(x, y)
        lon, lat = self._to_wgs84.transform(X, Y)
        # The dominant residual for an already-rectified product is the accuracy
        # of the rectification itself; we report at least one pixel.
        unc = max(self.pixel_size_m * self.pixel_size_uncertainty, self.pixel_size_m)
        return GeoFix(lat, lon, unc, self.method, _band(unc),
                      {"source_epsg": self.epsg,
                       "pixel_size_m": round(self.pixel_size_m, 4),
                       "note": "Uncertainty reflects raster resolution only; it does "
                               "not include the error of the original rectification."})

    def describe(self) -> dict:
        return {"available": True, "method": self.method, "epsg": self.epsg,
                "pixel_size_m": round(self.pixel_size_m, 4), "shape": list(self.shape)}


# ---------------------------------------------------------------------------
# Case B -- waterfall + navigation
# ---------------------------------------------------------------------------
@dataclass
class SonarGeometry:
    """Acquisition geometry needed to place a pixel on the seabed."""
    max_range_m: float                 # slant range at the outer edge of one channel
    nadir_col: float | None = None     # column of the nadir; None => image centre
    altitude_m: float | None = None    # towfish height above seabed
    across_track_flipped: bool = False # True if starboard is on the LEFT
    gps_accuracy_m: float = 5.0        # horizontal fix accuracy
    heading_accuracy_deg: float = 2.0
    layback_uncertainty_m: float = 0.0 # unknown tow cable geometry
    altitude_uncertainty_m: float = 0.5


class NavigationReference:
    """Row -> ping fix, column -> across-track ground range, then geodesic forward.

    The navigation table must provide, per ping: latitude, longitude, heading.
    Altitude is optional but strongly recommended: without it we cannot convert
    slant range to ground range and the across-track distance is overstated by
    up to the towfish altitude.
    """
    available = True
    method = "navigation_pingwise"

    def __init__(self, nav: "NavTable", image_shape: tuple[int, int],
                 geometry: SonarGeometry):
        from pyproj import Geod
        self.nav = nav
        self.h, self.w = image_shape
        self.g = geometry
        self._geod = Geod(ellps="WGS84")
        self.nadir_col = float(geometry.nadir_col if geometry.nadir_col is not None
                               else self.w / 2.0)

    def locate(self, x: float, y: float) -> GeoFix | None:
        fix = self.nav.at_row(y, self.h)
        if fix is None:
            return None
        lat0, lon0, heading, alt = fix

        alt = alt if alt is not None else self.g.altitude_m
        half = max(self.nadir_col, self.w - self.nadir_col)
        slant = abs(x - self.nadir_col) / max(half, 1e-6) * self.g.max_range_m

        if alt is None:
            ground = slant
            alt_note = ("Altitude unknown: slant range used directly as ground range. "
                        "Across-track distance is an OVER-estimate near nadir.")
            alt_err = self.g.max_range_m * 0.05
        else:
            ground = math.sqrt(max(slant * slant - alt * alt, 0.0))
            alt_note = f"Slant->ground corrected with altitude {alt:.2f} m."
            # d(ground)/d(alt) = -alt/ground
            alt_err = (abs(alt) / max(ground, 1e-3)) * self.g.altitude_uncertainty_m
            alt_err = min(alt_err, self.g.max_range_m)

        starboard = (x > self.nadir_col) != self.g.across_track_flipped
        bearing = (heading + (90.0 if starboard else -90.0)) % 360.0
        lon, lat, _ = self._geod.fwd(lon0, lat0, bearing, ground)

        # --- uncertainty budget, combined in quadrature ---
        heading_err = ground * math.sin(math.radians(self.g.heading_accuracy_deg))
        range_res = self.g.max_range_m / max(half, 1.0)     # one column of ground range
        terms = {
            "gps_m": self.g.gps_accuracy_m,
            "heading_m": heading_err,
            "layback_m": self.g.layback_uncertainty_m,
            "altitude_m": alt_err,
            "range_resolution_m": range_res,
        }
        unc = float(math.sqrt(sum(v * v for v in terms.values())))

        return GeoFix(lat, lon, unc, self.method, _band(unc), {
            "slant_range_m": round(slant, 2),
            "ground_range_m": round(ground, 2),
            "side": "starboard" if starboard else "port",
            "heading_deg": round(heading, 2),
            "altitude_m": None if alt is None else round(alt, 2),
            "error_budget_m": {k: round(v, 3) for k, v in terms.items()},
            "note": alt_note,
        })

    def describe(self) -> dict:
        return {"available": True, "method": self.method,
                "pings": len(self.nav), "max_range_m": self.g.max_range_m,
                "nadir_col": self.nadir_col,
                "altitude_known": self.g.altitude_m is not None or self.nav.has_altitude}
