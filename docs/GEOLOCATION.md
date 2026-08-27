# Geolocation

## The rule

**AQUA-SHIELD never invents a coordinate.** If the metadata to compute a position
is absent, the hazard is reported with `latitude: null`, `longitude: null`,
`geolocation_confidence: "UNAVAILABLE"` and an explicit note. The dashboard shows
*"Geolocation unavailable"*.

This is the most important design decision in the module. A fabricated latitude
looks like data, exports cleanly to CSV, and sends a vessel to open water.

`tests/test_integration.py::test_no_geolocation_means_null_coordinates_never_zeros`
enforces it.

## Three cases

### Case A — georectified raster (GeoTIFF)
The affine pixel→CRS map is read from `ModelTransformationTag`, or from
`ModelPixelScaleTag` + `ModelTiepointTag`; the EPSG code comes from
`GeoKeyDirectoryTag`. Coordinates are reprojected to WGS84 with `pyproj`.
This case also yields a **ground sample distance**, which is the only way physical
object dimensions can be reported.

Uncertainty: at least one pixel. *Stated limitation:* this reflects raster
resolution only and does **not** include the error of the original rectification,
which we cannot see from the file.

### Case B — waterfall + per-ping navigation

```
image row  ──▶ ping index ──▶ vessel/towfish fix (lat, lon, heading, altitude)
image col  ──▶ slant range ──▶ ground range ──▶ geodesic forward solution
```

- Row → ping by linear interpolation over the nav table.
- Column → slant range, scaled from the nadir column and the per-channel range.
- Slant → ground: `ground = sqrt(slant² − altitude²)`.
- Side: starboard if the column is beyond nadir (configurably flipped).
- Bearing: `heading ± 90°`; position from `pyproj.Geod.fwd` on the WGS84 ellipsoid.

Heading is interpolated **on the circle**, so a track crossing north
(350° → 10°) interpolates to 0°, not 180°. Unit tested.

If the nav file has no heading column, course-over-ground is derived from
successive fixes and flagged as `heading_derived_from_track: true` — that is the
vessel *track*, not the towfish heading, and they differ in a crosscurrent.

### Case C — image only
No coordinates. Reported as unavailable, with the reason.

## Uncertainty budget

Every Case-B fix carries its full error budget, combined in quadrature:

| Term | Source |
|---|---|
| `gps_m` | GPS/DGPS horizontal accuracy (operator-supplied) |
| `heading_m` | `ground_range × sin(heading_accuracy)` — grows with range |
| `layback_m` | Tow-cable geometry uncertainty (operator-supplied) |
| `altitude_m` | `(altitude / ground_range) × altitude_uncertainty` |
| `range_resolution_m` | One range-bin column, in ground-range metres |

```
uncertainty_m = sqrt(Σ term²)
```

Bands: `HIGH ≤ 10 m`, `MEDIUM ≤ 40 m`, `LOW` beyond.

### The nadir singularity — a real result, not a bug

The `altitude_m` term contains `altitude / ground_range`, which **diverges as
ground range → 0**. Directly beneath the towfish, a small altitude error produces
an enormous ground-range error: the slant-to-ground inversion is ill-conditioned
there.

The implementation lets this show. A detection at nadir is reported with large
uncertainty and `LOW` confidence, while one at mid-swath reports ~6 m and `HIGH`.
That matches sonar practice, where the nadir region is treated as unreliable and
usually discarded. Unit tested
(`test_nadir_is_flagged_as_ill_conditioned`).

## Deduplication interaction

Averaging N independent fixes of the same object reduces random error by ~√N, so
a hazard seen in 4 pings reports a tighter position than one seen once. We floor
the improvement at `base/2` because the systematic terms (layback, GPS bias) do
**not** average away.

## Outputs

- WGS84 latitude/longitude
- UTM easting/northing + zone, computed from the survey centroid
- Per-hazard `geoloc_uncertainty_m`
- GeoJSON for QGIS/ArcGIS — hazards without a fix are **omitted**, never placed at
  (0, 0), and the omitted count is recorded on the FeatureCollection.

## What has NOT been validated

**Geolocation accuracy has not been measured against ground truth.** The
MILCO/NOMBO dataset ships no navigation data, so there is nothing to validate
against. What *is* verified: the geometry, the sign conventions, the circular
heading interpolation, the error-budget arithmetic, and the refusal behaviour —
all unit tested. The demo's `04_georeferenced` scenario uses an explicitly
**synthetic** track, labelled as such in the file, the metadata and the UI.

Validating true positional accuracy requires a survey with both sonar and
surveyed object positions. That is the first thing to do with real NIOT data.
