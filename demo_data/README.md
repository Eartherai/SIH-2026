# AQUA-SHIELD demo data

Small curated scenarios for the dashboard and the live demo. All imagery is drawn from the **held-out test surveys** (2018 and 2021), so nothing shown in the demo was seen during training.

| Scenario | Purpose |
|---|---|
| `01_clear_targets` | Large, high-contrast man-made targets |
| `02_hard_targets` | Small / low-contrast targets - shows the failure mode |
| `03_natural_seabed` | No targets at all; every detection is a false positive |
| `04_georeferenced` | Geolocation, dedup and map, using a **synthetic** track |

## Navigation data

Only `04_georeferenced` has navigation, and it is **synthetic** - the source dataset ships no positions. It exercises the geolocation maths on a known geometry. The other scenarios correctly report *Geolocation unavailable*.

## Attribution

Imagery: Pessanha Santos, N. & Moura, R. (2024), 'Side-scan sonar imaging data of underwater vehicles for mine detection', Data in Brief 53:110132. figshare DOI 10.6084/m9.figshare.24574879. Licensed CC BY 4.0.
