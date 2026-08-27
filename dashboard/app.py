"""AQUA-SHIELD operator dashboard.

Design brief: the user is a survey operator or marine researcher, NOT an ML
engineer. Nothing on screen requires knowing what a tensor or a checkpoint is.
The four questions the interface must answer are:

    Where is the hazard?  What is it?  How confident are we?  Can I export it?

Runs fully offline once the model and demo data are present. No cloud inference,
no external API calls. Map tiles are the only optional network use and there is
an offline fallback that needs no tiles at all.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aquashield import __version__                                     # noqa: E402
from aquashield.confidence.calibration import PlattCalibrator          # noqa: E402
from aquashield.confidence.fp_filter import LearnedFPFilter            # noqa: E402
from aquashield.detection.detector import Detector                     # noqa: E402
from aquashield.detection.model_meta import read_meta                  # noqa: E402
from aquashield.detection.taxonomy import Taxonomy                     # noqa: E402
from aquashield.device import select_device                            # noqa: E402
from aquashield.geolocation import (NavigationReference, NoGeoReference,  # noqa: E402
                                    SonarGeometry, load_nav_csv)
from aquashield.pipeline import AquaShieldPipeline, PipelineConfig     # noqa: E402
from aquashield.reporting import (build_report, csv_string,            # noqa: E402
                                  write_geojson)
from aquashield.sonar.preprocess import PROFILES, PreprocessConfig     # noqa: E402

st.set_page_config(page_title="AQUA-SHIELD", page_icon="🛰️", layout="wide")

OFFLINE_MAP = os.environ.get("AQS_OFFLINE_MAP", "0") == "1"
DEMO_DIR = ROOT / "demo_data"
MODEL_DIR = ROOT / "models"


# ---------------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------------
def find_models() -> list[Path]:
    found = sorted(MODEL_DIR.glob("*.pt"))
    found += sorted(ROOT.glob("runs/**/weights/best.pt"))
    seen, out = set(), []
    for p in found:
        r = p.resolve()
        if r not in seen:
            seen.add(r)
            out.append(p)
    return out


@st.cache_resource(show_spinner=False)
def get_detector(weights: str, conf: float, imgsz: int):
    return Detector(weights, conf=conf, imgsz=imgsz)


@st.cache_resource(show_spinner=False)
def get_verification(fp_path: str, cal_path: str):
    return LearnedFPFilter.load(fp_path), PlattCalibrator.load(cal_path)


def decode_upload(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🛰️ AQUA-SHIELD")
st.sidebar.caption(f"v{__version__} · Detection → Verification → Localization → Action")

dev = select_device("auto")
st.sidebar.success(f"Compute: **{dev.device.upper()}** · {dev.reason}")

models = find_models()
if not models:
    st.sidebar.error("No model checkpoint found.")
    st.title("AQUA-SHIELD")
    st.error(
        "**No trained model is available.**\n\n"
        "AQUA-SHIELD will not run with a placeholder or simulated detector — every "
        "detection you see in this dashboard comes from real inference.\n\n"
        "Train one first:\n"
        "```bash\n"
        "python scripts/prepare_milco_nombo.py\n"
        "python scripts/train.py --exp-id E01 --epochs 150\n"
        "```"
    )
    st.stop()

st.sidebar.header("1 · Model")
model_choice = st.sidebar.selectbox(
    "Detector checkpoint", models,
    format_func=lambda p: f"{p.parent.parent.name}/{p.name}" if "weights" in str(p) else p.name)

st.sidebar.header("2 · Data source")
mode = st.sidebar.radio("Input", ["SIH Demo Mode", "Upload sonar imagery"],
                        label_visibility="collapsed")

frames: list[tuple[str, np.ndarray]] = []
nav_file = None
demo_meta: dict | None = None

if mode == "SIH Demo Mode":
    scenarios = sorted([p for p in DEMO_DIR.glob("*") if p.is_dir()]) if DEMO_DIR.exists() else []
    if not scenarios:
        st.sidebar.warning("No demo scenarios found. Run `python scripts/build_demo_data.py`.")
    else:
        sc = st.sidebar.selectbox("Scenario", scenarios, format_func=lambda p: p.name)
        meta_p = sc / "scenario.json"
        demo_meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        imgs = sorted(list((sc / "images").glob("*.jpg")) + list((sc / "images").glob("*.png")))
        max_n = st.sidebar.slider("Frames to process", 1, max(len(imgs), 1),
                                  min(len(imgs), 12))
        frames = [(p.stem, cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for p in imgs[:max_n]]
        nav_p = sc / "navigation.csv"
        nav_file = nav_p if nav_p.exists() else None
        if demo_meta:
            st.sidebar.info(demo_meta.get("short_description", ""))
else:
    ups = st.sidebar.file_uploader("Sonar frames (PNG/JPG/TIFF)",
                                   type=["png", "jpg", "jpeg", "tif", "tiff"],
                                   accept_multiple_files=True)
    for u in ups or []:
        img = decode_upload(u.getvalue())
        if img is None:
            st.sidebar.error(f"Could not decode {u.name} — skipped.")
        else:
            frames.append((Path(u.name).stem, img))
    nav_up = st.sidebar.file_uploader("Navigation CSV (optional)", type=["csv"])
    if nav_up:
        tmp = ROOT / "outputs" / "_uploaded_nav.csv"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(nav_up.getvalue())
        nav_file = tmp

st.sidebar.header("3 · Preprocessing")
# The correct profile is a property of the CHECKPOINT, not a free choice.
# Applying a chain the detector was never trained on degrades it severely
# (measured: F1 0.144 -> 0.012). We default to whatever this model was trained on.
_meta = read_meta(str(model_choice))
_trained_profile = _meta.get("preprocess_profile", "none")
if _meta.get("_assumed"):
    st.sidebar.caption(f"No metadata for this checkpoint — assuming it was trained "
                       f"on **{_trained_profile}** (raw) imagery.")
else:
    st.sidebar.caption(f"This checkpoint was trained on the **{_trained_profile}** profile.")
profile_name = st.sidebar.selectbox(
    "Profile", list(PROFILES.keys()),
    index=list(PROFILES).index(_trained_profile if _trained_profile in PROFILES else "none"),
    help="Match this to the profile the detector was trained on. Mismatching it "
         "shifts the input distribution and degrades detection sharply.")
if profile_name != _trained_profile:
    st.sidebar.warning(f"Profile '{profile_name}' does not match this checkpoint's "
                       f"training profile '{_trained_profile}'. Expect degraded detection.")
base = PROFILES[profile_name]
with st.sidebar.expander("Fine-tune stages"):
    pp = PreprocessConfig(
        dropout_handling=st.checkbox("Repair ping dropouts", base.dropout_handling),
        water_column_removal=st.checkbox("Remove water column (nadir)",
                                         base.water_column_removal),
        water_column_mode=base.water_column_mode,
        denoise=st.checkbox("Speckle denoise (Lee filter)", base.denoise),
        denoise_method=base.denoise_method,
        gain_normalization=st.checkbox("Across-track gain normalisation",
                                       base.gain_normalization),
        dynamic_range_normalization=st.checkbox("Dynamic-range stretch",
                                                base.dynamic_range_normalization),
        contrast_normalization=st.checkbox("CLAHE contrast", base.contrast_normalization),
    )

st.sidebar.header("4 · Detection & verification")
conf = st.sidebar.slider("Detector sensitivity (lower = more candidates)",
                         0.01, 0.90, 0.10, 0.01)
use_fp = st.sidebar.checkbox("False-positive filter", True)
use_tiling = st.sidebar.checkbox("Resolution-aware tiling", True)
min_conf_pct = st.sidebar.slider("Minimum reported confidence (%)", 0, 100, 0, 5)

st.sidebar.header("5 · Survey geometry")
with st.sidebar.expander("Sonar geometry (for geolocation)"):
    max_range = st.number_input("Slant range per channel (m)", 5.0, 500.0, 50.0, 5.0)
    gps_acc = st.number_input("GPS accuracy (m)", 0.5, 50.0, 5.0, 0.5)
    hdg_acc = st.number_input("Heading accuracy (deg)", 0.1, 30.0, 2.0, 0.1)
    layback = st.number_input("Layback uncertainty (m)", 0.0, 100.0, 3.0, 0.5)
    alt_m = st.number_input("Towfish altitude (m, 0 = unknown)", 0.0, 200.0, 0.0, 0.5)

run = st.sidebar.button("▶  Process survey", type="primary", width='stretch',
                        disabled=not frames)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("AQUA-SHIELD")
st.caption("Acoustic Intelligence for Underwater Anomaly, Debris & Marine-Hazard "
           "Localization — SIH 2026 · PS 26057 · MoES / NIOT")

if not frames:
    st.info("Select **SIH Demo Mode** or upload sonar frames in the sidebar to begin.")
    st.stop()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if run or "result" not in st.session_state:
    if not run:
        st.info("Press **Process survey** in the sidebar.")
        st.stop()

    det = get_detector(str(model_choice), conf, 640)
    tag = "milco_nombo"
    filt, cal = get_verification(str(MODEL_DIR / f"fp_filter_{tag}.json"),
                                 str(MODEL_DIR / f"calibration_{tag}.json"))

    georef = NoGeoReference()
    if nav_file is not None:
        try:
            nav = load_nav_csv(nav_file)
            geom = SonarGeometry(max_range_m=max_range, gps_accuracy_m=gps_acc,
                                 heading_accuracy_deg=hdg_acc,
                                 layback_uncertainty_m=layback,
                                 altitude_m=(alt_m if alt_m > 0 else None))
            georef = NavigationReference(nav, frames[0][1].shape[:2], geom)
        except Exception as e:                                    # noqa: BLE001
            st.warning(f"Navigation file could not be used ({e}). "
                       "Continuing without geolocation.")
            georef = NoGeoReference(f"Navigation file rejected: {e}")

    cfg = PipelineConfig(preprocess_config=pp, preprocess_profile=profile_name,
                         detector_conf=conf, use_fp_filter=use_fp,
                         min_report_confidence_pct=float(min_conf_pct),
                         tile_size=(640 if use_tiling else 100_000),
                         tile_overlap=(128 if use_tiling else 0))
    pipe = AquaShieldPipeline(det, cfg, fp_filter=(filt if use_fp else None),
                              calibrator=cal, taxonomy=Taxonomy(tag))

    bar = st.progress(0.0, "Processing…")
    t0 = time.perf_counter()
    survey_id = (demo_meta or {}).get("survey_id") or f"SURVEY-{int(time.time())}"
    res = pipe.process_survey(
        frames, survey_id=survey_id, georef=georef,
        progress=lambda i, n, fid: bar.progress(i / n, f"Frame {i}/{n} — {fid}"))
    bar.empty()
    st.session_state["result"] = res
    st.session_state["elapsed"] = time.perf_counter() - t0

res = st.session_state["result"]
s = res.summary

# ---------------------------------------------------------------------------
# Survey summary
# ---------------------------------------------------------------------------
c = st.columns(6)
c[0].metric("Frames", s["frames_processed"])
c[1].metric("Raw candidates", s["candidate_detections_raw"])
c[2].metric("Filtered out", s["detections_rejected_by_fp_filter"],
            f"-{s['fp_filter_rejection_rate']:.0%}" if s["candidate_detections_raw"] else None)
c[3].metric("Unique hazards", s["unique_hazards"])
c[4].metric("High priority", s["high_priority_hazards"])
c[5].metric("ms / frame", f"{s['mean_ms_per_frame']:.0f}")

if s["unique_hazards"] == 0:
    st.success("**No confident detections found.** The processed frames contain no "
               "anomalies above the current thresholds. Lower the detector "
               "sensitivity in the sidebar to widen the search.")

tabs = st.tabs(["🖼️ Detections", "🗺️ Map", "📋 Hazard register", "🔬 Evidence & QC",
                "⬇️ Export", "ℹ️ Provenance"])

# ---------------------------------------------------------------- detections
with tabs[0]:
    with_det = [f for f in res.frames if f.accepted] or res.frames
    labels = [f"{f.frame_id}  ({len(f.accepted)} accepted / "
              f"{len(f.rejected)} filtered)" for f in with_det]
    pick = st.selectbox("Frame", range(len(with_det)), format_func=lambda i: labels[i])
    fr = with_det[pick]
    left, right = st.columns([3, 2])
    with left:
        if fr.preview_bgr is not None:
            st.image(cv2.cvtColor(fr.preview_bgr, cv2.COLOR_BGR2RGB),
                     caption=f"{fr.frame_id} — yellow = man-made, green = ambiguous, "
                             "thin red = rejected by the false-positive filter",
                     width='stretch')
    with right:
        st.markdown("**Accepted detections**")
        if not fr.accepted:
            st.caption("None in this frame.")
        for a in fr.accepted:
            st.markdown(
                f"- **{a['detector_class']}** · {a['level2'].replace('_',' ')}  \n"
                f"  confidence **{a['confidence_pct']:.1f}%** ({a['confidence_band']})"
                f"{'' if a['calibrated'] else ' · *raw score, not calibrated*'}")
        if fr.rejected:
            with st.expander(f"Rejected by the FP filter ({len(fr.rejected)})"):
                for r in fr.rejected[:25]:
                    st.caption(f"{r['detector_class']} @ {r['box_xyxy']} — "
                               f"{r['rejected_because'][:150]}")

# ---------------------------------------------------------------------- map
with tabs[1]:
    located = [h for h in res.hazards if h.latitude is not None]
    if not located:
        st.warning("**Geolocation unavailable.** No navigation metadata or "
                   "georeferencing was supplied with this imagery, so no "
                   "coordinates have been produced.")
        st.caption("AQUA-SHIELD does not estimate positions it cannot measure — a "
                   "fabricated coordinate would send a cleanup vessel to open water. "
                   "Supply a navigation CSV (columns: lat, lon, heading, altitude) "
                   "or a GeoTIFF to enable the map.")
    else:
        df = pd.DataFrame([{"hazard_id": h.hazard_id, "lat": h.latitude,
                            "lon": h.longitude, "class": h.level2,
                            "confidence_pct": h.confidence_pct,
                            "priority": h.priority_score,
                            "priority_band": h.priority_band,
                            "uncertainty_m": h.geoloc_uncertainty_m,
                            "observations": h.observation_count} for h in located])
        if OFFLINE_MAP:
            st.info("Offline map mode — plotting coordinates without remote tiles.")
            st.scatter_chart(df, x="lon", y="lat", color="priority_band", size="priority")
        else:
            try:
                import folium
                from streamlit_folium import st_folium
                m = folium.Map(location=[df.lat.mean(), df.lon.mean()], zoom_start=15,
                               tiles="OpenStreetMap")
                colours = {"URGENT": "red", "HIGH": "orange",
                           "ELEVATED": "blue", "ROUTINE": "green"}
                for _, r in df.iterrows():
                    # circle = the actual positional uncertainty, drawn to scale
                    if r.uncertainty_m:
                        folium.Circle([r.lat, r.lon], radius=float(r.uncertainty_m),
                                      color=colours.get(r.priority_band, "gray"),
                                      fill=True, fill_opacity=0.12, weight=1).add_to(m)
                    folium.CircleMarker(
                        [r.lat, r.lon], radius=6,
                        color=colours.get(r.priority_band, "gray"), fill=True,
                        fill_opacity=0.9,
                        popup=folium.Popup(
                            f"<b>{r.hazard_id}</b><br>{r['class']}<br>"
                            f"confidence {r.confidence_pct:.0f}%<br>"
                            f"priority {r.priority:.0f} ({r.priority_band})<br>"
                            f"±{r.uncertainty_m:.0f} m · {r.observations} obs",
                            max_width=260)).add_to(m)
                st_folium(m, height=520, width='stretch')
                st.caption("Shaded circles show the reported positional uncertainty "
                           "at true scale — not a decoration.")
            except Exception as e:                                # noqa: BLE001
                st.warning(f"Interactive map unavailable ({e}). Showing plain plot.")
                st.scatter_chart(df, x="lon", y="lat", color="priority_band")
        st.dataframe(df, width='stretch', hide_index=True)
        n_missing = len(res.hazards) - len(located)
        if n_missing:
            st.caption(f"{n_missing} hazard(s) have no position fix and are not mapped.")

# ------------------------------------------------------------------ register
with tabs[2]:
    if not res.hazards:
        st.caption("No hazards to list.")
    else:
        rows = [{"hazard_id": h.hazard_id, "class": h.level2, "level1": h.level1,
                 "confidence_%": h.confidence_pct, "band": h.confidence_band,
                 "calibrated": h.calibrated, "priority": h.priority_score,
                 "priority_band": h.priority_band, "obs": h.observation_count,
                 "lat": h.latitude, "lon": h.longitude,
                 "±m": h.geoloc_uncertainty_m, "length_m": h.estimated_length_m,
                 "frames": ";".join(h.frame_ids[:3])} for h in res.hazards]
        df = pd.DataFrame(rows).sort_values("priority", ascending=False)
        bands = st.multiselect("Filter by priority",
                               ["URGENT", "HIGH", "ELEVATED", "ROUTINE"],
                               default=["URGENT", "HIGH", "ELEVATED", "ROUTINE"])
        classes = st.multiselect("Filter by class", sorted(df["class"].unique()),
                                 default=sorted(df["class"].unique()))
        view = df[df.priority_band.isin(bands) & df["class"].isin(classes)]
        st.dataframe(view, width='stretch', hide_index=True)

        pick = st.selectbox("Inspect hazard", view.hazard_id.tolist()) if len(view) else None
        if pick:
            h = next(x for x in res.hazards if x.hazard_id == pick)
            a, b = st.columns(2)
            with a:
                st.markdown(f"### {h.hazard_id}")
                st.markdown(f"- **Class** — {h.level2.replace('_',' ')} ({h.level1})")
                st.markdown(f"- **Detector said** — {h.detector_class} "
                            f"(raw score {h.raw_detector_score:.3f})")
                st.markdown(f"- **Confidence** — {h.confidence_pct:.1f}% "
                            f"({h.confidence_band})")
                st.markdown(f"- **Priority** — {h.priority_score:.0f} ({h.priority_band})")
                st.markdown(f"- **Observations** — {h.observation_count} "
                            f"({h.association_mode})")
                st.markdown("- **Size** — " + (
                    f"{h.estimated_length_m} × {h.estimated_width_m} m"
                    if h.estimated_length_m else
                    f"{h.bbox_x1-h.bbox_x0:.0f} × {h.bbox_y1-h.bbox_y0:.0f} px "
                    "(no ground sample distance)"))
                st.markdown("- **Position** — " + (
                    f"{h.latitude:.6f}, {h.longitude:.6f} ± {h.geoloc_uncertainty_m:.0f} m "
                    f"({h.geolocation_confidence})" if h.latitude is not None
                    else "*unavailable — no navigation metadata*"))
            with b:
                st.markdown("### Evidence")
                if h.evidence:
                    st.bar_chart(pd.DataFrame(
                        {"evidence": {k: float(v) for k, v in h.evidence.items()
                                      if isinstance(v, (int, float))}}))
                st.caption("These are separate indicators, not independent "
                           "probabilities. `model` is the detector score; the others "
                           "are measured image/context properties.")
                st.markdown("**Filter verdict**")
                st.caption(h.fp_filter_verdict or "—")
            for n in h.notes:
                st.warning(n)

# -------------------------------------------------------------- evidence/QC
with tabs[3]:
    q = pd.DataFrame([{"frame": f.frame_id, **{k: v for k, v in f.qc.items()
                                               if isinstance(v, (int, float, bool))}}
                      for f in res.frames])
    st.markdown("#### Frame quality control")
    st.dataframe(q, width='stretch', hide_index=True)
    st.markdown("#### Preprocessing applied")
    st.code("\n".join(res.frames[0].preprocess_steps) or "(none)", language="text")
    st.markdown("#### Timing breakdown (first frame, ms)")
    st.json(res.frames[0].timings_ms)
    notes = [n for f in res.frames for n in f.qc.get("notes", [])]
    if notes:
        st.markdown("#### QC notes")
        for n in sorted(set(notes))[:12]:
            st.caption("• " + n)

# ------------------------------------------------------------------- export
with tabs[4]:
    report = build_report(res.hazards, survey_id=res.survey_id,
                          summary=res.summary, provenance=res.provenance)
    a, b, c3 = st.columns(3)
    a.download_button("⬇️ JSON report", json.dumps(report, indent=2, default=str),
                      f"{res.survey_id}_hazards.json", "application/json",
                      width='stretch')
    b.download_button("⬇️ CSV report", csv_string(res.hazards),
                      f"{res.survey_id}_hazards.csv", "text/csv",
                      width='stretch')
    gj = ROOT / "outputs" / f"{res.survey_id}.geojson"
    write_geojson(res.hazards, gj)
    c3.download_button("⬇️ GeoJSON (QGIS)", gj.read_text(),
                       f"{res.survey_id}.geojson", "application/geo+json",
                       width='stretch')
    st.markdown("##### JSON preview")
    st.json({**report, "hazards": report["hazards"][:2]}, expanded=False)

# --------------------------------------------------------------- provenance
with tabs[5]:
    st.markdown("Everything that produced the numbers above.")
    st.json(res.provenance)
    st.markdown("#### Survey summary")
    st.json(res.summary)
    st.info("Confidence values are calibrated only when `calibration.fitted` is true "
            "above. Otherwise they are raw detector scores and must not be read as "
            "probabilities.")
