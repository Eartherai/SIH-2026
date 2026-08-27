"""The AQUA-SHIELD pipeline: Detection -> Verification -> Localization -> Action.

    RAW SONAR
      -> QUALITY CONTROL          sonar/qc.py
      -> PREPROCESSING            sonar/preprocess.py
      -> RESOLUTION-AWARE TILING  sonar/tiling.py
      -> DETECTION                detection/detector.py
      -> FALSE-POSITIVE FILTER    confidence/fp_filter.py   (+ features.py)
      -> CONFIDENCE CALIBRATION   confidence/calibration.py
      -> DEDUPLICATION            tracking/dedup.py
      -> GEOLOCALIZATION          geolocation/
      -> PRIORITY                 reporting/priority.py
      -> REPORT                   reporting/writers.py

Each stage is optional and independently switchable, so the ablation harness can
measure what each one is actually worth.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .confidence.calibration import IdentityCalibrator, PlattCalibrator, band as conf_band
from .confidence.features import extract as extract_features
from .confidence.fp_filter import LearnedFPFilter, RuleBasedFilter
from .detection.detector import Detector
from .detection.taxonomy import Taxonomy
from .geolocation.reference import NoGeoReference
from .reporting.priority import score_priority, PriorityWeights
from .reporting.schema import HazardRecord
from .sonar import qc as qcmod
from .sonar.preprocess import PROFILES, PreprocessConfig, preprocess
from .tracking.dedup import Observation, deduplicate


@dataclass
class PipelineConfig:
    # DEFAULT IS "none", AND THAT IS DELIBERATE.
    #
    # Preprocessing must match what the detector was TRAINED on. Applying a
    # preprocessing chain at inference to a model trained on raw frames shifts the
    # input distribution and destroys performance -- measured on the held-out test
    # surveys, F1 fell from 0.144 (raw) to 0.012 (standard profile), a 12x
    # degradation, while false-alarm frames rose from 80 to 186 of 473.
    #
    # Use `preprocess_profile_for_model()` to pick the profile a checkpoint was
    # actually trained with, rather than assuming one here.
    preprocess_profile: str = "none"
    preprocess_config: PreprocessConfig | None = None
    tile_size: int = 640
    tile_overlap: int = 128
    detector_conf: float = 0.10          # low: recall first, verify second
    detector_iou: float = 0.45
    imgsz: int = 640

    use_fp_filter: bool = True
    fp_threshold: float = 0.5
    use_calibration: bool = True

    dedup: bool = True
    dedup_radius_m: float = 12.0
    dedup_max_frame_gap: int = 3
    dedup_min_iou: float = 0.20

    min_report_confidence_pct: float = 0.0
    priority_weights: PriorityWeights = field(default_factory=PriorityWeights)
    taxonomy_source: str = "milco_nombo"


@dataclass
class FrameResult:
    frame_id: str
    frame_index: int
    image_shape: tuple[int, int]
    qc: dict
    preprocess_steps: list[str]
    raw_detections: list[dict]           # everything the model emitted
    accepted: list[dict]                 # survived verification
    rejected: list[dict]                 # with the reason it was rejected
    timings_ms: dict
    preview_bgr: np.ndarray | None = None


@dataclass
class SurveyResult:
    survey_id: str
    frames: list[FrameResult]
    hazards: list[HazardRecord]
    summary: dict
    provenance: dict


class AquaShieldPipeline:
    def __init__(self, detector: Detector, cfg: PipelineConfig | None = None,
                 fp_filter=None, calibrator=None, taxonomy: Taxonomy | None = None):
        self.det = detector
        self.cfg = cfg or PipelineConfig()
        self.fp = fp_filter if fp_filter is not None else RuleBasedFilter()
        self.cal = calibrator if calibrator is not None else IdentityCalibrator()
        self.tax = taxonomy or Taxonomy(self.cfg.taxonomy_source)
        self.pp_cfg = (self.cfg.preprocess_config
                       or PROFILES.get(self.cfg.preprocess_profile, PROFILES["standard"]))

    # ------------------------------------------------------------------ frame
    def process_frame(self, image: np.ndarray, frame_id: str, frame_index: int = 0,
                      make_preview: bool = True) -> FrameResult:
        t0 = time.perf_counter()
        gray_in = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        q = qcmod.assess(gray_in)
        t_qc = time.perf_counter()

        pre = preprocess(gray_in, self.pp_cfg)
        proc = pre.image
        t_pre = time.perf_counter()

        inf = self.det.detect(proc, self.cfg.tile_size, self.cfg.tile_overlap)
        t_inf = time.perf_counter()

        gray01 = proc.astype(np.float32) / 255.0
        nadir = None
        if q.water_column_detected and q.water_column_bounds:
            nadir = float(sum(q.water_column_bounds)) / 2.0

        raw_out, accepted, rejected = [], [], []
        if inf.detections:
            feats = [extract_features(gray01, d.box_xyxy, nadir) for d in inf.detections]
            F = np.stack([f.vector() for f in feats])
            raws = np.array([d.raw_score for d in inf.detections], np.float32)
            verdicts = self.fp.predict(F, raws) if self.cfg.use_fp_filter else None

            cal_scores = (self.cal.transform(raws) if self.cfg.use_calibration
                          else raws.astype(np.float64))
            calibrated = bool(getattr(self.cal, "fitted", False)) and self.cfg.use_calibration

            for k, (d, f) in enumerate(zip(inf.detections, feats)):
                tx = self.tax[d.class_id]
                base = {
                    "det_index": k,
                    "box_xyxy": [round(float(v), 2) for v in d.box_xyxy],
                    "raw_score": round(float(d.raw_score), 4),
                    "class_id": d.class_id,
                    "detector_class": d.class_name or tx.native_name,
                    "level1": tx.level1,
                    "level2": tx.level2,
                    "tile_index": d.tile_index,
                    "features": f.as_dict(),
                }
                raw_out.append(base)

                v = verdicts[k] if verdicts is not None else None
                # Verified confidence: start from the calibrated detector score and
                # let the physical-evidence filter move it. When the filter is only
                # the heuristic fallback we do NOT let it manufacture confidence -
                # it can only hold the score or halve it.
                score = float(cal_scores[k])
                if v is not None and getattr(self.fp, "fitted", False):
                    score = float(np.sqrt(max(score, 1e-9) * max(v.score, 1e-9)))
                elif v is not None:
                    score = float(v.score)

                rec = {**base,
                       "confidence_pct": round(100.0 * float(np.clip(score, 0, 1)), 2),
                       "confidence_band": conf_band(100.0 * score),
                       "calibrated": calibrated,
                       "fp_verdict": (v.reason if v else "fp filter disabled"),
                       "fp_score": (round(float(v.score), 4) if v else None),
                       "evidence": {
                           "model": round(float(d.raw_score), 4),
                           "shadow": f.shadow_ratio,
                           "texture": f.texture_homogeneity,
                           "data_quality": round(q.quality_score, 4),
                       }}
                if v is None or v.accepted:
                    accepted.append(rec)
                else:
                    rejected.append({**rec, "rejected_because": v.reason})
        t_post = time.perf_counter()

        preview = None
        if make_preview:
            preview = self._overlay(proc, accepted, rejected)

        return FrameResult(
            frame_id=frame_id, frame_index=frame_index, image_shape=gray_in.shape[:2],
            qc=q.as_dict(), preprocess_steps=pre.steps_applied,
            raw_detections=raw_out, accepted=accepted, rejected=rejected,
            timings_ms={
                "qc": round((t_qc - t0) * 1000, 2),
                "preprocess": round((t_pre - t_qc) * 1000, 2),
                "inference": round(inf.inference_ms, 2),
                "tiling": round(inf.preprocess_ms, 2),
                "detector_postprocess": round(inf.postprocess_ms, 2),
                "verification": round((t_post - t_inf) * 1000, 2),
                "frame_total": round((t_post - t0) * 1000, 2),
                "n_tiles": inf.n_tiles,
            },
            preview_bgr=preview,
        )

    @staticmethod
    def _overlay(gray: np.ndarray, accepted: list[dict], rejected: list[dict]) -> np.ndarray:
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for r in rejected:      # dim red, thin: filtered clutter, shown for transparency
            x0, y0, x1, y1 = (int(v) for v in r["box_xyxy"])
            cv2.rectangle(bgr, (x0, y0), (x1, y1), (60, 60, 190), 1)
        for a in accepted:
            x0, y0, x1, y1 = (int(v) for v in a["box_xyxy"])
            colour = (0, 200, 255) if a["level1"] == "MAN_MADE" else (0, 220, 120)
            cv2.rectangle(bgr, (x0, y0), (x1, y1), colour, 2)
            lbl = f"{a['detector_class']} {a['confidence_pct']:.0f}%"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            ty = max(y0 - 4, th + 2)
            cv2.rectangle(bgr, (x0, ty - th - 3), (x0 + tw + 4, ty + 2), colour, -1)
            cv2.putText(bgr, lbl, (x0 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (20, 20, 20), 1, cv2.LINE_AA)
        return bgr

    # ----------------------------------------------------------------- survey
    def process_survey(self, frames: list[tuple[str, np.ndarray]], *,
                       survey_id: str | None = None, georef=None,
                       make_previews: bool = True, progress=None) -> SurveyResult:
        survey_id = survey_id or f"SURVEY-{uuid.uuid4().hex[:8].upper()}"
        georef = georef or NoGeoReference()
        t0 = time.perf_counter()

        frame_results: list[FrameResult] = []
        observations: list[Observation] = []

        for i, (fid, img) in enumerate(frames):
            fr = self.process_frame(img, fid, i, make_preview=make_previews)
            frame_results.append(fr)
            for j, a in enumerate(fr.accepted):
                x0, y0, x1, y1 = a["box_xyxy"]
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                fix = georef.locate(cx, cy) if getattr(georef, "available", False) else None
                observations.append(Observation(
                    obs_id=f"{fid}#{j}", frame_id=fid, frame_index=i,
                    box_xyxy=(x0, y0, x1, y1), class_id=a["class_id"],
                    class_name=a["detector_class"], confidence=a["confidence_pct"] / 100.0,
                    latitude=fix.latitude if fix else None,
                    longitude=fix.longitude if fix else None,
                    geoloc_uncertainty_m=fix.uncertainty_m if fix else None,
                    extra={"record": a, "fix": fix.as_dict() if fix else None,
                           "quality": fr.qc["quality_score"]},
                ))
            if progress:
                progress(i + 1, len(frames), fid)

        hazard_groups = (deduplicate(observations, radius_m=self.cfg.dedup_radius_m,
                                     max_frame_gap=self.cfg.dedup_max_frame_gap,
                                     min_iou=self.cfg.dedup_min_iou)
                         if self.cfg.dedup else
                         deduplicate(observations, radius_m=0.0, max_frame_gap=0,
                                     min_iou=1.01))
        by_obs = {o.obs_id: o for o in observations}
        gsd = self._ground_sample_distance(georef)

        hazards: list[HazardRecord] = []
        for hz in hazard_groups:
            best = max((by_obs[o] for o in hz.observation_ids),
                       key=lambda o: o.confidence)
            rec = best.extra["record"]
            fix = best.extra["fix"]
            w_px, h_px = hz.bbox_span_px
            length_m = width_m = None
            if gsd:
                length_m = round(max(w_px, h_px) * gsd, 2)
                width_m = round(min(w_px, h_px) * gsd, 2)

            conf_pct = 100.0 * hz.best_confidence
            if conf_pct < self.cfg.min_report_confidence_pct:
                continue

            pr = score_priority(
                confidence_pct=conf_pct, level2_class=rec["level2"],
                estimated_length_m=length_m, observation_count=hz.observation_count,
                survey_quality=float(best.extra["quality"]),
                geolocated=hz.latitude is not None,
                geoloc_uncertainty_m=hz.geoloc_uncertainty_m,
                weights=self.cfg.priority_weights)

            notes: list[str] = []
            if hz.latitude is None:
                notes.append("Geolocation unavailable: no navigation metadata or "
                             "georeferencing for this survey. No position estimated.")
            if not rec["calibrated"]:
                notes.append("Confidence is a RAW detector score, not a calibrated "
                             "probability (no calibration fitted for this model).")
            if gsd is None:
                notes.append("Physical dimensions unavailable: ground sample distance "
                             "unknown. Bounding box is reported in pixels only.")

            e = dict(rec["evidence"])
            e["persistence"] = round(min(1.0, hz.observation_count / 8.0), 4)

            hazards.append(HazardRecord(
                hazard_id=hz.hazard_id, survey_id=survey_id,
                detector_class=rec["detector_class"], level1=rec["level1"],
                level2=rec["level2"], raw_detector_score=rec["raw_score"],
                confidence_pct=round(conf_pct, 2), confidence_band=conf_band(conf_pct),
                calibrated=rec["calibrated"],
                priority_score=round(pr.score, 1), priority_band=pr.band,
                bbox_x0=rec["box_xyxy"][0], bbox_y0=rec["box_xyxy"][1],
                bbox_x1=rec["box_xyxy"][2], bbox_y1=rec["box_xyxy"][3],
                estimated_length_m=length_m, estimated_width_m=width_m,
                ground_sample_distance_m=gsd,
                latitude=hz.latitude, longitude=hz.longitude,
                geolocation_method=(fix["method"] if fix else "none"),
                geolocation_confidence=(fix["confidence"] if fix else "UNAVAILABLE"),
                geoloc_uncertainty_m=(round(hz.geoloc_uncertainty_m, 2)
                                      if hz.geoloc_uncertainty_m else None),
                frame_ids=hz.frame_ids, observation_count=hz.observation_count,
                association_mode=hz.association_mode,
                evidence=e, fp_filter_verdict=rec.get("fp_verdict", ""),
                frame_quality_score=round(float(best.extra["quality"]), 4),
                notes=notes,
            ))

        if any(h.latitude is not None for h in hazards):
            self._add_utm(hazards)

        elapsed = time.perf_counter() - t0
        summary = self._summarise(frame_results, hazards, observations, elapsed)
        provenance = {
            "model_path": self.det.weights,
            "detector_backend": self.det.backend,
            "device": self.det.device,
            "detector_conf_threshold": self.cfg.detector_conf,
            "imgsz": self.cfg.imgsz,
            "tile_size": self.cfg.tile_size,
            "tile_overlap": self.cfg.tile_overlap,
            "preprocess_profile": self.cfg.preprocess_profile,
            "preprocess_config": self.pp_cfg.as_dict(),
            "fp_filter": self.fp.as_dict(),
            "calibration": self.cal.as_dict(),
            "taxonomy_source": self.cfg.taxonomy_source,
            "taxonomy_citation": self.tax.citation,
            "taxonomy_license": self.tax.license,
            "georeference": georef.describe(),
        }
        return SurveyResult(survey_id, frame_results, hazards, summary, provenance)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _ground_sample_distance(georef) -> float | None:
        """Metres per pixel, only when the georeference genuinely defines it."""
        gsd = getattr(georef, "pixel_size_m", None)
        return float(gsd) if gsd else None

    @staticmethod
    def _add_utm(hazards: list[HazardRecord]) -> None:
        from pyproj import CRS, Transformer
        located = [h for h in hazards if h.latitude is not None]
        if not located:
            return
        lat0 = float(np.mean([h.latitude for h in located]))
        lon0 = float(np.mean([h.longitude for h in located]))
        zone = int((lon0 + 180) / 6) + 1
        north = lat0 >= 0
        epsg = (32600 if north else 32700) + zone
        tr = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)
        for h in located:
            e, n = tr.transform(h.longitude, h.latitude)
            h.utm_easting, h.utm_northing = round(e, 2), round(n, 2)
            h.utm_zone = f"{zone}{'N' if north else 'S'}"

    @staticmethod
    def _summarise(frames, hazards, observations, elapsed: float) -> dict:
        n_raw = sum(len(f.raw_detections) for f in frames)
        n_acc = sum(len(f.accepted) for f in frames)
        n_rej = sum(len(f.rejected) for f in frames)
        by_l2: dict[str, int] = {}
        for h in hazards:
            by_l2[h.level2] = by_l2.get(h.level2, 0) + 1
        qs = [f.qc["quality_score"] for f in frames] or [0.0]
        return {
            "frames_processed": len(frames),
            "candidate_detections_raw": n_raw,
            "detections_accepted": n_acc,
            "detections_rejected_by_fp_filter": n_rej,
            "fp_filter_rejection_rate": (round(n_rej / n_raw, 4) if n_raw else 0.0),
            "observations": len(observations),
            "unique_hazards": len(hazards),
            "deduplication_ratio": (round(len(observations) / len(hazards), 3)
                                    if hazards else 0.0),
            "hazards_by_class": by_l2,
            "man_made_hazards": sum(1 for h in hazards if h.level1 == "MAN_MADE"),
            "ambiguous_hazards": sum(1 for h in hazards if h.level1 == "AMBIGUOUS"),
            "high_confidence_hazards": sum(1 for h in hazards
                                           if h.confidence_band in ("HIGH", "CRITICAL")),
            "high_priority_hazards": sum(1 for h in hazards
                                         if h.priority_band in ("HIGH", "URGENT")),
            "geolocated_hazards": sum(1 for h in hazards if h.latitude is not None),
            "mean_frame_quality": round(float(np.mean(qs)), 4),
            "processing_seconds": round(elapsed, 2),
            "mean_ms_per_frame": (round(1000 * elapsed / len(frames), 1) if frames else 0.0),
        }
