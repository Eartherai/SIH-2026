#!/usr/bin/env python3
"""Fill docs/BENCHMARKS.md from the recorded experiment files.

Documentation that is typed by hand drifts from the numbers it describes. Every
table between a `<!-- BENCHMARK:X -->` marker and the next heading is generated
from `experiments/*.json(l)`. If a record is missing, the section says so
explicitly instead of being left with a stale or invented table.
"""
from __future__ import annotations

import json
from pathlib import Path

EXP = Path("experiments")
DOC = Path("docs/BENCHMARKS.md")
MISSING = "_Not yet measured. Run the command in section 9 to populate this table._"


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def t_training() -> str:
    rows = load_jsonl(EXP / "registry.jsonl")
    if not rows:
        return MISSING
    seen, uniq = set(), []
    for r in reversed(rows):                       # keep the newest per experiment
        if r["experiment_id"] not in seen:
            seen.add(r["experiment_id"])
            uniq.append(r)
    uniq.reverse()
    out = ["| Experiment | Config | Epochs | mAP50 | mAP50-95 | Precision | Recall |",
           "|---|---|---|---|---|---|---|"]
    for r in uniq:
        m = r["metrics_test"]
        a = r.get("augment", {})
        cfg = (f"imgsz {r.get('imgsz')}, lr0 {r.get('lr0')}, "
               f"mosaic {a.get('mosaic')}, scale {a.get('scale')}")
        ep = r.get("epochs_completed", r.get("epochs_requested"))
        if r.get("stopped_early_by_operator"):
            ep = f"{ep}*"
        out.append(f"| `{r['experiment_id']}` | {cfg} | {ep} | "
                   f"**{m['mAP50']:.4f}** | {m['mAP50_95']:.4f} | "
                   f"{m['precision']:.4f} | {m['recall']:.4f} |")
    out += ["", "\\* stopped early by the operator; see `notes` in "
                "`experiments/registry.jsonl`.", "",
            "Per-class mAP50 and the full hyperparameters for every run are in the "
            "registry. All figures are on the **held-out test surveys**, never on "
            "the validation survey."]
    return "\n".join(out)


def t_preprocessing() -> str:
    d = load_json(EXP / "preprocessing_ablation.json")
    if not d:
        return MISSING
    out = ["| Detector trained on | Inference input | P | R | F1 | FP | FA-frames |",
           "|---|---|---|---|---|---|---|"]
    for r in d["rows"]:
        out.append(f"| {r['trained_on']} | {r['inference_on']} | {r['precision']:.4f} | "
                   f"{r['recall']:.4f} | {r['f1']:.4f} | {r['fp']} | "
                   f"{r['false_alarm_frames']}/{r['n_frames_empty']} |")
    if d.get("conclusion"):
        out += ["", f"**Conclusion.** {d['conclusion']}"]
    return "\n".join(out)


def t_ablation() -> str:
    d = load_json(EXP / "ablation.json")
    if not d:
        return MISSING
    ds = d["dataset"]
    out = [f"Model: `{Path(d['weights']).parent.parent.name}` · device `{d['device']}` · "
           f"{ds['frames']} frames ({ds['frames_with_targets']} with targets, "
           f"{ds['empty_frames']} empty, {ds['gt_objects']} objects) · "
           f"match IoU {d['match_iou_threshold']}", "",
           "| Variant | P | R | F1 | TP | FP | FN | FA-frames | ms/frame |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in d["variants"]:
        name = r["variant"].split("_", 1)[1].replace("_", " ")
        out.append(f"| {r['variant'].split('_')[0]}. {name} | {r['precision']:.4f} | "
                   f"{r['recall']:.4f} | {r['f1']:.4f} | {r['tp']} | {r['fp']} | "
                   f"{r['fn']} | {r['false_alarm_frames']}/{r['n_frames_empty']} | "
                   f"{r['latency_ms_mean']:.0f} |")
    if not d.get("learned_fp_filter_available"):
        out += ["", "> The learned FP filter was NOT available for this run; the "
                    "rule-based fallback was used. Fit it with "
                    "`scripts/fit_verification.py`."]
    out += ["", "**Read this table for direction and magnitude, not for a precise "
                f"ranking of adjacent rows.** With {ds['gt_objects']} test objects, "
                "differences of a few percent are within noise "
                "(`docs/LIMITATIONS.md`, §11)."]
    return "\n".join(out)


def t_latency() -> str:
    rows = load_jsonl(EXP / "benchmarks.jsonl")
    if not rows:
        return MISSING
    b = rows[-1]
    out = ["### Per-frame CPU stages (device independent)", "",
           "| Stage | mean | p95 |", "|---|---|---|"]
    for k, v in b["stages"].items():
        out.append(f"| {k.replace('_',' ')} | {v['mean_ms']:.2f} ms | {v['p95_ms']:.2f} ms |")
    out += ["", "### Inference and end-to-end", "",
            "| Device | Inference only | Full frame pipeline | Throughput | Peak RSS |",
            "|---|---|---|---|---|"]
    for dev, v in b["devices"].items():
        if not v.get("available"):
            out.append(f"| {dev} | _unavailable — {v.get('reason','')}_ | | | |")
            continue
        out.append(f"| **{dev}** | {v['inference_only']['mean_ms']:.2f} ms "
                   f"(p95 {v['inference_only']['p95_ms']:.2f}) | "
                   f"{v['end_to_end_per_frame']['mean_ms']:.2f} ms | "
                   f"{v['survey']['frames_per_second']:.2f} frames/s | "
                   f"{v['peak_rss_mb']:.0f} MB |")
    if b.get("mps_speedup_vs_cpu"):
        out += ["", f"**MPS speedup over CPU (inference only): "
                    f"{b['mps_speedup_vs_cpu']:.2f}×**"]
    out += ["", f"Model: {b['model_size_mb']} MB · {b['n_images']} frames · "
                f"shapes {b['image_shapes']}",
            "", "Peak RSS stayed far below the 24 GB unified-memory budget, so the "
                "pipeline is not memory-bound on this class of machine."]
    return "\n".join(out)


def t_edge() -> str:
    d = load_json(EXP / "edge_export.json")
    if not d:
        return MISSING
    out = [f"Source checkpoint: {d['source_size_mb']} MB at imgsz {d['imgsz']}", "",
           "| Format | Size | Export time | Runtime latency |", "|---|---|---|---|"]
    for fmt, e in d["exports"].items():
        if "error" in e:
            out.append(f"| {fmt} | _export failed: {e['error']}_ | | |")
            continue
        b = e.get("benchmark") or {}
        lat = (f"{b['mean_ms']:.2f} ms (p95 {b['p95_ms']:.2f})"
               if b.get("mean_ms") else f"_{b.get('error','not benchmarked')}_")
        out.append(f"| {fmt.upper()} | {e['size_mb']} MB | {e['export_seconds']}s | {lat} |")
    for e in d["exports"].values():
        if (e.get("benchmark") or {}).get("providers"):
            out += ["", f"ONNX Runtime providers: `{', '.join(e['benchmark']['providers'])}`"]
            break
    return "\n".join(out)


def t_robustness() -> str:
    d = load_json(EXP / "robustness.json")
    if not d:
        return MISSING
    out = [f"{d['frames']} held-out frames · match IoU {d['match_iou_threshold']}", "",
           "| Condition | Level | P | R | F1 | Recall retained | FA-frames |",
           "|---|---|---|---|---|---|---|"]
    for r in d["results"]:
        ret = r.get("recall_retained_vs_baseline")
        out.append(f"| {r['condition'].replace('_',' ')} | "
                   f"{'—' if r['level'] is None else r['level']} | "
                   f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | "
                   f"{'—' if ret is None else f'{ret:.2f}×'} | "
                   f"{r['false_alarm_frames']}/{r['n_frames_empty']} |")
    out += ["", f"> {d['note']}"]
    return "\n".join(out)


BUILDERS = {
    "TRAINING": t_training, "PREPROCESSING": t_preprocessing, "ABLATION": t_ablation,
    "LATENCY": t_latency, "EDGE": t_edge, "ROBUSTNESS": t_robustness,
}


def main() -> None:
    text = DOC.read_text()
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    filled = []
    while i < len(lines):
        line = lines[i]
        out.append(line)
        if line.strip().startswith("<!-- BENCHMARK:"):
            key = line.strip().removeprefix("<!-- BENCHMARK:").removesuffix("-->").strip()
            i += 1
            # skip previously generated content up to the next heading / rule
            while i < len(lines) and not lines[i].startswith(("## ", "---")):
                i += 1
            out.append("")
            out.append(BUILDERS[key]() if key in BUILDERS else MISSING)
            out.append("")
            filled.append(key)
            continue
        i += 1
    DOC.write_text("\n".join(out) + ("\n" if not text.endswith("\n") else ""))
    print(f"rendered {DOC}: {', '.join(filled)}")


if __name__ == "__main__":
    main()
