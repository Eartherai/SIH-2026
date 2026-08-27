#!/usr/bin/env bash
# Reproduce every measured number in docs/BENCHMARKS.md.
#
# Usage:  ./scripts/run_full_evaluation.sh <raw-weights> [preprocessed-weights]
#
# Runs, in order: verification fit -> pipeline ablation -> latency benchmark ->
# robustness study -> edge export -> (optional) matched preprocessing 2x2, then
# regenerates docs/BENCHMARKS.md from the recorded JSON.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

RAW_W="${1:?usage: run_full_evaluation.sh <raw-weights> [preprocessed-weights]}"
PP_W="${2:-}"

step() { printf "\n\033[1m==> %s\033[0m\n" "$*"; }

step "1/6  Fitting verification stage on the VALIDATION survey"
python scripts/fit_verification.py --weights "$RAW_W" --tag milco_nombo

step "2/6  Pipeline ablation on the HELD-OUT test surveys"
python scripts/evaluate.py --weights "$RAW_W" --out experiments/ablation.json

step "3/6  Latency & memory benchmark"
python scripts/benchmark.py --weights "$RAW_W" --n 30

step "4/6  Robustness under controlled degradation"
python scripts/robustness.py --weights "$RAW_W" --limit 120

step "5/6  Edge export"
python scripts/export_edge.py --weights "$RAW_W" --formats onnx

if [ -n "$PP_W" ]; then
  step "6/6  Matched preprocessing 2x2"
  python scripts/ablate_preprocessing.py --raw-weights "$RAW_W" --pp-weights "$PP_W"
else
  step "6/6  Matched preprocessing 2x2 -- SKIPPED (no preprocessed-weights argument)"
  echo "     Train one with:"
  echo "       python scripts/prepare_preprocessed.py --water-column-removal"
  echo "       python scripts/train.py --exp-id E06 --data data/processed/milco_nombo_yolo_pp/data.yaml"
fi

step "Rendering docs/BENCHMARKS.md from the recorded measurements"
python scripts/render_benchmarks.py

printf "\n\033[1mDone.\033[0m Every table in docs/BENCHMARKS.md is now generated from experiments/.\n"
