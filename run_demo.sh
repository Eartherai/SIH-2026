#!/usr/bin/env bash
# AQUA-SHIELD demo launcher.
# Validates the environment, verifies the model and demo data exist, then starts
# the dashboard. Fails loudly with an actionable message rather than starting a
# dashboard that has nothing real to show.
set -euo pipefail

cd "$(dirname "$0")"
BOLD=$'\033[1m'; RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; OFF=$'\033[0m'
say()  { printf "%s\n" "$*"; }
ok()   { printf "  ${GRN}OK${OFF}   %s\n" "$*"; }
warn() { printf "  ${YLW}WARN${OFF} %s\n" "$*"; }
die()  { printf "  ${RED}FAIL${OFF} %s\n\n" "$*"; exit 1; }

say "${BOLD}AQUA-SHIELD${OFF} — Acoustic Intelligence for Underwater Anomaly, Debris & Marine-Hazard Localization"
say "SIH 2026 · PS 26057 · MoES / NIOT"
say ""

# ---------------------------------------------------------------- 1. environment
say "${BOLD}1. Environment${OFF}"
if [ ! -d .venv ]; then
  die "No virtual environment found. Run:  ./setup.sh"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
ok "virtualenv active ($(python --version 2>&1))"

python - <<'PY' || die "PyTorch is not importable. Run ./setup.sh"
import sys, torch
print(f"  OK   torch {torch.__version__}")
PY

PYTHONPATH=src python -m aquashield.device | sed 's/^/  /'
say ""

# ---------------------------------------------------------------- 2. model
say "${BOLD}2. Model${OFF}"
MODEL="$(ls -1 models/*.pt 2>/dev/null | head -1 || true)"
if [ -z "$MODEL" ]; then
  MODEL="$(find runs -name best.pt 2>/dev/null | head -1 || true)"
fi
if [ -z "$MODEL" ]; then
  say ""
  die "No trained detector found.
       AQUA-SHIELD refuses to run on a placeholder model — every detection in the
       dashboard comes from real inference. Train one first:

         python scripts/prepare_milco_nombo.py
         python scripts/train.py --exp-id E01 --epochs 150"
fi
ok "detector: $MODEL"

for f in models/fp_filter_milco_nombo.json models/calibration_milco_nombo.json; do
  if [ -f "$f" ]; then ok "verification: $(basename "$f")"
  else warn "$(basename "$f") missing — falling back to the rule-based filter, and
         confidences will be reported as RAW detector scores (calibrated: false).
         Fit them with:  python scripts/fit_verification.py --weights $MODEL"
  fi
done
say ""

# ---------------------------------------------------------------- 3. demo data
say "${BOLD}3. Demo data${OFF}"
if [ ! -d demo_data ] || [ -z "$(ls -A demo_data 2>/dev/null)" ]; then
  warn "demo_data/ is empty — building it now"
  python scripts/build_demo_data.py || die "could not build demo data"
fi
N=$(find demo_data -maxdepth 1 -mindepth 1 -type d | wc -l | tr -d ' ')
ok "$N demo scenarios available"
say ""

# ---------------------------------------------------------------- 4. launch
say "${BOLD}4. Launching dashboard${OFF}"
say "  Opening http://localhost:8501"
say "  Offline map mode: ${AQS_OFFLINE_MAP:-0}  (export AQS_OFFLINE_MAP=1 to disable map tiles)"
say ""
exec streamlit run dashboard/app.py --server.headless false --browser.gatherUsageStats false
