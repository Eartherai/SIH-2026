#!/usr/bin/env bash
# One-time setup for AQUA-SHIELD on Apple Silicon (and anything else).
set -euo pipefail
cd "$(dirname "$0")"

echo "AQUA-SHIELD setup"
echo "================="

PY=""
if command -v uv >/dev/null 2>&1; then
  echo "-> uv found; creating a Python 3.12 environment"
  uv venv --python 3.12 .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install -r requirements.txt
else
  for c in python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1; then
      V=$("$c" -c 'import sys;print(sys.version_info[:2]>=(3,10))')
      [ "$V" = "True" ] && PY="$c" && break
    fi
  done
  [ -z "$PY" ] && { echo "ERROR: need Python >= 3.10. Install it, or install uv:"; \
                    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
  echo "-> using $PY"
  "$PY" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
fi

echo ""
PYTHONPATH=src python -m aquashield.device
echo ""
echo "Next:"
echo "  1. python scripts/download_datasets.py      # fetch MILCO/NOMBO (CC BY 4.0, ~218 MB)"
echo "  2. python scripts/prepare_milco_nombo.py    # build survey-level splits"
echo "  3. python scripts/train.py --exp-id E01 --epochs 150"
echo "  4. python scripts/fit_verification.py --weights runs/**/best.pt"
echo "  5. ./run_demo.sh"
