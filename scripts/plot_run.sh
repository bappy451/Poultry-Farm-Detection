#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/plot_run.sh RUN_DIR [options]

What it does:
  - Generates professional-grade training plots for a single run (loss curves, etc).

Options:
  RUN_DIR               Path to the run directory containing metrics.csv/json.
  --output-dir DIR      Directory to save plots. Default: RUN_DIR/plots.
  --python PATH         Python executable to use.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

. "${SCRIPT_DIR}/common.sh"

RUN_DIR=""
OUTPUT_DIR=""
PYTHON_BIN=""
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir=*)
      OUTPUT_DIR="${1#*=}"
      shift 1
      ;;
    --output-dir)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --output-dir requires a path." >&2
        exit 1
      fi
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --python=*)
      PYTHON_BIN="${1#*=}"
      shift 1
      ;;
    --python)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --python requires a path." >&2
        exit 1
      fi
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -z "${RUN_DIR}" && "$1" != --* ]]; then
        RUN_DIR="$1"
      else
        FORWARD_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ -z "${RUN_DIR}" ]]; then
  echo "Error: RUN_DIR is required." >&2
  usage
  exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${RUN_DIR}/plots"
fi

PYTHON_BIN=$(resolve_python "${PYTHON_BIN}" "${CODE_DIR}")

./scripts/poultry_geo.sh \
  --python "${PYTHON_BIN}" \
  plot-run "${RUN_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  "${FORWARD_ARGS[@]}"
