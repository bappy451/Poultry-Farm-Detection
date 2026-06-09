#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/plot_ssl_grid.sh --training-name NAME [options]

What it does:
  - Generates a professional-grade summary plot for an SSL grid search campaign.
  - Automatically finds the grid_search_results.csv in the trained_models folder.

Options:
  --training-name NAME  Name of the training campaign (folder in trained_models).
  --root-dir DIR        Root directory for trained models. Default: trained_models.
  --output-dir DIR      Directory to save plots. Default: <run_dir>/figures.
  --python PATH         Python executable to use.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

. "${SCRIPT_DIR}/common.sh"
TRAINING_NAME=""
ROOT_DIR="trained_models"
OUTPUT_DIR=""
PYTHON_BIN=""
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --training-name=*)
      TRAINING_NAME="${1#*=}"
      shift 1
      ;;
    --training-name)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --training-name requires a value." >&2
        exit 1
      fi
      TRAINING_NAME="$2"
      shift 2
      ;;
    --root-dir=*)
      ROOT_DIR="${1#*=}"
      shift 1
      ;;
    --root-dir)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --root-dir requires a path." >&2
        exit 1
      fi
      ROOT_DIR="$2"
      shift 2
      ;;
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
      if [[ -z "${TRAINING_NAME}" && "$1" != --* ]]; then
        TRAINING_NAME="$1"
      else
        FORWARD_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ -z "${TRAINING_NAME}" ]]; then
  echo "Error: --training-name is required." >&2
  usage
  exit 1
fi

RESULTS_CSV="${CODE_DIR}/${ROOT_DIR}/${TRAINING_NAME}/grid_search_results.csv"

if [[ ! -f "${RESULTS_CSV}" ]]; then
  echo "Error: Could not find results CSV at ${RESULTS_CSV}" >&2
  exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${CODE_DIR}/${ROOT_DIR}/${TRAINING_NAME}/figures"
fi

PYTHON_BIN=$(resolve_python "${PYTHON_BIN}" "${CODE_DIR}")

./scripts/poultry_geo.sh \
  --python "${PYTHON_BIN}" \
  ssl-grid-plot "${RESULTS_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  "${FORWARD_ARGS[@]}"
