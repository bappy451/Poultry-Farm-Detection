#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/spark_grid_train.sh --training-name NAME [options]

What it does:
  - runs stateful resume-aware SparK (SSL) grid search training
  - appends every run to one project log file by default

Options:
  --training-name NAME  Run identifier.
  --config PATH         Path to the SparK grid-search YAML config. Default: configs/spark_grid_search.yaml
  --epochs INT          Override runtime.epochs from the config.
  --root-dir PATH       Root directory for saved runs. Default: trained_models
  --python PATH         Python executable to use.
  --log-file PATH       Explicit log file path.
  --no-log              Disable automatic append logging for this run
  -h, --help            Show this wrapper help
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

. "${SCRIPT_DIR}/common.sh"

TRAINING_NAME=""
CONFIG_PATH="configs/spark_grid_search.yaml"
EPOCHS=""
ROOT_DIR="trained_models"
PYTHON_BIN=""
LOG_FILE=""
NO_LOG="0"
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
    --config=*)
      CONFIG_PATH="${1#*=}"
      shift 1
      ;;
    --config)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --config requires a path." >&2
        exit 1
      fi
      CONFIG_PATH="$2"
      shift 2
      ;;
    --epochs=*)
      EPOCHS="${1#*=}"
      shift 1
      ;;
    --epochs)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --epochs requires an integer." >&2
        exit 1
      fi
      EPOCHS="$2"
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
    --log-file=*)
      LOG_FILE="${1#*=}"
      shift 1
      ;;
    --log-file)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --log-file requires a path." >&2
        exit 1
      fi
      LOG_FILE="$2"
      shift 2
      ;;
    --no-log)
      NO_LOG="1"
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -z "${TRAINING_NAME}" ]]; then
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

PYTHON_BIN=$(resolve_python "${PYTHON_BIN}" "${CODE_DIR}")

WRAPPER_ARGS=(--python "${PYTHON_BIN}")
[[ -n "${LOG_FILE}" ]] && WRAPPER_ARGS+=(--log-file "${LOG_FILE}")
[[ "${NO_LOG}" == "1" ]] && WRAPPER_ARGS+=(--no-log)

./scripts/poultry_geo.sh "${WRAPPER_ARGS[@]}" \
  spark-grid-train "${TRAINING_NAME}" \
  --config "${CONFIG_PATH}" \
  --root-dir "${ROOT_DIR}" \
  ${EPOCHS:+--epochs "${EPOCHS}"} \
  "${FORWARD_ARGS[@]}"
