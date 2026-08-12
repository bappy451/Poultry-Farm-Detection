#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

show_usage() {
  cat <<EOF
Usage:
  ./scripts/run_dino_vit_compare.sh [--config PATH] [--dry-run] [options]

What it does:
  - Runs the 9-experiment DINO ViT-Base/16 Downstream Comparison Suite sequentially
  - Evaluates 3 initialization strategies (DINO pretrained, ImageNet supervised, Random scratch)
  - Across 3 target datasets (Poultry House Detection, GRSD Greenhouse, DOTA Aerial)
  - Displays a modern live Rich progress bar with ETA and summary tables
  - Generates comparison_results.csv incrementally in runs/dino_vit_compare/

Options:
  --config PATH   Path to suite YAML config. Default: configs/dino_vit_downstream_compare.yaml
  --dry-run       Run only 1 epoch/batch per experiment to verify pipeline correctness
  --help, -h      Show usage help
EOF
}

CONFIG_PATH="configs/dino_vit_downstream_compare.yaml"
DRY_RUN=""
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_usage
      exit 0
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="--dry-run"
      shift 1
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift 1
      ;;
  esac
done

if [[ -n "${POULTRY_PYTHON}" ]]; then
  PYTHON_EXE="${POULTRY_PYTHON}"
elif [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
  PYTHON_EXE="${REPO_DIR}/.venv/bin/python"
else
  PYTHON_EXE="python3"
fi

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH}"

echo "========================================================================"
echo " Starting DINO ViT-Base Downstream 9-Experiment Comparison Suite"
echo "========================================================================"
echo "Python: ${PYTHON_EXE}"
echo "Config: ${CONFIG_PATH}"

exec "${PYTHON_EXE}" -m downstream_finetune --dino-vit-compare --config "${CONFIG_PATH}" ${DRY_RUN} "${FORWARD_ARGS[@]}"
