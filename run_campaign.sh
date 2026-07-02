#!/usr/bin/env bash
set -e

# This script runs the full pretraining campaign (SparK and MIM) in sequence.
# It uses the equivalent bash wrappers for Linux environments.

echo "======================================================================"
echo "[1/2] STARTING SPARK GRID SEARCH"
echo "======================================================================"
./scripts/spark_grid_train.sh --training-name "spark-search" --config configs/spark_grid_search.yaml --verbose

echo ""
echo "[GENERATING SPARK SUMMARY PLOTS]"
./scripts/plot_ssl_grid.sh --training-name "spark-search"

echo ""
echo "======================================================================"
echo "[2/2] STARTING MIM GRID SEARCH"
echo "======================================================================"
./scripts/train_ssl_grid.sh --training-name "mim-search" --config configs/mim_grid_config.yaml --verbose

echo ""
echo "[GENERATING MIM SUMMARY PLOTS]"
./scripts/plot_ssl_grid.sh --training-name "mim-search"

echo ""
echo "======================================================================"
echo "PRETRAINING CAMPAIGN COMPLETED SUCCESSFULLY"
echo "======================================================================"
