#!/usr/bin/env bash
# Reproduces the SAG time-to-detect / detection-delay experiment.
#
# Streams edges with K-spaced checkpoint snapshots; per-anomaly delay is
# (first checkpoint at which score crosses validation threshold) -
# (first edge in stream order that touches the anomaly node).
#
# Methods:
#   SAG-full, SAG-no-decay, SAG-no-exposure, MIDAS  (streaming-compatible)
#   NetWalk: NOT streaming-compatible -> recorded as skipped
#            (see results/revision/UNIMPLEMENTED.md)
#
# Outputs (canonical, latest run; --overwrite to replace):
#   ./outputs/results/revision/time_to_detect_raw.csv
#   ./outputs/results/revision/time_to_detect_summary.csv
#   ./outputs/figures/revision/time_to_detect/<dataset>_ttd_cdf.{pdf,png}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASETS="${DATASETS:-YelpReviews,Enron,BlogCatalog,Cora}"
METHODS="${METHODS:-SAG-full,SAG-no-decay,SAG-no-exposure,MIDAS,NetWalk}"
SEEDS="${SEEDS:-0,1,2}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-200}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
TAG="${TAG:-}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:-}"

python -u experiments/run_time_to_detect.py \
    --datasets "${DATASETS}" \
    --methods "${METHODS}" \
    --seeds "${SEEDS}" \
    --num_checkpoints "${NUM_CHECKPOINTS}" \
    --output_dir "${OUTPUT_DIR}" \
    ${TAG:+--tag "${TAG}"} \
    ${OVERWRITE_FLAG}
