#!/usr/bin/env bash
# Reproduces the SAG scalability stress test (revision experiment 10).
#
# For each (dataset, method, seed, prefix_pct) we run on the first prefix_pct%
# of the edge stream and record runtime, per-edge latency, p95 batch latency,
# peak CPU/GPU memory, and the size of maintained state (node states + edge
# counter entries / sketch bucket count).
#
# Outputs (canonical, latest run; --overwrite to replace):
#   ./outputs/results/revision/scalability_raw.csv
#   ./outputs/results/revision/scalability_summary.csv
#   ./outputs/figures/revision/scalability/<dataset>__<metric>.{pdf,png}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASETS="${DATASETS:-YelpReviews,Enron,BlogCatalog}"
METHODS="${METHODS:-SAG-full,MIDAS,NetWalk}"
SEEDS="${SEEDS:-0,1,2}"
PREFIX_PCTS="${PREFIX_PCTS:-10,25,50,75,100}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
TAG="${TAG:-}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:-}"

python -u experiments/run_scalability_stress_test.py \
    --datasets "${DATASETS}" \
    --methods "${METHODS}" \
    --seeds "${SEEDS}" \
    --prefix_pcts "${PREFIX_PCTS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_dir "${OUTPUT_DIR}" \
    ${TAG:+--tag "${TAG}"} \
    ${OVERWRITE_FLAG}
