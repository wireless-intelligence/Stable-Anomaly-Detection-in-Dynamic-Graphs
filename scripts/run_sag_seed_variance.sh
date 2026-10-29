#!/usr/bin/env bash
# Reproduces the SAG meaningful multi-seed variance experiment.
#
# Sweeps four seed axes (stream_order, anomaly_injection,
# split, threshold) over seeds 0..9 on the four priority datasets. Skipped
# (dataset, axis) combinations are recorded with reasons in the raw CSV.
#
# Outputs (canonical, latest run):
#   ./outputs/results/revision/sag_seed_variance_raw.csv
#   ./outputs/results/revision/sag_seed_variance_summary.csv
# Plus a timestamped per-run copy at ./outputs/results/revision/seed_variance_<ts>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASETS="${DATASETS:-YelpReviews,Enron,Cora,BlogCatalog}"
AXES="${AXES:-stream_order,anomaly_injection,split,threshold}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
TAG="${TAG:-}"

python -u experiments/run_sag_seed_variance.py \
    --datasets "${DATASETS}" \
    --axes "${AXES}" \
    --seeds "${SEEDS}" \
    --output_dir "${OUTPUT_DIR}" \
    ${TAG:+--tag "${TAG}"}
