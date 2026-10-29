#!/usr/bin/env bash
# Reproduces the SAG degree/exposure-bucket analysis (revision).
#
# For each (dataset, method, seed) we run the method once over the streaming
# graph and report per-degree-quartile AUROC / AUPRC / Macro-F1 / FPR /
# anomaly_ratio / n_nodes. Variants compared:
#   SAG-full, SAG-no-exposure   (always)
#   MIDAS, NetWalk, GraphSAGE   (when registered in src/baselines)
#   ConsisGAD                   (NOT in this codebase -> see UNIMPLEMENTED.md)
#
# Outputs (canonical, latest run; --overwrite to replace):
#   ./outputs/results/revision/exposure_bucket_raw.csv
#   ./outputs/results/revision/exposure_bucket_summary.csv
#   ./outputs/figures/revision/exposure_bucket/<dataset>__<metric>.{pdf,png}
# Plus a timestamped per-run copy at
#   ./outputs/results/revision/exposure_bucket_<ts>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASETS="${DATASETS:-YelpReviews,Enron,BlogCatalog}"
METHODS="${METHODS:-SAG-full,SAG-no-exposure,MIDAS,NetWalk,GraphSAGE}"
SEEDS="${SEEDS:-0,1,2}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
TAG="${TAG:-}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:-}"

python -u experiments/run_exposure_bucket_analysis.py \
    --datasets "${DATASETS}" \
    --methods "${METHODS}" \
    --seeds "${SEEDS}" \
    --output_dir "${OUTPUT_DIR}" \
    ${TAG:+--tag "${TAG}"} \
    ${OVERWRITE_FLAG}
