#!/usr/bin/env bash
# Reproduces the SAG calibration & threshold analysis (revision experiment 12).
#
# 10 datasets x 5 normalization variants x 5 seeds. For each (dataset, variant,
# seed) we record ECE, AUROC/AUPRC, best-F1 threshold + macro-F1 / TPR / FPR
# at that threshold, plus reliability bins and PR-curve samples.
#
# Outputs (canonical, latest run; --overwrite to replace):
#   ./outputs/results/revision/calibration_raw.csv
#   ./outputs/results/revision/calibration_bins.csv
#   ./outputs/results/revision/calibration_pr.csv
#   ./outputs/results/revision/calibration_summary.csv
#   ./outputs/figures/revision/calibration/<dataset>__reliability.{pdf,png}
#   ./outputs/figures/revision/calibration/<dataset>__pr.{pdf,png}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASETS="${DATASETS:-Cora,Citeseer,Pubmed,BlogCatalog,Flickr,ACM,Amazon,YelpChi,YelpReviews,Enron}"
VARIANTS="${VARIANTS:-bayesian,raw,minmax,zscore_sigmoid,rank}"
SEEDS="${SEEDS:-0,1,2,3,4}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
TAG="${TAG:-}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:-}"

python -u experiments/run_calibration_analysis.py \
    --datasets "${DATASETS}" \
    --variants "${VARIANTS}" \
    --seeds "${SEEDS}" \
    --output_dir "${OUTPUT_DIR}" \
    ${TAG:+--tag "${TAG}"} \
    ${OVERWRITE_FLAG}
