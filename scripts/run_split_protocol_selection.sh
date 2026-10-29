#!/usr/bin/env bash
# Reproduces the split-protocol selection experiment.
#
# 4 candidate splits per dataset:
#   60_20_20      (first 60% / next 20% / last 20%, edge-chronological)
#   70_10_20      (first 70% / next 10% / last 20%)
#   70_20_10      (first 70% / next 20% / last 10%)
#   70_30_rank    (first 70% / last 30%, ranking-only -- no Macro-F1)
#
# Methods (CLI-configurable):
#   SAG-online-strict : single chronological streaming pass with snapshots
#   MIDAS             : streaming reference baseline
#   DOMINANT          : strong static GAD baseline (fit on train edges only)
#   ConsisGAD         : strong GNN/GAD baseline (fit on train edges only)
#
# Outputs (canonical, latest run; --overwrite to replace):
#   ./outputs/results/revision_strict/split_selection_metadata.csv
#   ./outputs/results/revision_strict/split_selection_results.csv
#   ./outputs/results/revision_strict/split_selection_summary.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASETS="${DATASETS:-Cora,Citeseer,Pubmed,BlogCatalog,YelpReviews,Enron,CTU-13}"
METHODS="${METHODS:-SAG-online-strict,MIDAS,DOMINANT,ANEMONE}"
SEEDS="${SEEDS:-42,43,44,45,46,47,48,49,50,51}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
TAG="${TAG:-}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:-}"

python -u experiments/run_split_protocol_selection.py \
    --datasets "${DATASETS}" \
    --methods "${METHODS}" \
    --seeds "${SEEDS}" \
    --output_dir "${OUTPUT_DIR}" \
    ${TAG:+--tag "${TAG}"} \
    ${OVERWRITE_FLAG}
