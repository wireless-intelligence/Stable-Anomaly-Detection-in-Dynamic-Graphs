#!/usr/bin/env bash
# Reproduces the SAG anomaly-type / strength sweep (revision experiment 9).
#
# Axes:
#   A_clique_size        clique_size in {5, 10, 15, 20, 25}
#   B_attr_percentile    attr_distance_percentile in {50, 70, 90, 95, 99}
#   C_mix_ratio          structural_only / attribute_only / 50_50 / 25_75 / 75_25
#   D_non_clique         star / chain / bipartite_burst / community_bridge
#                        (NotImplementedError stubs in AnomalyInjector;
#                         runner records each as skipped with reason ->
#                         see results/revision/UNIMPLEMENTED.md)
#
# Outputs (canonical, latest run; --overwrite to replace):
#   ./outputs/results/revision/anomaly_type_sweep_raw.csv
#   ./outputs/results/revision/anomaly_type_sweep_summary.csv
#   ./outputs/figures/revision/anomaly_type_sweep/<axis>__<metric>.{pdf,png}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASETS="${DATASETS:-Cora,Citeseer,Pubmed,BlogCatalog}"
AXES="${AXES:-A_clique_size,B_attr_percentile,C_mix_ratio,D_non_clique}"
SEEDS="${SEEDS:-0,1,2}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
TAG="${TAG:-}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:-}"

python -u experiments/run_anomaly_type_sweep.py \
    --datasets "${DATASETS}" \
    --axes "${AXES}" \
    --seeds "${SEEDS}" \
    --output_dir "${OUTPUT_DIR}" \
    ${TAG:+--tag "${TAG}"} \
    ${OVERWRITE_FLAG}
