#!/usr/bin/env bash
# Reproduces the SAG parameter sensitivity sweeps.
#
# One-parameter-at-a-time sensitivity over 9 hyperparameters x 4 priority
# datasets x N seeds (default 3). Each run records AUROC / AUPRC / Macro-F1
# and balanced variants, plus per-edge latency and peak memory.
#
# Outputs (canonical, latest run):
#   ./outputs/results/revision/parameter_sensitivity_raw.csv
#   ./outputs/results/revision/parameter_sensitivity_summary.csv
#   ./outputs/figures/revision/parameter_sensitivity/<param>.{pdf,png}
# Plus a timestamped per-run copy at
#   ./outputs/results/revision/parameter_sensitivity_<ts>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASETS="${DATASETS:-YelpReviews,Enron,Cora,BlogCatalog}"
PARAMS="${PARAMS:-alpha,gamma,R,delta,rho,P_A,P_a_given_A,P_a_given_not_A,beta_upd}"
SEEDS="${SEEDS:-0,1,2}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
TAG="${TAG:-}"

python -u experiments/run_parameter_sensitivity.py \
    --datasets "${DATASETS}" \
    --params "${PARAMS}" \
    --seeds "${SEEDS}" \
    --output_dir "${OUTPUT_DIR}" \
    ${TAG:+--tag "${TAG}"}
