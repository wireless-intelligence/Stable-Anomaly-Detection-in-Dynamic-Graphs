#!/usr/bin/env bash
# Reproduces the SAG failure-case and limitation analysis (revision exp 13).
#
# Sweeps 7 anomaly types per dataset, comparing SAG-full (in 4 fusion modes
# to expose per-predicate detectability) against GraphSAGE (message-passing)
# and MIDAS (streaming).  role_equivalence and motif anomalies are
# NotImplementedError stubs and recorded as skipped (see UNIMPLEMENTED.md).
#
# Outputs (canonical, latest run; --overwrite to replace):
#   ./outputs/results/revision/failure_case_raw.csv
#   ./outputs/results/revision/failure_case_summary.csv
#   ./outputs/results/revision/failure_case_summary.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASETS="${DATASETS:-Cora,Citeseer,Pubmed,BlogCatalog}"
ANOMALY_TYPES="${ANOMALY_TYPES:-clique_dense,star,chain,bipartite_burst,community_bridge,role_equivalence,motif}"
METHODS="${METHODS:-SAG-full,GraphSAGE,MIDAS}"
SEEDS="${SEEDS:-0,1,2}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
TAG="${TAG:-}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:-}"

python -u experiments/run_failure_case_analysis.py \
    --datasets "${DATASETS}" \
    --anomaly_types "${ANOMALY_TYPES}" \
    --methods "${METHODS}" \
    --seeds "${SEEDS}" \
    --output_dir "${OUTPUT_DIR}" \
    ${TAG:+--tag "${TAG}"} \
    ${OVERWRITE_FLAG}
