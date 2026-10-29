#!/usr/bin/env bash
# Reproduces the SAG predicate-ablation experiment.
#
# Sweeps seven variants over seeds 0..9 on six datasets, isolating the
# contribution of each predicate group (state, attribute, structural) and
# the Evidence Selection mechanism.
#
# Variants:
#   full | edge_only | structural_only | attribute_only |
#   no_attribute | no_exposure | no_evidence_selection
#
# Outputs (canonical, latest run):
#   ./outputs/results/revision/predicate_ablation_raw.csv
#   ./outputs/results/revision/predicate_ablation_summary.csv
# Plus a timestamped per-run copy at
#   ./outputs/results/revision/predicate_ablation_<ts>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASETS="${DATASETS:-Cora,Citeseer,Pubmed,BlogCatalog,YelpReviews,Enron}"
VARIANTS="${VARIANTS:-full,edge_only,structural_only,attribute_only,no_attribute,no_exposure,no_evidence_selection}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
TAG="${TAG:-}"

python -u experiments/run_predicate_ablation.py \
    --datasets "${DATASETS}" \
    --variants "${VARIANTS}" \
    --seeds "${SEEDS}" \
    --output_dir "${OUTPUT_DIR}" \
    ${TAG:+--tag "${TAG}"}
