#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
mkdir -p outputs/logs/ablation_all
LOG="outputs/logs/ablation_all/run_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

DATASETS="Cora Citeseer Pubmed BlogCatalog ACM Flickr Amazon Enron CTU-13 YelpReviews"

echo "==> Production fusion (matches main results table)" | tee -a "$LOG"
for D in $DATASETS; do
  echo "=== $D (production) ===" | tee -a "$LOG"
  python experiments/run_ablation.py --dataset "$D" --n_runs 1 --tag prod 2>&1 | tee -a "$LOG"
done

echo "==> h_only fusion (isolates streaming-state for paper ablation figure)" | tee -a "$LOG"
for D in $DATASETS; do
  echo "=== $D (h_only) ===" | tee -a "$LOG"
  python experiments/run_ablation.py --dataset "$D" --fusion_mode h_only --n_runs 1 --tag h_only 2>&1 | tee -a "$LOG"
done

echo "ALL DONE" | tee -a "$LOG"
