#!/usr/bin/env bash
set +e   # don't stop on failure
cd "$(dirname "$0")/.."
mkdir -p outputs/logs/comparison_all
LOG="outputs/logs/comparison_all/run_remaining_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

# Same as run_comparison_all.sh but only datasets that haven't run yet
METHODS="SAG,DOMINANT,AnomalyDAE,CoLA,CONAD,GraphSAGE,HGT,DySAT,MIDAS,NetWalk"

for D in Flickr Amazon Enron CTU-13 YelpReviews; do
  echo "=== $D ===" | tee -a "$LOG"
  python experiments/run_comparison.py --dataset "$D" --n_runs 1 --methods "$METHODS" 2>&1 | tee -a "$LOG"
done
echo "ALL DONE" | tee -a "$LOG"
