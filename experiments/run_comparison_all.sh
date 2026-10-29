#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
mkdir -p outputs/logs/comparison_all
LOG="outputs/logs/comparison_all/run_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

# 7 baselines that don't require torch_sparse:
#   GraphSAGE (PyG simple), DOMINANT (our local), DySAT, MIDAS, NetWalk, SedanSpot
# Plus our SAG. Skip pygod-only ones (HGT, ANOMALOUS, ONE, Radar, SCAN) since
# they require torch_sparse for some datasets and can OOM on big graphs.
METHODS="SAG,DOMINANT,AnomalyDAE,CoLA,CONAD,GraphSAGE,HGT,DySAT,MIDAS,NetWalk"

for D in Cora Citeseer Pubmed BlogCatalog ACM Flickr Amazon Enron CTU-13 YelpReviews; do
  echo "=== $D ===" | tee -a "$LOG"
  python experiments/run_comparison.py --dataset "$D" --n_runs 1 --methods "$METHODS" 2>&1 | tee -a "$LOG"
done
echo "ALL DONE" | tee -a "$LOG"
