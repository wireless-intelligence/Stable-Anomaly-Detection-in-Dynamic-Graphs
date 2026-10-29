#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export NEO4J_URI=bolt://127.0.0.1:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=changeme
mkdir -p outputs/logs/neo4j_all
LOG="outputs/logs/neo4j_all/run_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"
for D in Enron BlogCatalog Amazon; do
  echo "=== $D ===" | tee -a "$LOG"
  python experiments/neo4j_benchmark.py --dataset "$D" --methods 'SAG,GraphSAGE,DOMINANT,DySAT' 2>&1 | tee -a "$LOG"
done
echo "ALL DONE" | tee -a "$LOG"
