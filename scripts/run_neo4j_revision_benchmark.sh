#!/usr/bin/env bash
# Reproduces the SAG Neo4j benchmark with system-level metrics.
#
# Two modes (set with MODE env var or --mode flag):
#   local_only  (default)  -- on-disk pipeline; per-batch timing of the
#                              Python streaming path. Reported as
#                              neo4j_used=False so it isn't conflated with
#                              the DB-backed run.
#   neo4j                  -- pulls edges from a running Neo4j over bolt;
#                              per-batch latency includes Cypher round-trip
#                              + local processing. Will refuse to run if
#                              the driver cannot connect.
#
# Reported per (dataset, method, seed):
#   latency mean / median / p95 / p99 / min / max (us)
#   throughput (edges/s)
#   memory (Python process RSS, NOT Neo4j server-side memory or indexes)
#   warmup_edges, measurement_edges, batch_size, workers, device_path,
#   neo4j_used, memory_includes_neo4j_indexes  (always False here)
#
# Outputs (canonical, latest run; --overwrite to replace):
#   ./outputs/results/revision/neo4j_benchmark_raw.csv
#   ./outputs/results/revision/neo4j_benchmark_summary.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASETS="${DATASETS:-YelpReviews,Enron,BlogCatalog}"
METHODS="${METHODS:-SAG,MIDAS,GraphSAGE}"
SEEDS="${SEEDS:-0,1,2}"
MODE="${MODE:-local_only}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
WARMUP_EDGES="${WARMUP_EDGES:-2000}"
MEASUREMENT_EDGES="${MEASUREMENT_EDGES:-20000}"
WORKERS="${WORKERS:-1}"
NEO4J_URI="${NEO4J_URI:-bolt://127.0.0.1:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-neo4j}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
TAG="${TAG:-}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:-}"

python -u experiments/run_neo4j_revision_benchmark.py \
    --datasets "${DATASETS}" \
    --methods "${METHODS}" \
    --seeds "${SEEDS}" \
    --mode "${MODE}" \
    --batch_size "${BATCH_SIZE}" \
    --warmup_edges "${WARMUP_EDGES}" \
    --measurement_edges "${MEASUREMENT_EDGES}" \
    --workers "${WORKERS}" \
    --uri "${NEO4J_URI}" \
    --user "${NEO4J_USER}" \
    --password "${NEO4J_PASSWORD}" \
    --output_dir "${OUTPUT_DIR}" \
    ${TAG:+--tag "${TAG}"} \
    ${OVERWRITE_FLAG}
