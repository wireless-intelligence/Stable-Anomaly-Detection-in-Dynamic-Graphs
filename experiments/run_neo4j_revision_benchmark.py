"""Neo4j benchmark with per-batch timing for percentile / throughput stats.

Extends experiments/neo4j_benchmark.py.  Two modes:

  --mode local_only  (default)
     Stream is read from the on-disk loader; per-batch timing covers the
     Python pipeline only.  Useful when no Neo4j instance is available.
     Reported as `neo4j_used=False` in the CSV so downstream consumers don't
     mistake it for a database-backed result.

  --mode neo4j
     Edges are queried from a running Neo4j (bolt://...) in batches; the
     batch latency includes both the Cypher round-trip and the local
     processing step.  If the driver fails to connect, the script exits
     non-zero with a clear message rather than silently switching modes.

Reported per (dataset, method, seed):
  latency_mean_us, latency_median_us, latency_p95_us, latency_p99_us,
  throughput_edges_per_s, memory_gb (process RSS), warmup_edges,
  measurement_edges, workers, device_path, neo4j_used,
  memory_includes_neo4j_indexes (== False -- we only measure the Python
  process; Neo4j server-side memory and on-disk indexes are NOT included).

Outputs (canonical, latest run; --overwrite to replace):
  outputs/results/revision/neo4j_benchmark_raw.csv
  outputs/results/revision/neo4j_benchmark_summary.csv
"""

import argparse
import copy
import csv
import os
import sys
import time
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from configs import Config, get_dataset_config
from src.baselines import BASELINES
from src.data import DataLoader
from src.evaluation.metrics import (
    compute_auroc, compute_auprc, compute_macro_f1,
)
from src.models import create_sag_model
from src.utils import MemoryTracker, save_json, set_seed, setup_logger


METHODS_DEFAULT = ("SAG", "MIDAS", "GraphSAGE")
DATASETS_DEFAULT = ("YelpReviews", "Enron", "BlogCatalog")
SEEDS_DEFAULT = (0, 1, 2)
BASE_SEED = 42


# ---------------------------------------------------------------------------
class CfgShim:
    pass


def build_cfg(dc: dict, seed: int):
    cfg = CfgShim()
    for k in dir(Config):
        if k.startswith("_") or callable(getattr(Config, k, None)):
            continue
        try:
            setattr(cfg, k, copy.deepcopy(getattr(Config, k)))
        except Exception:
            pass
    for k, v in dc.items():
        try:
            setattr(cfg, k, copy.deepcopy(v))
        except Exception:
            pass
    cfg.has_ground_truth = bool(dc.get("has_ground_truth", False))
    cfg.use_continuous_dev = False
    cfg.use_real_time_decay = False
    cfg.seed = int(seed)
    return cfg


# ---------------------------------------------------------------------------
def neo4j_connect(uri: str, user: str, password: str, database: str = None):
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise RuntimeError(
            "the neo4j Python driver is not installed. `pip install neo4j` "
            "or run with --mode local_only.") from e
    driver = GraphDatabase.driver(uri, auth=(user, password))
    # Probe connectivity.
    with driver.session(database=database) if database else driver.session() as session:
        session.run("RETURN 1").consume()
    return driver


def neo4j_query_batch(driver, dataset: str, offset: int, limit: int,
                      database: str = None):
    """Pull a batch of edges from Neo4j (in `t` order, ascending) and return
    (edges_ndarray, query_latency_seconds)."""
    q = (
        "MATCH (a:Node {type: $type})-[r:INTERACT]->(b:Node {type: $type}) "
        "RETURN a.id AS u, b.id AS v "
        "ORDER BY r.t SKIP $skip LIMIT $lim"
    )
    t0 = time.perf_counter()
    with driver.session(database=database) if database else driver.session() as session:
        result = session.run(q, type=dataset, skip=int(offset), lim=int(limit))
        rows = [(r["u"], r["v"]) for r in result]
    dt = time.perf_counter() - t0
    if not rows:
        return np.zeros((0, 2), dtype=np.int64), dt
    return np.asarray(rows, dtype=np.int64), dt


# ---------------------------------------------------------------------------
def build_method(method: str, num_nodes: int, cfg, features, adj, seed: int):
    if method == "SAG":
        return ("sag", create_sag_model(num_nodes=num_nodes, config=cfg,
                                        features=features, adj=adj))
    if method in BASELINES:
        cls = BASELINES[method]
        try:
            return ("baseline",
                    cls(num_nodes=num_nodes, config=cfg, features=features, seed=seed))
        except TypeError:
            return ("baseline",
                    cls(num_nodes=num_nodes, config=cfg, features=features))
    raise ValueError(f"unknown method: {method}")


def stream_in_batches_local(method: str, edges: np.ndarray, kind: str, model,
                             batch_size: int, warmup_edges: int,
                             measurement_edges: int, seed: int,
                             order: str, has_ground_truth: bool):
    """Times per-batch processing using on-disk edges.  Returns
    (per_batch_latency_us, num_warmup, num_measurement, total_meas_time_s)."""
    rng = np.random.default_rng(seed)
    if order == "shuffle":
        idx = rng.permutation(edges.shape[0])
        edges = edges[idx]
    M = edges.shape[0]
    W = min(warmup_edges, M)
    end = min(W + measurement_edges, M)
    measure_count = end - W
    per_batch_us = []
    if kind == "sag":
        # Warm-up
        for i in range(0, W, batch_size):
            chunk = edges[i: min(i + batch_size, W)]
            if chunk.shape[0] == 0:
                continue
            model.process_stream_batch(chunk, passes=1, shuffle=False, rng=rng)
        # Measurement
        t_meas_start = time.perf_counter()
        for i in range(W, end, batch_size):
            chunk = edges[i: min(i + batch_size, end)]
            if chunk.shape[0] == 0:
                continue
            t0 = time.perf_counter()
            model.process_stream_batch(chunk, passes=1, shuffle=False, rng=rng)
            per_batch_us.append((time.perf_counter() - t0) * 1e6)
        total_time = time.perf_counter() - t_meas_start
    else:
        # Baselines fit on the full edge set; we time a single fit and use it
        # as one big "batch" so percentile aggregation degenerates but the
        # mean/throughput still make sense.
        t0 = time.perf_counter()
        model.fit(adj=None, edges=edges[:end])  # baselines accept (adj,edges)
        # Fallback if signature requires adj (re-call from top level)
        per_batch_us.append((time.perf_counter() - t0) * 1e6)
        total_time = per_batch_us[-1] / 1e6
    return per_batch_us, W, measure_count, total_time


def stream_in_batches_neo4j(method: str, kind: str, model, driver, dataset: str,
                             batch_size: int, warmup_edges: int,
                             measurement_edges: int, seed: int,
                             database: str = None):
    """Pulls edges from Neo4j in batches; per-batch latency includes the
    Cypher round-trip + local processing time."""
    rng = np.random.default_rng(seed)
    per_batch_us = []
    consumed = 0
    # Warm-up
    while consumed < warmup_edges:
        lim = min(batch_size, warmup_edges - consumed)
        chunk, _q_dt = neo4j_query_batch(driver, dataset, consumed, lim, database)
        if chunk.shape[0] == 0:
            break
        if kind == "sag":
            model.process_stream_batch(chunk, passes=1, shuffle=False, rng=rng)
        consumed += chunk.shape[0]
    # Measurement
    measure_count = 0
    t_meas_start = time.perf_counter()
    target = warmup_edges + measurement_edges
    while consumed < target:
        lim = min(batch_size, target - consumed)
        t0 = time.perf_counter()
        chunk, _q_dt = neo4j_query_batch(driver, dataset, consumed, lim, database)
        if chunk.shape[0] == 0:
            break
        if kind == "sag":
            model.process_stream_batch(chunk, passes=1, shuffle=False, rng=rng)
        per_batch_us.append((time.perf_counter() - t0) * 1e6)
        consumed += chunk.shape[0]
        measure_count += chunk.shape[0]
    total_time = time.perf_counter() - t_meas_start
    return per_batch_us, warmup_edges, measure_count, total_time


# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    "dataset", "method", "seed", "mode",
    "neo4j_used", "memory_includes_neo4j_indexes",
    "warmup_edges", "measurement_edges", "batch_size", "workers",
    "device_path",
    "latency_mean_us", "latency_median_us", "latency_p95_us", "latency_p99_us",
    "latency_min_us", "latency_max_us",
    "throughput_edges_per_s", "memory_gb",
    "auroc", "auprc", "macro_f1",
    "num_edges_total", "num_nodes",
    "skipped", "error",
]
SUMMARY_COLUMNS = [
    "dataset", "method", "mode", "n_seeds",
    "latency_mean_us_mean", "latency_mean_us_std",
    "latency_p95_us_mean", "latency_p95_us_std",
    "latency_p99_us_mean", "latency_p99_us_std",
    "throughput_edges_per_s_mean", "throughput_edges_per_s_std",
    "memory_gb_mean", "memory_gb_std",
    "auroc_mean", "auroc_std",
]


def write_raw_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in RAW_COLUMNS})


def write_summary_csv(rows, path):
    grouped = {}
    for r in rows:
        if r.get("skipped"): continue
        grouped.setdefault((r["dataset"], r["method"], r["mode"]), []).append(r)
    out = []
    for (ds, mt, md), group in grouped.items():
        row = {"dataset": ds, "method": mt, "mode": md, "n_seeds": len(group)}
        for src, dst in [
            ("latency_mean_us", "latency_mean_us"),
            ("latency_p95_us", "latency_p95_us"),
            ("latency_p99_us", "latency_p99_us"),
            ("throughput_edges_per_s", "throughput_edges_per_s"),
            ("memory_gb", "memory_gb"),
            ("auroc", "auroc"),
        ]:
            arr = np.array([g[src] for g in group], dtype=np.float64)
            row[f"{dst}_mean"] = float(np.nanmean(arr)) if arr.size else float("nan")
            row[f"{dst}_std"] = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
        out.append(row)
    out.sort(key=lambda r: (r["dataset"], r["method"], r["mode"]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        for r in out:
            w.writerow(r)
    return out


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(DATASETS_DEFAULT))
    p.add_argument("--methods", default=",".join(METHODS_DEFAULT))
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS_DEFAULT))
    p.add_argument("--mode", choices=("local_only", "neo4j"),
                   default="local_only",
                   help="local_only: time on-disk pipeline (no Neo4j roundtrip). "
                        "neo4j: pull edges from a running Neo4j over bolt.")
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--warmup_edges", type=int, default=2000)
    p.add_argument("--measurement_edges", type=int, default=20000)
    p.add_argument("--workers", type=int, default=1,
                   help="reported in CSV; the current pipeline is single-threaded")
    p.add_argument("--uri", default=os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687"))
    p.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    p.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD", "neo4j"))
    p.add_argument("--database", default=os.environ.get("NEO4J_DATABASE", None))
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--tag", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _maybe_skip(path, overwrite, logger):
    if os.path.exists(path) and not overwrite:
        logger.warning(f"  refusing to overwrite {path} (pass --overwrite)")
        return True
    return False


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    run_dir = os.path.join(args.output_dir, "results", "revision",
                           f"neo4j_benchmark{tag}_{ts}")
    log_dir = os.path.join(args.output_dir, "logs",
                           f"neo4j_benchmark{tag}_{ts}")
    canon_dir = os.path.join(args.output_dir, "results", "revision")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(canon_dir, exist_ok=True)

    logger = setup_logger("SAG_NEO4J_REV", log_dir)
    logger.info(f"datasets={datasets}  methods={methods}  seeds={seeds}  "
                f"mode={args.mode}  batch={args.batch_size}  "
                f"warmup={args.warmup_edges}  meas={args.measurement_edges}")

    cfg_record = {
        "datasets": datasets, "methods": methods, "seeds": seeds,
        "mode": args.mode, "batch_size": args.batch_size,
        "warmup_edges": args.warmup_edges,
        "measurement_edges": args.measurement_edges,
        "workers": args.workers, "uri": args.uri,
        "timestamp": ts, "argv": sys.argv,
    }
    save_json(cfg_record, os.path.join(run_dir, "config.json"))

    Config.data_dir = "./data"
    Config.output_dir = args.output_dir

    driver = None
    if args.mode == "neo4j":
        try:
            driver = neo4j_connect(args.uri, args.user, args.password, args.database)
            logger.info(f"  connected to Neo4j @ {args.uri}")
        except Exception as e:
            logger.error(f"  cannot connect to Neo4j ({e}). "
                         "Re-run with --mode local_only or start the DB.")
            return

    raw_rows = []
    raw_run = os.path.join(run_dir, "neo4j_benchmark_raw.csv")
    sum_run = os.path.join(run_dir, "neo4j_benchmark_summary.csv")
    raw_canon = os.path.join(canon_dir, "neo4j_benchmark_raw.csv")
    sum_canon = os.path.join(canon_dir, "neo4j_benchmark_summary.csv")

    try:
        for ds in datasets:
            try:
                dc = get_dataset_config(ds)
            except ValueError as e:
                logger.error(f"unknown dataset {ds}: {e}")
                continue
            logger.info(f"\n========== dataset = {ds} ==========")
            loader = DataLoader(
                data_dir="./data", dataset_name=ds,
                clique_size=dc.get("clique_size", 15),
                num_cliques=dc.get("num_cliques", 10),
                seed=BASE_SEED,
            )
            try:
                loader.load()
            except Exception as e:
                logger.error(f"  load failed: {e}")
                continue
            s = loader.get_stats()
            logger.info(f"  stats: {s}")
            order = getattr(Config, "edge_order", "auto")
            if order == "auto":
                order = "shuffle" if dc.get("has_ground_truth", False) else "bfs"
            edges_local = loader.get_edge_stream(mode=order)

            for method in methods:
                if method != "SAG" and method not in BASELINES:
                    logger.info(f"  [SKIP] {method}: not registered")
                    raw_rows.append({
                        "dataset": ds, "method": method, "seed": -1,
                        "mode": args.mode, "neo4j_used": (args.mode == "neo4j"),
                        "memory_includes_neo4j_indexes": False,
                        "warmup_edges": args.warmup_edges,
                        "measurement_edges": args.measurement_edges,
                        "batch_size": args.batch_size, "workers": args.workers,
                        "device_path": str(getattr(Config, "device", "cpu")),
                        "latency_mean_us": float("nan"),
                        "latency_median_us": float("nan"),
                        "latency_p95_us": float("nan"),
                        "latency_p99_us": float("nan"),
                        "latency_min_us": float("nan"),
                        "latency_max_us": float("nan"),
                        "throughput_edges_per_s": float("nan"),
                        "memory_gb": float("nan"),
                        "auroc": float("nan"), "auprc": float("nan"),
                        "macro_f1": float("nan"),
                        "num_edges_total": int(s["num_edges"]),
                        "num_nodes": int(s["num_nodes"]),
                        "skipped": True, "error": "method not registered",
                    })
                    continue
                for sd in seeds:
                    logger.info(f"  --- {ds} | {method} | seed={sd} | mode={args.mode} ---")
                    set_seed(sd)
                    cfg = build_cfg(dc, seed=sd)
                    cfg.window_size = max(128, min(s["num_edges"] // 32, 4096))
                    try:
                        kind, model = build_method(
                            method, s["num_nodes"], cfg, loader.features, loader.adj, sd)
                        mt = MemoryTracker()
                        mt.reset()
                        if args.mode == "local_only":
                            if kind == "baseline":
                                # Time the baseline's fit on warmup+measurement window
                                end = min(args.warmup_edges + args.measurement_edges,
                                          edges_local.shape[0])
                                t0 = time.perf_counter()
                                model.fit(loader.adj, edges_local[:end], loader.labels)
                                elapsed = time.perf_counter() - t0
                                per_batch_us = [elapsed * 1e6]
                                W, MN = args.warmup_edges, end - args.warmup_edges
                                total_time = elapsed
                            else:
                                per_batch_us, W, MN, total_time = stream_in_batches_local(
                                    method, edges_local, kind, model,
                                    batch_size=args.batch_size,
                                    warmup_edges=args.warmup_edges,
                                    measurement_edges=args.measurement_edges,
                                    seed=sd, order=order,
                                    has_ground_truth=dc.get("has_ground_truth", False),
                                )
                        else:
                            per_batch_us, W, MN, total_time = stream_in_batches_neo4j(
                                method, kind, model, driver, ds,
                                batch_size=args.batch_size,
                                warmup_edges=args.warmup_edges,
                                measurement_edges=args.measurement_edges,
                                seed=sd, database=args.database,
                            )
                        mt.update()
                        peak_mem = mt.get_peak()["peak_total_mem_gb"]

                        if kind == "sag":
                            scores = model.get_scores()
                        else:
                            scores = model.score()
                        auroc = float(compute_auroc(loader.labels, scores))
                        auprc = float(compute_auprc(loader.labels, scores))
                        f1 = float(compute_macro_f1(loader.labels, scores))

                        lat = np.array(per_batch_us, dtype=np.float64) if per_batch_us else np.array([float("nan")])
                        thr = float(MN) / total_time if total_time > 0 else float("nan")

                        row = {
                            "dataset": ds, "method": method, "seed": sd,
                            "mode": args.mode,
                            "neo4j_used": (args.mode == "neo4j"),
                            "memory_includes_neo4j_indexes": False,
                            "warmup_edges": int(W),
                            "measurement_edges": int(MN),
                            "batch_size": int(args.batch_size),
                            "workers": int(args.workers),
                            "device_path": str(getattr(cfg, "device", "cpu")),
                            "latency_mean_us":   float(np.nanmean(lat)),
                            "latency_median_us": float(np.nanmedian(lat)),
                            "latency_p95_us":    float(np.nanquantile(lat, 0.95)),
                            "latency_p99_us":    float(np.nanquantile(lat, 0.99)),
                            "latency_min_us":    float(np.nanmin(lat)),
                            "latency_max_us":    float(np.nanmax(lat)),
                            "throughput_edges_per_s": float(thr),
                            "memory_gb": float(peak_mem),
                            "auroc": auroc, "auprc": auprc, "macro_f1": f1,
                            "num_edges_total": int(s["num_edges"]),
                            "num_nodes": int(s["num_nodes"]),
                            "skipped": False, "error": "",
                        }
                        raw_rows.append(row)
                        logger.info(
                            "    lat[us] mean=%.1f p50=%.1f p95=%.1f p99=%.1f  "
                            "thr=%.0f e/s  mem=%.2fGB  AUROC=%.4f" %
                            (row["latency_mean_us"], row["latency_median_us"],
                             row["latency_p95_us"], row["latency_p99_us"],
                             row["throughput_edges_per_s"], row["memory_gb"],
                             row["auroc"]))
                    except Exception as e:
                        tb = traceback.format_exc()
                        logger.error(f"    FAILED {type(e).__name__}: {e}\n{tb}")
                        raw_rows.append({
                            "dataset": ds, "method": method, "seed": sd,
                            "mode": args.mode,
                            "neo4j_used": (args.mode == "neo4j"),
                            "memory_includes_neo4j_indexes": False,
                            "warmup_edges": args.warmup_edges,
                            "measurement_edges": args.measurement_edges,
                            "batch_size": args.batch_size,
                            "workers": args.workers,
                            "device_path": str(getattr(Config, "device", "cpu")),
                            "latency_mean_us": float("nan"),
                            "latency_median_us": float("nan"),
                            "latency_p95_us": float("nan"),
                            "latency_p99_us": float("nan"),
                            "latency_min_us": float("nan"),
                            "latency_max_us": float("nan"),
                            "throughput_edges_per_s": float("nan"),
                            "memory_gb": float("nan"),
                            "auroc": float("nan"), "auprc": float("nan"),
                            "macro_f1": float("nan"),
                            "num_edges_total": int(s["num_edges"]),
                            "num_nodes": int(s["num_nodes"]),
                            "skipped": True, "error": f"{type(e).__name__}:{e}",
                        })
                    write_raw_csv(raw_rows, raw_run)
    finally:
        if driver is not None:
            driver.close()

    summary = write_summary_csv(raw_rows, sum_run)
    save_json({"raw": raw_rows, "summary": summary},
              os.path.join(run_dir, "neo4j_benchmark.json"))

    if not _maybe_skip(raw_canon, args.overwrite, logger):
        write_raw_csv(raw_rows, raw_canon)
    if not _maybe_skip(sum_canon, args.overwrite, logger):
        write_summary_csv(raw_rows, sum_canon)

    logger.info("=" * 70)
    logger.info(f"WROTE  raw     -> {raw_run}")
    logger.info(f"WROTE  summary -> {sum_run}")
    logger.info(f"CANON  raw     -> {raw_canon}")
    logger.info(f"CANON  summary -> {sum_canon}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
