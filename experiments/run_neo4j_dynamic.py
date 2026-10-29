"""SAG system-level streaming on REAL dynamic graphs, through Neo4j.

Production dynamic-graph anomaly detection lives in a graph database, not a
numpy array. This runner ingests a real timestamped dataset (DGraphFin /
Elliptic++ / AMLSim / temporal edge lists) into Neo4j -- storing the GENUINE
per-edge timestamp on each :INTERACT relationship -- then re-streams edges from
the DB in true time order (``ORDER BY r.t``) and runs SAG as a strict single
pass (``passes=1, shuffle=False, edge_times=...``). Dynamic / streaming / static
baselines run over the same DB-sourced stream. Accuracy is scored only on
evaluable nodes (``eval_mask``); we also record ingest time, DB query latency,
per-edge latency and peak memory.

This is the dynamic-dataset analogue of ``neo4j_benchmark.py`` (which handled
the static ground-truth datasets) and shows SAG's
efficiency be shown under the same DB-level graph-access conditions as the
baselines.

Usage:
    export NEO4J_URI=bolt://127.0.0.1:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=changeme
    python experiments/run_neo4j_dynamic.py --dataset AMLSim \
        --methods SAG,TGN,JODIE,MIDAS --edge_limit 200000
"""

import argparse
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from configs import Config, get_dataset_config
from src.baselines import BASELINES, DYNAMIC_BASELINES, STREAMING_BASELINES
from src.data import TemporalDataLoader
from src.evaluation import compute_all_metrics
from src.models import create_sag_model
from src.utils import MemoryTracker, save_json, set_seed, setup_logger
from experiments.neo4j_benchmark import _driver


def _ingest_temporal(driver, name, num_nodes, labels, eval_mask, edges, times,
                     logger, database=None, batch=5000):
    with (driver.session(database=database) if database else driver.session()) as session:
        session.run("MATCH (n) DETACH DELETE n")
        session.run("CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.id)")
        logger.info(f"Creating {num_nodes} nodes...")
        lab = labels.tolist(); ev = eval_mask.tolist()
        for s in range(0, num_nodes, batch):
            e = min(s + batch, num_nodes)
            chunk = [{"id": i, "label": int(lab[i]), "ev": bool(ev[i])} for i in range(s, e)]
            session.run(
                "UNWIND $batch AS row CREATE (n:Node {id: row.id, label: row.label, "
                "eval: row.ev, type: $type})", batch=chunk, type=name)
        logger.info(f"Creating {edges.shape[0]} timestamped edges...")
        for s in range(0, edges.shape[0], batch):
            e = min(s + batch, edges.shape[0])
            chunk = [{"u": int(edges[i, 0]), "v": int(edges[i, 1]), "t": float(times[i])}
                     for i in range(s, e)]
            session.run(
                "UNWIND $batch AS row MATCH (a:Node {id: row.u, type: $type}), "
                "(b:Node {id: row.v, type: $type}) "
                "CREATE (a)-[:INTERACT {t: row.t}]->(b)", batch=chunk, type=name)


def _query_temporal_stream(driver, name, database=None, limit=None):
    with (driver.session(database=database) if database else driver.session()) as session:
        q = ("MATCH (a:Node {type: $type})-[r:INTERACT]->(b:Node {type: $type}) "
             "RETURN a.id AS u, b.id AS v, r.t AS t ORDER BY r.t")
        if limit is not None:
            q += f" LIMIT {int(limit)}"
        res = session.run(q, type=name)
        rows = [(r["u"], r["v"], r["t"]) for r in res]
    if not rows:
        return np.zeros((0, 2), np.int64), np.zeros(0, np.float64)
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, :2].astype(np.int64), arr[:, 2].astype(np.float64)


def paradigm_of(name):
    if name == "SAG":
        return "sag"
    if name in DYNAMIC_BASELINES:
        return "dynamic"
    if name in STREAMING_BASELINES:
        return "streaming"
    return "static"


def _benchmark(name, loader, cfg, edges, times, seed, logger):
    set_seed(seed)
    mt = MemoryTracker(); mt.reset()
    mask = loader.eval_mask
    par = paradigm_of(name)
    if name == "SAG":
        model = create_sag_model(num_nodes=loader.stats["num_nodes"], config=cfg,
                                 features=loader.features, adj=loader.adj)
        t0 = time.perf_counter()
        model.process_stream_batch(edges, passes=1, shuffle=False,
                                   rng=np.random.default_rng(seed), edge_times=times)
        elapsed = time.perf_counter() - t0
        scores = model.get_scores()
    elif name in DYNAMIC_BASELINES:
        model = DYNAMIC_BASELINES[name](num_nodes=loader.stats["num_nodes"], config=cfg,
                                        features=loader.features, seed=seed)
        t0 = time.perf_counter()
        model.fit_temporal(edges, times, loader.labels)
        scores = model.score()
        elapsed = time.perf_counter() - t0
    elif name in BASELINES:
        model = BASELINES[name](num_nodes=loader.stats["num_nodes"], config=cfg,
                                features=loader.features, seed=seed)
        t0 = time.perf_counter()
        model.fit(loader.adj, edges, loader.labels)
        scores = model.score()
        elapsed = time.perf_counter() - t0
    else:
        raise ValueError(f"unknown method {name}")
    mt.update()
    m = compute_all_metrics(loader.labels[mask], scores[mask])
    return {
        "auroc": m["auroc"], "auprc": m["auprc"], "auprc_bal": m["auprc_bal"],
        "macro_f1": m["macro_f1"], "macro_f1_bal": m["macro_f1_bal"],
        "query_latency_ms": elapsed * 1000.0,
        "per_edge_ms": elapsed * 1000.0 / max(1, edges.shape[0]),
        "memory_gb": mt.get_peak()["peak_total_mem_gb"],
        "paradigm": par,
    }


def parse_args():
    p = argparse.ArgumentParser(description="SAG Neo4j streaming on real dynamic graphs")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--uri", type=str, default=os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687"))
    p.add_argument("--user", type=str, default=os.environ.get("NEO4J_USER", "neo4j"))
    p.add_argument("--password", type=str, default=os.environ.get("NEO4J_PASSWORD", "neo4j"))
    p.add_argument("--database", type=str, default=os.environ.get("NEO4J_DATABASE", None))
    p.add_argument("--methods", type=str, default="SAG,TGN,JODIE,MIDAS")
    p.add_argument("--edge_limit", type=int, default=200000)
    p.add_argument("--skip_ingest", action="store_true")
    p.add_argument("--no_inject", action="store_true")
    p.add_argument("--n_runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    Config.data_dir = args.data_dir
    Config.output_dir = args.output_dir
    Config.seed = args.seed
    try:
        dc = get_dataset_config(args.dataset)
    except ValueError:
        dc = {}
    for k in ("alpha", "gamma", "rho", "kernel_dim"):
        if k in dc:
            setattr(Config, k, dc[k])
    Config.num_epochs = dc.get("num_epochs", 1)
    Config.has_ground_truth = True
    Config.fusion_mode = dc.get("fusion_mode", "auto")
    Config.edge_order = "temporal"
    Config.attr_mode = dc.get("attr_mode", "auto")
    Config.use_continuous_dev = bool(dc.get("use_continuous_dev", False))
    Config.dyn_epochs = int(dc.get("dyn_epochs", 3))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp = f"neo4j_dyn_{args.dataset}_{ts}"
    log_dir = os.path.join(args.output_dir, "logs", exp)
    result_dir = os.path.join(args.output_dir, "results", exp)
    os.makedirs(log_dir, exist_ok=True); os.makedirs(result_dir, exist_ok=True)
    logger = setup_logger("SAG_NEO4J_DYN", log_dir)

    set_seed(Config.seed)
    loader = TemporalDataLoader(data_dir=Config.data_dir, dataset_name=args.dataset,
                                seed=Config.seed, inject_anomalies=not args.no_inject)
    loader.load()
    logger.info(f"Loaded {args.dataset}: {loader.get_stats()}")

    edges_all, times_all = loader.get_temporal_stream()
    if args.edge_limit and edges_all.shape[0] > args.edge_limit:
        edges_all = edges_all[: args.edge_limit]
        times_all = times_all[: args.edge_limit]
        logger.info(f"Capped to earliest {args.edge_limit} edges for ingest")

    logger.info(f"Connecting to Neo4j at {args.uri}")
    driver = _driver(args.uri, args.user, args.password)
    try:
        if not args.skip_ingest:
            t0 = time.perf_counter()
            _ingest_temporal(driver, args.dataset, loader.stats["num_nodes"],
                             loader.labels, loader.eval_mask, edges_all, times_all,
                             logger, database=args.database)
            logger.info(f"Ingest finished in {time.perf_counter()-t0:.1f}s")

        t0 = time.perf_counter()
        edges, times = _query_temporal_stream(driver, args.dataset,
                                              database=args.database, limit=args.edge_limit)
        logger.info(f"Streamed {edges.shape[0]} edges from Neo4j in "
                    f"{(time.perf_counter()-t0)*1000:.1f}ms")

        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        results = {}
        for name in methods:
            logger.info(f"\n=== {name} ({paradigm_of(name)}) ===")
            if name != "SAG" and name not in DYNAMIC_BASELINES and name not in BASELINES:
                logger.warning(f"Unknown method {name}; skipping"); continue
            per_run = []
            try:
                for i in range(args.n_runs):
                    r = _benchmark(name, loader, Config, edges, times,
                                   Config.seed + i, logger)
                    per_run.append(r)
                    logger.info(f"  seed={Config.seed+i}: AUROC={r['auroc']:.4f} "
                                f"latency={r['query_latency_ms']:.1f}ms mem={r['memory_gb']:.2f}GB")
                agg = {}
                for k in ["auroc", "auprc", "auprc_bal", "macro_f1", "macro_f1_bal",
                          "query_latency_ms", "per_edge_ms", "memory_gb"]:
                    v = np.asarray([r[k] for r in per_run])
                    agg[k] = {"mean": float(v.mean()),
                              "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
                agg["paradigm"] = paradigm_of(name)
                results[name] = {"per_run": per_run, "aggregated": agg}
                logger.info(f"[{name}] AUROC={agg['auroc']['mean']*100:.1f}  "
                            f"latency={agg['query_latency_ms']['mean']:.1f}ms  "
                            f"mem={agg['memory_gb']['mean']:.2f}GB")
            except Exception as e:
                logger.error(f"[{name}] FAILED: {type(e).__name__}: {e}")
                results[name] = {"error": f"{type(e).__name__}: {e}",
                                 "paradigm": paradigm_of(name)}
    finally:
        driver.close()

    save_json({"dataset": args.dataset, "stats": loader.get_stats(),
               "edge_limit": args.edge_limit, "n_runs": args.n_runs,
               "results": results, "timestamp": ts},
              os.path.join(result_dir, "neo4j_dynamic_results.json"))
    logger.info(f"Saved to {result_dir}")


if __name__ == "__main__":
    main()
