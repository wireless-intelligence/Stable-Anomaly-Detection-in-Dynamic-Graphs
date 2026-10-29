"""SAG hyperparameter sensitivity in the Neo4j-streamed setting.

Pipeline per dataset:
  1. Ingest the graph into Neo4j (or reuse via --skip_ingest).
  2. For each (param, value) in PARAM_SWEEPS, re-stream edges from Neo4j
     and run SAG with that hyperparameter overridden. Record AUROC,
     AUPRC_bal, F1_bal, query latency, and peak memory.
  3. Persist a per-dataset JSON in outputs/results/neo4j_sens_<ds>_<ts>/.

Why this matters for the paper:
  Production deployments live in a graph DB, not a numpy array. Showing that
  SAG's accuracy is stable to hyperparameters under DB-level streaming
  (re-fetched edges, real query latency) is a stronger robustness claim than
  the in-memory sensitivity figure.
"""

import argparse
import copy
import json
import os
import sys
import time
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from configs import Config, get_dataset_config
from src.data import DataLoader
from src.evaluation import compute_all_metrics
from src.models import create_sag_model
from src.utils import MemoryTracker, save_json, set_seed, setup_logger

from experiments.neo4j_benchmark import _driver, _ingest_graph, _query_stream


PARAM_SWEEPS = {
    "alpha":      [0.5, 0.7, 1.0, 1.3, 1.5, 2.0],
    "gamma":      [0.80, 0.85, 0.90, 0.95, 0.99],
    "rho":        [1.2, 1.5, 2.0, 3.0, 5.0],
    "kernel_dim": [1, 2, 4, 8, 16],
}


class CfgShim:
    pass


def build_cfg(dc, sweep_param=None, fusion_mode_override=None):
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
    if sweep_param == "kernel_dim":
        cfg.kernel_lambdas = None
    if fusion_mode_override:
        cfg.fusion_mode = fusion_mode_override
        # When state-evolution components are isolated, skip the heavy
        # attribute-side machinery — its score is bypassed by h_only anyway,
        # so retraining the GCN-AE per call would just waste minutes.
        if fusion_mode_override == "h_only":
            cfg.use_trained_ae = False
            cfg.use_pca_mahal = False
            cfg.use_attr_boost = False
            cfg.attr_mode = "auto"
    return cfg


def run_sag_on_edges(loader, cfg, edges, seed=42):
    set_seed(seed)
    mt = MemoryTracker()
    mt.reset()
    model = create_sag_model(num_nodes=loader.stats["num_nodes"], config=cfg,
                             features=loader.features, adj=loader.adj)
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    order = getattr(cfg, "edge_order", "auto")
    if order == "auto":
        order = "shuffle" if bool(getattr(cfg, "has_ground_truth", False)) else "bfs"
    do_shuffle = (order == "shuffle")
    passes = (2 if edges.shape[0] > 50000 else 3) if do_shuffle else 1
    model.process_stream_batch(edges, passes=passes, shuffle=do_shuffle, rng=rng)
    elapsed = time.perf_counter() - t0
    mt.update()
    scores = model.get_scores()
    metrics = compute_all_metrics(loader.labels, scores)
    return {
        "auroc": float(metrics["auroc"]),
        "auprc_bal": float(metrics["auprc_bal"]),
        "macro_f1_bal": float(metrics["macro_f1_bal"]),
        "query_latency_ms": float(elapsed * 1000.0),
        "memory_gb": float(mt.get_peak()["peak_total_mem_gb"]),
    }


def parse_args():
    p = argparse.ArgumentParser(description="SAG hyperparameter sensitivity via Neo4j")
    p.add_argument("--datasets", default="Enron,BlogCatalog,Amazon")
    p.add_argument("--params", default="alpha,gamma,rho,kernel_dim")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--uri", default=os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687"))
    p.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    p.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD", "neo4j"))
    p.add_argument("--database", default=os.environ.get("NEO4J_DATABASE", None))
    p.add_argument("--edge_limit", type=int, default=200_000)
    p.add_argument("--skip_ingest", action="store_true",
                   help="reuse the graph already in Neo4j (must match --datasets[0])")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fusion_mode", default=None,
                   help="override fusion_mode (e.g. h_only)")
    return p.parse_args()


def main():
    args = parse_args()
    Config.data_dir = args.data_dir
    Config.output_dir = args.output_dir
    Config.seed = args.seed

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    params = [p.strip() for p in args.params.split(",") if p.strip()]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output_dir, "logs", f"neo4j_sens_{ts}")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger("SAG_NEO4J_SENS", log_dir)
    logger.info(f"Datasets: {datasets} | Params: {params} | "
                f"fusion_mode_override={args.fusion_mode} | "
                f"edge_limit={args.edge_limit}")

    driver = _driver(args.uri, args.user, args.password)
    try:
        for ds in datasets:
            dc = get_dataset_config(ds)
            for k in ("alpha", "gamma", "rho"):
                if k in dc:
                    setattr(Config, k, dc[k])
            Config.num_epochs = dc.get("num_epochs", 20)
            Config.has_ground_truth = bool(dc.get("has_ground_truth", False))
            Config.fusion_mode = dc.get("fusion_mode", "auto")
            Config.edge_order = dc.get("edge_order", "auto")
            Config.attr_mode = dc.get("attr_mode", "auto")
            Config.use_trained_ae = bool(dc.get("use_trained_ae", False))
            Config.trained_ae_cfg = dc.get("trained_ae_cfg", {})
            Config.use_attr_boost = bool(dc.get("use_attr_boost", False))
            Config.use_pca_mahal = bool(dc.get("use_pca_mahal", False))

            loader = DataLoader(data_dir=args.data_dir, dataset_name=ds,
                                clique_size=dc.get("clique_size", 15),
                                num_cliques=dc.get("num_cliques", 10),
                                seed=args.seed)
            loader.load()
            logger.info(f"\n=== {ds}: {loader.get_stats()} ===")

            # Ingest into Neo4j (replaces existing graph).
            if not args.skip_ingest:
                logger.info(f"  ingesting {ds} ...")
                t0 = time.perf_counter()
                full_edges = loader.get_edge_stream(shuffle=True)
                if args.edge_limit and full_edges.shape[0] > args.edge_limit:
                    full_edges = full_edges[: args.edge_limit]
                logger.info(f"  using {full_edges.shape[0]} edges")

                class _TmpLoader:
                    dataset_name = ds
                    labels = loader.labels
                    def get_stats(self):
                        return loader.get_stats()
                    def get_edge_stream(self, shuffle=True):
                        return full_edges
                _ingest_graph(driver, _TmpLoader(), logger, database=args.database)
                logger.info(f"  ingest done in {time.perf_counter()-t0:.1f}s")

            # Re-fetch edges from Neo4j once per dataset; reuse for the sweep.
            t0 = time.perf_counter()
            edges = _query_stream(driver, ds, database=args.database,
                                  limit=args.edge_limit)
            read_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(f"  fetched {edges.shape[0]} edges from Neo4j in {read_ms:.1f}ms")

            results = {"dataset": ds, "stats": loader.get_stats(),
                       "edges_used": int(edges.shape[0]),
                       "neo4j_read_ms": read_ms,
                       "params": {}}

            for param in params:
                if param not in PARAM_SWEEPS:
                    logger.warning(f"  unknown param {param}; skipping")
                    continue
                values = PARAM_SWEEPS[param]
                default_val = dc.get(param)
                if default_val is None and hasattr(Config, param):
                    default_val = getattr(Config, param)
                logger.info(f"\n  >>> sweep {param} (default={default_val}) values={values}")
                results["params"][param] = {"default": default_val, "values": []}
                for v in values:
                    cfg = build_cfg(dc, sweep_param=param,
                                    fusion_mode_override=args.fusion_mode)
                    setattr(cfg, param, v)
                    t0 = time.perf_counter()
                    try:
                        m = run_sag_on_edges(loader, cfg, edges, seed=args.seed)
                        m["value"] = v
                        m["elapsed_s"] = time.perf_counter() - t0
                        results["params"][param]["values"].append(m)
                        logger.info(
                            f"    {param}={v}: AUROC={m['auroc']*100:.2f} "
                            f"AUPRC_bal={m['auprc_bal']*100:.2f} "
                            f"F1_bal={m['macro_f1_bal']*100:.2f} "
                            f"latency={m['query_latency_ms']:.1f}ms "
                            f"mem={m['memory_gb']:.2f}GB ({m['elapsed_s']:.1f}s)"
                        )
                    except Exception as e:
                        tb = traceback.format_exc()
                        results["params"][param]["values"].append(
                            {"value": v, "error": str(e), "traceback": tb[:500]})
                        logger.error(f"    {param}={v}: FAILED {type(e).__name__}: {e}")

            exp_dir = os.path.join(args.output_dir, "results",
                                   f"neo4j_sens_{ds}_{ts}")
            os.makedirs(exp_dir, exist_ok=True)
            save_json(results, os.path.join(exp_dir, "neo4j_sensitivity.json"))
            logger.info(f"  saved to {exp_dir}")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
