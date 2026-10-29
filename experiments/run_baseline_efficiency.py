"""Baseline efficiency benchmark.

For each (dataset, method): measure
  - wall-clock time (fit + score, mean over n_runs)
  - peak memory (psutil RSS delta)
  - score quality (AUROC, AUPRC_bal) for sanity

Output:
    outputs/results/efficiency_<ts>/
        per_dataset_<dataset>.json
        efficiency_summary.json
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from configs import Config, get_dataset_config
from src.baselines import BASELINES
from src.data import DataLoader
from src.evaluation import compute_all_metrics
from src.models import create_sag_model
from src.utils import MemoryTracker, save_json, set_seed, setup_logger


DEFAULT_DATASETS = ["Cora", "Citeseer", "Pubmed", "BlogCatalog", "ACM",
                    "Flickr", "Amazon", "Enron"]
DEFAULT_METHODS  = ["DOMINANT", "CONAD", "CoLA", "AnomalyDAE", "GraphSAGE",
                    "DySAT", "MIDAS", "NetWalk",
                    "ConsisGAD", "GraphGPT", "ANEMONE", "ComGA", "PMP",
                    "GAAP", "Graphormer"]


def _setup_cfg(dc):
    cfg = Config
    for k in ("alpha", "gamma", "rho"):
        if k in dc:
            setattr(cfg, k, dc[k])
    cfg.num_epochs = dc.get("num_epochs", 20)
    cfg.has_ground_truth = bool(dc.get("has_ground_truth", False))
    cfg.fusion_mode = dc.get("fusion_mode", "auto")
    cfg.edge_order = dc.get("edge_order", "auto")
    cfg.attr_mode = dc.get("attr_mode", "auto")
    cfg.use_trained_ae = bool(dc.get("use_trained_ae", False))
    cfg.trained_ae_cfg = dc.get("trained_ae_cfg", {})
    cfg.use_attr_boost = bool(dc.get("use_attr_boost", False))
    cfg.use_pca_mahal = bool(dc.get("use_pca_mahal", False))
    return cfg


def _measure_one(method, loader, cfg, n_runs, max_seconds, logger):
    n = loader.stats["num_nodes"]
    edges = loader.get_edge_stream(mode="bfs")
    times, mems, aucs = [], [], []
    for run in range(n_runs):
        gc.collect()
        mt = MemoryTracker(); mt.reset()
        set_seed(42 + run)
        try:
            if method == "SAG":
                model = create_sag_model(num_nodes=n, config=cfg,
                                         features=loader.features,
                                         adj=loader.adj)
                t0 = time.time()
                model.process_stream_batch(edges, passes=1, shuffle=False,
                                           rng=np.random.default_rng(42 + run))
                scores = model.get_scores()
                elapsed = time.time() - t0
            else:
                cls = BASELINES[method]
                model = cls(num_nodes=n, config=Config,
                            features=loader.features, seed=42 + run)
                t0 = time.time()
                model.fit(loader.adj, edges, loader.labels)
                scores = model.score()
                elapsed = time.time() - t0
            mt.update()
            times.append(elapsed)
            mems.append(mt.get_peak()["peak_total_mem_gb"])
            aucs.append(float(compute_all_metrics(loader.labels, scores)["auroc"]))
            if max_seconds and elapsed > max_seconds:
                logger.warning(f"  {method}: {elapsed:.1f}s > limit {max_seconds}s — stopping early")
                break
        except Exception as e:
            logger.error(f"  {method} run{run} FAILED: {type(e).__name__}: {e}")
            return {"error": str(e)}
    if not times:
        return {"error": "no successful runs"}
    return {
        "n_runs":       len(times),
        "time_s_mean":  float(np.mean(times)),
        "time_s_std":   float(np.std(times, ddof=1) if len(times) > 1 else 0.0),
        "time_per_edge_us": float(np.mean(times) / max(1, edges.shape[0]) * 1e6),
        "memory_gb_mean": float(np.mean(mems)),
        "auroc_mean":   float(np.mean(aucs)),
    }


def run_efficiency(datasets, methods, exp_dir, logger, n_runs=2,
                   max_seconds=600):
    summary = {"datasets": {}}
    for ds in datasets:
        logger.info(f"=== {ds} ===")
        dc = get_dataset_config(ds)
        cfg = _setup_cfg(dc)
        loader = DataLoader(data_dir="./data", dataset_name=ds,
                            clique_size=dc.get("clique_size", 15),
                            num_cliques=dc.get("num_cliques", 5), seed=42)
        loader.load()
        per_ds = {}
        for m in ["SAG"] + methods:
            logger.info(f"  measuring {m}...")
            res = _measure_one(m, loader, cfg, n_runs, max_seconds, logger)
            per_ds[m] = res
            if "error" not in res:
                logger.info(f"    {m:12} {res['time_s_mean']:.2f}s  "
                            f"{res['memory_gb_mean']:.3f}GB  "
                            f"AUROC={res['auroc_mean']*100:.1f}")
        summary["datasets"][ds] = per_ds
        save_json(per_ds, os.path.join(exp_dir, f"per_dataset_{ds}.json"))
    save_json(summary, os.path.join(exp_dir, "efficiency_summary.json"))
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    p.add_argument("--n_runs", type=int, default=2)
    p.add_argument("--max_seconds", type=int, default=600)
    p.add_argument("--output_dir", default="./outputs")
    return p.parse_args()


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.output_dir, "results", f"efficiency_{ts}")
    log_dir = os.path.join(args.output_dir, "logs", f"efficiency_{ts}")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger("EFFICIENCY", log_dir)
    logger.info(f"Datasets: {datasets} | Methods (+SAG): {methods}")

    Config.data_dir = "./data"
    Config.output_dir = args.output_dir
    run_efficiency(datasets, methods, exp_dir, logger,
                   n_runs=args.n_runs, max_seconds=args.max_seconds)
    logger.info(f"Saved to {exp_dir}")


if __name__ == "__main__":
    main()
