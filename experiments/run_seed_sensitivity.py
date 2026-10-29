"""Seed-count sensitivity experiment.

Runs SAG with N independent seeds (default 150), then post-processes to
compute mean / std at cumulative cutoffs [5, 10, 20, 40, 60, 100, 150].
This shows how stable the reported numbers are vs the chosen n_runs.

Output:
  outputs/results/seed_sens_<ts>/
      per_dataset_<dataset>.json    raw 150-seed metrics
      seed_sens_summary.json        cumulative mean/std at each cutoff
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from configs import Config, get_dataset_config
from src.data import DataLoader
from src.evaluation import compute_all_metrics
from src.models import create_sag_model
from src.utils import save_json, set_seed, setup_logger


DEFAULT_DATASETS = ["Cora", "Citeseer", "BlogCatalog", "ACM", "Pubmed"]
DEFAULT_CUTOFFS = [5, 10, 20, 40, 60, 100, 150]


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
    cfg.iforest_blend_h_weight = float(dc.get("iforest_blend_h_weight", 0.3))
    return cfg


def run_one_dataset(dataset, max_seeds, exp_dir, logger):
    dc = get_dataset_config(dataset)
    cfg = _setup_cfg(dc)
    loader = DataLoader(data_dir="./data", dataset_name=dataset,
                        clique_size=dc.get("clique_size", 15),
                        num_cliques=dc.get("num_cliques", 5), seed=42)
    loader.load()
    n = loader.stats["num_nodes"]
    logger.info(f"=== {dataset} (n={n}, anom={loader.stats['num_anomalies']}) ===")

    order = getattr(cfg, "edge_order", "auto")
    if order == "auto":
        order = "shuffle" if bool(getattr(cfg, "has_ground_truth", False)) else "bfs"

    seeds = list(range(42, 42 + max_seeds))
    rows = []
    t0 = time.time()
    for i, sd in enumerate(seeds):
        set_seed(sd)
        model = create_sag_model(num_nodes=n, config=cfg,
                                 features=loader.features, adj=loader.adj)
        edges = loader.get_edge_stream(mode=order)
        do_shuffle = (order == "shuffle")
        passes = (2 if loader.stats["num_edges"] > 50000 else 3) if do_shuffle else 1
        model.process_stream_batch(edges, passes=passes, shuffle=do_shuffle,
                                   rng=np.random.default_rng(sd))
        scores = model.get_scores()
        mt = compute_all_metrics(loader.labels, scores)
        rows.append({
            "seed": sd,
            "auroc": float(mt["auroc"]),
            "auprc": float(mt["auprc"]),
            "auprc_bal": float(mt["auprc_bal"]),
            "macro_f1": float(mt["macro_f1"]),
            "macro_f1_bal": float(mt["macro_f1_bal"]),
        })
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(seeds) - i - 1)
            last_auc = np.mean([r["auroc"] for r in rows[-25:]]) * 100
            logger.info(f"  {i+1}/{len(seeds)} seeds done  "
                        f"(last-25 AUROC mean={last_auc:.2f})  "
                        f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")
    save_json({"dataset": dataset, "n_seeds": len(rows), "rows": rows,
               "stats": loader.get_stats()},
              os.path.join(exp_dir, f"per_dataset_{dataset}.json"))
    return rows


def cumulative_summary(per_ds_rows, cutoffs):
    summary = {}
    for cutoff in cutoffs:
        sub = per_ds_rows[:cutoff]
        if not sub: continue
        entry = {}
        for k in ("auroc", "auprc_bal", "macro_f1_bal"):
            vals = [r[k] for r in sub]
            entry[k] = {
                "mean": float(np.mean(vals) * 100),
                "std": float(np.std(vals, ddof=1) * 100) if len(vals) > 1 else 0.0,
            }
        summary[str(cutoff)] = entry
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    p.add_argument("--max_seeds", type=int, default=150)
    p.add_argument("--cutoffs", default=",".join(map(str, DEFAULT_CUTOFFS)))
    p.add_argument("--output_dir", default="./outputs")
    return p.parse_args()


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    cutoffs = [int(c.strip()) for c in args.cutoffs.split(",") if c.strip()]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.output_dir, "results", f"seed_sens_{ts}")
    log_dir = os.path.join(args.output_dir, "logs", f"seed_sens_{ts}")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger("SEED_SENS", log_dir)
    logger.info(f"Datasets: {datasets} | max_seeds: {args.max_seeds} | cutoffs: {cutoffs}")

    Config.data_dir = "./data"
    Config.output_dir = args.output_dir

    summary = {}
    for ds in datasets:
        try:
            rows = run_one_dataset(ds, args.max_seeds, exp_dir, logger)
            summary[ds] = cumulative_summary(rows, cutoffs)
        except Exception as e:
            logger.error(f"  {ds} FAILED: {type(e).__name__}: {e}")
            summary[ds] = {"error": str(e)}

    save_json({"datasets": summary, "cutoffs": cutoffs,
               "max_seeds": args.max_seeds},
              os.path.join(exp_dir, "seed_sens_summary.json"))

    logger.info("\n=== Cumulative summary (AUROC mean ± std) ===")
    header = f"{'Dataset':<14}  " + "  ".join(f"n={c:<3}" for c in cutoffs)
    logger.info(header)
    for ds in datasets:
        if "error" in summary.get(ds, {}):
            continue
        cells = []
        for c in cutoffs:
            entry = summary[ds].get(str(c))
            if entry is None: cells.append("    -    "); continue
            m, s = entry["auroc"]["mean"], entry["auroc"]["std"]
            cells.append(f"{m:5.2f}±{s:.2f}")
        logger.info(f"{ds:<14}  " + "  ".join(cells))
    logger.info(f"Saved to {exp_dir}")


if __name__ == "__main__":
    main()
