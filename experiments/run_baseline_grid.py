"""Grid search for baseline hyperparameters.

Each baseline has a small parameter grid (3-12 configs). For each (dataset, method)
we sweep the grid and keep the config with highest AUROC.

Output:
  outputs/results/grid_<ts>/<dataset>__<method>.json   per-cell results
  outputs/results/grid_<ts>/best_params.json           summary of best per cell
"""

import argparse
import json
import os
import sys
import time
import itertools
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from configs import Config, get_dataset_config
from src.baselines import BASELINES
from src.data import DataLoader
from src.evaluation import compute_all_metrics
from src.utils import set_seed, save_json, setup_logger


# ---- Parameter grids ----
GRIDS = {
    # All grids use OFFICIAL DEFAULT as the FIRST entry, then 2-3 perturbations.
    "DOMINANT": [
        {"epoch": 20, "hid_dim": 64, "num_layers": 2},   # pygod default
        {"epoch": 50, "hid_dim": 64, "num_layers": 2},
        {"epoch": 100, "hid_dim": 128, "num_layers": 3},
    ],
    "AnomalyDAE": [
        {"epoch": 20, "hid_dim": 64},                      # pygod default
        {"epoch": 50, "hid_dim": 64},
        {"epoch": 100, "hid_dim": 128},
    ],
    "CoLA": [
        {"epoch": 20, "hid_dim": 64},                      # pygod default
        {"epoch": 50, "hid_dim": 64},
        {"epoch": 100, "hid_dim": 128},
    ],
    "CONAD": [
        {"epoch": 20, "hid_dim": 64, "num_layers": 2},   # pygod default
        {"epoch": 50, "hid_dim": 64, "num_layers": 2},
        {"epoch": 100, "hid_dim": 128, "num_layers": 3},
    ],
    "GraphSAGE": [
        {"hidden_dim": 64, "num_layers": 2},               # default
        {"hidden_dim": 128, "num_layers": 2},
        {"hidden_dim": 64, "num_layers": 3},
    ],
    "HGT": [
        {"epoch": 20, "hid_dim": 64, "num_layers": 2},   # pygod-ish
        {"epoch": 50, "hid_dim": 64, "num_layers": 2},
        {"epoch": 100, "hid_dim": 128, "num_layers": 3},
    ],
    "DySAT": [
        {"hidden_dim": 64, "num_snapshots": 8, "num_heads": 2},   # default
        {"hidden_dim": 128, "num_snapshots": 8, "num_heads": 4},
        {"hidden_dim": 64, "num_snapshots": 16, "num_heads": 2},
    ],
    "MIDAS": [],
    "NetWalk": [
        {},
    ],
    "SedanSpot": [
        {"sample_size": 100},
        {"sample_size": 200},
    ],
    "GUIDE": [
        {"epoch": 20, "hid_dim": 64},
        {"epoch": 50, "hid_dim": 128},
    ],
    "ANOMALOUS": [
        {"epoch": 20},
        {"epoch": 50},
    ],
    "AdONE": [
        {"epoch": 20, "hid_dim": 64},
        {"epoch": 50, "hid_dim": 128},
    ],
    "ConsisGAD": [
        {"hidden_dim": 64,  "num_layers": 2, "lr": 5e-3, "weight_decay": 5e-4, "num_epochs": 200, "alpha_consis": 0.3},
        {"hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "weight_decay": 5e-4, "num_epochs": 300, "alpha_consis": 0.3},
        {"hidden_dim": 128, "num_layers": 3, "lr": 5e-3, "weight_decay": 0,    "num_epochs": 200, "alpha_consis": 0.5},
    ],
    "GraphGPT": [
        {"hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "weight_decay": 0,    "num_epochs": 200, "beta_attn": 0.5},
        {"hidden_dim": 256, "num_layers": 2, "lr": 5e-4, "weight_decay": 0,    "num_epochs": 300, "beta_attn": 0.4},
        {"hidden_dim": 128, "num_layers": 3, "lr": 1e-3, "weight_decay": 5e-4, "num_epochs": 200, "beta_attn": 0.7},
    ],
    "ANEMONE": [
        {"hidden_dim": 64,  "num_layers": 2, "lr": 1e-3, "weight_decay": 0,    "num_epochs": 200, "gamma_global": 0.5},
        {"hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "weight_decay": 0,    "num_epochs": 300, "gamma_global": 0.3},
        {"hidden_dim": 64,  "num_layers": 3, "lr": 5e-4, "weight_decay": 5e-4, "num_epochs": 200, "gamma_global": 0.7},
    ],
    "ComGA": [
        {"hidden_dim": 64,  "num_layers": 2, "lr": 5e-3, "weight_decay": 5e-4, "num_epochs": 200, "num_communities": 8,  "alpha_comm": 0.2},
        {"hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "weight_decay": 5e-4, "num_epochs": 300, "num_communities": 16, "alpha_comm": 0.3},
    ],
    "PMP": [
        {"hidden_dim": 64,  "lr": 5e-3, "weight_decay": 5e-4, "num_epochs": 200, "beta_disagree": 0.5},
        {"hidden_dim": 128, "lr": 1e-3, "weight_decay": 5e-4, "num_epochs": 300, "beta_disagree": 0.7},
    ],
    "GAAP": [
        {"hidden_dim": 64,  "num_protos": 4, "lr": 5e-3, "weight_decay": 5e-4, "num_epochs": 200, "alpha_attr": 0.5},
        {"hidden_dim": 128, "num_protos": 8, "lr": 1e-3, "weight_decay": 5e-4, "num_epochs": 300, "alpha_attr": 0.4},
    ],
    "Graphormer": [
        {"hidden_dim": 64,  "num_heads": 4, "spd_clip": 4, "lr": 5e-4, "weight_decay": 5e-4, "num_epochs": 100, "mask_ratio": 0.15},
        {"hidden_dim": 128, "num_heads": 8, "spd_clip": 4, "lr": 1e-4, "weight_decay": 5e-4, "num_epochs": 150, "mask_ratio": 0.20},
    ],
    "TAPE": [
        {"hidden_dim": 64,  "lr": 5e-3, "weight_decay": 5e-4, "num_epochs": 200, "alpha_text": 0.4},
        {"hidden_dim": 128, "lr": 1e-3, "weight_decay": 5e-4, "num_epochs": 300, "alpha_text": 0.5},
    ],
    "HiGPT": [
        {"hidden_dim": 64,  "lr": 5e-3, "weight_decay": 5e-4, "num_epochs": 200},
        {"hidden_dim": 128, "lr": 1e-3, "weight_decay": 5e-4, "num_epochs": 300},
    ],
    "InstructGLM": [
        {"hidden_dim": 64,  "lr": 5e-3, "weight_decay": 5e-4, "num_epochs": 200},
        {"hidden_dim": 128, "lr": 1e-3, "weight_decay": 5e-4, "num_epochs": 300},
    ],
}


def setup_cfg(dc):
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
    cfg.use_continuous_dev = False
    cfg.use_real_time_decay = False
    return cfg


def run_one_config(method, params, loader, dc, max_seconds=None):
    """Run one (method, params) on the dataset; return metrics dict or {error: ...}."""
    cls = BASELINES.get(method)
    if cls is None:
        return {"error": f"unknown method {method}"}
    set_seed(42)
    edges = loader.get_edge_stream(mode="bfs")
    if hasattr(edges, "shape"):
        edges_arr = np.asarray(edges, dtype=np.int64)
    else:
        edges_arr = np.asarray(list(edges), dtype=np.int64)
    try:
        # most baselines accept (num_nodes, config, features, seed=, **params)
        kwargs = dict(params)
        kwargs.setdefault("seed", 42)
        try:
            m = cls(num_nodes=loader.stats["num_nodes"], config=Config,
                    features=loader.features, **kwargs)
        except TypeError:
            # Try without seed (some baselines lack it)
            kwargs.pop("seed", None)
            m = cls(num_nodes=loader.stats["num_nodes"], config=Config,
                    features=loader.features, **kwargs)
        t0 = time.time()
        m.fit(loader.adj, edges_arr, loader.labels)
        scores = m.score()
        elapsed = time.time() - t0
        metrics = compute_all_metrics(loader.labels, scores)
        return {
            "auroc": float(metrics["auroc"]),
            "auprc_bal": float(metrics["auprc_bal"]),
            "macro_f1_bal": float(metrics["macro_f1_bal"]),
            "elapsed_s": float(elapsed),
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()[:500]}


def grid_search(dataset, methods, exp_dir, logger, max_seconds_per_run=None):
    dc = get_dataset_config(dataset)
    loader = DataLoader(data_dir="./data", dataset_name=dataset,
                        clique_size=dc.get("clique_size", 15),
                        num_cliques=dc.get("num_cliques", 10), seed=42)
    loader.load()
    setup_cfg(dc)
    logger.info(f"=== {dataset} stats: {loader.get_stats()} ===")

    all_best = {}
    for method in methods:
        if method not in GRIDS:
            logger.warning(f"  skip {method}: no grid defined")
            continue
        grid = GRIDS[method]
        cell = {"method": method, "results": []}
        best = None
        for i, params in enumerate(grid):
            t0 = time.time()
            r = run_one_config(method, params, loader, dc,
                               max_seconds=max_seconds_per_run)
            cell["results"].append({"params": params, **r})
            if "error" in r:
                logger.error(f"  {method} cfg{i+1}/{len(grid)} FAILED: {r['error'][:100]}")
                continue
            logger.info(f"  {method} cfg{i+1}/{len(grid)}: AUROC={r['auroc']*100:.2f} "
                        f"AUPRC_bal={r['auprc_bal']*100:.2f} ({r['elapsed_s']:.1f}s) "
                        f"params={params}")
            if best is None or r["auroc"] > best["auroc"]:
                best = {"params": params, **r}
        if best is not None:
            cell["best"] = best
            all_best[method] = best
            logger.info(f"  >>> {method} BEST: AUROC={best['auroc']*100:.2f} params={best['params']}")
        with open(os.path.join(exp_dir, f"{dataset}__{method}.json"), "w") as f:
            json.dump(cell, f, indent=2)
    return all_best


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="Cora")
    p.add_argument("--methods", default="DOMINANT")
    p.add_argument("--output_dir", default="./outputs")
    return p.parse_args()


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.output_dir, "results", f"grid_{ts}")
    log_dir = os.path.join(args.output_dir, "logs", f"grid_{ts}")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger("SAG_GRID", log_dir)
    logger.info(f"Datasets: {datasets} | Methods: {methods}")

    Config.data_dir = "./data"
    Config.output_dir = args.output_dir

    all_best = {}
    for ds in datasets:
        all_best[ds] = grid_search(ds, methods, exp_dir, logger)
        with open(os.path.join(exp_dir, "best_params.json"), "w") as f:
            json.dump(all_best, f, indent=2)
    logger.info(f"Saved to {exp_dir}")


if __name__ == "__main__":
    main()
