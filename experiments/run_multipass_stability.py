"""SAG multi-pass / concept-drift stability.

Run SAG over the same dataset for N passes (with shuffle) and record after
each pass:
  - AUROC / AUPRC_bal / F1_bal
  - Memory (peak working-set MB)
  - L2 norm of the state buffer ||h_v|| averaged across nodes
  - Per-node score variance over the last K passes (does the score drift?)

This responds to the "Stable" claim in the paper title: SAG should give
near-identical metrics across passes and bounded h_v norm.
"""

import argparse
import os
import sys
import json
import time
import tracemalloc
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from configs import Config, get_dataset_config
from src.data import DataLoader
from src.evaluation import compute_all_metrics
from src.models import create_sag_model
from src.utils import set_seed, save_json, setup_logger


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
    cfg.use_continuous_dev = False
    cfg.use_real_time_decay = False
    return cfg


def run_passes(name, num_passes, output_path, logger):
    dc = get_dataset_config(name)
    loader = DataLoader(data_dir="./data", dataset_name=name,
                        clique_size=dc.get("clique_size", 15),
                        num_cliques=dc.get("num_cliques", 10), seed=42)
    loader.load()
    s = loader.get_stats()
    logger.info(f"=== {name} stats: {s}")

    cfg = setup_cfg(dc)
    cfg.window_size = max(128, min(s["num_edges"] // 32, 4096))
    set_seed(42)
    model = create_sag_model(num_nodes=s["num_nodes"], config=cfg,
                             features=loader.features, adj=loader.adj)

    order = getattr(cfg, "edge_order", "auto")
    if order == "auto":
        order = "shuffle" if bool(getattr(cfg, "has_ground_truth", False)) else "bfs"
    do_shuffle = (order == "shuffle")

    snapshots = []
    score_history = []  # for variance across passes
    for p in range(num_passes):
        tracemalloc.start()
        t0 = time.time()
        edges_arr = np.asarray(list(loader.get_edge_stream(mode=order)),
                                dtype=np.int64)
        if do_shuffle:
            np.random.default_rng(42 + p).shuffle(edges_arr)
        model.process_stream_batch(edges_arr, passes=1, shuffle=False,
                                   rng=np.random.default_rng(42 + p))
        elapsed = time.time() - t0
        cur_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        scores = model.get_scores()
        m = compute_all_metrics(loader.labels, scores)
        # h_v stats
        h_norm_mean = None
        try:
            h_buf = model.state.h
            h_norm_mean = float(np.linalg.norm(h_buf, axis=1).mean())
        except Exception:
            pass

        score_history.append(scores.copy())
        # score variance across last 3 passes (drift indicator)
        last_var = None
        if len(score_history) >= 2:
            arr = np.stack(score_history[-3:], axis=0)
            # rank-corrected variance: std of rank/N per node
            ranks = np.argsort(np.argsort(arr, axis=1), axis=1) / arr.shape[1]
            last_var = float(ranks.std(axis=0).mean())

        snapshot = {
            "pass": p + 1,
            "auroc": float(m["auroc"]),
            "auprc_bal": float(m["auprc_bal"]),
            "macro_f1_bal": float(m["macro_f1_bal"]),
            "elapsed_s": elapsed,
            "peak_mem_mb": peak_mem / 1024 / 1024,
            "h_norm_mean": h_norm_mean,
            "rank_drift_std": last_var,
        }
        snapshots.append(snapshot)
        logger.info(f"  pass {p+1:2d}: AUROC={m['auroc']*100:.2f} "
                    f"AUPRC_bal={m['auprc_bal']*100:.2f} "
                    f"F1_bal={m['macro_f1_bal']*100:.2f} "
                    f"|h|={h_norm_mean if h_norm_mean else 'n/a'} "
                    f"drift={last_var if last_var is not None else 'n/a'} "
                    f"({elapsed:.1f}s)")
        with open(output_path, "w") as f:
            json.dump({"dataset": name, "stats": s,
                       "snapshots": snapshots}, f, indent=2,
                      default=lambda o: float(o) if isinstance(o, np.floating) else None)
    return snapshots


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="Cora,BlogCatalog")
    p.add_argument("--num_passes", type=int, default=10)
    p.add_argument("--output_dir", default="./outputs")
    return p.parse_args()


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.output_dir, "results", f"multipass_{ts}")
    log_dir = os.path.join(args.output_dir, "logs", f"multipass_{ts}")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger("SAG_PASS", log_dir)
    logger.info(f"Datasets: {datasets} | passes: {args.num_passes}")

    Config.data_dir = "./data"
    Config.output_dir = args.output_dir

    all_results = {}
    for ds in datasets:
        out_path = os.path.join(exp_dir, f"{ds}.json")
        all_results[ds] = run_passes(ds, args.num_passes, out_path, logger)
    save_json(all_results, os.path.join(exp_dir, "all_multipass.json"))
    logger.info(f"Saved to {exp_dir}")


if __name__ == "__main__":
    main()
