import argparse
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
from src.utils import MemoryTracker, Timer, save_json, set_seed, setup_logger


ALL_DATASETS = [
    "Amazon",
    "YelpChi",
    "Enron",
    "BlogCatalog",
    "Flickr",
    "ACM",
    "Cora",
    "Citeseer",
    "Pubmed",
    "Arxiv",
    "CTU-13",
]


def run_sag(name, cfg, n_runs, logger):
    dc = get_dataset_config(name)
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
    cfg.iforest_blend_h_weight = float(dc.get("iforest_blend_h_weight", 0.3))
    cfg.use_attr_boost = bool(dc.get("use_attr_boost", False))
    cfg.use_continuous_dev = bool(dc.get("use_continuous_dev", False))
    cfg.use_real_time_decay = bool(dc.get("use_real_time_decay", False))
    cfg.use_pca_mahal = bool(dc.get("use_pca_mahal", False))
    cfg.pm_blend_weight = float(dc.get("pm_blend_weight", 0.7))

    loader = DataLoader(
        data_dir=cfg.data_dir,
        dataset_name=name,
        clique_size=dc.get("clique_size", 15),
        num_cliques=dc.get("num_cliques", 10),
        seed=cfg.seed,
    )
    loader.load()
    s = loader.get_stats()
    logger.info(f"[{name}] stats {s}")

    cfg.window_size = max(128, min(s["num_edges"] // 32, 4096))

    metrics = {"auroc": [], "auprc": [], "auprc_bal": [], "macro_f1": [], "macro_f1_bal": []}
    pk_runs = []
    rk_runs = []
    lat, mem = [], []
    order = getattr(cfg, "edge_order", "auto")
    if order == "auto":
        order = "shuffle" if bool(getattr(cfg, "has_ground_truth", False)) else "bfs"
    for i in range(n_runs):
        seed = cfg.seed + i
        set_seed(seed)
        model = create_sag_model(
            num_nodes=s["num_nodes"], config=cfg,
            features=loader.features, adj=loader.adj,
        )
        edges = loader.get_edge_stream(mode=order)
        edge_times = loader.get_edge_times(edges) if order == "temporal" and bool(getattr(cfg, "use_real_time_decay", False)) else None
        mt = MemoryTracker()
        mt.reset()
        t = Timer()
        t.reset()
        t.start()
        do_shuffle = (order == "shuffle")
        passes = (2 if s["num_edges"] > 50000 else 3) if do_shuffle else 1
        model.process_stream_batch(edges, passes=passes, shuffle=do_shuffle, rng=np.random.default_rng(seed), edge_times=edge_times)
        t.stop()
        mt.update()
        scores = model.get_scores()
        m = compute_all_metrics(loader.labels, scores)
        for k in metrics:
            metrics[k].append(m[k])
        pk_runs.append(m.get("precision_at_k", {}))
        rk_runs.append(m.get("recall_at_k", {}))
        lat.append(t.get_elapsed_ms() / max(1, edges.shape[0]) * 1000.0)
        mem.append(mt.get_peak()["peak_total_mem_gb"])
    agg = {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1) if len(v) > 1 else 0.0)}
           for k, v in metrics.items()}
    if pk_runs:
        all_ks = sorted({int(k) for d in pk_runs for k in d.keys()})
        agg["precision_at_k"] = {}
        agg["recall_at_k"] = {}
        for k in all_ks:
            pk_vals = [d[k] for d in pk_runs if k in d]
            rk_vals = [d[k] for d in rk_runs if k in d]
            if pk_vals:
                agg["precision_at_k"][str(k)] = {
                    "mean": float(np.mean(pk_vals)),
                    "std": float(np.std(pk_vals, ddof=1) if len(pk_vals) > 1 else 0.0),
                }
            if rk_vals:
                agg["recall_at_k"][str(k)] = {
                    "mean": float(np.mean(rk_vals)),
                    "std": float(np.std(rk_vals, ddof=1) if len(rk_vals) > 1 else 0.0),
                }
    agg["latency_us"] = {"mean": float(np.mean(lat)), "std": float(np.std(lat, ddof=1) if len(lat) > 1 else 0.0)}
    agg["memory_gb"] = {"mean": float(np.mean(mem)), "std": float(np.std(mem, ddof=1) if len(mem) > 1 else 0.0)}
    agg["stats"] = s
    return agg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--n_runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--datasets", type=str, default="all")
    return p.parse_args()


def main():
    args = parse_args()
    Config.data_dir = args.data_dir
    Config.output_dir = args.output_dir
    Config.seed = args.seed

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp = f"all_{ts}"
    log_dir = os.path.join(args.output_dir, "logs", exp)
    os.makedirs(log_dir, exist_ok=True)
    result_dir = os.path.join(args.output_dir, "results", exp)
    os.makedirs(result_dir, exist_ok=True)
    logger = setup_logger("SAG_ALL", log_dir)

    datasets = ALL_DATASETS if args.datasets == "all" else [d.strip() for d in args.datasets.split(",")]

    all_results = {}
    for name in datasets:
        logger.info(f"\n{'='*60}\n{name}\n{'='*60}")
        try:
            agg = run_sag(name, Config, args.n_runs, logger)
            all_results[name] = agg
            logger.info(
                "%-14s AUROC=%5.1f AUPRC=%5.1f/%5.1f F1=%5.1f/%5.1f  lat=%.0fus mem=%.2fGB"
                % (
                    name,
                    agg["auroc"]["mean"] * 100,
                    agg["auprc"]["mean"] * 100,
                    agg["auprc_bal"]["mean"] * 100,
                    agg["macro_f1"]["mean"] * 100,
                    agg["macro_f1_bal"]["mean"] * 100,
                    agg["latency_us"]["mean"],
                    agg["memory_gb"]["mean"],
                )
            )
        except FileNotFoundError as e:
            logger.warning(f"[{name}] skipped: {e}")
        except Exception as e:
            logger.error(f"[{name}] error: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    for name in datasets:
        if name not in all_results:
            logger.info(f"{name:<14}  (missing/error)")
            continue
        a = all_results[name]
        logger.info(
            "%-14s AUROC=%5.1f  AUPRC_imb=%5.1f AUPRC_bal=%5.1f  F1_imb=%5.1f F1_bal=%5.1f"
            % (
                name,
                a["auroc"]["mean"] * 100,
                a["auprc"]["mean"] * 100,
                a["auprc_bal"]["mean"] * 100,
                a["macro_f1"]["mean"] * 100,
                a["macro_f1_bal"]["mean"] * 100,
            )
        )
    save_json(
        {"datasets": all_results, "timestamp": ts, "n_runs": args.n_runs},
        os.path.join(result_dir, "all_datasets.json"),
    )
    logger.info(f"Saved to {result_dir}")


if __name__ == "__main__":
    main()
