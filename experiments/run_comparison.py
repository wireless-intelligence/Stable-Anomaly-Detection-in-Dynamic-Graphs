import argparse
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
from src.utils import MemoryTracker, Timer, save_json, set_seed, setup_logger


def run_sag(loader, cfg, n_runs: int, logger):
    seeds = [cfg.seed + i for i in range(n_runs)]
    res = []
    lat = []
    mem = []
    order = getattr(cfg, "edge_order", "auto")
    if order == "auto":
        order = "shuffle" if bool(getattr(cfg, "has_ground_truth", False)) else "bfs"
    for s in seeds:
        set_seed(s)
        timer = Timer()
        mt = MemoryTracker()
        model = create_sag_model(
            num_nodes=loader.stats["num_nodes"], config=cfg,
            features=loader.features, adj=loader.adj,
        )
        timer.reset()
        timer.start()
        mt.reset()
        edges = loader.get_edge_stream(mode=order)
        do_shuffle = (order == "shuffle")
        passes = (2 if loader.stats["num_edges"] > 50000 else 3) if do_shuffle else 1
        rng = np.random.default_rng(s)
        model.process_stream_batch(edges, passes=passes, shuffle=do_shuffle, rng=rng)
        timer.stop()
        mt.update()
        scores = model.get_scores()
        m = compute_all_metrics(loader.labels, scores)
        res.append(m)
        total_ms = timer.get_elapsed_ms()
        per_edge_ms = total_ms / max(1, edges.shape[0]) * 1000.0
        lat.append(per_edge_ms)
        mem.append(mt.get_peak()["peak_total_mem_gb"])
        logger.info(
            f"  SAG run seed={s}: AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} F1={m['macro_f1']:.4f}"
        )
    return _aggregate(res, lat, mem)


def run_baseline(name, cls, loader, cfg, n_runs: int, logger):
    seeds = [cfg.seed + i for i in range(n_runs)]
    res = []
    lat = []
    mem = []
    for s in seeds:
        set_seed(s)
        timer = Timer()
        mt = MemoryTracker()
        model = cls(num_nodes=loader.stats["num_nodes"], config=cfg, features=loader.features, seed=s)
        timer.reset()
        timer.start()
        mt.reset()
        edges = loader.get_edge_stream(shuffle=True)
        model.fit(loader.adj, edges, loader.labels)
        scores = model.score()
        timer.stop()
        mt.update()
        m = compute_all_metrics(loader.labels, scores)
        res.append(m)
        total_ms = timer.get_elapsed_ms()
        lat.append(total_ms / max(1, edges.shape[0]) * 1000.0)
        mem.append(mt.get_peak()["peak_total_mem_gb"])
        logger.info(
            f"  {name} run seed={s}: AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} F1={m['macro_f1']:.4f}"
        )
    return _aggregate(res, lat, mem)


def _aggregate(results, lat, mem):
    keys = ["auroc", "auprc", "auprc_bal", "macro_f1", "macro_f1_bal"]
    agg = {}
    for k in keys:
        vals = [r.get(k, 0.0) for r in results]
        agg[k] = {"values": vals}
        v = np.asarray(vals)
        agg[k]["mean"] = float(v.mean())
        agg[k]["std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    agg["latency_us"] = {"mean": float(np.mean(lat)), "std": float(np.std(lat, ddof=1) if len(lat) > 1 else 0.0)}
    agg["memory_gb"] = {"mean": float(np.mean(mem)), "std": float(np.std(mem, ddof=1) if len(mem) > 1 else 0.0)}
    return agg


def fmt(agg):
    return (
        f"AUROC={agg['auroc']['mean']*100:5.1f}  "
        f"AUPRC_bal={agg['auprc_bal']['mean']*100:5.1f}  "
        f"F1_bal={agg['macro_f1_bal']['mean']*100:5.1f}"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--n_runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--methods",
        type=str,
        default="SAG,GraphSAGE,DOMINANT,MIDAS,DySAT,SedanSpot",
    )
    return p.parse_args()


def main():
    args = parse_args()
    Config.data_dir = args.data_dir
    Config.output_dir = args.output_dir
    Config.seed = args.seed
    dc = get_dataset_config(args.dataset)
    Config.num_epochs = dc.get("num_epochs", 20)
    Config.has_ground_truth = bool(dc.get("has_ground_truth", False))
    for k in ("alpha", "gamma", "rho"):
        if k in dc:
            setattr(Config, k, dc[k])
    Config.fusion_mode = dc.get("fusion_mode", "auto")
    Config.edge_order = dc.get("edge_order", "auto")
    Config.attr_mode = dc.get("attr_mode", "auto")
    Config.use_trained_ae = bool(dc.get("use_trained_ae", False))
    Config.trained_ae_cfg = dc.get("trained_ae_cfg", {})
    Config.iforest_blend_h_weight = float(dc.get("iforest_blend_h_weight", 0.3))
    Config.use_attr_boost = bool(dc.get("use_attr_boost", False))
    Config.use_continuous_dev = bool(dc.get("use_continuous_dev", False))
    Config.use_real_time_decay = bool(dc.get("use_real_time_decay", False))
    Config.use_pca_mahal = bool(dc.get("use_pca_mahal", False))
    Config.pm_blend_weight = float(dc.get("pm_blend_weight", 0.7))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp = f"{args.dataset}_cmp_{ts}"
    log_dir = os.path.join(args.output_dir, "logs", exp)
    os.makedirs(log_dir, exist_ok=True)
    result_dir = os.path.join(args.output_dir, "results", exp)
    os.makedirs(result_dir, exist_ok=True)
    logger = setup_logger("SAG_CMP", log_dir)
    logger.info(f"Comparison on {args.dataset} with n_runs={args.n_runs}")

    set_seed(Config.seed)
    loader = DataLoader(
        data_dir=Config.data_dir,
        dataset_name=args.dataset,
        clique_size=dc.get("clique_size", 15),
        num_cliques=dc.get("num_cliques", 10),
        seed=Config.seed,
    )
    loader.load()
    logger.info(f"Dataset stats: {loader.get_stats()}")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    all_results = {}
    for name in methods:
        logger.info(f"\n=== Running {name} ===")
        t0 = time.time()
        try:
            if name == "SAG":
                agg = run_sag(loader, Config, args.n_runs, logger)
            elif name in BASELINES:
                agg = run_baseline(name, BASELINES[name], loader, Config, args.n_runs, logger)
            else:
                logger.warning(f"Unknown method {name}; skipping")
                continue
            logger.info(f"[{name}] {fmt(agg)}   (elapsed {time.time()-t0:.1f}s)")
            all_results[name] = agg
        except Exception as e:
            logger.error(f"[{name}] FAILED: {type(e).__name__}: {e}")
            all_results[name] = {"error": f"{type(e).__name__}: {e}"}

    save_json(
        {
            "dataset": args.dataset,
            "stats": loader.get_stats(),
            "n_runs": args.n_runs,
            "results": all_results,
            "timestamp": ts,
        },
        os.path.join(result_dir, "comparison.json"),
    )
    logger.info(f"Saved results to {result_dir}")


if __name__ == "__main__":
    main()
