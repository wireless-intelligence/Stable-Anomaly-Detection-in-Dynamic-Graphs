import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from configs import Config, get_dataset_config
from src.data import DataLoader
from src.evaluation import compute_all_metrics
from src.models import ABLATION_NAMES, ABLATION_TYPES, create_sag_model
from src.utils import MemoryTracker, Timer, save_json, set_seed, setup_logger


def run_one(ablation_key, loader, cfg, n_runs, logger):
    vals = {"auroc": [], "auprc": [], "auprc_bal": [], "macro_f1": [], "macro_f1_bal": []}
    lat, mem = [], []
    seeds = [cfg.seed + i for i in range(n_runs)]
    order = getattr(cfg, "edge_order", "auto")
    if order == "auto":
        order = "shuffle" if bool(getattr(cfg, "has_ground_truth", False)) else "bfs"
    for s in seeds:
        set_seed(s)
        model = create_sag_model(
            num_nodes=loader.stats["num_nodes"],
            config=cfg,
            features=loader.features,
            adj=loader.adj,
            ablation=ABLATION_TYPES[ablation_key],
        )
        timer = Timer()
        mt = MemoryTracker()
        mt.reset()
        timer.start()
        edges = loader.get_edge_stream(mode=order)
        rng = np.random.default_rng(s)
        do_shuffle = (order == "shuffle")
        passes = (2 if loader.stats["num_edges"] > 50000 else 3) if do_shuffle else 1
        model.process_stream_batch(edges, passes=passes, shuffle=do_shuffle, rng=rng)
        timer.stop()
        mt.update()
        scores = model.get_scores()
        m = compute_all_metrics(loader.labels, scores)
        for k in vals:
            vals[k].append(m[k])
        lat.append(timer.get_elapsed_ms() / max(1, edges.shape[0]) * 1000.0)
        mem.append(mt.get_peak()["peak_total_mem_gb"])
        logger.info(f"  [{ablation_key}] seed={s}: AUROC={m['auroc']:.4f} F1={m['macro_f1']:.4f} F1_bal={m['macro_f1_bal']:.4f} AUPRC_bal={m['auprc_bal']:.4f}")
    out = {
        k: {
            "mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "values": v,
        }
        for k, v in vals.items()
    }
    out["latency_us"] = {"mean": float(np.mean(lat)), "std": float(np.std(lat, ddof=1) if len(lat) > 1 else 0.0)}
    out["memory_gb"] = {"mean": float(np.mean(mem)), "std": float(np.std(mem, ddof=1) if len(mem) > 1 else 0.0)}
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--n_runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ablations", type=str, default="all")
    p.add_argument("--fusion_mode", type=str, default=None,
                   help="Override dataset's fusion_mode (e.g. 'h_only' to isolate "
                        "state-evolution components). When unset, uses dataset config.")
    p.add_argument("--tag", type=str, default=None,
                   help="Tag appended to result/log dir name (e.g. 'h_only').")
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

    if args.fusion_mode:
        Config.fusion_mode = args.fusion_mode

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_suffix = f"_{args.tag}" if args.tag else ""
    exp = f"{args.dataset}_ablation{tag_suffix}_{ts}"
    log_dir = os.path.join(args.output_dir, "logs", exp)
    os.makedirs(log_dir, exist_ok=True)
    result_dir = os.path.join(args.output_dir, "results", exp)
    os.makedirs(result_dir, exist_ok=True)
    logger = setup_logger("SAG_ABLATION", log_dir)

    loader = DataLoader(
        data_dir=Config.data_dir,
        dataset_name=args.dataset,
        clique_size=dc.get("clique_size", 15),
        num_cliques=dc.get("num_cliques", 10),
        seed=Config.seed,
    )
    loader.load()
    logger.info(f"Stats: {loader.get_stats()}")

    if args.ablations == "all":
        keys = list(ABLATION_TYPES.keys())
    else:
        keys = [k.strip() for k in args.ablations.split(",")]

    all_res = {}
    for key in keys:
        logger.info(f"\n=== {ABLATION_NAMES.get(key, key)} ===")
        all_res[key] = {"name": ABLATION_NAMES.get(key, key), "aggregated": run_one(key, loader, Config, args.n_runs, logger)}

    logger.info("\n=== Ablation Summary (paper-aligned: balanced metrics) ===")
    for k in keys:
        agg = all_res[k]["aggregated"]
        logger.info(
            "%-24s AUROC=%5.1f±%.1f  AUPRC_bal=%5.1f±%.1f  F1_bal=%5.1f±%.1f  (imb AUPRC=%5.1f F1=%5.1f)"
            % (
                all_res[k]["name"],
                agg["auroc"]["mean"] * 100,
                agg["auroc"]["std"] * 100,
                agg["auprc_bal"]["mean"] * 100,
                agg["auprc_bal"]["std"] * 100,
                agg["macro_f1_bal"]["mean"] * 100,
                agg["macro_f1_bal"]["std"] * 100,
                agg["auprc"]["mean"] * 100,
                agg["macro_f1"]["mean"] * 100,
            )
        )

    save_json(
        {
            "dataset": args.dataset,
            "n_runs": args.n_runs,
            "results": all_res,
            "timestamp": ts,
            "stats": loader.get_stats(),
        },
        os.path.join(result_dir, "ablation.json"),
    )
    logger.info(f"Saved to {result_dir}")


if __name__ == "__main__":
    main()
