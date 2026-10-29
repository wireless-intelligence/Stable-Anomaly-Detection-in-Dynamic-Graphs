"""Dynamic (temporal) anomaly-detection comparison on REAL timestamped graphs.

Key differences vs ``run_comparison.py`` (static replay protocol):

  * Datasets carry genuine per-edge timestamps and native node labels
    (DGraphFin / Elliptic++ / AMLSim / temporal edge lists) -- no shuffled
    multi-epoch warm-up.
  * SAG is run as a STRICT single pass in true time order
    (``shuffle=False, passes=1, edge_times=...``).
  * Baselines are grouped and reported by paradigm:
      - dynamic   : temporal GNNs (TGN, JODIE) trained self-supervised on the
                    timestamped stream;
      - streaming : native edge-stream detectors (MIDAS, SedanSpot, NetWalk);
      - static    : full-graph transductive GAD (DOMINANT, CoLA, ...) run once
                    on the accumulated graph -- explicitly flagged as adapted.
  * Metrics are computed ONLY over evaluable nodes (``eval_mask``), excluding
    background/unknown classes.

Usage:
    python experiments/run_dynamic_comparison.py --dataset DGraphFin \
        --methods SAG,TGN,JODIE,MIDAS,SedanSpot,DOMINANT --n_runs 3 \
        --max_edges 2000000
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
from src.utils import MemoryTracker, Timer, save_json, set_seed, setup_logger


def _eval(labels, scores, mask):
    m = compute_all_metrics(labels[mask], scores[mask])
    return m


def _aggregate(results, lat, mem):
    keys = ["auroc", "auprc", "auprc_bal", "macro_f1", "macro_f1_bal"]
    agg = {}
    for k in keys:
        vals = [r.get(k, 0.0) for r in results]
        v = np.asarray(vals)
        agg[k] = {"values": vals, "mean": float(v.mean()),
                  "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    agg["latency_us"] = {"mean": float(np.mean(lat)) if lat else 0.0,
                         "std": float(np.std(lat, ddof=1)) if len(lat) > 1 else 0.0}
    agg["memory_gb"] = {"mean": float(np.mean(mem)) if mem else 0.0,
                        "std": float(np.std(mem, ddof=1)) if len(mem) > 1 else 0.0}
    return agg


def run_sag(loader, cfg, n_runs, logger):
    seeds = [cfg.seed + i for i in range(n_runs)]
    edges, times = loader.get_temporal_stream()
    mask = loader.eval_mask
    res, lat, mem = [], [], []
    for s in seeds:
        set_seed(s)
        timer, mt = Timer(), MemoryTracker()
        model = create_sag_model(num_nodes=loader.stats["num_nodes"], config=cfg,
                                 features=loader.features, adj=loader.adj)
        timer.reset(); timer.start(); mt.reset()
        # strict single-pass streaming in true temporal order (no replay)
        model.process_stream_batch(edges, passes=1, shuffle=False,
                                   rng=np.random.default_rng(s), edge_times=times)
        timer.stop(); mt.update()
        scores = model.get_scores()
        m = _eval(loader.labels, scores, mask)
        res.append(m)
        lat.append(timer.get_elapsed_ms() / max(1, edges.shape[0]) * 1000.0)
        mem.append(mt.get_peak()["peak_total_mem_gb"])
        logger.info(f"  SAG seed={s}: AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} "
                    f"F1={m['macro_f1']:.4f}")
    return _aggregate(res, lat, mem)


def run_dynamic(name, cls, loader, cfg, n_runs, logger):
    seeds = [cfg.seed + i for i in range(n_runs)]
    edges, times = loader.get_temporal_stream()
    mask = loader.eval_mask
    res, lat, mem = [], [], []
    for s in seeds:
        set_seed(s)
        timer, mt = Timer(), MemoryTracker()
        model = cls(num_nodes=loader.stats["num_nodes"], config=cfg,
                    features=loader.features, seed=s)
        timer.reset(); timer.start(); mt.reset()
        model.fit_temporal(edges, times, loader.labels)
        scores = model.score()
        timer.stop(); mt.update()
        m = _eval(loader.labels, scores, mask)
        res.append(m)
        lat.append(timer.get_elapsed_ms() / max(1, edges.shape[0]) * 1000.0)
        mem.append(mt.get_peak()["peak_total_mem_gb"])
        logger.info(f"  {name} seed={s}: AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} "
                    f"F1={m['macro_f1']:.4f}")
    return _aggregate(res, lat, mem)


def run_static(name, cls, loader, cfg, n_runs, logger):
    """Static / streaming baselines using the BaselineModel.fit(adj, edges, labels)
    interface. Streaming ones get the time-ordered stream; static GAD get the
    accumulated adjacency (adapted, reported as such)."""
    seeds = [cfg.seed + i for i in range(n_runs)]
    edges = loader.get_edge_stream(shuffle=False)  # true temporal order
    mask = loader.eval_mask
    res, lat, mem = [], [], []
    for s in seeds:
        set_seed(s)
        timer, mt = Timer(), MemoryTracker()
        model = cls(num_nodes=loader.stats["num_nodes"], config=cfg,
                    features=loader.features, seed=s)
        timer.reset(); timer.start(); mt.reset()
        model.fit(loader.adj, edges, loader.labels)
        scores = model.score()
        timer.stop(); mt.update()
        m = _eval(loader.labels, scores, mask)
        res.append(m)
        lat.append(timer.get_elapsed_ms() / max(1, edges.shape[0]) * 1000.0)
        mem.append(mt.get_peak()["peak_total_mem_gb"])
        logger.info(f"  {name} seed={s}: AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} "
                    f"F1={m['macro_f1']:.4f}")
    return _aggregate(res, lat, mem)


def paradigm_of(name):
    if name == "SAG":
        return "sag"
    if name in DYNAMIC_BASELINES:
        return "dynamic"
    if name in STREAMING_BASELINES:
        return "streaming"
    return "static"


def fmt(agg):
    return (f"AUROC={agg['auroc']['mean']*100:5.1f}  "
            f"AUPRC={agg['auprc']['mean']*100:5.1f}  "
            f"F1_bal={agg['macro_f1_bal']['mean']*100:5.1f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True,
                   help="DGraphFin | Elliptic-tx | Elliptic-wallet | AMLSim | "
                        "Bitcoin-Alpha | ...")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--n_runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--methods", type=str, default="SAG,TGN,JODIE,MIDAS,SedanSpot")
    p.add_argument("--max_edges", type=int, default=0,
                   help="cap stream length (earliest-by-time prefix); 0 = all")
    p.add_argument("--no_inject", action="store_true",
                   help="disable synthetic anomaly injection for unlabeled "
                        "temporal edge lists")
    return p.parse_args()


def _apply_dataset_config(name):
    try:
        dc = get_dataset_config(name)
    except ValueError:
        dc = {}
    Config.num_epochs = dc.get("num_epochs", 1)
    Config.has_ground_truth = True
    for k in ("alpha", "gamma", "rho", "kernel_dim"):
        if k in dc:
            setattr(Config, k, dc[k])
    Config.fusion_mode = dc.get("fusion_mode", "auto")
    Config.attr_mode = dc.get("attr_mode", "auto")
    Config.edge_order = "temporal"
    Config.use_continuous_dev = bool(dc.get("use_continuous_dev", False))
    Config.use_trained_ae = bool(dc.get("use_trained_ae", False))
    Config.trained_ae_cfg = dc.get("trained_ae_cfg", {})
    Config.use_attr_boost = bool(dc.get("use_attr_boost", False))
    Config.use_pca_mahal = bool(dc.get("use_pca_mahal", False))
    Config.dyn_epochs = int(dc.get("dyn_epochs", 1))
    return dc


def main():
    args = parse_args()
    Config.data_dir = args.data_dir
    Config.output_dir = args.output_dir
    Config.seed = args.seed
    _apply_dataset_config(args.dataset)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp = f"{args.dataset}_dyn_{ts}"
    log_dir = os.path.join(args.output_dir, "logs", exp)
    result_dir = os.path.join(args.output_dir, "results", exp)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    logger = setup_logger("SAG_DYN", log_dir)
    logger.info(f"Dynamic comparison on {args.dataset} n_runs={args.n_runs}")

    set_seed(Config.seed)
    loader = TemporalDataLoader(data_dir=Config.data_dir, dataset_name=args.dataset,
                                seed=Config.seed, inject_anomalies=not args.no_inject)
    loader.load()

    if args.max_edges and loader.edges.shape[0] > args.max_edges:
        edges, times = loader.tg.sorted_by_time()
        loader.tg.edges = edges[: args.max_edges]
        loader.tg.edge_times = times[: args.max_edges]
        loader.edges = loader.tg.edges
        loader.edge_times = loader.tg.edge_times
        loader.adj = loader.tg.build_adj()
        loader.stats["num_edges"] = int(loader.edges.shape[0])
        logger.info(f"Capped stream to earliest {args.max_edges} edges")

    logger.info(f"Dataset stats: {loader.get_stats()}")
    if loader.stats.get("synthetic_labels"):
        logger.warning("Labels are SYNTHETIC temporal anomalies over real "
                       "structure -- report as such, not as native labels.")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    all_results = {}
    for name in methods:
        logger.info(f"\n=== {name} ({paradigm_of(name)}) ===")
        t0 = time.time()
        try:
            if name == "SAG":
                agg = run_sag(loader, Config, args.n_runs, logger)
            elif name in DYNAMIC_BASELINES:
                agg = run_dynamic(name, DYNAMIC_BASELINES[name], loader, Config,
                                  args.n_runs, logger)
            elif name in BASELINES:
                agg = run_static(name, BASELINES[name], loader, Config,
                                 args.n_runs, logger)
            else:
                logger.warning(f"Unknown method {name}; skipping")
                continue
            agg["paradigm"] = paradigm_of(name)
            logger.info(f"[{name}] {fmt(agg)}  (elapsed {time.time()-t0:.1f}s)")
            all_results[name] = agg
        except Exception as e:
            logger.error(f"[{name}] FAILED: {type(e).__name__}: {e}")
            all_results[name] = {"error": f"{type(e).__name__}: {e}",
                                 "paradigm": paradigm_of(name)}

    save_json(
        {
            "dataset": args.dataset,
            "stats": loader.get_stats(),
            "n_runs": args.n_runs,
            "results": all_results,
            "timestamp": ts,
        },
        os.path.join(result_dir, "dynamic_comparison.json"),
    )
    logger.info(f"Saved results to {result_dir}")


if __name__ == "__main__":
    main()
