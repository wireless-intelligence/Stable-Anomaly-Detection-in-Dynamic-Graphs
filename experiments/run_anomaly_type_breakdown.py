"""Anomaly-type breakdown experiment.

Characterises which anomaly types
SAG can / cannot detect by sweeping 10 injection types across selected
clean-base datasets and 4 representative baselines.

Output:
    outputs/results/anomaly_breakdown_<ts>/
        per_type_<dataset>.json   raw AUROC per (method, type)
        breakdown_summary.json    consolidated table
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
from src.baselines import BASELINES
from src.data import DataLoader
from src.data.anomaly_types import INJECTORS, inject_by_name
from src.evaluation import compute_all_metrics
from src.models import create_sag_model
from src.utils import save_json, set_seed, setup_logger


CLEAN_DATASETS = ["Cora", "Citeseer", "Pubmed"]
ANOMALY_TYPES = list(INJECTORS.keys())
BASELINE_METHODS = ["DOMINANT", "CONAD", "GraphSAGE", "ConsisGAD"]


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


def _load_clean(dataset):
    dc = get_dataset_config(dataset)
    loader = DataLoader(data_dir="./data", dataset_name=dataset,
                        clique_size=dc.get("clique_size", 15),
                        num_cliques=dc.get("num_cliques", 5),
                        inject_anomaly=False, seed=42)
    loader.load()
    return loader, dc


def _edge_array(adj):
    coo = adj.tocoo()
    mask = coo.row < coo.col
    return np.column_stack([coo.row[mask], coo.col[mask]]).astype(np.int64)


def _run_sag(num_nodes, adj, features, edges, cfg):
    set_seed(42)
    model = create_sag_model(num_nodes=num_nodes, config=cfg,
                             features=features, adj=adj)
    model.process_stream_batch(edges, passes=2, shuffle=True,
                               rng=np.random.default_rng(42))
    return model.get_scores()


def _run_baseline(name, num_nodes, adj, features, edges):
    cls = BASELINES.get(name)
    if cls is None:
        return None
    set_seed(42)
    try:
        m = cls(num_nodes=num_nodes, config=Config, features=features, seed=42)
        m.fit(adj, edges, None)
        return m.score()
    except Exception as e:
        print(f"    baseline {name} FAILED: {type(e).__name__}: {e}")
        return None


def run_breakdown(datasets, methods, anomaly_types, exp_dir, logger,
                  num_anomalies=60):
    summary = {"datasets": {}}
    for ds in datasets:
        logger.info(f"=== {ds} ===")
        clean_loader, dc = _load_clean(ds)
        cfg = _setup_cfg(dc)
        n = clean_loader.stats["num_nodes"]
        per_type = {}
        for atype in anomaly_types:
            logger.info(f"  injecting type={atype}")
            t0 = time.time()
            adj_new, feat_new, labels = inject_by_name(
                atype, clean_loader.adj, clean_loader.features,
                num_anomalies=num_anomalies, seed=42)
            edges = _edge_array(adj_new)
            inj_t = time.time() - t0
            type_results = {"_n_anomalies": int(labels.sum()),
                            "_inject_time_s": inj_t,
                            "methods": {}}
            t0 = time.time()
            sag_scores = _run_sag(n, adj_new, feat_new, edges, cfg)
            sag_metrics = compute_all_metrics(labels, sag_scores)
            type_results["methods"]["SAG"] = {
                "auroc": float(sag_metrics["auroc"]),
                "auprc_bal": float(sag_metrics["auprc_bal"]),
                "f1_bal": float(sag_metrics["macro_f1_bal"]),
                "elapsed_s": float(time.time() - t0),
            }
            logger.info(f"    SAG       AUROC={sag_metrics['auroc']*100:.1f}")
            for mname in methods:
                t0 = time.time()
                bs = _run_baseline(mname, n, adj_new, feat_new, edges)
                if bs is None:
                    type_results["methods"][mname] = {"error": "fit_failed"}
                    continue
                bm = compute_all_metrics(labels, bs)
                type_results["methods"][mname] = {
                    "auroc": float(bm["auroc"]),
                    "auprc_bal": float(bm["auprc_bal"]),
                    "f1_bal": float(bm["macro_f1_bal"]),
                    "elapsed_s": float(time.time() - t0),
                }
                logger.info(f"    {mname:10}AUROC={bm['auroc']*100:.1f}")
            per_type[atype] = type_results
        summary["datasets"][ds] = per_type
        save_json(per_type, os.path.join(exp_dir, f"per_type_{ds}.json"))

    save_json(summary, os.path.join(exp_dir, "breakdown_summary.json"))
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(CLEAN_DATASETS))
    p.add_argument("--methods", default=",".join(BASELINE_METHODS))
    p.add_argument("--types", default=",".join(ANOMALY_TYPES))
    p.add_argument("--num_anomalies", type=int, default=60)
    p.add_argument("--output_dir", default="./outputs")
    return p.parse_args()


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    types = [t.strip() for t in args.types.split(",") if t.strip()]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.output_dir, "results", f"anomaly_breakdown_{ts}")
    log_dir = os.path.join(args.output_dir, "logs", f"anomaly_breakdown_{ts}")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger("ANOM_BREAK", log_dir)
    logger.info(f"Datasets: {datasets} | Methods: {methods} | Types: {types}")

    Config.data_dir = "./data"
    Config.output_dir = args.output_dir
    run_breakdown(datasets, methods, types, exp_dir, logger,
                  num_anomalies=args.num_anomalies)
    logger.info(f"Saved to {exp_dir}")


if __name__ == "__main__":
    main()
