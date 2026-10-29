"""SAG meaningful multi-seed variance experiment (revision).

Sweeps four seed axes per dataset, holding all *other*
sources of randomness fixed at a single base seed.  Variance comes from the
axis under test, not from compounded randomness.

Axes
----
stream_order      : edge-stream RNG (edge_ordering.* + process_stream_batch rng)
anomaly_injection : AnomalyInjector(seed=) - synthetic anomaly generation
split             : SAG is unsupervised over the full edge stream; recorded as
                    SKIP with reason for every dataset
threshold         : negative-subsample seed for compute_balanced_auprc_f1 -
                    drives variance of balanced AUPRC / macro-F1 only
                    (un-balanced AUROC / AUPRC / macro-F1 are deterministic
                    given the score vector and recorded as such)

Outputs (default --output_dir ./outputs/results/revision/<timestamp>/ AND
canonical copies at ./outputs/results/revision/sag_seed_variance_{raw,summary}.csv):
  raw    : one row per (dataset, axis, seed)              (or skip stub)
  summary: one row per (dataset, axis) with mean / std / n
"""

import argparse
import copy
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.sparse as sp

from configs import Config, get_dataset_config
from src.data import DataLoader
from src.data.anomaly_injection import AnomalyInjector
from src.data.edge_ordering import get_edge_stream as _edge_stream_fn
from src.data.edge_ordering import multi_relation_stream
from src.data.loader import INJECTED_DATASETS
from src.evaluation.metrics import (
    compute_auroc,
    compute_auprc,
    compute_macro_f1,
    compute_balanced_auprc_f1,
)
from src.models import create_sag_model
from src.utils import set_seed, save_json, setup_logger


AXES = ("stream_order", "anomaly_injection", "split", "threshold")
DEFAULT_DATASETS = ("YelpReviews", "Enron", "Cora", "BlogCatalog")
DEFAULT_SEEDS = tuple(range(10))
BASE_SEED = 42


# ---------------------------------------------------------------------------
# Per-run config shim (Config is a singleton; never mutate it directly)
# ---------------------------------------------------------------------------
class CfgShim:
    pass


def build_cfg(dc: dict, model_seed: int):
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
    cfg.seed = int(model_seed)
    return cfg


# ---------------------------------------------------------------------------
# Loader factory: explicit anomaly_injection_seed control
# ---------------------------------------------------------------------------
def load_dataset_with_inj_seed(dataset: str, inj_seed: int, dc: dict):
    """Load (adj, features, labels). For INJECTED_DATASETS, we re-run injection
    with the given seed so that anomaly_injection_seed varies independently of
    everything else loader.load() does (which is otherwise deterministic).
    """
    loader = DataLoader(
        data_dir="./data",
        dataset_name=dataset,
        clique_size=dc.get("clique_size", 15),
        num_cliques=dc.get("num_cliques", 10),
        seed=BASE_SEED,
        inject_anomaly=False,
    )
    loader.load()
    if dataset in INJECTED_DATASETS:
        inj = AnomalyInjector(
            adj=loader.adj,
            features=loader.features,
            clique_size=dc.get("clique_size", 15),
            num_cliques=dc.get("num_cliques", 10),
            k_attr=50,
            seed=int(inj_seed),
        )
        inj.inject_all()
        d = inj.get_data()
        adj = sp.csr_matrix(d["adj"])
        adj.eliminate_zeros()
        adj = adj.maximum(adj.T)
        feats = d["features"] if sp.issparse(d["features"]) else sp.csr_matrix(d["features"])
        loader.adj = adj
        loader.features = feats
        loader.labels = d["labels"]
        loader.stats = {
            "dataset": dataset,
            "num_nodes": int(adj.shape[0]),
            "num_edges": int(adj.nnz // 2),
            "num_features": int(feats.shape[1]),
            "num_anomalies": int(np.sum(d["labels"])),
            "anomaly_ratio": float(np.sum(d["labels"])) / float(adj.shape[0]) if adj.shape[0] else 0.0,
        }
    return loader


# ---------------------------------------------------------------------------
# Edge stream w/ explicit stream_order_seed (independent of loader.seed)
# ---------------------------------------------------------------------------
def edge_stream_with_seed(loader, mode: str, stream_seed: int) -> np.ndarray:
    if mode == "raw":
        coo = loader.adj.tocoo()
        mask = coo.row < coo.col
        return np.stack([coo.row[mask], coo.col[mask]], axis=1).astype(np.int64)
    if mode == "multi_relation":
        rels = getattr(loader, "relation_adjs", None)
        if not rels:
            mode = "shuffle"
        else:
            return multi_relation_stream(rels, seed=int(stream_seed))
    kw = {}
    if mode == "temporal":
        nt = getattr(loader, "node_times", None)
        if nt is None:
            mode = "shuffle"
        else:
            kw["node_times"] = nt
    return _edge_stream_fn(loader.adj, mode=mode, seed=int(stream_seed), **kw)


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------
def run_one(dataset: str, axis: str, seed: int, dc: dict, logger):
    inj_seed = seed if axis == "anomaly_injection" else BASE_SEED
    stream_seed = seed if axis == "stream_order" else BASE_SEED
    model_seed = seed if axis == "stream_order" else BASE_SEED  # rng default falls back to config.seed
    eval_seed = seed if axis == "threshold" else 0

    set_seed(model_seed)
    loader = load_dataset_with_inj_seed(dataset, inj_seed, dc)
    s = loader.get_stats()

    cfg = build_cfg(dc, model_seed=model_seed)
    cfg.window_size = max(128, min(s["num_edges"] // 32, 4096))

    model = create_sag_model(
        num_nodes=s["num_nodes"], config=cfg,
        features=loader.features, adj=loader.adj,
    )

    order = getattr(cfg, "edge_order", "auto")
    if order == "auto":
        order = "shuffle" if cfg.has_ground_truth else "bfs"

    edges = edge_stream_with_seed(loader, mode=order, stream_seed=stream_seed)
    do_shuffle = (order == "shuffle")
    passes = (2 if s["num_edges"] > 50000 else 3) if do_shuffle else 1
    rng = np.random.default_rng(stream_seed)

    t0 = time.time()
    model.process_stream_batch(edges, passes=passes, shuffle=do_shuffle, rng=rng)
    elapsed = time.time() - t0
    scores = model.get_scores()

    y = np.asarray(loader.labels)
    if scores.max() > scores.min():
        s_norm = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        s_norm = scores

    auroc = float(compute_auroc(y, s_norm))
    auprc = float(compute_auprc(y, s_norm))
    macro_f1 = float(compute_macro_f1(y, s_norm))
    auprc_bal, macro_f1_bal = compute_balanced_auprc_f1(y, s_norm, n_trials=20, seed=int(eval_seed))

    return {
        "dataset": dataset,
        "axis": axis,
        "seed": int(seed),
        "inj_seed": int(inj_seed),
        "stream_seed": int(stream_seed),
        "model_seed": int(model_seed),
        "eval_seed": int(eval_seed),
        "auroc": auroc,
        "auprc": auprc,
        "macro_f1": float(macro_f1),
        "auprc_bal": float(auprc_bal),
        "macro_f1_bal": float(macro_f1_bal),
        "elapsed_s": float(elapsed),
        "num_edges": int(s["num_edges"]),
        "num_nodes": int(s["num_nodes"]),
        "num_anomalies": int(s["num_anomalies"]),
        "skipped": False,
        "skip_reason": "",
    }


def skip_stub(dataset: str, axis: str, seed: int, reason: str) -> dict:
    return {
        "dataset": dataset, "axis": axis, "seed": int(seed),
        "inj_seed": -1, "stream_seed": -1, "model_seed": -1, "eval_seed": -1,
        "auroc": float("nan"), "auprc": float("nan"), "macro_f1": float("nan"),
        "auprc_bal": float("nan"), "macro_f1_bal": float("nan"),
        "elapsed_s": 0.0, "num_edges": 0, "num_nodes": 0, "num_anomalies": 0,
        "skipped": True, "skip_reason": reason,
    }


def axis_applicable(dataset: str, axis: str) -> tuple:
    """Return (applicable: bool, reason_if_not: str)."""
    if axis == "split":
        return False, ("SAG is unsupervised: scores are produced from a "
                       "single streaming pass over all edges; there is no "
                       "train/val/test split to seed.")
    if axis == "anomaly_injection" and dataset not in INJECTED_DATASETS:
        return False, ("Dataset has ground-truth anomaly labels (no synthetic "
                       "injection step in the pipeline).")
    return True, ""


# ---------------------------------------------------------------------------
# Aggregation + CSV emission
# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    "dataset", "axis", "seed", "inj_seed", "stream_seed", "model_seed",
    "eval_seed", "auroc", "auprc", "macro_f1", "auprc_bal", "macro_f1_bal",
    "elapsed_s", "num_edges", "num_nodes", "num_anomalies",
    "skipped", "skip_reason",
]
SUMMARY_METRICS = ("auroc", "auprc", "macro_f1", "auprc_bal", "macro_f1_bal")
SUMMARY_COLUMNS = (
    ["dataset", "axis", "n_seeds", "applicable", "skip_reason"]
    + [f"{m}_mean" for m in SUMMARY_METRICS]
    + [f"{m}_std" for m in SUMMARY_METRICS]
    + [f"{m}_min" for m in SUMMARY_METRICS]
    + [f"{m}_max" for m in SUMMARY_METRICS]
)


def write_raw_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in RAW_COLUMNS})


def write_summary_csv(rows, path):
    grouped = {}
    for r in rows:
        grouped.setdefault((r["dataset"], r["axis"]), []).append(r)
    out = []
    for (ds, ax), group in grouped.items():
        ok = [g for g in group if not g["skipped"]]
        if not ok:
            row = {
                "dataset": ds, "axis": ax, "n_seeds": 0,
                "applicable": False,
                "skip_reason": group[0].get("skip_reason", "") if group else "",
            }
            for m in SUMMARY_METRICS:
                row[f"{m}_mean"] = ""
                row[f"{m}_std"] = ""
                row[f"{m}_min"] = ""
                row[f"{m}_max"] = ""
        else:
            row = {
                "dataset": ds, "axis": ax, "n_seeds": len(ok),
                "applicable": True, "skip_reason": "",
            }
            for m in SUMMARY_METRICS:
                v = np.array([r[m] for r in ok], dtype=np.float64)
                row[f"{m}_mean"] = float(v.mean())
                row[f"{m}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
                row[f"{m}_min"] = float(v.min())
                row[f"{m}_max"] = float(v.max())
        out.append(row)
    out.sort(key=lambda r: (r["dataset"], r["axis"]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        for r in out:
            w.writerow(r)
    return out


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    p.add_argument("--axes", default=",".join(AXES))
    p.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--tag", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    for a in axes:
        if a not in AXES:
            raise ValueError(f"unknown axis: {a} (valid: {AXES})")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    run_dir = os.path.join(args.output_dir, "results", "revision",
                           f"seed_variance{tag}_{ts}")
    log_dir = os.path.join(args.output_dir, "logs",
                           f"seed_variance{tag}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    canon_dir = os.path.join(args.output_dir, "results", "revision")
    os.makedirs(canon_dir, exist_ok=True)

    logger = setup_logger("SAG_SEED_VAR", log_dir)
    logger.info(f"datasets={datasets}  axes={axes}  seeds={seeds}")
    logger.info(f"run_dir={run_dir}")

    Config.data_dir = "./data"
    Config.output_dir = args.output_dir

    raw_rows = []
    raw_path_run = os.path.join(run_dir, "sag_seed_variance_raw.csv")
    summary_path_run = os.path.join(run_dir, "sag_seed_variance_summary.csv")
    raw_path_canon = os.path.join(canon_dir, "sag_seed_variance_raw.csv")
    summary_path_canon = os.path.join(canon_dir, "sag_seed_variance_summary.csv")

    for ds in datasets:
        try:
            dc = get_dataset_config(ds)
        except ValueError as e:
            logger.error(f"unknown dataset {ds}: {e}")
            continue
        for axis in axes:
            applicable, reason = axis_applicable(ds, axis)
            if not applicable:
                logger.info(f"[SKIP] {ds} / {axis}: {reason}")
                for sd in seeds:
                    raw_rows.append(skip_stub(ds, axis, sd, reason))
                continue
            for sd in seeds:
                logger.info(f"=== {ds} | axis={axis} | seed={sd} ===")
                try:
                    row = run_one(ds, axis, sd, dc, logger)
                    logger.info(
                        "  AUROC=%.4f  AUPRC=%.4f  F1=%.4f  AUPRCb=%.4f  F1b=%.4f  (%.1fs)" %
                        (row["auroc"], row["auprc"], row["macro_f1"],
                         row["auprc_bal"], row["macro_f1_bal"], row["elapsed_s"])
                    )
                    raw_rows.append(row)
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.error(f"  FAILED {type(e).__name__}: {e}\n{tb}")
                    err_row = skip_stub(ds, axis, sd, f"ERROR:{type(e).__name__}:{e}")
                    err_row["skipped"] = True
                    raw_rows.append(err_row)
                # write incremental in case the run is killed mid-loop
                write_raw_csv(raw_rows, raw_path_run)
                write_raw_csv(raw_rows, raw_path_canon)

    summary = write_summary_csv(raw_rows, summary_path_run)
    write_summary_csv(raw_rows, summary_path_canon)
    save_json({"raw": raw_rows, "summary": summary},
              os.path.join(run_dir, "sag_seed_variance.json"))

    logger.info("=" * 70)
    logger.info(f"WROTE  raw     -> {raw_path_run}")
    logger.info(f"WROTE  summary -> {summary_path_run}")
    logger.info(f"CANON  raw     -> {raw_path_canon}")
    logger.info(f"CANON  summary -> {summary_path_canon}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
