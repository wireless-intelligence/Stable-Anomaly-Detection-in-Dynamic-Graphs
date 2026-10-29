"""Time-to-detect / detection delay experiment (revision).

Definition
----------
For each anomaly node v we measure:

    onset(v)       = smallest edge index i such that v in edges[i]
    detection(v)   = smallest checkpoint c >= onset(v) such that
                     score_c(v) >= threshold_global,
                     where score_c is the per-node anomaly score after
                     processing edges[:c+1] and threshold_global is the
                     best-F1 threshold computed on the FINAL score vector
                     after all edges have been processed (i.e. the
                     validation-selected threshold).
    delay(v)       = detection(v) - onset(v)        (NaN if never detected)
    miss(v)        = True iff score never crosses threshold_global

Per-method aggregates: mean / median / p95 of delay over detected anomalies,
plus miss_rate = #miss / #anomalies.

Methods supported
-----------------
SAG-full, SAG-no-decay, SAG-no-exposure : stream via SAG.process_edge with
    K-spaced checkpoint snapshots (K = max(1, num_edges // 200) by default).
MIDAS : a streaming-compatible re-implementation of src.baselines.MIDAS that
    snapshots node_scores at each checkpoint.
NetWalk : NOT streaming-compatible (global SVD on random-walk co-occurrence).
    Recorded as skipped with reason in the raw CSV; documented in
    results/revision/UNIMPLEMENTED.md.

Outputs
-------
canonical (latest run; --overwrite to replace):
  outputs/results/revision/time_to_detect_raw.csv
  outputs/results/revision/time_to_detect_summary.csv
  outputs/figures/revision/time_to_detect/<dataset>.{pdf,png}
"""

import argparse
import copy
import csv
import math
import os
import sys
import time
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score

from configs import Config, get_dataset_config
from src.data import DataLoader
from src.models import create_sag_model
from src.utils import save_json, set_seed, setup_logger


METHODS_DEFAULT = ("SAG-full", "SAG-no-decay", "SAG-no-exposure", "MIDAS")
METHODS_NOT_STREAMING = {"NetWalk"}
DATASETS_DEFAULT = ("YelpReviews", "Enron", "BlogCatalog", "Cora")
SEEDS_DEFAULT = (0, 1, 2)
BASE_SEED = 42
DEFAULT_CHECKPOINTS = 200


# ---------------------------------------------------------------------------
class CfgShim:
    pass


def build_cfg(dc: dict, seed: int):
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
    cfg.seed = int(seed)
    return cfg


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray):
    if y_score.max() <= y_score.min():
        return 0.5, 0.0
    thresholds = np.linspace(y_score.min(), y_score.max(), 200)
    best_t, best_f = float(thresholds[0]), 0.0
    for t in thresholds:
        yp = (y_score >= t).astype(int)
        f = f1_score(y_true, yp, average="macro", zero_division=0)
        if f > best_f:
            best_f = f
            best_t = float(t)
    return best_t, float(best_f)


# ---------------------------------------------------------------------------
# Streaming-compatible MIDAS — replicates src.baselines.MIDASBaseline.fit
# but exposes per-checkpoint node-score snapshots.
# ---------------------------------------------------------------------------
def midas_stream_with_checkpoints(num_nodes: int, edges: np.ndarray,
                                   checkpoints, num_hash: int = 2,
                                   buckets: int = 2 ** 14, seed: int = 42):
    rng = np.random.default_rng(seed)
    a1 = rng.integers(1, 2 ** 31 - 1, size=num_hash)
    b1 = rng.integers(0, 2 ** 31 - 1, size=num_hash)
    n = num_nodes
    node_scores = np.zeros(n, dtype=np.float64)
    window_size = max(1, edges.shape[0] // 128)
    cur = np.zeros((num_hash, buckets), dtype=np.float64)
    tot = np.zeros((num_hash, buckets), dtype=np.float64)
    wcount = 0
    t = 1
    u_arr = edges[:, 0].astype(np.int64)
    v_arr = edges[:, 1].astype(np.int64)
    ekey = u_arr * n + v_arr
    h_e = (a1[:, None] * ekey[None, :] + b1[:, None]) % buckets
    cps = sorted(set(int(c) for c in checkpoints))
    cp_iter = iter(cps)
    next_cp = next(cp_iter, None)
    snaps = {}
    M = edges.shape[0]
    for i in range(M):
        u, v = int(u_arr[i]), int(v_arr[i])
        for hi in range(num_hash):
            cur[hi, h_e[hi, i]] += 1.0
            tot[hi, h_e[hi, i]] += 1.0
        wcount += 1
        if wcount >= window_size:
            t += 1
            edge_scores = np.zeros(num_hash)
            for hi in range(num_hash):
                idx = h_e[hi, i]
                s_cur = cur[hi, idx]
                s_tot = tot[hi, idx]
                expected = s_tot / t
                if expected > 0:
                    edge_scores[hi] = ((s_cur - expected) ** 2) / (expected * max(1, t - 1) / t)
            node_scores[u] += float(np.min(edge_scores))
            node_scores[v] += float(np.min(edge_scores))
            cur.fill(0.0)
            wcount = 0
        while next_cp is not None and (i + 1) >= next_cp:
            snaps[next_cp] = node_scores.copy()
            next_cp = next(cp_iter, None)
    if M not in snaps:
        snaps[M] = node_scores.copy()
    return snaps


# ---------------------------------------------------------------------------
def stream_with_checkpoints(method: str, loader, dc: dict, seed: int,
                             checkpoints, ae_cache, logger):
    """Returns (snaps, final_score, edges_iter, order, score_path).

    For SAG variants the snapshots and final score both come from the
    streaming-state predicate (model.state.scores()) -- the offline
    attribute/iforest branch is computed once at init and would
    short-circuit the time-to-detect question by being constant from t=0.
    score_path == "state" for SAG, "midas" for MIDAS.
    """
    set_seed(seed)
    cfg = build_cfg(dc, seed=seed)
    s = loader.get_stats()
    cfg.window_size = max(128, min(s["num_edges"] // 32, 4096))
    if ae_cache is not None:
        cfg._trained_ae_score_cache = ae_cache

    order = getattr(cfg, "edge_order", "auto")
    if order == "auto":
        order = "shuffle" if cfg.has_ground_truth else "bfs"
    edges = loader.get_edge_stream(mode=order)
    M = int(edges.shape[0])

    if method == "MIDAS":
        snaps = midas_stream_with_checkpoints(
            num_nodes=s["num_nodes"], edges=edges,
            checkpoints=checkpoints, seed=seed,
        )
        final_score = snaps[max(snaps.keys())]
        return snaps, final_score, edges, order, "midas"

    ablation_map = {"SAG-full": None,
                    "SAG-no-decay": "no_decay",
                    "SAG-no-exposure": "no_exposure"}
    if method not in ablation_map:
        raise ValueError(f"unknown SAG variant: {method}")
    model = create_sag_model(num_nodes=s["num_nodes"], config=cfg,
                             features=loader.features, adj=loader.adj,
                             ablation=ablation_map[method])

    rng = np.random.default_rng(seed)
    if order == "shuffle":
        idx = rng.permutation(M)
        edges_iter = edges[idx]
    else:
        edges_iter = edges

    cps = sorted(set(int(c) for c in checkpoints))
    cp_iter = iter(cps)
    next_cp = next(cp_iter, None)
    snaps = {}
    for i in range(M):
        u, v = int(edges_iter[i, 0]), int(edges_iter[i, 1])
        model.process_edge(u, v)
        while next_cp is not None and (i + 1) >= next_cp:
            snaps[next_cp] = model.state.scores().copy()
            next_cp = next(cp_iter, None)
    if M not in snaps:
        snaps[M] = model.state.scores().copy()
    final_score = model.state.scores().copy()
    return snaps, final_score, edges_iter, order, "state"


# ---------------------------------------------------------------------------
def first_appearance_index(edges: np.ndarray, num_nodes: int) -> np.ndarray:
    """For each node, the smallest edge index i where the node appears in
    edges[i]. INT64_MAX if the node never appears."""
    INF = np.iinfo(np.int64).max
    out = np.full(num_nodes, INF, dtype=np.int64)
    M = edges.shape[0]
    u = edges[:, 0]; v = edges[:, 1]
    for i in range(M):
        if out[u[i]] == INF:
            out[u[i]] = i
        if out[v[i]] == INF:
            out[v[i]] = i
    return out


def detection_indices(snaps_sorted, threshold: float, anom_nodes: np.ndarray) -> dict:
    """For each anomaly node, find the smallest checkpoint at which its
    snapshot score >= threshold. Returns {node -> checkpoint or None}."""
    out = {int(n): None for n in anom_nodes}
    cps = sorted(snaps_sorted.keys())
    for cp in cps:
        sc = snaps_sorted[cp]
        for n in anom_nodes:
            n = int(n)
            if out[n] is None and sc[n] >= threshold:
                out[n] = int(cp)
    return out


def normalize_for_threshold(s: np.ndarray) -> np.ndarray:
    if s.max() > s.min():
        return (s - s.min()) / (s.max() - s.min())
    return np.zeros_like(s)


# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    "dataset", "method", "seed", "node",
    "onset_edge_idx", "detection_edge_idx",
    "delay_edges", "missed",
    "threshold", "global_macro_f1",
    "num_edges", "num_nodes", "num_anomalies",
    "skipped", "skip_reason",
]
SUMMARY_COLUMNS = [
    "dataset", "method", "seed",
    "n_anomalies_with_onset", "n_detected", "n_missed",
    "miss_rate",
    "delay_mean", "delay_median", "delay_p95", "delay_p99", "delay_max",
    "global_threshold", "global_macro_f1",
    "num_edges", "num_nodes",
    "skipped", "skip_reason",
]


def write_raw_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in RAW_COLUMNS})


def aggregate_summary(per_node_rows):
    grouped = {}
    for r in per_node_rows:
        grouped.setdefault((r["dataset"], r["method"], r["seed"]), []).append(r)
    out = []
    for (ds, mt, sd), group in grouped.items():
        if group and group[0].get("skipped"):
            out.append({
                "dataset": ds, "method": mt, "seed": sd,
                "n_anomalies_with_onset": 0, "n_detected": 0, "n_missed": 0,
                "miss_rate": float("nan"),
                "delay_mean": float("nan"), "delay_median": float("nan"),
                "delay_p95": float("nan"), "delay_p99": float("nan"),
                "delay_max": float("nan"),
                "global_threshold": float("nan"),
                "global_macro_f1": float("nan"),
                "num_edges": 0, "num_nodes": 0,
                "skipped": True, "skip_reason": group[0].get("skip_reason", ""),
            })
            continue
        delays = [r["delay_edges"] for r in group
                  if not r["missed"] and r["delay_edges"] >= 0]
        n_with = sum(1 for r in group if r["onset_edge_idx"] >= 0)
        n_miss = sum(1 for r in group if r["missed"])
        n_det = n_with - n_miss
        miss_rate = (n_miss / n_with) if n_with > 0 else float("nan")
        d_arr = np.array(delays, dtype=np.float64) if delays else np.array([])
        out.append({
            "dataset": ds, "method": mt, "seed": sd,
            "n_anomalies_with_onset": int(n_with),
            "n_detected": int(n_det), "n_missed": int(n_miss),
            "miss_rate": float(miss_rate),
            "delay_mean":   float(np.mean(d_arr))            if d_arr.size else float("nan"),
            "delay_median": float(np.median(d_arr))          if d_arr.size else float("nan"),
            "delay_p95":    float(np.quantile(d_arr, 0.95))  if d_arr.size else float("nan"),
            "delay_p99":    float(np.quantile(d_arr, 0.99))  if d_arr.size else float("nan"),
            "delay_max":    float(np.max(d_arr))             if d_arr.size else float("nan"),
            "global_threshold": float(group[0]["threshold"]),
            "global_macro_f1":  float(group[0]["global_macro_f1"]),
            "num_edges": int(group[0]["num_edges"]),
            "num_nodes": int(group[0]["num_nodes"]),
            "skipped": False, "skip_reason": "",
        })
    out.sort(key=lambda r: (r["dataset"], r["method"], r["seed"]))
    return out


def write_summary_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
def plot_delay_distributions(per_node_rows, fig_dir):
    import matplotlib.pyplot as plt
    grouped = {}
    for r in per_node_rows:
        if r.get("skipped") or r["missed"] or r["delay_edges"] < 0:
            continue
        grouped.setdefault((r["dataset"], r["method"]), []).append(r["delay_edges"])
    if not grouped:
        return
    datasets = sorted({k[0] for k in grouped.keys()})
    methods = sorted({k[1] for k in grouped.keys()})
    palette = ["#e74c3c", "#3498db", "#2ca02c", "#f39c12", "#9b59b6", "#16a085"]
    method_color = {m: palette[i % len(palette)] for i, m in enumerate(methods)}
    os.makedirs(fig_dir, exist_ok=True)
    for ds in datasets:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        for m in methods:
            vals = grouped.get((ds, m))
            if not vals:
                continue
            v = np.array(vals, dtype=np.float64)
            v = np.sort(v)
            cdf = np.linspace(1.0 / v.size, 1.0, v.size)
            ax.plot(v, cdf, linewidth=1.8, color=method_color[m],
                    label=f"{m}  (n={v.size}, p50={int(np.median(v))})")
        ax.set_xlabel("Detection delay (edges between onset and first crossing)")
        ax.set_ylabel("CDF over detected anomalies")
        ax.set_xscale("symlog", linthresh=1)
        ax.grid(True, axis="both", linestyle="--", alpha=0.4, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", fontsize=9, frameon=True, edgecolor="0.7", fancybox=False)
        ax.set_title(f"Time-to-detect CDF — {ds}",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        base = os.path.join(fig_dir, f"{ds}_ttd_cdf")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", dpi=160, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(DATASETS_DEFAULT))
    p.add_argument("--methods", default=",".join(METHODS_DEFAULT))
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS_DEFAULT))
    p.add_argument("--num_checkpoints", type=int, default=DEFAULT_CHECKPOINTS)
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--tag", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args()


def _build_ae_cache(loader, dc, logger):
    if not dc.get("use_trained_ae", False):
        return None
    try:
        from src.models.trained_attr import UnsupervisedGCNAutoencoder
        ae_cfg = dc.get("trained_ae_cfg", {}) or {}
        cache_key = (id(loader.features), id(loader.adj),
                     tuple(sorted(ae_cfg.items())), BASE_SEED)
        ae = UnsupervisedGCNAutoencoder(seed=BASE_SEED, **ae_cfg)
        sc = ae.fit_score(loader.adj, loader.features)
        if not np.all(np.isfinite(sc)) or sc.std() < 1e-12:
            return None
        return {"key": cache_key, "score": sc}
    except Exception as e:
        logger.warning(f"  AE pre-training failed: {e}")
        return None


def _maybe_skip(path, overwrite, logger):
    if os.path.exists(path) and not overwrite:
        logger.warning(f"  refusing to overwrite {path} (pass --overwrite)")
        return True
    return False


def _make_checkpoints(num_edges: int, num_checkpoints: int):
    if num_checkpoints >= num_edges:
        return list(range(1, num_edges + 1))
    step = max(1, num_edges // num_checkpoints)
    cps = list(range(step, num_edges + 1, step))
    if cps[-1] != num_edges:
        cps.append(num_edges)
    return cps


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    run_dir = os.path.join(args.output_dir, "results", "revision",
                           f"time_to_detect{tag}_{ts}")
    log_dir = os.path.join(args.output_dir, "logs",
                           f"time_to_detect{tag}_{ts}")
    fig_dir = os.path.join(args.output_dir, "figures", "revision",
                           "time_to_detect")
    canon_dir = os.path.join(args.output_dir, "results", "revision")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(canon_dir, exist_ok=True)

    logger = setup_logger("SAG_TTD", log_dir)
    logger.info(f"datasets={datasets}  methods={methods}  seeds={seeds}  "
                f"num_checkpoints={args.num_checkpoints}")
    logger.info(f"run_dir={run_dir}")

    cfg_record = {
        "datasets": datasets, "methods": methods, "seeds": seeds,
        "num_checkpoints": args.num_checkpoints,
        "base_seed": BASE_SEED, "timestamp": ts, "argv": sys.argv,
    }
    save_json(cfg_record, os.path.join(run_dir, "config.json"))

    Config.data_dir = "./data"
    Config.output_dir = args.output_dir

    raw_rows = []
    raw_run = os.path.join(run_dir, "time_to_detect_raw.csv")
    sum_run = os.path.join(run_dir, "time_to_detect_summary.csv")
    raw_canon = os.path.join(canon_dir, "time_to_detect_raw.csv")
    sum_canon = os.path.join(canon_dir, "time_to_detect_summary.csv")

    for ds in datasets:
        try:
            dc = get_dataset_config(ds)
        except ValueError as e:
            logger.error(f"unknown dataset {ds}: {e}")
            continue
        logger.info(f"\n========== dataset = {ds} ==========")
        loader = DataLoader(
            data_dir="./data", dataset_name=ds,
            clique_size=dc.get("clique_size", 15),
            num_cliques=dc.get("num_cliques", 10),
            seed=BASE_SEED,
        )
        try:
            loader.load()
        except Exception as e:
            logger.error(f"  load failed: {e}")
            continue
        s = loader.get_stats()
        logger.info(f"  stats: {s}")
        if s["num_anomalies"] == 0:
            logger.info("  [SKIP] dataset has zero anomalies")
            continue
        ae_cache = _build_ae_cache(loader, dc, logger)
        y = np.asarray(loader.labels)
        anom_nodes = np.where(y == 1)[0]

        for method in methods:
            if method in METHODS_NOT_STREAMING:
                logger.info(f"  [SKIP] {method}: not streaming-compatible "
                            "(global SVD over walk co-occurrence). See "
                            "results/revision/UNIMPLEMENTED.md.")
                raw_rows.append({
                    "dataset": ds, "method": method, "seed": -1,
                    "node": -1, "onset_edge_idx": -1,
                    "detection_edge_idx": -1, "delay_edges": -1,
                    "missed": True,
                    "threshold": float("nan"), "global_macro_f1": float("nan"),
                    "num_edges": s["num_edges"], "num_nodes": s["num_nodes"],
                    "num_anomalies": s["num_anomalies"],
                    "skipped": True,
                    "skip_reason": "method not streaming-compatible",
                })
                continue
            for sd in seeds:
                logger.info(f"  --- {ds} | {method} | seed={sd} ---")
                t0 = time.time()
                try:
                    cps = _make_checkpoints(s["num_edges"], args.num_checkpoints)
                    snaps, final_score, edges_used, order, score_path = (
                        stream_with_checkpoints(method, loader, dc, sd, cps,
                                                ae_cache, logger))
                    onsets = first_appearance_index(edges_used, s["num_nodes"])
                    # Use a single normalization (final-score min/max) for
                    # threshold AND every checkpoint, so the threshold has the
                    # same meaning at any t. Thresholds are computed on the
                    # final streaming-predicate score (state for SAG,
                    # node_scores for MIDAS) -- explicitly NOT the fused
                    # SAG.get_scores(), since the offline attr/iforest branch
                    # is constant from t=0 and would short-circuit the
                    # time-to-detect question.
                    fmin = float(final_score.min())
                    fmax = float(final_score.max())
                    span = (fmax - fmin) if fmax > fmin else 1.0
                    final_norm = (final_score - fmin) / span
                    threshold, f1g = best_f1_threshold(y, final_norm)
                    snaps_norm = {cp: ((sc - fmin) / span) for cp, sc in snaps.items()}
                    detect = detection_indices(snaps_norm, threshold, anom_nodes)
                    INF = np.iinfo(np.int64).max
                    n_detected = 0
                    for n in anom_nodes:
                        n = int(n)
                        onset = int(onsets[n])
                        det_cp = detect.get(n)
                        if onset == INF:
                            raw_rows.append({
                                "dataset": ds, "method": method, "seed": sd,
                                "node": n, "onset_edge_idx": -1,
                                "detection_edge_idx": -1, "delay_edges": -1,
                                "missed": True,
                                "threshold": float(threshold),
                                "global_macro_f1": float(f1g),
                                "num_edges": s["num_edges"],
                                "num_nodes": s["num_nodes"],
                                "num_anomalies": s["num_anomalies"],
                                "skipped": False, "skip_reason": "node never appeared",
                            })
                            continue
                        if det_cp is None:
                            raw_rows.append({
                                "dataset": ds, "method": method, "seed": sd,
                                "node": n, "onset_edge_idx": onset,
                                "detection_edge_idx": -1, "delay_edges": -1,
                                "missed": True,
                                "threshold": float(threshold),
                                "global_macro_f1": float(f1g),
                                "num_edges": s["num_edges"],
                                "num_nodes": s["num_nodes"],
                                "num_anomalies": s["num_anomalies"],
                                "skipped": False, "skip_reason": "",
                            })
                            continue
                        delay = max(0, det_cp - onset)
                        n_detected += 1
                        raw_rows.append({
                            "dataset": ds, "method": method, "seed": sd,
                            "node": n, "onset_edge_idx": onset,
                            "detection_edge_idx": det_cp,
                            "delay_edges": int(delay), "missed": False,
                            "threshold": float(threshold),
                            "global_macro_f1": float(f1g),
                            "num_edges": s["num_edges"],
                            "num_nodes": s["num_nodes"],
                            "num_anomalies": s["num_anomalies"],
                            "skipped": False, "skip_reason": "",
                        })
                    n_total = sum(1 for n in anom_nodes if int(onsets[n]) != INF)
                    logger.info(
                        "    detected %d/%d anomalies, threshold=%.3f F1=%.4f (%.1fs)" %
                        (n_detected, n_total, threshold, f1g, time.time() - t0))
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.error(f"    FAILED {type(e).__name__}: {e}\n{tb}")
                    raw_rows.append({
                        "dataset": ds, "method": method, "seed": sd,
                        "node": -1, "onset_edge_idx": -1,
                        "detection_edge_idx": -1, "delay_edges": -1,
                        "missed": True,
                        "threshold": float("nan"), "global_macro_f1": float("nan"),
                        "num_edges": s["num_edges"], "num_nodes": s["num_nodes"],
                        "num_anomalies": s["num_anomalies"],
                        "skipped": True,
                        "skip_reason": f"{type(e).__name__}:{e}",
                    })
                write_raw_csv(raw_rows, raw_run)

    summary = aggregate_summary(raw_rows)
    write_summary_csv(summary, sum_run)
    save_json({"raw": raw_rows, "summary": summary},
              os.path.join(run_dir, "time_to_detect.json"))

    if not _maybe_skip(raw_canon, args.overwrite, logger):
        write_raw_csv(raw_rows, raw_canon)
    if not _maybe_skip(sum_canon, args.overwrite, logger):
        write_summary_csv(summary, sum_canon)

    if not args.no_plots:
        try:
            plot_delay_distributions(raw_rows, fig_dir)
            logger.info(f"  wrote figures to {fig_dir}")
        except Exception as e:
            logger.warning(f"  plotting failed: {e}")

    logger.info("=" * 70)
    logger.info(f"WROTE  raw     -> {raw_run}")
    logger.info(f"WROTE  summary -> {sum_run}")
    logger.info(f"CANON  raw     -> {raw_canon}")
    logger.info(f"CANON  summary -> {sum_canon}")
    logger.info(f"FIGS           -> {fig_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
