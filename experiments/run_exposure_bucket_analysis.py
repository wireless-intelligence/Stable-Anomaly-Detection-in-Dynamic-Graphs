"""Degree / exposure bucket analysis (revision).

Splits nodes into degree quartiles (Q1..Q4 = 0-25/25-50/50-75/75-100%) and
reports per-bucket detection performance for SAG variants and (when
available) baselines. Directly tests whether exposure scaling mitigates
structural imbalance: SAG-full vs SAG-no-exposure across buckets.

Per-bucket metrics:
  AUROC, AUPRC, Macro-F1 (at the dataset's global best-F1 threshold), FPR
  at that threshold, anomaly_ratio, n_nodes, n_anomalies.

All scores come from the same single streaming pass per (dataset, method,
seed); the ONLY thing the bucket index changes is which subset of nodes
the metric is computed over.

Outputs:
  outputs/results/revision/exposure_bucket_raw.csv
  outputs/results/revision/exposure_bucket_summary.csv
  outputs/figures/revision/exposure_bucket/<dataset>__<metric>.{pdf,png}
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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix

from configs import Config, get_dataset_config
from src.baselines import BASELINES
from src.data import DataLoader
from src.models import create_sag_model
from src.utils import save_json, set_seed, setup_logger


METHODS_DEFAULT = ("SAG-full", "SAG-no-exposure", "MIDAS", "NetWalk", "GraphSAGE")
DATASETS_DEFAULT = ("YelpReviews", "Enron", "BlogCatalog")
SEEDS_DEFAULT = (0, 1, 2)
BASE_SEED = 42
BUCKET_LABELS = ("Q1", "Q2", "Q3", "Q4")  # 0-25, 25-50, 50-75, 75-100


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


def run_method(method: str, loader, dc: dict, seed: int, ae_cache, logger) -> np.ndarray:
    """Run one method, return per-node anomaly scores (np.ndarray length=N)."""
    set_seed(seed)
    cfg = build_cfg(dc, seed=seed)
    s = loader.get_stats()
    cfg.window_size = max(128, min(s["num_edges"] // 32, 4096))
    if ae_cache is not None:
        cfg._trained_ae_score_cache = ae_cache

    if method == "SAG-full":
        model = create_sag_model(num_nodes=s["num_nodes"], config=cfg,
                                 features=loader.features, adj=loader.adj)
        ablation = None
    elif method == "SAG-no-exposure":
        model = create_sag_model(num_nodes=s["num_nodes"], config=cfg,
                                 features=loader.features, adj=loader.adj,
                                 ablation="no_exposure")
        ablation = "no_exposure"
    elif method in BASELINES:
        order = getattr(cfg, "edge_order", "auto")
        if order == "auto":
            order = "shuffle" if cfg.has_ground_truth else "bfs"
        edges = loader.get_edge_stream(mode=order)
        baseline_cls = BASELINES[method]
        try:
            bm = baseline_cls(num_nodes=s["num_nodes"], config=cfg,
                              features=loader.features, seed=seed)
        except TypeError:
            bm = baseline_cls(num_nodes=s["num_nodes"], config=cfg,
                              features=loader.features)
        bm.fit(loader.adj, edges, loader.labels)
        scores = bm.score().astype(np.float64)
        if scores.shape[0] != s["num_nodes"]:
            raise RuntimeError(
                f"baseline {method} returned {scores.shape[0]} scores, "
                f"expected {s['num_nodes']}")
        return scores
    else:
        raise ValueError(f"unknown method: {method}")

    order = getattr(cfg, "edge_order", "auto")
    if order == "auto":
        order = "shuffle" if cfg.has_ground_truth else "bfs"
    edges = loader.get_edge_stream(mode=order)
    do_shuffle = (order == "shuffle")
    passes = (2 if s["num_edges"] > 50000 else 3) if do_shuffle else 1
    rng = np.random.default_rng(seed)
    model.process_stream_batch(edges, passes=passes, shuffle=do_shuffle, rng=rng)
    scores = model.get_scores()
    return scores.astype(np.float64)


# ---------------------------------------------------------------------------
def degree_buckets(adj) -> tuple:
    """Return (deg, bucket_idx) where bucket_idx[i] in {0,1,2,3} corresponds
    to Q1..Q4 by node degree (lower degree => Q1)."""
    A = sp.csr_matrix(adj)
    deg = np.asarray(A.sum(axis=1)).flatten()
    qs = np.quantile(deg, [0.25, 0.50, 0.75])
    b = np.zeros(deg.shape[0], dtype=np.int8)
    b[deg > qs[0]] = 1
    b[deg > qs[1]] = 2
    b[deg > qs[2]] = 3
    return deg, b


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray):
    """Returns (best_threshold, best_f1)."""
    if y_score.max() <= y_score.min():
        return 0.5, 0.0
    thresholds = np.linspace(y_score.min(), y_score.max(), 200)
    best_t, best_f = 0.5, 0.0
    for t in thresholds:
        yp = (y_score >= t).astype(int)
        f = f1_score(y_true, yp, average="macro", zero_division=0)
        if f > best_f:
            best_f = f
            best_t = float(t)
    return best_t, float(best_f)


def per_bucket_metrics(y_true, y_score, buckets, threshold):
    out = []
    for q, lab in enumerate(BUCKET_LABELS):
        mask = (buckets == q)
        n_nodes = int(mask.sum())
        n_anom = int(np.sum(y_true[mask]))
        if n_nodes == 0:
            out.append({
                "bucket": lab, "n_nodes": 0, "n_anomalies": 0,
                "anomaly_ratio": 0.0,
                "auroc": float("nan"), "auprc": float("nan"),
                "macro_f1": float("nan"), "fpr": float("nan"),
            })
            continue
        ys, ss = y_true[mask], y_score[mask]
        anomaly_ratio = float(n_anom) / float(n_nodes)
        if n_anom == 0 or n_anom == n_nodes:
            auroc = float("nan"); auprc = float("nan")
        else:
            try:
                auroc = float(roc_auc_score(ys, ss))
            except ValueError:
                auroc = float("nan")
            try:
                auprc = float(average_precision_score(ys, ss))
            except ValueError:
                auprc = float("nan")
        yp = (ss >= threshold).astype(int)
        try:
            macro_f1 = float(f1_score(ys, yp, average="macro", zero_division=0))
        except ValueError:
            macro_f1 = float("nan")
        if (n_nodes - n_anom) > 0:
            tn = int(np.sum((yp == 0) & (ys == 0)))
            fp = int(np.sum((yp == 1) & (ys == 0)))
            fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else float("nan")
        else:
            fpr = float("nan")
        out.append({
            "bucket": lab, "n_nodes": n_nodes, "n_anomalies": n_anom,
            "anomaly_ratio": anomaly_ratio,
            "auroc": auroc, "auprc": auprc,
            "macro_f1": macro_f1, "fpr": fpr,
        })
    return out


# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    "dataset", "method", "seed", "bucket",
    "n_nodes", "n_anomalies", "anomaly_ratio",
    "auroc", "auprc", "macro_f1", "fpr",
    "global_threshold", "global_macro_f1",
    "deg_q25", "deg_q50", "deg_q75", "deg_max",
    "skipped", "error",
]
SUMMARY_METRICS = ("auroc", "auprc", "macro_f1", "fpr",
                   "anomaly_ratio", "n_nodes", "n_anomalies")
SUMMARY_COLUMNS = (
    ["dataset", "method", "bucket", "n_seeds"]
    + [f"{m}_mean" for m in SUMMARY_METRICS]
    + [f"{m}_std" for m in SUMMARY_METRICS]
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
        if r.get("skipped"): continue
        grouped.setdefault((r["dataset"], r["method"], r["bucket"]), []).append(r)
    out = []
    for (ds, mt, bk), group in grouped.items():
        row = {"dataset": ds, "method": mt, "bucket": bk, "n_seeds": len(group)}
        for m in SUMMARY_METRICS:
            arr = np.array([g[m] for g in group], dtype=np.float64)
            row[f"{m}_mean"] = float(np.nanmean(arr)) if arr.size else float("nan")
            row[f"{m}_std"] = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
        out.append(row)
    bucket_idx = {b: i for i, b in enumerate(BUCKET_LABELS)}
    out.sort(key=lambda r: (r["dataset"], r["method"], bucket_idx.get(r["bucket"], 99)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        for r in out:
            w.writerow(r)
    return out


# ---------------------------------------------------------------------------
def plot_buckets(summary, fig_dir):
    import matplotlib.pyplot as plt
    datasets = sorted({r["dataset"] for r in summary})
    methods = sorted({r["method"] for r in summary})
    palette = ["#e74c3c", "#3498db", "#2ca02c", "#f39c12", "#9b59b6", "#16a085", "#e67e22"]
    method_color = {m: palette[i % len(palette)] for i, m in enumerate(methods)}
    metrics = [("auroc_mean", "auroc_std", "AUROC"),
               ("macro_f1_mean", "macro_f1_std", "Macro-F1"),
               ("fpr_mean", "fpr_std", "FPR")]
    os.makedirs(fig_dir, exist_ok=True)
    bucket_idx = {b: i for i, b in enumerate(BUCKET_LABELS)}
    for ds in datasets:
        for mk, sk, ylabel in metrics:
            fig, ax = plt.subplots(figsize=(7.0, 3.6))
            for m in methods:
                rows = sorted([r for r in summary
                               if r["dataset"] == ds and r["method"] == m],
                              key=lambda r: bucket_idx.get(r["bucket"], 99))
                if not rows: continue
                xs = [bucket_idx[r["bucket"]] for r in rows]
                ys = [r[mk] for r in rows]
                es = [r[sk] for r in rows]
                ax.errorbar(xs, ys, yerr=es,
                            marker="o", markersize=5, linewidth=1.6, capsize=3,
                            color=method_color[m], label=m)
            ax.set_xticks(list(range(len(BUCKET_LABELS))))
            ax.set_xticklabels(BUCKET_LABELS)
            ax.set_xlabel("Degree quartile (Q1=lowest, Q4=highest)")
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", linestyle="--", alpha=0.45, linewidth=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(loc="best", frameon=True, edgecolor="0.7",
                      fancybox=False, fontsize=9)
            ax.set_title(f"{ds} — {ylabel} per degree bucket",
                         fontsize=12, fontweight="bold")
            fig.tight_layout()
            base = os.path.join(fig_dir, f"{ds}__{ylabel.replace(' ', '_')}")
            fig.savefig(base + ".pdf", bbox_inches="tight")
            fig.savefig(base + ".png", dpi=160, bbox_inches="tight")
            plt.close(fig)


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(DATASETS_DEFAULT))
    p.add_argument("--methods", default=",".join(METHODS_DEFAULT))
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS_DEFAULT))
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--tag", default=None)
    p.add_argument("--overwrite", action="store_true",
                   help="overwrite canonical CSV/figure files if present")
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
        logger.info("  pre-training AE (one-time)...")
        t0 = time.time()
        ae = UnsupervisedGCNAutoencoder(seed=BASE_SEED, **ae_cfg)
        sc = ae.fit_score(loader.adj, loader.features)
        if not np.all(np.isfinite(sc)) or sc.std() < 1e-12:
            return None
        logger.info(f"  AE trained in {time.time()-t0:.1f}s")
        return {"key": cache_key, "score": sc}
    except Exception as e:
        logger.warning(f"  AE pre-training failed: {e}")
        return None


def _maybe_skip(path: str, overwrite: bool, logger) -> bool:
    if os.path.exists(path) and not overwrite:
        logger.warning(f"  refusing to overwrite {path} (pass --overwrite to replace)")
        return True
    return False


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    run_dir = os.path.join(args.output_dir, "results", "revision",
                           f"exposure_bucket{tag}_{ts}")
    log_dir = os.path.join(args.output_dir, "logs",
                           f"exposure_bucket{tag}_{ts}")
    fig_dir = os.path.join(args.output_dir, "figures", "revision",
                           "exposure_bucket")
    canon_dir = os.path.join(args.output_dir, "results", "revision")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(canon_dir, exist_ok=True)

    logger = setup_logger("SAG_EXP_BUCKET", log_dir)
    logger.info(f"datasets={datasets}  methods={methods}  seeds={seeds}")
    logger.info(f"run_dir={run_dir}")

    Config.data_dir = "./data"
    Config.output_dir = args.output_dir

    raw_rows = []
    raw_run = os.path.join(run_dir, "exposure_bucket_raw.csv")
    sum_run = os.path.join(run_dir, "exposure_bucket_summary.csv")
    raw_canon = os.path.join(canon_dir, "exposure_bucket_raw.csv")
    sum_canon = os.path.join(canon_dir, "exposure_bucket_summary.csv")

    cfg_record = {
        "datasets": datasets, "methods": methods, "seeds": seeds,
        "base_seed": BASE_SEED, "bucket_labels": list(BUCKET_LABELS),
        "timestamp": ts, "argv": sys.argv,
    }
    save_json(cfg_record, os.path.join(run_dir, "config.json"))

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
        logger.info(f"  stats: {loader.get_stats()}")
        deg, buckets = degree_buckets(loader.adj)
        deg_q25 = float(np.quantile(deg, 0.25))
        deg_q50 = float(np.quantile(deg, 0.50))
        deg_q75 = float(np.quantile(deg, 0.75))
        deg_max = float(deg.max())
        ae_cache = _build_ae_cache(loader, dc, logger)
        y = np.asarray(loader.labels)

        for method in methods:
            if (method not in ("SAG-full", "SAG-no-exposure")
                    and method not in BASELINES):
                logger.info(f"  [SKIP] {method}: not registered in BASELINES")
                for sd in seeds:
                    for lab in BUCKET_LABELS:
                        raw_rows.append({
                            "dataset": ds, "method": method, "seed": sd,
                            "bucket": lab, "n_nodes": 0, "n_anomalies": 0,
                            "anomaly_ratio": float("nan"),
                            "auroc": float("nan"), "auprc": float("nan"),
                            "macro_f1": float("nan"), "fpr": float("nan"),
                            "global_threshold": float("nan"),
                            "global_macro_f1": float("nan"),
                            "deg_q25": deg_q25, "deg_q50": deg_q50,
                            "deg_q75": deg_q75, "deg_max": deg_max,
                            "skipped": True,
                            "error": "method not implemented in this codebase",
                        })
                continue
            for sd in seeds:
                logger.info(f"  --- {ds} | {method} | seed={sd} ---")
                try:
                    t0 = time.time()
                    scores = run_method(method, loader, dc, sd, ae_cache, logger)
                    if scores.max() > scores.min():
                        s_norm = (scores - scores.min()) / (scores.max() - scores.min())
                    else:
                        s_norm = scores
                    thr, f1g = best_f1_threshold(y, s_norm)
                    bucket_results = per_bucket_metrics(y, s_norm, buckets, thr)
                    logger.info(
                        "    global F1=%.4f thr=%.3f  (%.1fs)" %
                        (f1g, thr, time.time() - t0))
                    for br in bucket_results:
                        raw_rows.append({
                            "dataset": ds, "method": method, "seed": sd,
                            **br,
                            "global_threshold": thr,
                            "global_macro_f1": f1g,
                            "deg_q25": deg_q25, "deg_q50": deg_q50,
                            "deg_q75": deg_q75, "deg_max": deg_max,
                            "skipped": False, "error": "",
                        })
                        logger.info(
                            "      %s n=%-7d ratio=%.3f AUROC=%.4f F1=%.4f FPR=%.4f" %
                            (br["bucket"], br["n_nodes"], br["anomaly_ratio"],
                             br["auroc"] if br["auroc"] == br["auroc"] else float("nan"),
                             br["macro_f1"], br["fpr"] if br["fpr"] == br["fpr"] else float("nan"))
                        )
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.error(f"    FAILED {type(e).__name__}: {e}\n{tb}")
                    for lab in BUCKET_LABELS:
                        raw_rows.append({
                            "dataset": ds, "method": method, "seed": sd,
                            "bucket": lab, "n_nodes": 0, "n_anomalies": 0,
                            "anomaly_ratio": float("nan"),
                            "auroc": float("nan"), "auprc": float("nan"),
                            "macro_f1": float("nan"), "fpr": float("nan"),
                            "global_threshold": float("nan"),
                            "global_macro_f1": float("nan"),
                            "deg_q25": deg_q25, "deg_q50": deg_q50,
                            "deg_q75": deg_q75, "deg_max": deg_max,
                            "skipped": True,
                            "error": f"{type(e).__name__}:{e}",
                        })
                write_raw_csv(raw_rows, raw_run)

    summary = write_summary_csv(raw_rows, sum_run)
    save_json({"raw": raw_rows, "summary": summary},
              os.path.join(run_dir, "exposure_bucket.json"))

    if not _maybe_skip(raw_canon, args.overwrite, logger):
        write_raw_csv(raw_rows, raw_canon)
    if not _maybe_skip(sum_canon, args.overwrite, logger):
        write_summary_csv(raw_rows, sum_canon)

    if not args.no_plots:
        try:
            plot_buckets(summary, fig_dir)
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
