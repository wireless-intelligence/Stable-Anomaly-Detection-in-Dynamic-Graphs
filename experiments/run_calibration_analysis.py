"""Calibration & threshold analysis (revision experiment 12).

For each (dataset, variant, seed) we run SAG-full once, expose the raw
streaming-state predicate, project it into [0,1] via four different
normalizations, and report:

  * ECE   (expected calibration error, 10 equal-mass bins)
  * AUROC, AUPRC                                (sanity)
  * best-F1 threshold + macro-F1 / TPR / FPR at that threshold
  * reliability-bin samples                     (bin -> avg_score, frac_pos)
  * PR-curve samples                            (recall -> precision)

Threshold stability is reported per (dataset, variant) as
  mean / std / min / max / CV  of the per-seed best-F1 threshold.

Variants
--------
  bayesian        : SAG's default fused get_scores()  (rank-based fusion)
  raw             : model.state.scores()               (no normalization)
  minmax          : (raw - min) / (max - min)
  zscore_sigmoid  : (raw - mean)/std passed through sigmoid
  rank            : rank-based [0,1]                   (paper Eq. 4)

Outputs (canonical, latest run; --overwrite to replace):
  outputs/results/revision/calibration_raw.csv          (per seed/variant)
  outputs/results/revision/calibration_bins.csv         (reliability bins)
  outputs/results/revision/calibration_pr.csv           (PR-curve samples)
  outputs/results/revision/calibration_summary.csv      (mean/std)
  outputs/figures/revision/calibration/<dataset>__reliability.{pdf,png}
  outputs/figures/revision/calibration/<dataset>__pr.{pdf,png}
"""

import argparse
import copy
import csv
import os
import sys
import time
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, confusion_matrix,
)

from configs import Config, get_dataset_config
from src.data import DataLoader
from src.models import create_sag_model
from src.utils import save_json, set_seed, setup_logger


DATASETS_DEFAULT = ("Cora", "Citeseer", "Pubmed", "BlogCatalog", "Flickr",
                    "ACM", "Amazon", "YelpChi", "YelpReviews", "Enron")
SEEDS_DEFAULT = (0, 1, 2, 3, 4)
VARIANTS_DEFAULT = ("bayesian", "raw", "minmax", "zscore_sigmoid", "rank")
N_BINS_DEFAULT = 10
PR_SAMPLES_DEFAULT = 50
BASE_SEED = 42


# ---------------------------------------------------------------------------
class CfgShim:
    pass


def build_cfg(dc, seed):
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


def _build_ae_cache(loader, dc, logger):
    if not dc.get("use_trained_ae", False):
        return None
    try:
        from src.models.trained_attr import UnsupervisedGCNAutoencoder
        ae_cfg = dc.get("trained_ae_cfg", {}) or {}
        ae = UnsupervisedGCNAutoencoder(seed=BASE_SEED, **ae_cfg)
        sc = ae.fit_score(loader.adj, loader.features)
        if not np.all(np.isfinite(sc)) or sc.std() < 1e-12:
            return None
        return {"key": (id(loader.features), id(loader.adj),
                        tuple(sorted(ae_cfg.items())), BASE_SEED),
                "score": sc}
    except Exception as e:
        logger.warning(f"  AE pre-training failed: {e}")
        return None


# ---------------------------------------------------------------------------
def run_sag_get_scores(loader, dc, seed, ae_cache):
    set_seed(seed)
    cfg = build_cfg(dc, seed=seed)
    s = loader.get_stats()
    cfg.window_size = max(128, min(s["num_edges"] // 32, 4096))
    if ae_cache is not None:
        cfg._trained_ae_score_cache = ae_cache
    model = create_sag_model(num_nodes=s["num_nodes"], config=cfg,
                             features=loader.features, adj=loader.adj)
    order = getattr(cfg, "edge_order", "auto")
    if order == "auto":
        order = "shuffle" if cfg.has_ground_truth else "bfs"
    edges = loader.get_edge_stream(mode=order)
    do_shuffle = (order == "shuffle")
    passes = (2 if s["num_edges"] > 50000 else 3) if do_shuffle else 1
    rng = np.random.default_rng(seed)
    model.process_stream_batch(edges, passes=passes, shuffle=do_shuffle, rng=rng)
    raw = np.asarray(model.state.scores(), dtype=np.float64)
    bayesian = np.asarray(model.get_scores(), dtype=np.float64)
    return raw, bayesian


# ---------------------------------------------------------------------------
def project(raw_state: np.ndarray, bayesian: np.ndarray, variant: str) -> np.ndarray:
    if variant == "bayesian":
        s = bayesian
        if s.max() > s.min():
            return (s - s.min()) / (s.max() - s.min())
        return np.zeros_like(s)
    if variant == "raw":
        # already a real-valued raw score; min-max it ONLY at the very end so
        # ECE has a [0,1] interpretation (otherwise ECE is meaningless on
        # an unbounded scale). Note: 'raw' is the same projection as 'minmax'
        # but we keep it as a separate row for the user's requested set.
        s = raw_state
        if s.max() > s.min():
            return (s - s.min()) / (s.max() - s.min())
        return np.zeros_like(s)
    if variant == "minmax":
        s = raw_state
        if s.max() > s.min():
            return (s - s.min()) / (s.max() - s.min())
        return np.zeros_like(s)
    if variant == "zscore_sigmoid":
        s = raw_state
        mu, sd = float(s.mean()), float(s.std())
        if sd < 1e-12:
            return np.zeros_like(s)
        z = (s - mu) / sd
        return 1.0 / (1.0 + np.exp(-z))
    if variant == "rank":
        order = np.argsort(raw_state, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(raw_state.shape[0], dtype=np.float64)
        if raw_state.shape[0] > 1:
            ranks /= (raw_state.shape[0] - 1)
        return ranks
    raise ValueError(f"unknown variant: {variant}")


def expected_calibration_error(y, p, n_bins=N_BINS_DEFAULT):
    """Equal-mass binning ECE."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if p.shape[0] == 0:
        return float("nan"), []
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    rows = []
    ece = 0.0
    N = p.shape[0]
    for b_idx, idx in enumerate(bins):
        if idx.size == 0:
            continue
        avg_p = float(p[idx].mean())
        frac_y = float(y[idx].mean())
        weight = idx.size / N
        ece += weight * abs(avg_p - frac_y)
        rows.append({
            "bin": b_idx, "n": int(idx.size),
            "avg_score": avg_p, "frac_pos": frac_y,
            "score_min": float(p[idx].min()), "score_max": float(p[idx].max()),
        })
    return float(ece), rows


def best_f1_threshold(y, p):
    if p.max() <= p.min():
        return 0.5, 0.0
    th = np.linspace(p.min(), p.max(), 200)
    best_t, best_f = float(th[0]), 0.0
    for t in th:
        yp = (p >= t).astype(int)
        f = f1_score(y, yp, average="macro", zero_division=0)
        if f > best_f:
            best_f = f
            best_t = float(t)
    return best_t, float(best_f)


def tpr_fpr_at(y, p, t):
    yp = (p >= t).astype(int)
    cm = confusion_matrix(y, yp, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    return float(tpr), float(fpr), tp, fp, fn, tn


def pr_samples(y, p, n_samples=PR_SAMPLES_DEFAULT):
    pr, rc, _ = precision_recall_curve(y, p)
    targets = np.linspace(0.0, 1.0, n_samples)
    out = []
    pr = np.asarray(pr); rc = np.asarray(rc)
    order = np.argsort(rc)
    rc_s = rc[order]; pr_s = pr[order]
    for t in targets:
        i = int(np.searchsorted(rc_s, t, side="left"))
        if i >= len(rc_s): i = len(rc_s) - 1
        out.append({"recall_target": float(t),
                    "recall_actual": float(rc_s[i]),
                    "precision": float(pr_s[i])})
    return out


# ---------------------------------------------------------------------------
RAW_COLS = [
    "dataset", "variant", "seed",
    "auroc", "auprc", "ece", "best_f1_threshold", "best_macro_f1",
    "tpr_at_thr", "fpr_at_thr",
    "tp", "fp", "fn", "tn",
    "score_min", "score_max", "score_mean", "score_std",
    "num_nodes", "num_anomalies",
    "skipped", "error",
]
BIN_COLS = [
    "dataset", "variant", "seed", "bin",
    "n", "avg_score", "frac_pos", "score_min", "score_max",
]
PR_COLS = [
    "dataset", "variant", "seed",
    "recall_target", "recall_actual", "precision",
]
SUMMARY_COLS = [
    "dataset", "variant", "n_seeds",
    "auroc_mean", "auroc_std",
    "auprc_mean", "auprc_std",
    "ece_mean", "ece_std",
    "thr_mean", "thr_std", "thr_min", "thr_max", "thr_cv",
    "best_macro_f1_mean", "best_macro_f1_std",
    "tpr_mean", "tpr_std", "fpr_mean", "fpr_std",
]


def write_csv(rows, cols, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def aggregate_summary(raw_rows):
    grouped = {}
    for r in raw_rows:
        if r.get("skipped"): continue
        grouped.setdefault((r["dataset"], r["variant"]), []).append(r)
    out = []
    for (ds, var), group in grouped.items():
        n = len(group)
        def arr(k): return np.array([float(g[k]) for g in group], dtype=np.float64)
        thr = arr("best_f1_threshold")
        thr_cv = float(thr.std(ddof=1) / abs(thr.mean())) if (n > 1 and abs(thr.mean()) > 1e-12) else 0.0
        out.append({
            "dataset": ds, "variant": var, "n_seeds": n,
            "auroc_mean": float(arr("auroc").mean()),
            "auroc_std":  float(arr("auroc").std(ddof=1)) if n > 1 else 0.0,
            "auprc_mean": float(arr("auprc").mean()),
            "auprc_std":  float(arr("auprc").std(ddof=1)) if n > 1 else 0.0,
            "ece_mean":   float(arr("ece").mean()),
            "ece_std":    float(arr("ece").std(ddof=1)) if n > 1 else 0.0,
            "thr_mean":   float(thr.mean()),
            "thr_std":    float(thr.std(ddof=1)) if n > 1 else 0.0,
            "thr_min":    float(thr.min()),
            "thr_max":    float(thr.max()),
            "thr_cv":     thr_cv,
            "best_macro_f1_mean": float(arr("best_macro_f1").mean()),
            "best_macro_f1_std":  float(arr("best_macro_f1").std(ddof=1)) if n > 1 else 0.0,
            "tpr_mean": float(arr("tpr_at_thr").mean()),
            "tpr_std":  float(arr("tpr_at_thr").std(ddof=1)) if n > 1 else 0.0,
            "fpr_mean": float(arr("fpr_at_thr").mean()),
            "fpr_std":  float(arr("fpr_at_thr").std(ddof=1)) if n > 1 else 0.0,
        })
    out.sort(key=lambda r: (r["dataset"], r["variant"]))
    return out


# ---------------------------------------------------------------------------
def plot_dataset(dataset, bin_rows, pr_rows, fig_dir):
    import matplotlib.pyplot as plt
    palette = {"bayesian": "#e74c3c", "raw": "#3498db",
               "minmax": "#2ca02c", "zscore_sigmoid": "#f39c12",
               "rank": "#9b59b6"}
    os.makedirs(fig_dir, exist_ok=True)

    # Reliability diagram (mean across seeds per bin)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1.2,
            label="perfect")
    by_var = {}
    for r in bin_rows:
        if r["dataset"] != dataset: continue
        by_var.setdefault(r["variant"], []).append(r)
    for var, rows in by_var.items():
        bins = sorted({int(r["bin"]) for r in rows})
        xs, ys = [], []
        for b in bins:
            sub = [r for r in rows if int(r["bin"]) == b]
            if not sub: continue
            xs.append(float(np.mean([float(r["avg_score"]) for r in sub])))
            ys.append(float(np.mean([float(r["frac_pos"]) for r in sub])))
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.6,
                color=palette.get(var, "#888"), label=var)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("predicted score (bin avg)")
    ax.set_ylabel("empirical anomaly fraction")
    ax.grid(True, axis="both", linestyle="--", alpha=0.45, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=9, frameon=True, edgecolor="0.7", fancybox=False)
    ax.set_title(f"Reliability diagram — {dataset}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    base = os.path.join(fig_dir, f"{dataset}__reliability")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # PR curve (mean across seeds per recall target)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    by_var = {}
    for r in pr_rows:
        if r["dataset"] != dataset: continue
        by_var.setdefault(r["variant"], []).append(r)
    for var, rows in by_var.items():
        targets = sorted({float(r["recall_target"]) for r in rows})
        xs, ys = [], []
        for t in targets:
            sub = [r for r in rows if abs(float(r["recall_target"]) - t) < 1e-9]
            if not sub: continue
            xs.append(t)
            ys.append(float(np.mean([float(r["precision"]) for r in sub])))
        ax.plot(xs, ys, linewidth=1.7, color=palette.get(var, "#888"), label=var)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.grid(True, axis="both", linestyle="--", alpha=0.45, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=9, frameon=True, edgecolor="0.7", fancybox=False)
    ax.set_title(f"PR curve — {dataset}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    base = os.path.join(fig_dir, f"{dataset}__pr")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(DATASETS_DEFAULT))
    p.add_argument("--variants", default=",".join(VARIANTS_DEFAULT))
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS_DEFAULT))
    p.add_argument("--n_bins", type=int, default=N_BINS_DEFAULT)
    p.add_argument("--pr_samples", type=int, default=PR_SAMPLES_DEFAULT)
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--tag", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args()


def _maybe_skip(path, overwrite, logger):
    if os.path.exists(path) and not overwrite:
        logger.warning(f"  refusing to overwrite {path} (pass --overwrite)")
        return True
    return False


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    run_dir = os.path.join(args.output_dir, "results", "revision",
                           f"calibration{tag}_{ts}")
    log_dir = os.path.join(args.output_dir, "logs", f"calibration{tag}_{ts}")
    fig_dir = os.path.join(args.output_dir, "figures", "revision", "calibration")
    canon_dir = os.path.join(args.output_dir, "results", "revision")
    os.makedirs(run_dir, exist_ok=True); os.makedirs(log_dir, exist_ok=True)
    os.makedirs(canon_dir, exist_ok=True)

    logger = setup_logger("SAG_CALIB", log_dir)
    logger.info(f"datasets={datasets}  variants={variants}  seeds={seeds}")
    save_json({"datasets": datasets, "variants": variants, "seeds": seeds,
               "n_bins": args.n_bins, "pr_samples": args.pr_samples,
               "argv": sys.argv, "timestamp": ts},
              os.path.join(run_dir, "config.json"))

    Config.data_dir = "./data"
    Config.output_dir = args.output_dir

    raw_rows, bin_rows, pr_rows = [], [], []
    raw_run = os.path.join(run_dir, "calibration_raw.csv")
    bins_run = os.path.join(run_dir, "calibration_bins.csv")
    pr_run = os.path.join(run_dir, "calibration_pr.csv")
    sum_run = os.path.join(run_dir, "calibration_summary.csv")
    raw_canon = os.path.join(canon_dir, "calibration_raw.csv")
    bins_canon = os.path.join(canon_dir, "calibration_bins.csv")
    pr_canon = os.path.join(canon_dir, "calibration_pr.csv")
    sum_canon = os.path.join(canon_dir, "calibration_summary.csv")

    for ds in datasets:
        try:
            dc = get_dataset_config(ds)
        except ValueError as e:
            logger.error(f"unknown dataset {ds}: {e}")
            continue
        logger.info(f"\n========== {ds} ==========")
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
        ae_cache = _build_ae_cache(loader, dc, logger)
        y = np.asarray(loader.labels)
        s = loader.get_stats()

        for sd in seeds:
            try:
                t0 = time.time()
                raw, bayesian = run_sag_get_scores(loader, dc, sd, ae_cache)
                logger.info(f"  seed={sd}: SAG run in {time.time()-t0:.1f}s")
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"  FAILED {ds} seed={sd}: {e}\n{tb}")
                for v in variants:
                    raw_rows.append({
                        "dataset": ds, "variant": v, "seed": sd,
                        "auroc": float("nan"), "auprc": float("nan"),
                        "ece": float("nan"), "best_f1_threshold": float("nan"),
                        "best_macro_f1": float("nan"),
                        "tpr_at_thr": float("nan"), "fpr_at_thr": float("nan"),
                        "tp": 0, "fp": 0, "fn": 0, "tn": 0,
                        "score_min": float("nan"), "score_max": float("nan"),
                        "score_mean": float("nan"), "score_std": float("nan"),
                        "num_nodes": int(s["num_nodes"]),
                        "num_anomalies": int(s["num_anomalies"]),
                        "skipped": True,
                        "error": f"{type(e).__name__}:{e}",
                    })
                continue

            for variant in variants:
                try:
                    p = project(raw, bayesian, variant)
                    auroc = float(roc_auc_score(y, p)) if y.sum() not in (0, len(y)) else float("nan")
                    auprc = float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")
                    ece, b_rows = expected_calibration_error(y, p, args.n_bins)
                    thr, bf1 = best_f1_threshold(y, p)
                    tpr, fpr, tp, fp, fn, tn = tpr_fpr_at(y, p, thr)
                    raw_rows.append({
                        "dataset": ds, "variant": variant, "seed": sd,
                        "auroc": auroc, "auprc": auprc, "ece": ece,
                        "best_f1_threshold": thr, "best_macro_f1": bf1,
                        "tpr_at_thr": tpr, "fpr_at_thr": fpr,
                        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                        "score_min": float(p.min()), "score_max": float(p.max()),
                        "score_mean": float(p.mean()), "score_std": float(p.std()),
                        "num_nodes": int(s["num_nodes"]),
                        "num_anomalies": int(s["num_anomalies"]),
                        "skipped": False, "error": "",
                    })
                    for br in b_rows:
                        bin_rows.append({"dataset": ds, "variant": variant,
                                         "seed": sd, **br})
                    for pr in pr_samples(y, p, args.pr_samples):
                        pr_rows.append({"dataset": ds, "variant": variant,
                                        "seed": sd, **pr})
                    logger.info(
                        "    %s ECE=%.4f thr=%.3f F1=%.4f TPR=%.3f FPR=%.3f" %
                        (variant, ece, thr, bf1, tpr, fpr))
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.error(f"    FAILED variant={variant}: {e}\n{tb}")
                    raw_rows.append({
                        "dataset": ds, "variant": variant, "seed": sd,
                        "auroc": float("nan"), "auprc": float("nan"),
                        "ece": float("nan"), "best_f1_threshold": float("nan"),
                        "best_macro_f1": float("nan"),
                        "tpr_at_thr": float("nan"), "fpr_at_thr": float("nan"),
                        "tp": 0, "fp": 0, "fn": 0, "tn": 0,
                        "score_min": float("nan"), "score_max": float("nan"),
                        "score_mean": float("nan"), "score_std": float("nan"),
                        "num_nodes": int(s["num_nodes"]),
                        "num_anomalies": int(s["num_anomalies"]),
                        "skipped": True, "error": f"{type(e).__name__}:{e}",
                    })
            write_csv(raw_rows, RAW_COLS, raw_run)
            write_csv(bin_rows, BIN_COLS, bins_run)
            write_csv(pr_rows, PR_COLS, pr_run)

    summary = aggregate_summary(raw_rows)
    write_csv(summary, SUMMARY_COLS, sum_run)
    save_json({"raw": raw_rows, "bins": bin_rows, "pr": pr_rows,
               "summary": summary},
              os.path.join(run_dir, "calibration.json"))

    if not _maybe_skip(raw_canon, args.overwrite, logger):
        write_csv(raw_rows, RAW_COLS, raw_canon)
    if not _maybe_skip(bins_canon, args.overwrite, logger):
        write_csv(bin_rows, BIN_COLS, bins_canon)
    if not _maybe_skip(pr_canon, args.overwrite, logger):
        write_csv(pr_rows, PR_COLS, pr_canon)
    if not _maybe_skip(sum_canon, args.overwrite, logger):
        write_csv(summary, SUMMARY_COLS, sum_canon)

    if not args.no_plots:
        for ds in {r["dataset"] for r in raw_rows if not r.get("skipped")}:
            try:
                plot_dataset(ds, bin_rows, pr_rows, fig_dir)
            except Exception as e:
                logger.warning(f"  plotting failed for {ds}: {e}")

    logger.info("=" * 70)
    logger.info(f"WROTE  raw     -> {raw_run}")
    logger.info(f"WROTE  bins    -> {bins_run}")
    logger.info(f"WROTE  pr      -> {pr_run}")
    logger.info(f"WROTE  summary -> {sum_run}")
    logger.info(f"CANON          -> {canon_dir}")
    logger.info(f"FIGS           -> {fig_dir}")


if __name__ == "__main__":
    main()
