"""SAG one-parameter-at-a-time sensitivity sweeps (revision).

For each (dataset, parameter, value, seed) we run SAG once with that single
parameter overridden away from the dataset's default while every other config
field stays at its native value.  Variance comes from re-seeded edge-stream
RNG only; randomness in the rest of the pipeline is held fixed.

Parameters swept (paper symbol --> cfg field):
  alpha             alpha
  gamma             gamma
  R                 kernel_dim
  delta             delta
  rho               rho
  P_A               prior_anomaly_rate
  P_a_given_A       likelihood_anomaly
  P_a_given_not_A   likelihood_normal
  beta_upd          beta_upd

Outputs (canonical, latest run):
  outputs/results/revision/parameter_sensitivity_raw.csv
  outputs/results/revision/parameter_sensitivity_summary.csv
  outputs/figures/revision/parameter_sensitivity/<param>.pdf  (and .png)

Per-run timestamped copy under outputs/results/revision/parameter_sensitivity_<ts>/.
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

from configs import Config, get_dataset_config
from src.data import DataLoader
from src.evaluation.metrics import (
    compute_auroc, compute_auprc, compute_macro_f1, compute_balanced_auprc_f1,
)
from src.models import create_sag_model
from src.utils import MemoryTracker, Timer, save_json, set_seed, setup_logger


# ---------------------------------------------------------------------------
# Parameter catalogue
# ---------------------------------------------------------------------------
PARAM_SWEEPS = {
    "alpha":             [0.25, 0.5, 1.0, 2.0, 4.0],
    "gamma":             [0.80, 0.90, 0.95, 0.98, 0.99],
    "R":                 [1, 2, 4, 8, 16],
    "delta":             [0.1, 0.5, 1.0, 2.0],
    "rho":               [1.0, 2.0, 3.0, 4.0],
    "P_A":               [0.01, 0.05, 0.10, 0.20],
    "P_a_given_A":       [0.7, 0.8, 0.9],
    "P_a_given_not_A":   [0.05, 0.10, 0.20],
    "beta_upd":          [0.25, 0.5, 1.0, 2.0],
}
PARAM_TO_CFG = {
    "alpha":           "alpha",
    "gamma":           "gamma",
    "R":               "kernel_dim",
    "delta":           "delta",
    "rho":             "rho",
    "P_A":             "prior_anomaly_rate",
    "P_a_given_A":     "likelihood_anomaly",
    "P_a_given_not_A": "likelihood_normal",
    "beta_upd":        "beta_upd",
}
PARAM_DISPLAY = {
    "alpha":           r"$\alpha$",
    "gamma":           r"$\gamma$",
    "R":               r"$R$ (kernel dim)",
    "delta":           r"$\delta$",
    "rho":             r"$\rho$",
    "P_A":             r"$P(A)$",
    "P_a_given_A":     r"$P(a|A)$",
    "P_a_given_not_A": r"$P(a|\neg A)$",
    "beta_upd":        r"$\beta_{\mathrm{upd}}$",
}

DEFAULT_DATASETS = ("YelpReviews", "Enron", "Cora", "BlogCatalog")
DEFAULT_SEEDS = (0, 1, 2)
BASE_SEED = 42


# ---------------------------------------------------------------------------
# Per-run config shim
# ---------------------------------------------------------------------------
class CfgShim:
    pass


def build_cfg(dc: dict, seed: int, sweep_param: str, value):
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
    cfg_field = PARAM_TO_CFG[sweep_param]
    if sweep_param == "R":
        setattr(cfg, cfg_field, int(value))
        cfg.kernel_lambdas = None
    else:
        setattr(cfg, cfg_field, float(value))
    return cfg


# ---------------------------------------------------------------------------
# Single-seed run with latency / memory capture
# ---------------------------------------------------------------------------
def run_one(dataset: str, sweep_param: str, value, seed: int, dc: dict,
            loader, ae_cache, logger):
    set_seed(seed)
    cfg = build_cfg(dc, seed=seed, sweep_param=sweep_param, value=value)
    s = loader.get_stats()
    cfg.window_size = max(128, min(s["num_edges"] // 32, 4096))
    if ae_cache is not None:
        cfg._trained_ae_score_cache = ae_cache

    model = create_sag_model(
        num_nodes=s["num_nodes"], config=cfg,
        features=loader.features, adj=loader.adj,
    )

    order = getattr(cfg, "edge_order", "auto")
    if order == "auto":
        order = "shuffle" if cfg.has_ground_truth else "bfs"
    edges = loader.get_edge_stream(mode=order)
    do_shuffle = (order == "shuffle")
    passes = (2 if s["num_edges"] > 50000 else 3) if do_shuffle else 1
    rng = np.random.default_rng(seed)

    timer = Timer()
    mt = MemoryTracker()
    mt.reset()
    timer.start()
    model.process_stream_batch(edges, passes=passes, shuffle=do_shuffle, rng=rng)
    timer.stop()
    mt.update()

    elapsed_ms = timer.get_elapsed_ms()
    peak_mem_gb = mt.get_peak()["peak_total_mem_gb"]
    latency_us_per_edge = (elapsed_ms / max(1, edges.shape[0])) * 1000.0

    scores = model.get_scores()
    y = np.asarray(loader.labels)
    if scores.max() > scores.min():
        s_norm = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        s_norm = scores

    auroc = float(compute_auroc(y, s_norm))
    auprc = float(compute_auprc(y, s_norm))
    macro_f1 = float(compute_macro_f1(y, s_norm))
    auprc_bal, macro_f1_bal = compute_balanced_auprc_f1(y, s_norm, n_trials=20, seed=0)

    return {
        "dataset": dataset, "param": sweep_param, "value": float(value),
        "seed": int(seed),
        "auroc": auroc, "auprc": auprc, "macro_f1": macro_f1,
        "auprc_bal": float(auprc_bal), "macro_f1_bal": float(macro_f1_bal),
        "latency_us_per_edge": float(latency_us_per_edge),
        "elapsed_ms": float(elapsed_ms),
        "peak_mem_gb": float(peak_mem_gb),
        "num_edges": int(s["num_edges"]),
        "num_nodes": int(s["num_nodes"]),
        "num_anomalies": int(s["num_anomalies"]),
        "skipped": False, "error": "",
    }


def error_stub(dataset, sweep_param, value, seed, msg):
    return {
        "dataset": dataset, "param": sweep_param, "value": float(value),
        "seed": int(seed),
        "auroc": float("nan"), "auprc": float("nan"), "macro_f1": float("nan"),
        "auprc_bal": float("nan"), "macro_f1_bal": float("nan"),
        "latency_us_per_edge": float("nan"),
        "elapsed_ms": float("nan"), "peak_mem_gb": float("nan"),
        "num_edges": 0, "num_nodes": 0, "num_anomalies": 0,
        "skipped": True, "error": msg,
    }


# ---------------------------------------------------------------------------
# CSV emission
# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    "dataset", "param", "value", "seed",
    "auroc", "auprc", "macro_f1", "auprc_bal", "macro_f1_bal",
    "latency_us_per_edge", "elapsed_ms", "peak_mem_gb",
    "num_edges", "num_nodes", "num_anomalies",
    "skipped", "error",
]
SUMMARY_METRICS = ("auroc", "auprc", "macro_f1", "auprc_bal", "macro_f1_bal",
                   "latency_us_per_edge", "peak_mem_gb")
SUMMARY_COLUMNS = (
    ["dataset", "param", "value", "n_seeds"]
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
        if r["skipped"]:
            continue
        grouped.setdefault((r["dataset"], r["param"], r["value"]), []).append(r)
    out = []
    for (ds, p, v), group in grouped.items():
        row = {"dataset": ds, "param": p, "value": v, "n_seeds": len(group)}
        for m in SUMMARY_METRICS:
            arr = np.array([g[m] for g in group], dtype=np.float64)
            row[f"{m}_mean"] = float(np.nanmean(arr)) if arr.size else float("nan")
            row[f"{m}_std"] = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
        out.append(row)
    out.sort(key=lambda r: (r["param"], r["dataset"], r["value"]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        for r in out:
            w.writerow(r)
    return out


# ---------------------------------------------------------------------------
# Plotting (one figure per parameter, datasets overlaid)
# ---------------------------------------------------------------------------
def plot_param(summary_rows, sweep_param, fig_dir):
    import matplotlib.pyplot as plt
    rows = [r for r in summary_rows if r["param"] == sweep_param]
    if not rows:
        return
    datasets = sorted({r["dataset"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    metrics = [("auroc_mean", "auroc_std", "AUROC"),
               ("auprc_bal_mean", "auprc_bal_std", "AUPRC (balanced)"),
               ("macro_f1_bal_mean", "macro_f1_bal_std", "Macro-F1 (balanced)")]
    palette = ["#e74c3c", "#3498db", "#2ca02c", "#f39c12", "#9b59b6", "#16a085"]
    for ax, (mk, sk, ylabel) in zip(axes, metrics):
        for i, ds in enumerate(datasets):
            sub = sorted([r for r in rows if r["dataset"] == ds],
                         key=lambda r: r["value"])
            xs = [r["value"] for r in sub]
            ys = [r[mk] for r in sub]
            es = [r[sk] for r in sub]
            ax.errorbar(xs, ys, yerr=es,
                        marker="o", markersize=5, linewidth=1.6, capsize=3,
                        color=palette[i % len(palette)], label=ds)
        ax.set_xlabel(PARAM_DISPLAY.get(sweep_param, sweep_param))
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linestyle="--", alpha=0.45, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if sweep_param in ("R",) or (xs and max(xs) / max(min(xs), 1e-12) > 50):
            ax.set_xscale("log")
    axes[-1].legend(loc="best", frameon=True, edgecolor="0.7",
                    fancybox=False, fontsize=9)
    fig.suptitle(f"Sensitivity to {PARAM_DISPLAY.get(sweep_param, sweep_param)}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(fig_dir, exist_ok=True)
    out_pdf = os.path.join(fig_dir, f"{sweep_param}.pdf")
    out_png = os.path.join(fig_dir, f"{sweep_param}.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    p.add_argument("--params", default=",".join(PARAM_SWEEPS.keys()))
    p.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--tag", default=None)
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--overwrite", action="store_true")
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
        ae_score = ae.fit_score(loader.adj, loader.features)
        if not np.all(np.isfinite(ae_score)) or ae_score.std() < 1e-12:
            logger.info("  AE produced degenerate output; skipping cache")
            return None
        logger.info(f"  AE trained in {time.time()-t0:.1f}s")
        return {"key": cache_key, "score": ae_score}
    except Exception as e:
        logger.warning(f"  AE pre-training failed: {e}; falling back to no cache")
        return None


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    params = [p.strip() for p in args.params.split(",") if p.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    for p in params:
        if p not in PARAM_SWEEPS:
            raise ValueError(f"unknown param: {p} (valid: {list(PARAM_SWEEPS)})")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    run_dir = os.path.join(args.output_dir, "results", "revision",
                           f"parameter_sensitivity{tag}_{ts}")
    log_dir = os.path.join(args.output_dir, "logs",
                           f"parameter_sensitivity{tag}_{ts}")
    fig_dir = os.path.join(args.output_dir, "figures", "revision",
                           "parameter_sensitivity")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    canon_dir = os.path.join(args.output_dir, "results", "revision")
    os.makedirs(canon_dir, exist_ok=True)

    logger = setup_logger("SAG_PARAM_SENS", log_dir)
    logger.info(f"datasets={datasets}  params={params}  seeds={seeds}")
    logger.info(f"run_dir={run_dir}   fig_dir={fig_dir}")

    Config.data_dir = "./data"
    Config.output_dir = args.output_dir

    raw_rows = []
    raw_run = os.path.join(run_dir, "parameter_sensitivity_raw.csv")
    sum_run = os.path.join(run_dir, "parameter_sensitivity_summary.csv")
    raw_canon = os.path.join(canon_dir, "parameter_sensitivity_raw.csv")
    sum_canon = os.path.join(canon_dir, "parameter_sensitivity_summary.csv")

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
            logger.error(f"  cannot load {ds}: {e}")
            for p in params:
                for v in PARAM_SWEEPS[p]:
                    for sd in seeds:
                        raw_rows.append(error_stub(ds, p, v, sd, f"load_error:{type(e).__name__}:{e}"))
            write_raw_csv(raw_rows, raw_run)
            write_raw_csv(raw_rows, raw_canon)
            continue
        logger.info(f"  stats: {loader.get_stats()}")
        ae_cache = _build_ae_cache(loader, dc, logger)

        for p in params:
            logger.info(f"\n  --- sweep {p}  values={PARAM_SWEEPS[p]} ---")
            for v in PARAM_SWEEPS[p]:
                for sd in seeds:
                    try:
                        row = run_one(ds, p, v, sd, dc, loader, ae_cache, logger)
                        logger.info(
                            "    %s=%s seed=%d AUROC=%.4f F1=%.4f F1b=%.4f lat=%.2fus mem=%.3fGB" %
                            (p, str(v), sd, row["auroc"], row["macro_f1"],
                             row["macro_f1_bal"], row["latency_us_per_edge"],
                             row["peak_mem_gb"])
                        )
                        raw_rows.append(row)
                    except Exception as e:
                        tb = traceback.format_exc()
                        logger.error(f"    {p}={v} seed={sd} FAILED {type(e).__name__}: {e}\n{tb}")
                        raw_rows.append(error_stub(ds, p, v, sd, f"ERROR:{type(e).__name__}:{e}"))
                    write_raw_csv(raw_rows, raw_run)
                    write_raw_csv(raw_rows, raw_canon)

    summary = write_summary_csv(raw_rows, sum_run)
    write_summary_csv(raw_rows, sum_canon)
    save_json({"raw": raw_rows, "summary": summary},
              os.path.join(run_dir, "parameter_sensitivity.json"))

    if not args.no_plots:
        logger.info("\n=== plotting ===")
        for p in params:
            try:
                plot_param(summary, p, fig_dir)
                logger.info(f"  wrote {fig_dir}/{p}.pdf|.png")
            except Exception as e:
                logger.warning(f"  plot failed for {p}: {e}")

    logger.info("=" * 70)
    logger.info(f"WROTE  raw     -> {raw_run}")
    logger.info(f"WROTE  summary -> {sum_run}")
    logger.info(f"CANON  raw     -> {raw_canon}")
    logger.info(f"CANON  summary -> {sum_canon}")
    logger.info(f"FIGS           -> {fig_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
