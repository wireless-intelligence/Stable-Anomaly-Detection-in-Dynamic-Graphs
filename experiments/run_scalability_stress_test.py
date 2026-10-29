"""Scalability stress test (revision experiment 10).

Runs each method on progressively larger stream prefixes (10, 25, 50, 75,
100% of edges) and records:

  * total_runtime_s              wall time of the streaming pass
  * latency_mean_us_per_edge     total_runtime / num_edges_used * 1e6
  * latency_p95_us               p95 of per-batch latency
  * peak_cpu_memory_gb           Python process RSS via MemoryTracker
  * peak_gpu_memory_gb           torch.cuda.max_memory_allocated/1e9 if available
  * num_node_states              SAG: state.s.shape (rows); MIDAS: num_nodes
  * num_edge_counter_entries     SAG: len(scorer.edge_curr) + len(edge_prev);
                                 MIDAS: count of non-zero hash buckets
  * auroc / auprc / macro_f1     final-prefix evaluation metrics

Prefix-evaluation note: AUROC / AUPRC / Macro-F1 at < 100% are computed on
the same node-level label vector as the 100% case; nodes whose only
appearance is past the prefix simply contribute their initial score (0).
This is faithful to the streaming setting (you cannot predict on data you
have not seen).

Outputs (canonical, latest run; --overwrite to replace):
  outputs/results/revision/scalability_raw.csv
  outputs/results/revision/scalability_summary.csv
  outputs/figures/revision/scalability/<dataset>__<metric>.{pdf,png}
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
from src.baselines import BASELINES
from src.data import DataLoader
from src.evaluation.metrics import (
    compute_auroc, compute_auprc, compute_macro_f1, compute_balanced_auprc_f1,
)
from src.models import create_sag_model
from src.utils import MemoryTracker, save_json, set_seed, setup_logger


METHODS_DEFAULT = ("SAG-full", "MIDAS", "NetWalk")
DATASETS_DEFAULT = ("YelpReviews", "Enron", "BlogCatalog")
SEEDS_DEFAULT = (0, 1, 2)
PREFIX_DEFAULT = (10, 25, 50, 75, 100)
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


def normalize(s):
    s = np.asarray(s, dtype=np.float64)
    if s.max() > s.min():
        return (s - s.min()) / (s.max() - s.min())
    return np.zeros_like(s)


def metrics_for(y, scores):
    s = normalize(scores)
    auprc_bal, f1_bal = compute_balanced_auprc_f1(y, s, n_trials=20, seed=0)
    return {
        "auroc": float(compute_auroc(y, s)),
        "auprc": float(compute_auprc(y, s)),
        "macro_f1": float(compute_macro_f1(y, s)),
        "auprc_bal": float(auprc_bal),
        "macro_f1_bal": float(f1_bal),
    }


def gpu_peak_gb_reset():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def gpu_peak_gb_read():
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / (1024 ** 3)
    except Exception:
        pass
    return float("nan")


# ---------------------------------------------------------------------------
def run_sag_prefix(loader, dc, seed, prefix_pct, batch_size: int = 2048):
    cfg = build_cfg(dc, seed=seed)
    s = loader.get_stats()
    cfg.window_size = max(128, min(s["num_edges"] // 32, 4096))
    set_seed(seed)
    model = create_sag_model(num_nodes=s["num_nodes"], config=cfg,
                             features=loader.features, adj=loader.adj)
    order = getattr(cfg, "edge_order", "auto")
    if order == "auto":
        order = "shuffle" if cfg.has_ground_truth else "bfs"
    edges_full = loader.get_edge_stream(mode=order)
    M = edges_full.shape[0]
    take = max(1, int(round(M * (prefix_pct / 100.0))))
    edges = edges_full[:take]
    rng = np.random.default_rng(seed)

    mt = MemoryTracker(); mt.reset()
    gpu_peak_gb_reset()
    per_batch_us = []
    t0 = time.perf_counter()
    for i in range(0, edges.shape[0], batch_size):
        chunk = edges[i: i + batch_size]
        if chunk.shape[0] == 0:
            continue
        b0 = time.perf_counter()
        model.process_stream_batch(chunk, passes=1, shuffle=False, rng=rng)
        per_batch_us.append((time.perf_counter() - b0) * 1e6)
    total = time.perf_counter() - t0
    mt.update()
    peak_cpu = mt.get_peak()["peak_total_mem_gb"]
    peak_gpu = gpu_peak_gb_read()

    scores = model.get_scores()
    state_arr = getattr(model.state, "s", None)
    n_node_states = int(state_arr.shape[0]) if state_arr is not None else int(s["num_nodes"])
    n_edge_entries = (
        len(getattr(model.scorer, "edge_curr", {}) or {}) +
        len(getattr(model.scorer, "edge_prev", {}) or {})
    )
    return {
        "scores": scores, "total_runtime_s": float(total),
        "per_batch_us": per_batch_us,
        "peak_cpu_memory_gb": float(peak_cpu),
        "peak_gpu_memory_gb": float(peak_gpu),
        "num_node_states": int(n_node_states),
        "num_edge_counter_entries": int(n_edge_entries),
        "num_edges_used": int(edges.shape[0]),
    }


def run_baseline_prefix(method, loader, dc, seed, prefix_pct):
    cfg = build_cfg(dc, seed=seed)
    s = loader.get_stats()
    set_seed(seed)
    cls = BASELINES[method]
    try:
        bm = cls(num_nodes=s["num_nodes"], config=cfg,
                 features=loader.features, seed=seed)
    except TypeError:
        bm = cls(num_nodes=s["num_nodes"], config=cfg,
                 features=loader.features)
    order = getattr(cfg, "edge_order", "auto")
    if order == "auto":
        order = "shuffle" if cfg.has_ground_truth else "bfs"
    edges_full = loader.get_edge_stream(mode=order)
    M = edges_full.shape[0]
    take = max(1, int(round(M * (prefix_pct / 100.0))))
    edges = edges_full[:take]

    mt = MemoryTracker(); mt.reset()
    gpu_peak_gb_reset()
    t0 = time.perf_counter()
    bm.fit(loader.adj, edges, loader.labels)
    total = time.perf_counter() - t0
    mt.update()
    peak_cpu = mt.get_peak()["peak_total_mem_gb"]
    peak_gpu = gpu_peak_gb_read()
    scores = bm.score()

    n_node_states = int(s["num_nodes"])
    n_edge_entries = float("nan")
    if method == "MIDAS":
        # We can't read the count-min sketch counts back from the public
        # API; report num_hash * buckets as the maintained sketch size.
        n_edge_entries = int(getattr(bm, "num_hash", 2) * getattr(bm, "buckets", 2 ** 14))
    return {
        "scores": scores, "total_runtime_s": float(total),
        "per_batch_us": [total * 1e6],
        "peak_cpu_memory_gb": float(peak_cpu),
        "peak_gpu_memory_gb": float(peak_gpu),
        "num_node_states": int(n_node_states),
        "num_edge_counter_entries": int(n_edge_entries) if isinstance(n_edge_entries, int) else n_edge_entries,
        "num_edges_used": int(edges.shape[0]),
    }


# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    "dataset", "method", "seed", "prefix_pct",
    "num_edges_used", "total_runtime_s",
    "latency_mean_us_per_edge", "latency_p95_us", "latency_p99_us",
    "peak_cpu_memory_gb", "peak_gpu_memory_gb",
    "num_node_states", "num_edge_counter_entries",
    "auroc", "auprc", "macro_f1", "auprc_bal", "macro_f1_bal",
    "skipped", "error",
]
SUMMARY_COLUMNS = [
    "dataset", "method", "prefix_pct", "n_seeds",
    "total_runtime_s_mean", "total_runtime_s_std",
    "latency_mean_us_per_edge_mean", "latency_mean_us_per_edge_std",
    "latency_p95_us_mean", "latency_p95_us_std",
    "peak_cpu_memory_gb_mean", "peak_cpu_memory_gb_std",
    "peak_gpu_memory_gb_mean",
    "num_node_states_mean", "num_edge_counter_entries_mean",
    "auroc_mean", "auroc_std",
]


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
        grouped.setdefault((r["dataset"], r["method"], r["prefix_pct"]), []).append(r)
    out = []
    for (ds, mt, pp), group in grouped.items():
        row = {"dataset": ds, "method": mt, "prefix_pct": pp,
               "n_seeds": len(group)}
        for src, dst in [
            ("total_runtime_s", "total_runtime_s"),
            ("latency_mean_us_per_edge", "latency_mean_us_per_edge"),
            ("latency_p95_us", "latency_p95_us"),
            ("peak_cpu_memory_gb", "peak_cpu_memory_gb"),
            ("auroc", "auroc"),
        ]:
            arr = np.array([g[src] for g in group], dtype=np.float64)
            row[f"{dst}_mean"] = float(np.nanmean(arr)) if arr.size else float("nan")
            row[f"{dst}_std"] = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
        for src, dst in [
            ("peak_gpu_memory_gb", "peak_gpu_memory_gb"),
            ("num_node_states", "num_node_states"),
            ("num_edge_counter_entries", "num_edge_counter_entries"),
        ]:
            arr = np.array([float(g[src]) if str(g[src]) not in ("", "nan") else float("nan")
                            for g in group], dtype=np.float64)
            row[f"{dst}_mean"] = float(np.nanmean(arr)) if arr.size else float("nan")
        out.append(row)
    out.sort(key=lambda r: (r["dataset"], r["method"], int(r["prefix_pct"])))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        for r in out:
            w.writerow(r)
    return out


# ---------------------------------------------------------------------------
def plot_summary(summary, fig_dir):
    import matplotlib.pyplot as plt
    datasets = sorted({r["dataset"] for r in summary})
    methods = sorted({r["method"] for r in summary})
    palette = ["#e74c3c", "#3498db", "#2ca02c", "#f39c12", "#9b59b6", "#16a085"]
    method_color = {m: palette[i % len(palette)] for i, m in enumerate(methods)}
    metric_specs = [
        ("total_runtime_s_mean", "Total runtime (s)", False),
        ("latency_mean_us_per_edge_mean", "Per-edge latency (us)", True),
        ("peak_cpu_memory_gb_mean", "Peak CPU memory (GB)", False),
        ("num_edge_counter_entries_mean", "Maintained edge-counter entries", True),
    ]
    os.makedirs(fig_dir, exist_ok=True)
    for ds in datasets:
        for mk, ylabel, log_y in metric_specs:
            fig, ax = plt.subplots(figsize=(7.0, 3.8))
            for m in methods:
                rows = sorted([r for r in summary
                               if r["dataset"] == ds and r["method"] == m],
                              key=lambda r: int(r["prefix_pct"]))
                if not rows: continue
                xs = [int(r["prefix_pct"]) for r in rows]
                ys = [r.get(mk, float("nan")) for r in rows]
                ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.7,
                        color=method_color[m], label=m)
            ax.set_xlabel("Stream prefix (% of edges)")
            ax.set_ylabel(ylabel)
            if log_y:
                ax.set_yscale("log")
            ax.grid(True, axis="y", linestyle="--", alpha=0.45, linewidth=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(loc="best", frameon=True, edgecolor="0.7",
                      fancybox=False, fontsize=9)
            ax.set_title(f"{ds} — {ylabel}",
                         fontsize=12, fontweight="bold")
            fig.tight_layout()
            base = os.path.join(fig_dir,
                f"{ds}__{ylabel.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}")
            fig.savefig(base + ".pdf", bbox_inches="tight")
            fig.savefig(base + ".png", dpi=160, bbox_inches="tight")
            plt.close(fig)


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(DATASETS_DEFAULT))
    p.add_argument("--methods", default=",".join(METHODS_DEFAULT))
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS_DEFAULT))
    p.add_argument("--prefix_pcts", default=",".join(str(p) for p in PREFIX_DEFAULT))
    p.add_argument("--batch_size", type=int, default=2048)
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
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    pcts = [int(p.strip()) for p in args.prefix_pcts.split(",") if p.strip()]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    run_dir = os.path.join(args.output_dir, "results", "revision",
                           f"scalability{tag}_{ts}")
    log_dir = os.path.join(args.output_dir, "logs",
                           f"scalability{tag}_{ts}")
    fig_dir = os.path.join(args.output_dir, "figures", "revision", "scalability")
    canon_dir = os.path.join(args.output_dir, "results", "revision")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(canon_dir, exist_ok=True)

    logger = setup_logger("SAG_SCALE", log_dir)
    logger.info(f"datasets={datasets}  methods={methods}  seeds={seeds}  pcts={pcts}")

    save_json({"datasets": datasets, "methods": methods, "seeds": seeds,
               "prefix_pcts": pcts, "batch_size": args.batch_size,
               "argv": sys.argv, "timestamp": ts},
              os.path.join(run_dir, "config.json"))

    Config.data_dir = "./data"
    Config.output_dir = args.output_dir

    raw_rows = []
    raw_run = os.path.join(run_dir, "scalability_raw.csv")
    sum_run = os.path.join(run_dir, "scalability_summary.csv")
    raw_canon = os.path.join(canon_dir, "scalability_raw.csv")
    sum_canon = os.path.join(canon_dir, "scalability_summary.csv")

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

        for method in methods:
            if method != "SAG-full" and method not in BASELINES:
                logger.info(f"  [SKIP] {method}: not registered")
                for pct in pcts:
                    for sd in seeds:
                        raw_rows.append({
                            "dataset": ds, "method": method, "seed": sd,
                            "prefix_pct": pct,
                            "num_edges_used": 0, "total_runtime_s": float("nan"),
                            "latency_mean_us_per_edge": float("nan"),
                            "latency_p95_us": float("nan"),
                            "latency_p99_us": float("nan"),
                            "peak_cpu_memory_gb": float("nan"),
                            "peak_gpu_memory_gb": float("nan"),
                            "num_node_states": 0,
                            "num_edge_counter_entries": "",
                            "auroc": float("nan"), "auprc": float("nan"),
                            "macro_f1": float("nan"), "auprc_bal": float("nan"),
                            "macro_f1_bal": float("nan"),
                            "skipped": True, "error": "method not registered",
                        })
                continue
            for pct in pcts:
                for sd in seeds:
                    logger.info(f"  --- {ds} | {method} | seed={sd} | prefix={pct}% ---")
                    try:
                        if method == "SAG-full":
                            res = run_sag_prefix(loader, dc, sd, pct,
                                                 batch_size=args.batch_size)
                        else:
                            res = run_baseline_prefix(method, loader, dc, sd, pct)
                        y = np.asarray(loader.labels)
                        m = metrics_for(y, res["scores"])
                        per_batch = np.array(res["per_batch_us"], dtype=np.float64) \
                            if res["per_batch_us"] else np.array([float("nan")])
                        per_edge_us = (res["total_runtime_s"] * 1e6 / max(1, res["num_edges_used"]))
                        row = {
                            "dataset": ds, "method": method, "seed": sd,
                            "prefix_pct": pct,
                            "num_edges_used": int(res["num_edges_used"]),
                            "total_runtime_s": float(res["total_runtime_s"]),
                            "latency_mean_us_per_edge": float(per_edge_us),
                            "latency_p95_us": float(np.nanquantile(per_batch, 0.95)),
                            "latency_p99_us": float(np.nanquantile(per_batch, 0.99)),
                            "peak_cpu_memory_gb": float(res["peak_cpu_memory_gb"]),
                            "peak_gpu_memory_gb": float(res["peak_gpu_memory_gb"]),
                            "num_node_states": int(res["num_node_states"]),
                            "num_edge_counter_entries": res["num_edge_counter_entries"],
                            "auroc": m["auroc"], "auprc": m["auprc"],
                            "macro_f1": m["macro_f1"],
                            "auprc_bal": m["auprc_bal"],
                            "macro_f1_bal": m["macro_f1_bal"],
                            "skipped": False, "error": "",
                        }
                        raw_rows.append(row)
                        logger.info(
                            "    edges=%d runtime=%.2fs lat=%.1fus mem=%.2fGB AUROC=%.4f"
                            " (n_states=%d, n_edge_entries=%s)" %
                            (row["num_edges_used"], row["total_runtime_s"],
                             row["latency_mean_us_per_edge"], row["peak_cpu_memory_gb"],
                             row["auroc"], row["num_node_states"],
                             str(row["num_edge_counter_entries"])))
                    except Exception as e:
                        tb = traceback.format_exc()
                        logger.error(f"    FAILED {type(e).__name__}: {e}\n{tb}")
                        raw_rows.append({
                            "dataset": ds, "method": method, "seed": sd,
                            "prefix_pct": pct,
                            "num_edges_used": 0,
                            "total_runtime_s": float("nan"),
                            "latency_mean_us_per_edge": float("nan"),
                            "latency_p95_us": float("nan"),
                            "latency_p99_us": float("nan"),
                            "peak_cpu_memory_gb": float("nan"),
                            "peak_gpu_memory_gb": float("nan"),
                            "num_node_states": 0,
                            "num_edge_counter_entries": "",
                            "auroc": float("nan"), "auprc": float("nan"),
                            "macro_f1": float("nan"), "auprc_bal": float("nan"),
                            "macro_f1_bal": float("nan"),
                            "skipped": True,
                            "error": f"{type(e).__name__}:{e}",
                        })
                    write_raw_csv(raw_rows, raw_run)

    summary = write_summary_csv(raw_rows, sum_run)
    save_json({"raw": raw_rows, "summary": summary},
              os.path.join(run_dir, "scalability.json"))

    if not _maybe_skip(raw_canon, args.overwrite, logger):
        write_raw_csv(raw_rows, raw_canon)
    if not _maybe_skip(sum_canon, args.overwrite, logger):
        write_summary_csv(raw_rows, sum_canon)

    if not args.no_plots:
        try:
            plot_summary(summary, fig_dir)
            logger.info(f"  wrote figures to {fig_dir}")
        except Exception as e:
            logger.warning(f"  plotting failed: {e}")

    logger.info("=" * 70)
    logger.info(f"WROTE  raw     -> {raw_run}")
    logger.info(f"WROTE  summary -> {sum_run}")
    logger.info(f"CANON  raw     -> {raw_canon}")
    logger.info(f"CANON  summary -> {sum_canon}")
    logger.info(f"FIGS           -> {fig_dir}")


if __name__ == "__main__":
    main()
