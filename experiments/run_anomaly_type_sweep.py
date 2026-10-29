"""Robustness to anomaly type and strength (revision experiment 9).

Sweeps four axes on the four injected-anomaly datasets:

A. structural strength : clique_size in {5,10,15,20,25}
                          (num_cliques = dataset default)
B. attribute strength  : attr_distance_percentile in {50,70,90,95,99}
C. mixed ratio         : structural_only, attribute_only, 50/50, 25/75, 75/25
D. non-clique types    : star, chain, bipartite_burst, community_bridge
                          (NotImplementedError stubs in AnomalyInjector;
                           runner records each as skipped with reason.)

Every (dataset, axis, value, seed) starts from a freshly loaded raw graph
and runs SAG-full once.  Variance comes from the seed (which controls both
the injection RNG and the streaming RNG).

Outputs (canonical, latest run; --overwrite to replace):
  outputs/results/revision/anomaly_type_sweep_raw.csv
  outputs/results/revision/anomaly_type_sweep_summary.csv
  outputs/figures/revision/anomaly_type_sweep/<axis>__<dataset>__<metric>.{pdf,png}
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
import scipy.sparse as sp

from configs import Config, get_dataset_config
from src.data import DataLoader
from src.data.anomaly_injection import AnomalyInjector
from src.evaluation.metrics import (
    compute_auroc, compute_auprc, compute_macro_f1, compute_balanced_auprc_f1,
)
from src.models import create_sag_model
from src.utils import save_json, set_seed, setup_logger


DATASETS_DEFAULT = ("Cora", "Citeseer", "Pubmed", "BlogCatalog")
SEEDS_DEFAULT = (0, 1, 2)
BASE_SEED = 42

# Axis A : structural-strength clique-size sweep
A_VALUES = (5, 10, 15, 20, 25)
# Axis B : attribute-distance-percentile sweep
B_VALUES = (50, 70, 90, 95, 99)
# Axis C : mixed-ratio sweep ; (struct_frac, attr_frac) pairs of total budget
C_VALUES = (("structural_only", 1.0, 0.0),
            ("attribute_only",  0.0, 1.0),
            ("split_50_50",     0.5, 0.5),
            ("split_25_75",     0.25, 0.75),
            ("split_75_25",     0.75, 0.25))
# Axis D : non-clique structural types (TODO stubs)
D_VALUES = ("star", "chain", "bipartite_burst", "community_bridge")


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


# ---------------------------------------------------------------------------
def load_raw_graph(dataset: str, dc: dict):
    """Load (raw_adj, raw_features) WITHOUT running anomaly injection."""
    loader = DataLoader(
        data_dir="./data", dataset_name=dataset,
        clique_size=dc.get("clique_size", 15),
        num_cliques=dc.get("num_cliques", 10),
        seed=BASE_SEED, inject_anomaly=False,
    )
    loader.load()
    return loader


def run_sag_on(loader_like, dc, seed, num_anomalies):
    set_seed(seed)
    cfg = build_cfg(dc, seed=seed)
    n = loader_like["num_nodes"]
    cfg.window_size = max(128, min(loader_like["num_edges"] // 32, 4096))
    model = create_sag_model(num_nodes=n, config=cfg,
                             features=loader_like["features"],
                             adj=loader_like["adj"])
    order = getattr(cfg, "edge_order", "auto")
    if order == "auto":
        order = "shuffle" if cfg.has_ground_truth else "bfs"
    edges = _edges_from_adj(loader_like["adj"], mode=order, seed=seed)
    do_shuffle = (order == "shuffle")
    passes = (2 if loader_like["num_edges"] > 50000 else 3) if do_shuffle else 1
    rng = np.random.default_rng(seed)
    model.process_stream_batch(edges, passes=passes, shuffle=do_shuffle, rng=rng)
    return model.get_scores()


def _edges_from_adj(adj, mode, seed):
    coo = sp.csr_matrix(adj).tocoo()
    mask = coo.row < coo.col
    e = np.stack([coo.row[mask], coo.col[mask]], axis=1).astype(np.int64)
    rng = np.random.default_rng(seed)
    if mode == "shuffle":
        rng.shuffle(e)
    return e


def metrics_for(y, scores) -> dict:
    s = np.asarray(scores, dtype=np.float64)
    if s.max() > s.min():
        s = (s - s.min()) / (s.max() - s.min())
    auprc_bal, f1_bal = compute_balanced_auprc_f1(y, s, n_trials=20, seed=0)
    return {
        "auroc": float(compute_auroc(y, s)),
        "auprc": float(compute_auprc(y, s)),
        "macro_f1": float(compute_macro_f1(y, s)),
        "auprc_bal": float(auprc_bal),
        "macro_f1_bal": float(f1_bal),
    }


# ---------------------------------------------------------------------------
def inject_for_axis(raw_loader, axis: str, value, dc: dict, seed: int):
    """Returns (loader_like_dict, num_struct, num_attr) or raises
    NotImplementedError for axis D types we have stubbed out."""
    base_clique = int(dc.get("clique_size", 15))
    base_num_cliques = int(dc.get("num_cliques", 10))
    if axis == "A_clique_size":
        cs = int(value)
        inj = AnomalyInjector(adj=raw_loader.adj, features=raw_loader.features,
                              clique_size=cs, num_cliques=base_num_cliques,
                              k_attr=50, seed=seed)
        inj.inject_structural()
        inj.inject_attribute(num_anomalies=cs * base_num_cliques)
    elif axis == "B_attr_percentile":
        pct = float(value)
        inj = AnomalyInjector(adj=raw_loader.adj, features=raw_loader.features,
                              clique_size=base_clique, num_cliques=base_num_cliques,
                              k_attr=50, seed=seed,
                              attr_distance_percentile=pct)
        inj.inject_structural()
        inj.inject_attribute()
    elif axis == "C_mix_ratio":
        name, sf, af = value
        budget = base_clique * base_num_cliques
        n_struct_anoms = int(round(sf * budget))
        n_attr = int(round(af * budget))
        nc_eff = max(0, n_struct_anoms // max(1, base_clique))
        inj = AnomalyInjector(adj=raw_loader.adj, features=raw_loader.features,
                              clique_size=base_clique, num_cliques=nc_eff,
                              k_attr=50, seed=seed)
        if nc_eff > 0:
            inj.inject_structural()
        if n_attr > 0:
            inj.inject_attribute(num_anomalies=n_attr)
    elif axis == "D_non_clique":
        inj = AnomalyInjector(adj=raw_loader.adj, features=raw_loader.features,
                              clique_size=base_clique, num_cliques=base_num_cliques,
                              k_attr=50, seed=seed)
        if value == "star":
            inj.inject_star(num_stars=base_num_cliques,
                            leaves_per_star=base_clique)
        elif value == "chain":
            inj.inject_chain(num_chains=base_num_cliques,
                             chain_length=base_clique)
        elif value == "bipartite_burst":
            inj.inject_bipartite_burst(num_groups=base_num_cliques,
                                       left_size=base_clique // 2,
                                       right_size=base_clique // 2)
        elif value == "community_bridge":
            inj.inject_community_bridge(num_bridges=base_num_cliques)
        else:
            raise ValueError(f"unknown D value: {value}")
    else:
        raise ValueError(f"unknown axis: {axis}")
    data = inj.get_data()
    adj = sp.csr_matrix(data["adj"])
    adj.eliminate_zeros(); adj = adj.maximum(adj.T)
    feat = data["features"] if sp.issparse(data["features"]) else sp.csr_matrix(data["features"])
    n_struct = data["num_structural"]; n_attr = data["num_attribute"]
    loader_like = {
        "adj": adj, "features": feat, "labels": data["labels"],
        "num_nodes": int(adj.shape[0]),
        "num_edges": int(adj.nnz // 2),
        "num_anomalies": int(np.sum(data["labels"])),
    }
    return loader_like, n_struct, n_attr


# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    "dataset", "axis", "axis_value_id", "axis_value", "seed",
    "auroc", "auprc", "macro_f1", "auprc_bal", "macro_f1_bal",
    "num_structural_anomalies", "num_attribute_anomalies",
    "num_anomalies_total", "num_nodes", "num_edges",
    "elapsed_s",
    "skipped", "skip_reason",
]
SUMMARY_METRICS = ("auroc", "auprc", "macro_f1", "auprc_bal", "macro_f1_bal",
                   "num_structural_anomalies", "num_attribute_anomalies")
SUMMARY_COLUMNS = (
    ["dataset", "axis", "axis_value_id", "axis_value", "n_seeds",
     "applicable", "skip_reason"]
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
        grouped.setdefault((r["dataset"], r["axis"], r["axis_value_id"]), []).append(r)
    out = []
    for (ds, ax, vid), group in grouped.items():
        ok = [g for g in group if not g["skipped"]]
        if not ok:
            row = {"dataset": ds, "axis": ax, "axis_value_id": vid,
                   "axis_value": group[0]["axis_value"],
                   "n_seeds": 0, "applicable": False,
                   "skip_reason": group[0].get("skip_reason", "")}
            for m in SUMMARY_METRICS:
                row[f"{m}_mean"] = ""
                row[f"{m}_std"] = ""
        else:
            row = {"dataset": ds, "axis": ax, "axis_value_id": vid,
                   "axis_value": ok[0]["axis_value"],
                   "n_seeds": len(ok), "applicable": True, "skip_reason": ""}
            for m in SUMMARY_METRICS:
                v = np.array([r[m] for r in ok], dtype=np.float64)
                row[f"{m}_mean"] = float(v.mean())
                row[f"{m}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        out.append(row)
    out.sort(key=lambda r: (r["axis"], r["dataset"], r["axis_value_id"]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        for r in out:
            w.writerow(r)
    return out


# ---------------------------------------------------------------------------
def plot_axis(summary, axis, fig_dir):
    import matplotlib.pyplot as plt
    rows = [r for r in summary if r["axis"] == axis and r["applicable"]]
    if not rows:
        return
    datasets = sorted({r["dataset"] for r in rows})
    palette = ["#e74c3c", "#3498db", "#2ca02c", "#f39c12", "#9b59b6", "#16a085"]
    metric_specs = [("auroc_mean", "auroc_std", "AUROC"),
                    ("auprc_bal_mean", "auprc_bal_std", "AUPRC (balanced)"),
                    ("macro_f1_bal_mean", "macro_f1_bal_std", "Macro-F1 (balanced)")]
    for mk, sk, ylabel in metric_specs:
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        for i, ds in enumerate(datasets):
            sub = sorted([r for r in rows if r["dataset"] == ds],
                         key=lambda r: int(r["axis_value_id"])
                            if str(r["axis_value_id"]).isdigit() else 0)
            if not sub: continue
            xs_id = [str(r["axis_value"]) for r in sub]
            ys = [r[mk] for r in sub]
            es = [r[sk] for r in sub]
            ax.errorbar(range(len(xs_id)), ys, yerr=es,
                        marker="o", markersize=5, linewidth=1.6, capsize=3,
                        color=palette[i % len(palette)], label=ds)
        ax.set_xticks(range(len(xs_id)))
        ax.set_xticklabels(xs_id, rotation=15)
        ax.set_xlabel(axis.replace("_", " "))
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linestyle="--", alpha=0.45, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", frameon=True, edgecolor="0.7",
                  fancybox=False, fontsize=9)
        ax.set_title(f"{axis} — {ylabel}",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        os.makedirs(fig_dir, exist_ok=True)
        base = os.path.join(fig_dir,
                            f"{axis}__{ylabel.replace(' ', '_')}".replace("/", "_"))
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", dpi=160, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(DATASETS_DEFAULT))
    p.add_argument("--axes", default="A_clique_size,B_attr_percentile,C_mix_ratio,D_non_clique")
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS_DEFAULT))
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


def _values_for(axis):
    return {"A_clique_size": A_VALUES, "B_attr_percentile": B_VALUES,
            "C_mix_ratio": C_VALUES, "D_non_clique": D_VALUES}[axis]


def _value_id(axis, value):
    if axis == "C_mix_ratio":
        return value[0]
    return str(value)


def _value_for_csv(axis, value):
    if axis == "C_mix_ratio":
        name, sf, af = value
        return f"{name}(struct={sf:.2f},attr={af:.2f})"
    return str(value)


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    run_dir = os.path.join(args.output_dir, "results", "revision",
                           f"anomaly_type_sweep{tag}_{ts}")
    log_dir = os.path.join(args.output_dir, "logs",
                           f"anomaly_type_sweep{tag}_{ts}")
    fig_dir = os.path.join(args.output_dir, "figures", "revision",
                           "anomaly_type_sweep")
    canon_dir = os.path.join(args.output_dir, "results", "revision")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(canon_dir, exist_ok=True)

    logger = setup_logger("SAG_ANOM_SWEEP", log_dir)
    logger.info(f"datasets={datasets}  axes={axes}  seeds={seeds}")
    logger.info(f"run_dir={run_dir}")

    save_json({"datasets": datasets, "axes": axes, "seeds": seeds,
               "argv": sys.argv, "timestamp": ts},
              os.path.join(run_dir, "config.json"))

    Config.data_dir = "./data"
    Config.output_dir = args.output_dir

    raw_rows = []
    raw_run = os.path.join(run_dir, "anomaly_type_sweep_raw.csv")
    sum_run = os.path.join(run_dir, "anomaly_type_sweep_summary.csv")
    raw_canon = os.path.join(canon_dir, "anomaly_type_sweep_raw.csv")
    sum_canon = os.path.join(canon_dir, "anomaly_type_sweep_summary.csv")

    for ds in datasets:
        try:
            dc = get_dataset_config(ds)
        except ValueError as e:
            logger.error(f"unknown dataset {ds}: {e}")
            continue
        logger.info(f"\n========== dataset = {ds} ==========")
        try:
            raw_loader = load_raw_graph(ds, dc)
        except Exception as e:
            logger.error(f"  load failed: {e}")
            continue

        for axis in axes:
            logger.info(f"\n  --- axis = {axis} ---")
            for value in _values_for(axis):
                vid = _value_id(axis, value)
                vstr = _value_for_csv(axis, value)
                for sd in seeds:
                    t0 = time.time()
                    try:
                        loader_like, n_s, n_a = inject_for_axis(
                            raw_loader, axis, value, dc, sd)
                        scores = run_sag_on(loader_like, dc, sd,
                                            num_anomalies=n_s + n_a)
                        m = metrics_for(loader_like["labels"], scores)
                        raw_rows.append({
                            "dataset": ds, "axis": axis,
                            "axis_value_id": vid, "axis_value": vstr,
                            "seed": sd, **m,
                            "num_structural_anomalies": int(n_s),
                            "num_attribute_anomalies": int(n_a),
                            "num_anomalies_total": int(n_s + n_a),
                            "num_nodes": int(loader_like["num_nodes"]),
                            "num_edges": int(loader_like["num_edges"]),
                            "elapsed_s": float(time.time() - t0),
                            "skipped": False, "skip_reason": "",
                        })
                        logger.info(
                            "    %s=%s seed=%d AUROC=%.4f F1b=%.4f n_s=%d n_a=%d (%.1fs)" %
                            (axis, vstr, sd, m["auroc"], m["macro_f1_bal"],
                             n_s, n_a, time.time() - t0))
                    except NotImplementedError as e:
                        raw_rows.append({
                            "dataset": ds, "axis": axis,
                            "axis_value_id": vid, "axis_value": vstr,
                            "seed": sd,
                            "auroc": float("nan"), "auprc": float("nan"),
                            "macro_f1": float("nan"), "auprc_bal": float("nan"),
                            "macro_f1_bal": float("nan"),
                            "num_structural_anomalies": 0,
                            "num_attribute_anomalies": 0,
                            "num_anomalies_total": 0,
                            "num_nodes": 0, "num_edges": 0,
                            "elapsed_s": 0.0, "skipped": True,
                            "skip_reason": "anomaly type not implemented",
                        })
                        logger.info(f"    {axis}={vstr} seed={sd} NotImplemented (stubbed)")
                    except Exception as e:
                        tb = traceback.format_exc()
                        logger.error(f"    {axis}={vstr} seed={sd} FAILED {type(e).__name__}: {e}\n{tb}")
                        raw_rows.append({
                            "dataset": ds, "axis": axis,
                            "axis_value_id": vid, "axis_value": vstr,
                            "seed": sd,
                            "auroc": float("nan"), "auprc": float("nan"),
                            "macro_f1": float("nan"), "auprc_bal": float("nan"),
                            "macro_f1_bal": float("nan"),
                            "num_structural_anomalies": 0,
                            "num_attribute_anomalies": 0,
                            "num_anomalies_total": 0,
                            "num_nodes": 0, "num_edges": 0,
                            "elapsed_s": 0.0, "skipped": True,
                            "skip_reason": f"ERROR:{type(e).__name__}:{e}",
                        })
                    write_raw_csv(raw_rows, raw_run)

    summary = write_summary_csv(raw_rows, sum_run)
    save_json({"raw": raw_rows, "summary": summary},
              os.path.join(run_dir, "anomaly_type_sweep.json"))

    if not _maybe_skip(raw_canon, args.overwrite, logger):
        write_raw_csv(raw_rows, raw_canon)
    if not _maybe_skip(sum_canon, args.overwrite, logger):
        write_summary_csv(raw_rows, sum_canon)

    if not args.no_plots:
        for axis in axes:
            try:
                plot_axis(summary, axis, fig_dir)
            except Exception as e:
                logger.warning(f"  plotting failed for {axis}: {e}")

    logger.info("=" * 70)
    logger.info(f"WROTE  raw     -> {raw_run}")
    logger.info(f"WROTE  summary -> {sum_run}")
    logger.info(f"CANON  raw     -> {raw_canon}")
    logger.info(f"CANON  summary -> {sum_canon}")
    logger.info(f"FIGS           -> {fig_dir}")


if __name__ == "__main__":
    main()
