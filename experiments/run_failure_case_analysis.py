"""Failure-case and limitation analysis (revision experiment 13).

For each (dataset, anomaly_type, seed) we (1) inject the anomaly type,
(2) run SAG-full + a message-passing baseline + a streaming baseline, and
(3) run SAG in the four single-predicate fusion modes
{auto, h_only, attr_only, struct_only}. The per-predicate AUROC tells us
WHICH evidence source was responsible (or, when SAG-full is weak, which
predicate would be needed but is missing).

Anomaly types
-------------
clique_dense       : the original CoLA-style clique injection (local)
star               : hub anomalies                                   (multi-hop)
chain              : long-path anomalies                              (multi-hop)
bipartite_burst    : complete-bipartite burst                         (mixed-hop)
community_bridge   : cross-community connectors                       (global)
role_equivalence   : NOT IMPLEMENTED -> recorded as skipped
motif              : NOT IMPLEMENTED -> recorded as skipped

Outputs (canonical, latest run; --overwrite to replace):
  outputs/results/revision/failure_case_raw.csv
  outputs/results/revision/failure_case_summary.csv
  outputs/results/revision/failure_case_summary.md
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
from src.baselines import BASELINES
from src.data import DataLoader
from src.data.anomaly_injection import AnomalyInjector
from src.evaluation.metrics import (
    compute_auroc, compute_auprc, compute_macro_f1, compute_balanced_auprc_f1,
)
from src.models import create_sag_model
from src.utils import save_json, set_seed, setup_logger


ANOMALY_TYPES_DEFAULT = (
    "clique_dense", "star", "chain", "bipartite_burst",
    "community_bridge", "role_equivalence", "motif",
)
PREDICATE_MODES = ("auto", "h_only", "attr_only", "struct_only")
METHODS_DEFAULT = ("SAG-full", "GraphSAGE", "MIDAS")
DATASETS_DEFAULT = ("Cora", "Citeseer", "Pubmed", "BlogCatalog")
SEEDS_DEFAULT = (0, 1, 2)
BASE_SEED = 42


# ---------------------------------------------------------------------------
class CfgShim:
    pass


def build_cfg(dc, seed, fusion_mode_override=None):
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
    if fusion_mode_override is not None:
        cfg.fusion_mode = fusion_mode_override
    return cfg


def load_raw(dataset, dc):
    loader = DataLoader(
        data_dir="./data", dataset_name=dataset,
        clique_size=dc.get("clique_size", 15),
        num_cliques=dc.get("num_cliques", 10),
        seed=BASE_SEED, inject_anomaly=False,
    )
    loader.load()
    return loader


def inject(raw_loader, anomaly_type: str, dc: dict, seed: int):
    cs = int(dc.get("clique_size", 15))
    nc = int(dc.get("num_cliques", 10))
    inj = AnomalyInjector(adj=raw_loader.adj, features=raw_loader.features,
                          clique_size=cs, num_cliques=nc, k_attr=50, seed=seed)
    if anomaly_type == "clique_dense":
        inj.inject_structural()
    elif anomaly_type == "star":
        inj.inject_star(num_stars=nc, leaves_per_star=cs)
    elif anomaly_type == "chain":
        inj.inject_chain(num_chains=nc, chain_length=cs)
    elif anomaly_type == "bipartite_burst":
        left = max(2, cs // 2)
        right = max(2, cs - left)
        inj.inject_bipartite_burst(num_groups=nc, left_size=left, right_size=right)
    elif anomaly_type == "community_bridge":
        inj.inject_community_bridge(num_bridges=nc * cs)
    elif anomaly_type == "role_equivalence":
        inj.inject_role_equivalence(num_pairs=nc * cs)
    elif anomaly_type == "motif":
        inj.inject_motif(motif_pattern="triangle", num_instances=nc * cs)
    else:
        raise ValueError(f"unknown anomaly_type: {anomaly_type}")
    data = inj.get_data()
    adj = sp.csr_matrix(data["adj"]); adj.eliminate_zeros(); adj = adj.maximum(adj.T)
    feat = data["features"] if sp.issparse(data["features"]) else sp.csr_matrix(data["features"])
    return {
        "adj": adj, "features": feat, "labels": data["labels"],
        "num_nodes": int(adj.shape[0]),
        "num_edges": int(adj.nnz // 2),
        "num_anomalies": int(np.sum(data["labels"])),
        "num_structural": data["num_structural"],
        "num_attribute": data["num_attribute"],
    }


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


def _edges_from_adj(adj, mode, seed):
    coo = sp.csr_matrix(adj).tocoo()
    mask = coo.row < coo.col
    e = np.stack([coo.row[mask], coo.col[mask]], axis=1).astype(np.int64)
    rng = np.random.default_rng(seed)
    if mode == "shuffle":
        rng.shuffle(e)
    return e


def run_method(method: str, anomdata: dict, dc: dict, seed: int,
               fusion_mode_override=None):
    set_seed(seed)
    cfg = build_cfg(dc, seed=seed, fusion_mode_override=fusion_mode_override)
    n = anomdata["num_nodes"]
    cfg.window_size = max(128, min(anomdata["num_edges"] // 32, 4096))
    if method == "SAG-full":
        model = create_sag_model(num_nodes=n, config=cfg,
                                 features=anomdata["features"],
                                 adj=anomdata["adj"])
        order = getattr(cfg, "edge_order", "auto")
        if order == "auto":
            order = "shuffle" if cfg.has_ground_truth else "bfs"
        edges = _edges_from_adj(anomdata["adj"], order, seed)
        do_shuffle = (order == "shuffle")
        passes = (2 if anomdata["num_edges"] > 50000 else 3) if do_shuffle else 1
        rng = np.random.default_rng(seed)
        model.process_stream_batch(edges, passes=passes, shuffle=do_shuffle, rng=rng)
        return model.get_scores()
    if method in BASELINES:
        cls = BASELINES[method]
        try:
            bm = cls(num_nodes=n, config=cfg,
                     features=anomdata["features"], seed=seed)
        except TypeError:
            bm = cls(num_nodes=n, config=cfg,
                     features=anomdata["features"])
        order = getattr(cfg, "edge_order", "auto")
        if order == "auto":
            order = "shuffle" if cfg.has_ground_truth else "bfs"
        edges = _edges_from_adj(anomdata["adj"], order, seed)
        bm.fit(anomdata["adj"], edges, anomdata["labels"])
        return bm.score()
    raise ValueError(f"unknown method: {method}")


# ---------------------------------------------------------------------------
RAW_COLS = [
    "dataset", "anomaly_type", "method", "fusion_mode", "seed",
    "auroc", "auprc", "macro_f1", "auprc_bal", "macro_f1_bal",
    "num_nodes", "num_edges", "num_anomalies",
    "num_structural", "num_attribute",
    "elapsed_s", "skipped", "error",
]
SUMMARY_METRICS = ("auroc", "auprc", "macro_f1", "auprc_bal", "macro_f1_bal")
SUMMARY_COLS = (
    ["dataset", "anomaly_type", "method", "fusion_mode", "n_seeds",
     "applicable", "skip_reason"]
    + [f"{m}_mean" for m in SUMMARY_METRICS]
    + [f"{m}_std" for m in SUMMARY_METRICS]
)


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
        grouped.setdefault((r["dataset"], r["anomaly_type"], r["method"],
                            r["fusion_mode"]), []).append(r)
    out = []
    for (ds, at, mt, fm), group in grouped.items():
        ok = [g for g in group if not g["skipped"]]
        if not ok:
            row = {"dataset": ds, "anomaly_type": at, "method": mt,
                   "fusion_mode": fm, "n_seeds": 0,
                   "applicable": False,
                   "skip_reason": group[0].get("error", "")}
            for m in SUMMARY_METRICS:
                row[f"{m}_mean"] = ""
                row[f"{m}_std"] = ""
        else:
            row = {"dataset": ds, "anomaly_type": at, "method": mt,
                   "fusion_mode": fm, "n_seeds": len(ok),
                   "applicable": True, "skip_reason": ""}
            for m in SUMMARY_METRICS:
                v = np.array([r[m] for r in ok], dtype=np.float64)
                row[f"{m}_mean"] = float(v.mean())
                row[f"{m}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        out.append(row)
    out.sort(key=lambda r: (r["dataset"], r["anomaly_type"], r["method"],
                            r["fusion_mode"]))
    return out


# ---------------------------------------------------------------------------
def write_summary_md(summary_rows, raw_rows, path, datasets, anomaly_types):
    """Narrate where SAG-full underperforms and which predicate would help."""
    by_key = {}
    for r in summary_rows:
        if not r["applicable"]:
            continue
        by_key[(r["dataset"], r["anomaly_type"], r["method"], r["fusion_mode"])] = r
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# SAG failure-case and limitation analysis\n\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write(
            "Per-predicate AUROC tells us **which evidence source detects an "
            "anomaly type**. The auto fusion mode is SAG-full's actual output; "
            "the three single-predicate modes (`h_only`, `attr_only`, "
            "`struct_only`) tell us what each predicate alone can do, which "
            "in turn identifies the predicate the fusion would need to be "
            "good at to rescue a weak case.\n\n")
        for ds in datasets:
            f.write(f"## {ds}\n\n")
            for at in anomaly_types:
                stub = by_key.get((ds, at, "SAG-full", "auto"))
                if stub is None:
                    skip = next((r for r in summary_rows
                                 if r["dataset"] == ds and r["anomaly_type"] == at
                                 and not r["applicable"]), None)
                    if skip is not None:
                        f.write(f"### {at}\n\n_skipped:_ "
                                f"`{skip.get('skip_reason', 'not implemented')}`\n\n")
                    continue
                f.write(f"### {at}\n\n")
                f.write("| method | fusion | AUROC | AUPRC_bal | F1_bal |\n"
                        "|---|---|---|---|---|\n")
                # SAG-full per-predicate
                for fm in PREDICATE_MODES:
                    r = by_key.get((ds, at, "SAG-full", fm))
                    if r is None: continue
                    f.write("| SAG-full | `{fm}` | {a:.4f} | {b:.4f} | {c:.4f} |\n".format(
                        fm=fm, a=r["auroc_mean"], b=r["auprc_bal_mean"],
                        c=r["macro_f1_bal_mean"]))
                # Other methods
                for mt in ("GraphSAGE", "MIDAS"):
                    r = by_key.get((ds, at, mt, "auto"))
                    if r is None: continue
                    f.write("| {mt} | n/a | {a:.4f} | {b:.4f} | {c:.4f} |\n".format(
                        mt=mt, a=r["auroc_mean"], b=r["auprc_bal_mean"],
                        c=r["macro_f1_bal_mean"]))
                # Diagnosis line
                full = by_key.get((ds, at, "SAG-full", "auto"))
                preds = {fm: by_key.get((ds, at, "SAG-full", fm))
                         for fm in ("h_only", "attr_only", "struct_only")}
                if full is not None and any(preds.values()):
                    best = max((p for p in preds.items() if p[1] is not None),
                               key=lambda kv: kv[1]["auroc_mean"])
                    worst = min((p for p in preds.items() if p[1] is not None),
                                key=lambda kv: kv[1]["auroc_mean"])
                    fa = full["auroc_mean"]
                    f.write(f"\n_Diagnosis_: SAG-full AUROC = **{fa:.4f}**. "
                            f"Strongest predicate: `{best[0]}` "
                            f"(AUROC={best[1]['auroc_mean']:.4f}). "
                            f"Weakest: `{worst[0]}` "
                            f"(AUROC={worst[1]['auroc_mean']:.4f}). ")
                    if fa < 0.7:
                        f.write(
                            "Full-fusion AUROC is below 0.70 -- this is a "
                            "**failure case**. The strongest single predicate "
                            "lifts performance, so a downstream fix is to "
                            "either (a) add a predicate that captures this "
                            "anomaly type's signal directly (e.g. a multi-hop "
                            "structural statistic for community-bridge / "
                            "long-path anomalies) or (b) reweight the "
                            "Evidence Selection so the strongest predicate "
                            "isn't suppressed by the rank-consistency check.\n\n")
                    else:
                        f.write("Full-fusion AUROC is at or above 0.70 -- "
                                "the existing predicate set covers this "
                                "anomaly type adequately.\n\n")


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(DATASETS_DEFAULT))
    p.add_argument("--anomaly_types", default=",".join(ANOMALY_TYPES_DEFAULT))
    p.add_argument("--methods", default=",".join(METHODS_DEFAULT))
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS_DEFAULT))
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--tag", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _maybe_skip(path, overwrite, logger):
    if os.path.exists(path) and not overwrite:
        logger.warning(f"  refusing to overwrite {path} (pass --overwrite)")
        return True
    return False


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    types = [t.strip() for t in args.anomaly_types.split(",") if t.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    run_dir = os.path.join(args.output_dir, "results", "revision",
                           f"failure_case{tag}_{ts}")
    log_dir = os.path.join(args.output_dir, "logs", f"failure_case{tag}_{ts}")
    canon_dir = os.path.join(args.output_dir, "results", "revision")
    os.makedirs(run_dir, exist_ok=True); os.makedirs(log_dir, exist_ok=True)
    os.makedirs(canon_dir, exist_ok=True)

    logger = setup_logger("SAG_FAIL_CASE", log_dir)
    save_json({"datasets": datasets, "anomaly_types": types, "methods": methods,
               "seeds": seeds, "argv": sys.argv, "timestamp": ts},
              os.path.join(run_dir, "config.json"))

    Config.data_dir = "./data"
    Config.output_dir = args.output_dir

    raw_rows = []
    raw_run = os.path.join(run_dir, "failure_case_raw.csv")
    sum_run = os.path.join(run_dir, "failure_case_summary.csv")
    md_run = os.path.join(run_dir, "failure_case_summary.md")
    raw_canon = os.path.join(canon_dir, "failure_case_raw.csv")
    sum_canon = os.path.join(canon_dir, "failure_case_summary.csv")
    md_canon = os.path.join(canon_dir, "failure_case_summary.md")

    for ds in datasets:
        try:
            dc = get_dataset_config(ds)
        except ValueError as e:
            logger.error(f"unknown dataset {ds}: {e}")
            continue
        logger.info(f"\n========== {ds} ==========")
        try:
            raw_loader = load_raw(ds, dc)
        except Exception as e:
            logger.error(f"  load failed: {e}")
            continue

        for at in types:
            for sd in seeds:
                logger.info(f"\n  --- {ds} | {at} | seed={sd} ---")
                try:
                    anomdata = inject(raw_loader, at, dc, sd)
                except NotImplementedError as e:
                    logger.info(f"    NotImplemented for {at}: {e}")
                    for method in methods:
                        for fm in (PREDICATE_MODES if method == "SAG-full" else ("auto",)):
                            raw_rows.append({
                                "dataset": ds, "anomaly_type": at,
                                "method": method, "fusion_mode": fm, "seed": sd,
                                "auroc": float("nan"), "auprc": float("nan"),
                                "macro_f1": float("nan"), "auprc_bal": float("nan"),
                                "macro_f1_bal": float("nan"),
                                "num_nodes": 0, "num_edges": 0, "num_anomalies": 0,
                                "num_structural": 0, "num_attribute": 0,
                                "elapsed_s": 0.0, "skipped": True,
                                "error": "anomaly type not implemented",
                            })
                    continue
                except Exception as e:
                    logger.error(f"    inject failed: {e}")
                    continue

                y = anomdata["labels"]
                for method in methods:
                    if method == "SAG-full":
                        for fm in PREDICATE_MODES:
                            t0 = time.time()
                            try:
                                sc = run_method(method, anomdata, dc, sd,
                                                fusion_mode_override=fm)
                                m = metrics_for(y, sc)
                                raw_rows.append({
                                    "dataset": ds, "anomaly_type": at,
                                    "method": method, "fusion_mode": fm, "seed": sd,
                                    **m,
                                    "num_nodes": anomdata["num_nodes"],
                                    "num_edges": anomdata["num_edges"],
                                    "num_anomalies": anomdata["num_anomalies"],
                                    "num_structural": anomdata["num_structural"],
                                    "num_attribute": anomdata["num_attribute"],
                                    "elapsed_s": float(time.time() - t0),
                                    "skipped": False, "error": "",
                                })
                                logger.info(
                                    "    SAG-full[%s] AUROC=%.4f F1b=%.4f (%.1fs)" %
                                    (fm, m["auroc"], m["macro_f1_bal"], time.time() - t0))
                            except Exception as e:
                                tb = traceback.format_exc()
                                logger.error(f"    SAG-full[{fm}] FAILED: {e}\n{tb}")
                                raw_rows.append({
                                    "dataset": ds, "anomaly_type": at,
                                    "method": method, "fusion_mode": fm, "seed": sd,
                                    "auroc": float("nan"), "auprc": float("nan"),
                                    "macro_f1": float("nan"), "auprc_bal": float("nan"),
                                    "macro_f1_bal": float("nan"),
                                    "num_nodes": anomdata["num_nodes"],
                                    "num_edges": anomdata["num_edges"],
                                    "num_anomalies": anomdata["num_anomalies"],
                                    "num_structural": anomdata["num_structural"],
                                    "num_attribute": anomdata["num_attribute"],
                                    "elapsed_s": 0.0, "skipped": True,
                                    "error": f"{type(e).__name__}:{e}",
                                })
                    else:
                        if method not in BASELINES:
                            raw_rows.append({
                                "dataset": ds, "anomaly_type": at,
                                "method": method, "fusion_mode": "auto", "seed": sd,
                                "auroc": float("nan"), "auprc": float("nan"),
                                "macro_f1": float("nan"), "auprc_bal": float("nan"),
                                "macro_f1_bal": float("nan"),
                                "num_nodes": anomdata["num_nodes"],
                                "num_edges": anomdata["num_edges"],
                                "num_anomalies": anomdata["num_anomalies"],
                                "num_structural": anomdata["num_structural"],
                                "num_attribute": anomdata["num_attribute"],
                                "elapsed_s": 0.0, "skipped": True,
                                "error": "method not registered in BASELINES",
                            })
                            continue
                        t0 = time.time()
                        try:
                            sc = run_method(method, anomdata, dc, sd)
                            m = metrics_for(y, sc)
                            raw_rows.append({
                                "dataset": ds, "anomaly_type": at,
                                "method": method, "fusion_mode": "auto", "seed": sd,
                                **m,
                                "num_nodes": anomdata["num_nodes"],
                                "num_edges": anomdata["num_edges"],
                                "num_anomalies": anomdata["num_anomalies"],
                                "num_structural": anomdata["num_structural"],
                                "num_attribute": anomdata["num_attribute"],
                                "elapsed_s": float(time.time() - t0),
                                "skipped": False, "error": "",
                            })
                            logger.info(
                                "    %s AUROC=%.4f F1b=%.4f (%.1fs)" %
                                (method, m["auroc"], m["macro_f1_bal"], time.time() - t0))
                        except Exception as e:
                            tb = traceback.format_exc()
                            logger.error(f"    {method} FAILED: {e}\n{tb}")
                            raw_rows.append({
                                "dataset": ds, "anomaly_type": at,
                                "method": method, "fusion_mode": "auto", "seed": sd,
                                "auroc": float("nan"), "auprc": float("nan"),
                                "macro_f1": float("nan"), "auprc_bal": float("nan"),
                                "macro_f1_bal": float("nan"),
                                "num_nodes": anomdata["num_nodes"],
                                "num_edges": anomdata["num_edges"],
                                "num_anomalies": anomdata["num_anomalies"],
                                "num_structural": anomdata["num_structural"],
                                "num_attribute": anomdata["num_attribute"],
                                "elapsed_s": 0.0, "skipped": True,
                                "error": f"{type(e).__name__}:{e}",
                            })
                write_csv(raw_rows, RAW_COLS, raw_run)

    summary = aggregate_summary(raw_rows)
    write_csv(summary, SUMMARY_COLS, sum_run)
    write_summary_md(summary, raw_rows, md_run, datasets, types)
    save_json({"raw": raw_rows, "summary": summary},
              os.path.join(run_dir, "failure_case.json"))

    if not _maybe_skip(raw_canon, args.overwrite, logger):
        write_csv(raw_rows, RAW_COLS, raw_canon)
    if not _maybe_skip(sum_canon, args.overwrite, logger):
        write_csv(summary, SUMMARY_COLS, sum_canon)
    if not _maybe_skip(md_canon, args.overwrite, logger):
        write_summary_md(summary, raw_rows, md_canon, datasets, types)

    logger.info("=" * 70)
    logger.info(f"WROTE  raw     -> {raw_run}")
    logger.info(f"WROTE  summary -> {sum_run}")
    logger.info(f"WROTE  md      -> {md_run}")
    logger.info(f"CANON          -> {canon_dir}")


if __name__ == "__main__":
    main()
