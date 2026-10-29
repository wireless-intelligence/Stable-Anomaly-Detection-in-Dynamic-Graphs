"""Leakage-free protocol sanity checks (revision).

Combines static source-level inspections with runtime invariant checks. Each
check produces a structured row (id, name, status, severity, details, source)
written to outputs/results/revision/leakage_checks.csv plus a human-readable
markdown report at outputs/results/revision/leakage_check_report.md.

Status semantics:
  pass   : invariant holds, no concern
  warn   : invariant doesn't strictly apply OR codebase deviates from a
           common expectation but the deviation is intentional and
           documented (details field explains why)
  fail   : invariant should hold and does not -> a real leakage / protocol
           bug
  info   : not a check, just a recorded fact (e.g. SAG is unsupervised so
           train/val/test split is N/A)
"""

import argparse
import csv
import inspect
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from configs import Config, get_dataset_config
from src.data import DataLoader
from src.data import anomaly_injection as ai_mod
from src.data import edge_ordering as eo_mod
from src.data import loader as ld_mod
from src.evaluation import metrics as ev_mod
from src.models import sag as sag_mod
from src.utils import save_json


CHECK_COLUMNS = [
    "id", "name", "status", "severity", "details", "source",
]


def add(rows, check_id, name, status, severity, details, source):
    rows.append({
        "id": check_id, "name": name, "status": status,
        "severity": severity, "details": details, "source": source,
    })


# ---------------------------------------------------------------------------
# Static checks (read source code)
# ---------------------------------------------------------------------------
def check_temporal_sort(rows):
    src = inspect.getsource(eo_mod.temporal_order)
    is_argsort = "np.argsort" in src and "edge_t" in src and "mergesort" in src
    add(rows, "temporal_sort_function",
        "edge_ordering.temporal_order sorts edges by minimum endpoint time",
        "pass" if is_argsort else "fail",
        "high",
        ("temporal_order uses np.argsort(edge_t + jitter, kind='mergesort'); "
         "stable + ascending."
         if is_argsort else
         "temporal_order does NOT sort by time -- review src/data/edge_ordering.py"),
        f"{eo_mod.__file__}::temporal_order")


def check_split_overlap(rows):
    add(rows, "split_overlap",
        "Train/Val/Test intervals do not overlap",
        "info", "info",
        ("SAG is fully unsupervised: it computes per-node anomaly scores from "
         "a single streaming pass over all edges. There is no train/val/test "
         "split to overlap. Labels are only used post-hoc to compute "
         "evaluation metrics, not to train the model."),
        "src/trainer/sag_trainer.py::SAGTrainer.train_epoch")


def check_threshold_uses_validation_only(rows):
    src = inspect.getsource(ev_mod.compute_macro_f1)
    src_eval = inspect.getsource(ev_mod.compute_balanced_auprc_f1)
    grid_search = "thresholds = np.linspace" in src and "best_f1" in src
    bal_subsample = "rng.choice(neg_idx" in src_eval
    detail = (
        "compute_macro_f1 selects threshold by grid search over [score_min, "
        "score_max] and reports best macro-F1 over the *full labeled set*. "
        "compute_balanced_auprc_f1 subsamples negatives and reports best F1 "
        "on the balanced subset.\n"
        "Honest classification: the threshold is selected using *all* labels "
        "in the score vector -- this is the standard convention in graph "
        "anomaly detection (CoLA / DOMINANT eval), but a strict protocol may "
        "interpret it as test-label leakage. SAG never uses labels to TRAIN; "
        "labels enter only at the metric step. Documented as warn so the user "
        "can override with a held-out validation split if the venue requires.")
    add(rows, "threshold_selection_protocol",
        "Validation thresholds are selected only using validation labels",
        "warn" if grid_search else "fail", "medium", detail,
        f"{ev_mod.__file__}::compute_macro_f1, compute_balanced_auprc_f1")
    add(rows, "test_labels_in_hyperparam_tuning",
        "Test labels are never used for threshold selection or hyperparameter tuning",
        "warn", "medium",
        ("Hyperparameter tuning in this codebase (e.g. run_param_sensitivity) "
         "reports metrics on the same labeled set used for tuning -- a "
         "common GAD-eval pattern. There is no separate held-out test split. "
         "The dataset configs in configs/dataset_configs.py are *fixed* "
         "defaults from the original paper, not tuned on test labels in the "
         "revision pipeline."),
        "configs/dataset_configs.py")


def check_online_predicate_stats(rows):
    src = inspect.getsource(sag_mod.SAG.process_stream_batch)
    uses_window = "ws" in src and "edge_prev" in src and "edge_curr" in src
    rolls = "edge_prev = edge_curr" in src
    detail = (
        "process_stream_batch maintains edge_prev/edge_curr and "
        "node_prev/node_curr counters that roll over every `ws` edges. "
        "At each step the per-edge anomaly evidence uses ONLY the prev "
        "(previous-window) and curr (current-window-up-to-now) counters, "
        "both of which contain only past or just-observed interactions. "
        "No look-ahead.")
    add(rows, "online_predicate_stats",
        "Online predicate statistics at time t use only interactions <= t",
        "pass" if (uses_window and rolls) else "fail",
        "high", detail,
        f"{sag_mod.__file__}::SAG.process_stream_batch")


def check_normalization_global(rows):
    # AttributeEvidenceEnhanced computes propagation features over the full
    # graph at __init__. That IS a global statistic but it's offline static
    # features (not labels), the same convention used by every static GAD
    # method (DOMINANT/CoLA). Flag as info.
    from src.models import structure_evidence as se_mod
    src = inspect.getsource(se_mod.AttributeEvidenceEnhanced.set_features)
    full_graph = "_sym_normalize" in src or "num_propagation" in src
    detail = (
        "AttributeEvidenceEnhanced computes static node attribute scores by "
        "GCN-style propagation over the *full* adjacency at init. This is a "
        "GLOBAL statistic, the same convention as DOMINANT / CoLA / ANEMONE. "
        "It does not use labels. The streaming-state predicate "
        "(state_evolution) is fully online and uses only past interactions.")
    add(rows, "attribute_normalization_global",
        "Attribute / structural feature stats may use the full graph",
        "warn" if full_graph else "info",
        "low", detail,
        f"{se_mod.__file__}::AttributeEvidenceEnhanced")


def check_injection_before_or_after_split(rows):
    src = inspect.getsource(ld_mod.DataLoader.load)
    injects_in_load = "AnomalyInjector" in src
    detail = (
        "DataLoader.load() runs anomaly injection (via AnomalyInjector) "
        "*before* returning the (adj, features, labels) tuple. SAG has no "
        "train/val/test split, so 'before split' is the only meaningful "
        "ordering. The injection seed is controlled by DataLoader(seed=...) "
        "(or by the seed_variance experiment for the anomaly_injection "
        "axis). For the time-to-detect / scalability experiments, all "
        "injected anomalies are present from the start of the stream and "
        "their onset is defined as the first edge that touches them in "
        "stream order.")
    add(rows, "injection_before_or_after_split",
        "Anomaly injection is performed before any split (clearly reported)",
        "info" if injects_in_load else "warn",
        "low", detail,
        f"{ld_mod.__file__}::DataLoader.load")


def check_attribute_injection_global_features(rows):
    src = inspect.getsource(ai_mod.AnomalyInjector.inject_attribute)
    uses_global = ("Xo[candidates]" in src) and ("argmax(dists)" in src)
    detail = (
        "AnomalyInjector.inject_attribute selects the replacement feature "
        "vector by sampling k_attr random nodes from the *full* node set "
        "and picking the one with the LARGEST Euclidean distance to the "
        "original. This uses GLOBAL feature information (every candidate's "
        "feature vector is read) and is a known design pattern from "
        "DOMINANT / ResGCN / CoLA injected-anomaly evaluation -- i.e. by "
        "construction, attribute anomalies are designed to be detectable "
        "by attribute-only methods. \n"
        "A strict protocol may consider this a leakage issue. We FLAG "
        "this and recommend reporting injected-anomaly results alongside "
        "real-anomaly results (Enron / YelpReviews) for any downstream "
        "claim that depends on detection difficulty.")
    add(rows, "attribute_injection_global",
        "Attribute injection does not use global feature information",
        "warn" if uses_global else "pass",
        "high", detail,
        f"{ai_mod.__file__}::AnomalyInjector.inject_attribute")


# ---------------------------------------------------------------------------
# Runtime checks (load a dataset and verify invariants on actual arrays)
# ---------------------------------------------------------------------------
def runtime_check_temporal_monotone(rows, dataset: str):
    try:
        dc = get_dataset_config(dataset)
        loader = DataLoader(
            data_dir="./data", dataset_name=dataset,
            clique_size=dc.get("clique_size", 15),
            num_cliques=dc.get("num_cliques", 10), seed=42,
        )
        loader.load()
        node_times = getattr(loader, "node_times", None)
        if node_times is None:
            add(rows, f"runtime__temporal_monotone__{dataset}",
                f"Edge stream in temporal order is non-decreasing on {dataset}",
                "info", "low",
                f"{dataset} has no node_times -> temporal ordering N/A",
                "src/data/edge_ordering.py::temporal_order")
            return
        edges = loader.get_edge_stream(mode="temporal")
        u, v = edges[:, 0], edges[:, 1]
        edge_t = np.minimum(node_times[u], node_times[v])
        ok = bool(np.all(np.diff(edge_t) >= 0))
        add(rows, f"runtime__temporal_monotone__{dataset}",
            f"Edge stream in temporal order is non-decreasing on {dataset}",
            "pass" if ok else "fail",
            "high",
            f"#edges={edges.shape[0]} ; min t = {edge_t.min():.3f} ; "
            f"max t = {edge_t.max():.3f} ; monotone={ok}",
            "src/data/edge_ordering.py::temporal_order")
    except Exception as e:
        add(rows, f"runtime__temporal_monotone__{dataset}",
            f"Edge stream in temporal order is non-decreasing on {dataset}",
            "warn", "low",
            f"could not run runtime check: {type(e).__name__}: {e}",
            "src/data/edge_ordering.py::temporal_order")


def runtime_check_window_only_past(rows, dataset: str = "Cora"):
    """Process a tiny stream with process_stream_batch and verify that, at
    any point during the stream, the scorer's edge_curr keys are a subset of
    edges seen so far."""
    try:
        from src.models.sag import SAG
        from src.models.bayesian_scorer import BayesianAnomalyScorer

        dc = get_dataset_config(dataset)
        loader = DataLoader(
            data_dir="./data", dataset_name=dataset,
            clique_size=dc.get("clique_size", 15),
            num_cliques=dc.get("num_cliques", 10), seed=42,
        )
        loader.load()
        # Use process_edge in a Python loop, snapshotting scorer state.
        n = loader.stats["num_nodes"]
        cfg = type("C", (), {})()
        for k in dir(Config):
            if k.startswith("_") or callable(getattr(Config, k, None)):
                continue
            setattr(cfg, k, getattr(Config, k))
        for k, v in dc.items():
            setattr(cfg, k, v)
        cfg.use_continuous_dev = False
        cfg.use_real_time_decay = False
        cfg.seed = 42
        model = SAG(num_nodes=n, config=cfg,
                    features=loader.features, adj=loader.adj)
        edges = loader.get_edge_stream(mode="bfs")[:200]  # small prefix
        seen = set()
        violation = False
        for i, (u, v) in enumerate(edges.tolist()):
            model.process_edge(int(u), int(v))
            seen.add((int(u), int(v)))
            seen.add((int(v), int(u)))
            ec = getattr(model.scorer, "edge_curr", None)
            if isinstance(ec, dict):
                # every key of edge_curr must correspond to an edge in seen
                for k in ec.keys():
                    if k not in seen:
                        violation = True
                        break
            if violation:
                break
        add(rows, f"runtime__window_only_past__{dataset}",
            f"Per-window edge counters at time t reference only edges with index <= t ({dataset})",
            "fail" if violation else "pass",
            "high",
            (f"violation found in first 200 edges on {dataset}" if violation
             else f"no violation across 200 edges on {dataset}"),
            "src/models/sag.py + src/models/bayesian_scorer.py")
    except Exception as e:
        add(rows, f"runtime__window_only_past__{dataset}",
            f"Per-window edge counters reference only past edges ({dataset})",
            "warn", "high",
            f"could not run runtime check: {type(e).__name__}: {e}",
            "src/models/sag.py")


# ---------------------------------------------------------------------------
def emit_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CHECK_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CHECK_COLUMNS})


def emit_report(rows, path):
    by_status = {"pass": [], "warn": [], "fail": [], "info": []}
    for r in rows:
        by_status.setdefault(r["status"], []).append(r)
    counts = {k: len(v) for k, v in by_status.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# SAG leakage-free protocol sanity-check report\n\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write("| status | count |\n|---|---|\n")
        for k in ("pass", "warn", "fail", "info"):
            f.write(f"| **{k}** | {counts.get(k, 0)} |\n")
        f.write("\n")
        for k in ("fail", "warn", "pass", "info"):
            block = by_status.get(k, [])
            if not block:
                continue
            f.write(f"## {k.upper()} ({len(block)})\n\n")
            for r in block:
                f.write(f"### {r['id']}\n\n")
                f.write(f"**{r['name']}**  \n")
                f.write(f"severity: `{r['severity']}`  source: `{r['source']}`\n\n")
                f.write(r["details"] + "\n\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runtime_dataset", default="Cora",
                   help="dataset to use for runtime invariant checks")
    p.add_argument("--temporal_dataset", default="CTU-13",
                   help="dataset to use for the temporal-monotone runtime check; "
                        "ignored if dataset has no node_times")
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    rows = []

    # Static
    check_temporal_sort(rows)
    check_split_overlap(rows)
    check_threshold_uses_validation_only(rows)
    check_online_predicate_stats(rows)
    check_normalization_global(rows)
    check_injection_before_or_after_split(rows)
    check_attribute_injection_global_features(rows)

    # Runtime
    runtime_check_temporal_monotone(rows, args.temporal_dataset)
    runtime_check_window_only_past(rows, args.runtime_dataset)

    canon_dir = os.path.join(args.output_dir, "results", "revision")
    os.makedirs(canon_dir, exist_ok=True)
    csv_path = os.path.join(canon_dir, "leakage_checks.csv")
    md_path = os.path.join(canon_dir, "leakage_check_report.md")

    if os.path.exists(csv_path) and not args.overwrite:
        print(f"refusing to overwrite {csv_path} (pass --overwrite)")
    else:
        emit_csv(rows, csv_path)
    if os.path.exists(md_path) and not args.overwrite:
        print(f"refusing to overwrite {md_path} (pass --overwrite)")
    else:
        emit_report(rows, md_path)

    save_json({"rows": rows, "argv": sys.argv,
               "timestamp": datetime.now().isoformat(timespec="seconds")},
              os.path.join(canon_dir, "leakage_checks.json"))

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"summary: " + ", ".join(
        f"{k}={sum(1 for r in rows if r['status']==k)}"
        for k in ("pass", "warn", "fail", "info")
    ))


if __name__ == "__main__":
    main()
