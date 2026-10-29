"""Split-protocol selection experiment.

SAG is evaluated transductively without a train/val/test split. To compare
with strict split protocols, this script evaluates four candidate
splits on each priority dataset and records the data needed to defend the
choice of main protocol -- WITHOUT cherry-picking the split that maximises
SAG performance.

Splits
------
  60_20_20      first 60% / next 20% / last 20%        edge-chronological
  70_10_20      first 70% / next 10% / last 20%
  70_20_10      first 70% / next 20% / last 10%
  70_30_rank    first 70% / last 30%                   ranking-only (no val)

The edge stream order is the dataset's native order (CTU-13 -> temporal,
ground-truth datasets -> shuffle(seed=42), injected datasets -> bfs(seed=42)).
A node is assigned to phase P iff its FIRST appearance in the edge stream
is in phase P. Anomaly counts per phase = #anomaly nodes whose first
appearance is in that phase.

For each (dataset, split, method, seed) we run methods that can honestly
respect the split:

  SAG-online-strict : single chronological streaming pass with snapshots at
                      train-end and val-end; threshold selected on val
                      labels; final metrics on test labels (or eval labels
                      for ranking_only).
  MIDAS             : same streaming protocol (re-implemented inline with
                      checkpoint snapshots).
  DOMINANT (fallback) : strong static GAD baseline; fit on train edges only
                        (graph built from train_edges with same node count);
                        score the full node set; evaluate on val/test nodes.
  ConsisGAD         : strong GNN/GAD baseline; same static fit-on-train
                      protocol.

NetWalk / Graphormer etc. are skipped here for cost reasons -- can be added
via --methods.

Outputs (canonical, latest run; --overwrite to replace):
  outputs/results/revision_strict/split_selection_metadata.csv
  outputs/results/revision_strict/split_selection_results.csv
  outputs/results/revision_strict/split_selection_summary.md
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
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, confusion_matrix,
)

from configs import Config, get_dataset_config
from src.baselines import BASELINES
from src.data import DataLoader
from src.evaluation.metrics import compute_balanced_auprc_f1
from src.models import create_sag_model
from src.utils import save_json, set_seed, setup_logger


SPLITS = [
    ("60_20_20",   (0.60, 0.20, 0.20), False),
    ("70_10_20",   (0.70, 0.10, 0.20), False),
    ("70_20_10",   (0.70, 0.20, 0.10), False),
    ("70_30_rank", (0.70, 0.00, 0.30), True),
]
METHODS_DEFAULT = ("SAG-online-strict", "MIDAS", "DOMINANT", "ANEMONE")
DATASETS_DEFAULT = ("Cora", "Citeseer", "Pubmed", "BlogCatalog",
                    "YelpReviews", "Enron", "CTU-13")
SEEDS_DEFAULT = list(range(42, 52))   # 42..51, 10 seeds
BASE_LOAD_SEED = 42

# Sufficiency rules of thumb
MIN_VAL_ANOMALIES_FOR_THRESHOLD = 10
MIN_TEST_ANOMALIES_FOR_F1 = 30
MIN_ANOMALIES_FOR_AUROC = 5
MIN_TEST_ANOMALIES_FOR_PATK = 5


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


# ---------------------------------------------------------------------------
def split_boundaries(num_edges: int, fractions):
    """Return (end_train, end_val, end_test) edge indices."""
    f_tr, f_va, f_te = fractions
    total = f_tr + f_va + f_te
    assert abs(total - 1.0) < 1e-6, f"fractions must sum to 1, got {total}"
    end_tr = int(round(num_edges * f_tr))
    end_va = end_tr + int(round(num_edges * f_va))
    end_te = num_edges
    return end_tr, end_va, end_te


def first_appearance_phase(edges: np.ndarray, num_nodes: int,
                            end_tr: int, end_va: int, end_te: int) -> np.ndarray:
    """Return array of length num_nodes: 0=train, 1=val, 2=test, 3=never."""
    INF = np.iinfo(np.int64).max
    first = np.full(num_nodes, INF, dtype=np.int64)
    M = edges.shape[0]
    u = edges[:, 0]; v = edges[:, 1]
    for i in range(M):
        if first[u[i]] == INF: first[u[i]] = i
        if first[v[i]] == INF: first[v[i]] = i
    phase = np.full(num_nodes, 3, dtype=np.int8)
    phase[first <  end_tr] = 0
    mask_v = (first >= end_tr) & (first < end_va)
    phase[mask_v] = 1
    mask_te = (first >= end_va) & (first < end_te)
    phase[mask_te] = 2
    return phase


def compute_split_metadata(dataset, edges, labels, num_nodes, split_name,
                           fractions, ranking_only):
    end_tr, end_va, end_te = split_boundaries(edges.shape[0], fractions)
    phase = first_appearance_phase(edges, num_nodes, end_tr, end_va, end_te)
    train_nodes = np.where(phase == 0)[0]
    val_nodes   = np.where(phase == 1)[0]
    test_nodes  = np.where(phase == 2)[0]
    never_nodes = np.where(phase == 3)[0]

    def stat(node_set, edge_count):
        n = int(node_set.size)
        a = int(np.sum(labels[node_set])) if n > 0 else 0
        ratio = (a / n) if n > 0 else 0.0
        return n, a, ratio, edge_count

    n_tr, a_tr, r_tr, _ = stat(train_nodes, end_tr)
    n_va, a_va, r_va, _ = stat(val_nodes,   max(0, end_va - end_tr))
    n_te, a_te, r_te, _ = stat(test_nodes,  max(0, end_te - end_va))
    n_nv, a_nv, r_nv, _ = stat(never_nodes, 0)

    if ranking_only:
        # treat val as empty; "test" maps to the eval span (= last 30%)
        # Re-stat: any node first-appearing in eval phase counts as eval.
        end_eval = end_te
        # In ranking_only, val phase has 0 fraction; what's classified as
        # phase=1 above would all be empty. The 30% goes to phase=2.
        applicable_macro_f1 = False
        applicable_auroc_auprc = (a_te >= MIN_ANOMALIES_FOR_AUROC and (n_te - a_te) >= MIN_ANOMALIES_FOR_AUROC)
        applicable_patk = (a_te >= MIN_TEST_ANOMALIES_FOR_PATK)
        val_sufficient = False
        test_sufficient = (a_te >= MIN_TEST_ANOMALIES_FOR_F1)
    else:
        applicable_macro_f1 = (a_va >= MIN_ANOMALIES_FOR_AUROC and a_te >= MIN_ANOMALIES_FOR_AUROC)
        applicable_auroc_auprc = (a_te >= MIN_ANOMALIES_FOR_AUROC and (n_te - a_te) >= MIN_ANOMALIES_FOR_AUROC)
        applicable_patk = (a_te >= MIN_TEST_ANOMALIES_FOR_PATK)
        val_sufficient = (a_va >= MIN_VAL_ANOMALIES_FOR_THRESHOLD)
        test_sufficient = (a_te >= MIN_TEST_ANOMALIES_FOR_F1)

    return {
        "dataset": dataset, "split": split_name,
        "num_edges_total": int(edges.shape[0]),
        "end_train_edge_idx": int(end_tr),
        "end_val_edge_idx":   int(end_va),
        "end_test_edge_idx":  int(end_te),
        "train_edges": int(end_tr),
        "val_edges":   int(max(0, end_va - end_tr)),
        "test_edges":  int(max(0, end_te - end_va)),
        "train_nodes": n_tr, "train_anomalies": a_tr, "train_anomaly_ratio": r_tr,
        "val_nodes":   n_va, "val_anomalies":   a_va, "val_anomaly_ratio":   r_va,
        "test_nodes":  n_te, "test_anomalies":  a_te, "test_anomaly_ratio":  r_te,
        "never_nodes": n_nv, "never_anomalies": a_nv, "never_anomaly_ratio": r_nv,
        "ranking_only": bool(ranking_only),
        "val_anomaly_sufficient_for_threshold":  bool(val_sufficient),
        "test_anomaly_sufficient_for_F1":        bool(test_sufficient),
        "applicable_macro_f1":   bool(applicable_macro_f1),
        "applicable_auroc_auprc": bool(applicable_auroc_auprc),
        "applicable_precision_at_k": bool(applicable_patk),
    }, phase


# ---------------------------------------------------------------------------
def normalize(s):
    s = np.asarray(s, dtype=np.float64)
    if s.max() > s.min():
        return (s - s.min()) / (s.max() - s.min())
    return np.zeros_like(s)


def best_f1_threshold(y, p):
    if y.sum() == 0 or y.sum() == len(y) or p.max() <= p.min():
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


def evaluate_split(scores_val, scores_test, val_nodes, test_nodes, labels,
                   ranking_only=False):
    out = {
        "auroc": float("nan"), "auprc": float("nan"),
        "macro_f1": float("nan"), "macro_f1_bal": float("nan"),
        "auprc_bal": float("nan"),
        "tpr_at_thr": float("nan"), "fpr_at_thr": float("nan"),
        "precision_at_K": float("nan"), "recall_at_K": float("nan"),
        "best_f1_threshold": float("nan"), "best_macro_f1": float("nan"),
        "K": 0,
    }
    if test_nodes.size == 0:
        return out
    y_test = labels[test_nodes]
    s_test = normalize(scores_test[test_nodes])
    if y_test.sum() in (0, len(y_test)):
        return out
    try:    out["auroc"] = float(roc_auc_score(y_test, s_test))
    except ValueError: pass
    try:    out["auprc"] = float(average_precision_score(y_test, s_test))
    except ValueError: pass
    auprc_bal, f1_bal = compute_balanced_auprc_f1(y_test, s_test, n_trials=20, seed=0)
    out["auprc_bal"]    = float(auprc_bal)
    out["macro_f1_bal"] = float(f1_bal)

    K = int(y_test.sum())
    out["K"] = K
    if K > 0 and K <= len(s_test):
        topk_idx = np.argsort(-s_test)[:K]
        out["precision_at_K"] = float(np.sum(y_test[topk_idx]) / K)
        out["recall_at_K"]    = float(np.sum(y_test[topk_idx]) / max(1, y_test.sum()))

    if ranking_only:
        return out

    if val_nodes.size == 0:
        return out
    y_val = labels[val_nodes]
    s_val = normalize(scores_val[val_nodes])
    if y_val.sum() in (0, len(y_val)):
        return out

    thr, f1_val = best_f1_threshold(y_val, s_val)
    out["best_f1_threshold"] = float(thr)
    out["best_macro_f1"]     = float(f1_val)

    yp = (s_test >= thr).astype(int)
    try:
        f1_test = f1_score(y_test, yp, average="macro", zero_division=0)
        out["macro_f1"] = float(f1_test)
    except ValueError: pass
    cm = confusion_matrix(y_test, yp, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    out["tpr_at_thr"] = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    out["fpr_at_thr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
def run_sag_online_strict(loader, dc, edges, end_tr, end_va, end_te, seed,
                           ae_cache):
    """Single chronological streaming pass with snapshots."""
    set_seed(seed)
    cfg = build_cfg(dc, seed=seed)
    s = loader.get_stats()
    cfg.window_size = max(128, min(s["num_edges"] // 32, 4096))
    if ae_cache is not None:
        cfg._trained_ae_score_cache = ae_cache
    model = create_sag_model(num_nodes=s["num_nodes"], config=cfg,
                             features=loader.features, adj=loader.adj)
    rng = np.random.default_rng(seed)

    # Process train phase
    if end_tr > 0:
        model.process_stream_batch(edges[:end_tr], passes=1, shuffle=False, rng=rng)
    score_train = model.state.scores().copy()

    # Process val phase
    if end_va > end_tr:
        model.process_stream_batch(edges[end_tr:end_va], passes=1, shuffle=False, rng=rng)
    score_val = model.state.scores().copy()

    # Process test phase
    if end_te > end_va:
        model.process_stream_batch(edges[end_va:end_te], passes=1, shuffle=False, rng=rng)
    score_test = model.state.scores().copy()

    return score_train, score_val, score_test


def run_midas_strict(num_nodes, edges, end_tr, end_va, end_te, seed,
                      num_hash=2, buckets=2 ** 14):
    rng = np.random.default_rng(seed)
    a1 = rng.integers(1, 2 ** 31 - 1, size=num_hash)
    b1 = rng.integers(0, 2 ** 31 - 1, size=num_hash)
    n = num_nodes
    node_scores = np.zeros(n, dtype=np.float64)
    window_size = max(1, edges.shape[0] // 128)
    cur = np.zeros((num_hash, buckets), dtype=np.float64)
    tot = np.zeros((num_hash, buckets), dtype=np.float64)
    wcount = 0; t = 1
    u_arr = edges[:, 0].astype(np.int64); v_arr = edges[:, 1].astype(np.int64)
    ekey = u_arr * n + v_arr
    h_e = (a1[:, None] * ekey[None, :] + b1[:, None]) % buckets
    snaps = {}
    boundaries = sorted(set([end_tr, end_va, end_te]))
    cp_iter = iter(boundaries)
    next_cp = next(cp_iter, None)
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
                s_cur = cur[hi, idx]; s_tot = tot[hi, idx]
                expected = s_tot / t
                if expected > 0:
                    edge_scores[hi] = ((s_cur - expected) ** 2) / (expected * max(1, t - 1) / t)
            node_scores[u] += float(np.min(edge_scores))
            node_scores[v] += float(np.min(edge_scores))
            cur.fill(0.0); wcount = 0
        while next_cp is not None and (i + 1) >= next_cp:
            snaps[next_cp] = node_scores.copy()
            next_cp = next(cp_iter, None)
    if M not in snaps:
        snaps[M] = node_scores.copy()

    # Default to the last available snapshot if a boundary wasn't hit.
    def get(idx):
        if idx in snaps:
            return snaps[idx]
        keys = [k for k in snaps if k >= idx]
        if keys: return snaps[min(keys)]
        return snaps[max(snaps.keys())]
    return get(end_tr), get(end_va), get(end_te)


def build_train_graph(num_nodes: int, edges_train: np.ndarray):
    """Build a CSR adjacency from train edges only (full N x N), symmetrized,
    self-loop-free."""
    if edges_train.shape[0] == 0:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float32)
    rows = edges_train[:, 0].astype(np.int64)
    cols = edges_train[:, 1].astype(np.int64)
    data = np.ones(rows.shape[0], dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    A = A.maximum(A.T)
    A.setdiag(0); A.eliminate_zeros()
    A.data[:] = 1.0
    return A


def run_static_baseline_strict(method, loader, dc, edges, end_tr, seed):
    """Fit static baseline on train_edges only; same train graph for val and
    test scoring -- the static method has a SINGLE score vector."""
    set_seed(seed)
    cfg = build_cfg(dc, seed=seed)
    A_train = build_train_graph(loader.stats["num_nodes"], edges[:end_tr])
    if method not in BASELINES:
        raise ValueError(f"baseline {method} not in BASELINES")
    cls = BASELINES[method]
    try:
        bm = cls(num_nodes=loader.stats["num_nodes"], config=cfg,
                 features=loader.features, seed=seed)
    except TypeError:
        bm = cls(num_nodes=loader.stats["num_nodes"], config=cfg,
                 features=loader.features)
    bm.fit(A_train, edges[:end_tr], None)
    sc = np.asarray(bm.score(), dtype=np.float64)
    if sc.shape[0] != loader.stats["num_nodes"]:
        raise RuntimeError(
            f"{method} returned {sc.shape[0]} scores, expected "
            f"{loader.stats['num_nodes']}")
    # static method has one score vector; use it for both val and test
    return sc, sc, sc


# ---------------------------------------------------------------------------
META_COLUMNS = [
    "dataset", "split",
    "num_edges_total",
    "end_train_edge_idx", "end_val_edge_idx", "end_test_edge_idx",
    "train_edges", "val_edges", "test_edges",
    "train_nodes", "train_anomalies", "train_anomaly_ratio",
    "val_nodes", "val_anomalies", "val_anomaly_ratio",
    "test_nodes", "test_anomalies", "test_anomaly_ratio",
    "never_nodes", "never_anomalies", "never_anomaly_ratio",
    "ranking_only",
    "val_anomaly_sufficient_for_threshold",
    "test_anomaly_sufficient_for_F1",
    "applicable_macro_f1",
    "applicable_auroc_auprc",
    "applicable_precision_at_k",
]
RESULT_COLUMNS = [
    "dataset", "split", "method", "seed", "ranking_only",
    "auroc", "auprc", "auprc_bal", "macro_f1", "macro_f1_bal",
    "best_f1_threshold", "best_macro_f1",
    "tpr_at_thr", "fpr_at_thr",
    "precision_at_K", "recall_at_K", "K",
    "train_anomalies", "val_anomalies", "test_anomalies",
    "elapsed_s", "skipped", "error",
]


def write_csv(rows, cols, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


# ---------------------------------------------------------------------------
def write_summary_md(meta_rows, result_rows, path):
    """Write the markdown analysis. Selection criteria documented in code."""
    import numpy as np
    by_split = {}
    for r in result_rows:
        if r.get("skipped"): continue
        by_split.setdefault((r["dataset"], r["split"], r["method"]), []).append(r)

    # Aggregate per (dataset, split, method) the AUROC mean/std/n
    agg = {}
    for k, rs in by_split.items():
        v = np.array([float(r["auroc"]) for r in rs if r["auroc"] != "" and not np.isnan(float(r["auroc"]))], dtype=np.float64)
        if v.size > 0:
            agg[k] = {
                "n": int(v.size),
                "mean": float(v.mean()),
                "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            }

    splits = sorted({k[1] for k in agg.keys()})
    methods = sorted({k[2] for k in agg.keys()})
    datasets = sorted({k[0] for k in agg.keys()})

    # per-split sufficiency / applicability totals
    suf = {}
    for m in meta_rows:
        s = m["split"]
        suf.setdefault(s, {"n_dataset": 0, "val_ok": 0, "test_ok": 0,
                            "f1_ok": 0, "auroc_ok": 0, "patk_ok": 0,
                            "rank_only": int(bool(m["ranking_only"]))})
        suf[s]["n_dataset"] += 1
        suf[s]["val_ok"]    += int(bool(m["val_anomaly_sufficient_for_threshold"]))
        suf[s]["test_ok"]   += int(bool(m["test_anomaly_sufficient_for_F1"]))
        suf[s]["f1_ok"]     += int(bool(m["applicable_macro_f1"]))
        suf[s]["auroc_ok"]  += int(bool(m["applicable_auroc_auprc"]))
        suf[s]["patk_ok"]   += int(bool(m["applicable_precision_at_k"]))

    # average AUROC std per split (lower = more stable)
    split_std_agg = {s: [] for s in splits}
    for (ds, s, mt), a in agg.items():
        split_std_agg[s].append(a["std"])
    split_std_mean = {s: (float(np.mean(split_std_agg[s])) if split_std_agg[s] else float("nan"))
                      for s in splits}

    # cherry-pick risk: ranking of SAG vs the worst baseline per split
    sag_advantage_per_split = {s: [] for s in splits}
    for ds in datasets:
        for s in splits:
            sag_key = (ds, s, "SAG-online-strict")
            if sag_key not in agg: continue
            others = [agg[(ds, s, m)] for m in methods if m != "SAG-online-strict" and (ds, s, m) in agg]
            if not others: continue
            sag = agg[sag_key]["mean"]
            best_other = max(o["mean"] for o in others)
            sag_advantage_per_split[s].append(sag - best_other)
    split_adv_mean = {s: (float(np.mean(sag_advantage_per_split[s])) if sag_advantage_per_split[s] else float("nan"))
                      for s in splits}
    # cherry-pick risk = max advantage minus min advantage
    cherry_risk = (max(split_adv_mean.values(), default=0)
                   - min(split_adv_mean.values(), default=0)) if split_adv_mean else 0.0

    # write
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "w", encoding="utf-8")
    f.write("# Split-protocol selection — code-grounded summary\n\n")
    f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}  \n")
    f.write("Selection criteria are NOT based on SAG performance; they are\n")
    f.write("based on (1) sufficient anomaly counts in val/test, (2) cross-seed\n")
    f.write("stability, (3) applicability to all metrics needed by the paper,\n")
    f.write("(4) protocol robustness.\n\n")

    f.write("## Sufficiency / applicability count by split\n\n")
    f.write("| split | datasets | val_ok | test_ok | F1_ok | AUROC_ok | P@K_ok | mean AUROC std | mean SAG advantage |\n")
    f.write("|---|---|---|---|---|---|---|---|---|\n")
    for s in splits:
        ss = suf.get(s, {})
        std = split_std_mean.get(s, float("nan"))
        adv = split_adv_mean.get(s, float("nan"))
        f.write(f"| `{s}` | {ss.get('n_dataset',0)} | {ss.get('val_ok','-')}/{ss.get('n_dataset',0)} | "
                f"{ss.get('test_ok','-')}/{ss.get('n_dataset',0)} | "
                f"{ss.get('f1_ok','-')}/{ss.get('n_dataset',0)} | "
                f"{ss.get('auroc_ok','-')}/{ss.get('n_dataset',0)} | "
                f"{ss.get('patk_ok','-')}/{ss.get('n_dataset',0)} | "
                f"{std:.4f} | {adv:+.4f} |\n")
    f.write("\n")
    f.write("Legend:  val_ok = #datasets where val_anomalies >= 10 (enough to "
            "select threshold);  test_ok = #datasets where test_anomalies >= 30 "
            "(stable F1 estimate);  F1_ok / AUROC_ok / P@K_ok = #datasets where "
            "the metric is *applicable* under the split (positives + negatives "
            "exist in test).  AUROC std is averaged across (dataset, method) "
            "cells.  SAG advantage = SAG_AUROC - max(other-method AUROC).\n\n")

    # Per-(dataset, split) applicability flags
    f.write("## Per-dataset metadata (sufficiency flags)\n\n")
    f.write("| dataset | split | train_edges | val_edges | test_edges | val_anom | test_anom | val_ok | test_ok | F1 | AUROC | P@K |\n")
    f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for m in sorted(meta_rows, key=lambda m: (m["dataset"], m["split"])):
        f.write(f"| {m['dataset']} | `{m['split']}` | {m['train_edges']:,} | "
                f"{m['val_edges']:,} | {m['test_edges']:,} | {m['val_anomalies']} | "
                f"{m['test_anomalies']} | {'✓' if m['val_anomaly_sufficient_for_threshold'] else '✗'} | "
                f"{'✓' if m['test_anomaly_sufficient_for_F1'] else '✗'} | "
                f"{'✓' if m['applicable_macro_f1'] else '✗'} | "
                f"{'✓' if m['applicable_auroc_auprc'] else '✗'} | "
                f"{'✓' if m['applicable_precision_at_k'] else '✗'} |\n")
    f.write("\n")

    # Decision questions
    f.write("## Selection questions\n\n")

    # Q1: most-applicable split
    best_apply = max(splits, key=lambda s: suf[s]["f1_ok"] + suf[s]["test_ok"] + suf[s]["val_ok"]) if splits else "—"
    f.write(f"**Q: Most-applicable split (val/test/F1 sufficient on most datasets)?**  \n  → `{best_apply}`\n\n")

    # Q2: smallest val
    smallest_val = min(splits, key=lambda s: suf[s]["val_ok"]) if splits else "—"
    f.write(f"**Q: Which split has insufficient val anomalies most often?**  \n  → `{smallest_val}` (val_ok = {suf.get(smallest_val,{}).get('val_ok','-')}/{suf.get(smallest_val,{}).get('n_dataset',0)})\n\n")

    # Q3: smallest test
    smallest_test = min(splits, key=lambda s: suf[s]["test_ok"]) if splits else "—"
    f.write(f"**Q: Which split has insufficient test anomalies most often?**  \n  → `{smallest_test}` (test_ok = {suf.get(smallest_test,{}).get('test_ok','-')}/{suf.get(smallest_test,{}).get('n_dataset',0)})\n\n")

    # Q4: most stable
    if splits and not all(np.isnan(split_std_mean[s]) for s in splits):
        most_stable = min(splits, key=lambda s: split_std_mean[s] if split_std_mean[s] == split_std_mean[s] else 1e9)
        f.write(f"**Q: Which split has the smallest cross-seed AUROC std?**  \n  → `{most_stable}` (mean std = {split_std_mean[most_stable]:.4f})\n\n")
    else:
        f.write("**Q: Which split is most stable across seeds?**  → UNKNOWN (no AUROC results yet)\n\n")

    # Q5: SAG advantage stability
    f.write(f"**Q: Is SAG's advantage stable across splits?**  \n")
    if all(not np.isnan(split_adv_mean[s]) for s in splits) and splits:
        f.write(f"  Range of SAG-vs-best-other advantage across splits = {cherry_risk:+.4f}  \n")
        if cherry_risk < 0.02:
            f.write("  → YES, SAG advantage is split-invariant (range < 0.02 AUROC). Low cherry-pick risk.\n\n")
        elif cherry_risk < 0.05:
            f.write("  → PARTIAL. Some split-dependence but acceptable.\n\n")
        else:
            f.write("  → NO. SAG advantage varies across splits — cherry-pick risk HIGH.\n\n")
    else:
        f.write("  → UNKNOWN until results available.\n\n")

    # Q6: cherry-pick risk
    f.write(f"**Q: Does any split look like a cherry-pick (SAG outperforms only there)?**  \n")
    if all(not np.isnan(split_adv_mean[s]) for s in splits) and splits:
        sorted_splits = sorted(split_adv_mean, key=lambda s: split_adv_mean[s], reverse=True)
        if sorted_splits:
            top = sorted_splits[0]
            f.write(f"  Highest-advantage split: `{top}` (advantage = {split_adv_mean[top]:+.4f})\n")
            if len(sorted_splits) > 1:
                gap = split_adv_mean[top] - split_adv_mean[sorted_splits[1]]
                if gap > 0.05:
                    f.write(f"  → FLAG: gap to next split = {gap:+.4f} AUROC. Avoid this split as the main protocol.\n\n")
                else:
                    f.write(f"  → No alarming gap (next split: `{sorted_splits[1]}`, gap = {gap:+.4f}).\n\n")
        else:
            f.write("  → UNKNOWN.\n\n")
    else:
        f.write("  → UNKNOWN until results available.\n\n")

    # Q7-8: recommendations
    f.write("## Recommendations\n\n")
    if splits:
        # Recommended main = max of (val_ok + test_ok + applicability) ties broken by lowest std
        def score_split(s):
            ss = suf.get(s, {})
            n = max(1, ss.get("n_dataset", 1))
            applicability = (ss.get("val_ok", 0) + ss.get("test_ok", 0)
                             + ss.get("f1_ok", 0) + ss.get("auroc_ok", 0)
                             + ss.get("patk_ok", 0)) / (5 * n)
            std_pen = split_std_mean.get(s, 0.0)
            if std_pen != std_pen: std_pen = 0.0
            return applicability - std_pen
        ranked = sorted(splits, key=score_split, reverse=True)
        main = ranked[0]
        appendix = ranked[1] if len(ranked) > 1 else "—"
        f.write(f"- **Recommended main protocol**: `{main}`  \n")
        f.write(f"  (best balance of val+test sufficiency + cross-seed stability)\n\n")
        f.write(f"- **Recommended appendix robustness protocol**: `{appendix}`  \n")
        f.write(f"  (run as robustness check; report SAG numbers at this split too)\n\n")
        # Also explicitly call out 70_30_rank
        if "70_30_rank" in splits:
            f.write("- **70_30 ranking-only**: useful as a robustness check for "
                    "ranking metrics (AUROC / AUPRC / P@K) when validation "
                    "cannot be carved out of small datasets. NOT recommended "
                    "as the main protocol because Macro-F1 is not applicable.\n\n")
    else:
        f.write("UNKNOWN — no splits passed.\n\n")

    f.write("## Per-(dataset, split, method) AUROC mean ± std\n\n")
    if agg:
        f.write("| dataset | split | method | n | AUROC mean | std |\n")
        f.write("|---|---|---|---|---|---|\n")
        for ds in datasets:
            for s in splits:
                for m in methods:
                    a = agg.get((ds, s, m))
                    if a is None: continue
                    f.write(f"| {ds} | `{s}` | {m} | {a['n']} | {a['mean']:.4f} | {a['std']:.4f} |\n")
        f.write("\n")
    f.close()


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(DATASETS_DEFAULT))
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


def _build_ae_cache(loader, dc, logger):
    if not dc.get("use_trained_ae", False):
        return None
    try:
        from src.models.trained_attr import UnsupervisedGCNAutoencoder
        ae_cfg = dc.get("trained_ae_cfg", {}) or {}
        cache_key = (id(loader.features), id(loader.adj),
                     tuple(sorted(ae_cfg.items())), BASE_LOAD_SEED)
        ae = UnsupervisedGCNAutoencoder(seed=BASE_LOAD_SEED, **ae_cfg)
        sc = ae.fit_score(loader.adj, loader.features)
        if not np.all(np.isfinite(sc)) or sc.std() < 1e-12:
            return None
        return {"key": cache_key, "score": sc}
    except Exception as e:
        logger.warning(f"  AE pre-training failed: {e}")
        return None


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    run_dir = os.path.join(args.output_dir, "results", "revision_strict",
                           f"split_selection{tag}_{ts}")
    log_dir = os.path.join(args.output_dir, "logs",
                           f"split_selection{tag}_{ts}")
    canon_dir = os.path.join(args.output_dir, "results", "revision_strict")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(canon_dir, exist_ok=True)

    logger = setup_logger("SAG_SPLIT_SEL", log_dir)
    logger.info(f"datasets={datasets}  methods={methods}  seeds={seeds}")

    save_json({"datasets": datasets, "methods": methods, "seeds": seeds,
               "splits": [list(s) for s in SPLITS],
               "argv": sys.argv, "timestamp": ts},
              os.path.join(run_dir, "config.json"))

    Config.data_dir = "./data"
    Config.output_dir = args.output_dir

    meta_rows, result_rows = [], []
    meta_run = os.path.join(run_dir, "split_selection_metadata.csv")
    res_run  = os.path.join(run_dir, "split_selection_results.csv")
    sum_run  = os.path.join(run_dir, "split_selection_summary.md")
    meta_canon = os.path.join(canon_dir, "split_selection_metadata.csv")
    res_canon  = os.path.join(canon_dir, "split_selection_results.csv")
    sum_canon  = os.path.join(canon_dir, "split_selection_summary.md")

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
            seed=BASE_LOAD_SEED,
        )
        try:
            loader.load()
        except Exception as e:
            logger.error(f"  load failed: {e}")
            continue
        s = loader.get_stats()
        ae_cache = _build_ae_cache(loader, dc, logger)
        labels = np.asarray(loader.labels)
        order = getattr(Config, "edge_order", "auto")
        if order == "auto":
            order = "shuffle" if s.get("num_anomalies", 0) > 0 and dc.get("has_ground_truth") else "bfs"
        order = dc.get("edge_order", "auto")
        if order == "auto":
            order = "shuffle" if dc.get("has_ground_truth", False) else "bfs"
        edges = loader.get_edge_stream(mode=order)
        logger.info(f"  stats={s}  edges_used_for_split={edges.shape[0]}")

        for split_name, fractions, ranking_only in SPLITS:
            meta, phase = compute_split_metadata(
                ds, edges, labels, s["num_nodes"], split_name, fractions, ranking_only)
            meta_rows.append(meta)
            end_tr = meta["end_train_edge_idx"]
            end_va = meta["end_val_edge_idx"]
            end_te = meta["end_test_edge_idx"]
            val_nodes  = np.where(phase == 1)[0]
            test_nodes = np.where(phase == 2)[0]
            if ranking_only:
                eval_nodes = test_nodes  # phase=2 in ranking_only is the eval span
            train_anom = meta["train_anomalies"]
            val_anom   = meta["val_anomalies"]
            test_anom  = meta["test_anomalies"]

            for method in methods:
                for sd in seeds:
                    logger.info(f"  --- {ds} | {split_name} | {method} | seed={sd} ---")
                    t0 = time.time()
                    try:
                        if method == "SAG-online-strict":
                            _, score_val, score_test = run_sag_online_strict(
                                loader, dc, edges, end_tr, end_va, end_te, sd, ae_cache)
                        elif method == "MIDAS":
                            _, score_val, score_test = run_midas_strict(
                                s["num_nodes"], edges, end_tr, end_va, end_te, sd)
                        elif method in BASELINES:
                            score_val, score_test, _ = run_static_baseline_strict(
                                method, loader, dc, edges, end_tr, sd)
                            score_test = score_val  # static => same vector
                        else:
                            raise ValueError(f"unknown method {method}")

                        m = evaluate_split(score_val, score_test,
                                           val_nodes if not ranking_only else np.array([], dtype=np.int64),
                                           test_nodes, labels,
                                           ranking_only=ranking_only)
                        result_rows.append({
                            "dataset": ds, "split": split_name, "method": method,
                            "seed": sd, "ranking_only": bool(ranking_only),
                            **m,
                            "train_anomalies": train_anom,
                            "val_anomalies":   val_anom,
                            "test_anomalies":  test_anom,
                            "elapsed_s": float(time.time() - t0),
                            "skipped": False, "error": "",
                        })
                        logger.info("    AUROC=%.4f F1=%.4f Pa=%.4f K=%d (%.1fs)" %
                                    (m["auroc"], m["macro_f1"], m["precision_at_K"],
                                     m["K"], time.time() - t0))
                    except Exception as e:
                        tb = traceback.format_exc()
                        logger.error(f"    FAILED {type(e).__name__}: {e}\n{tb}")
                        result_rows.append({
                            "dataset": ds, "split": split_name, "method": method,
                            "seed": sd, "ranking_only": bool(ranking_only),
                            "auroc": float("nan"), "auprc": float("nan"),
                            "auprc_bal": float("nan"), "macro_f1": float("nan"),
                            "macro_f1_bal": float("nan"),
                            "best_f1_threshold": float("nan"),
                            "best_macro_f1": float("nan"),
                            "tpr_at_thr": float("nan"), "fpr_at_thr": float("nan"),
                            "precision_at_K": float("nan"),
                            "recall_at_K": float("nan"), "K": 0,
                            "train_anomalies": train_anom,
                            "val_anomalies":   val_anom,
                            "test_anomalies":  test_anom,
                            "elapsed_s": 0.0, "skipped": True,
                            "error": f"{type(e).__name__}:{e}",
                        })
                    write_csv(meta_rows, META_COLUMNS, meta_run)
                    write_csv(result_rows, RESULT_COLUMNS, res_run)
        # write summary after each dataset for kill-safety
        write_summary_md(meta_rows, result_rows, sum_run)

    save_json({"meta": meta_rows, "results": result_rows},
              os.path.join(run_dir, "split_selection.json"))

    if not _maybe_skip(meta_canon, args.overwrite, logger):
        write_csv(meta_rows, META_COLUMNS, meta_canon)
    if not _maybe_skip(res_canon, args.overwrite, logger):
        write_csv(result_rows, RESULT_COLUMNS, res_canon)
    if not _maybe_skip(sum_canon, args.overwrite, logger):
        write_summary_md(meta_rows, result_rows, sum_canon)

    logger.info("=" * 70)
    logger.info(f"WROTE  metadata -> {meta_run}")
    logger.info(f"WROTE  results  -> {res_run}")
    logger.info(f"WROTE  summary  -> {sum_run}")
    logger.info(f"CANON  metadata -> {meta_canon}")
    logger.info(f"CANON  results  -> {res_canon}")
    logger.info(f"CANON  summary  -> {sum_canon}")


if __name__ == "__main__":
    main()
