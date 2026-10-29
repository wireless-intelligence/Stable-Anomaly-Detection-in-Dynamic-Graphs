"""Community / scale evaluation.

Two regimes:
  mode A — fixed scale (~5K nodes), varied community structure (LFR / SBM
           with different mu / community-size distributions).
  mode B — fixed dataset family (BlogCatalog), varied scale via stratified
           subsampling (1K / 2K / 5K / full).

For each setting, run SAG + 4 baselines + ground-truth comparison; outputs
a per-setting score + AUROC table consumed by the paired figure.

Output:
    outputs/results/comm_scale_<ts>/
        mode_a_<setting>.json     (5 community variants)
        mode_b_<frac>.json        (4 scale fractions)
        comm_scale_summary.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.sparse as sp

from configs import Config, get_dataset_config
from src.baselines import BASELINES
from src.data import DataLoader
from src.data.anomaly_types import inject_clique
from src.evaluation import compute_all_metrics
from src.models import create_sag_model
from src.utils import save_json, set_seed, setup_logger


DEFAULT_METHODS = ["DOMINANT", "CONAD", "GraphSAGE", "ConsisGAD"]


def _setup_cfg(dc):
    cfg = Config
    for k in ("alpha", "gamma", "rho"):
        if k in dc:
            setattr(cfg, k, dc[k])
    cfg.num_epochs = dc.get("num_epochs", 20)
    cfg.has_ground_truth = bool(dc.get("has_ground_truth", False))
    cfg.fusion_mode = "auto"
    cfg.edge_order = "auto"
    cfg.attr_mode = "auto"
    cfg.use_trained_ae = False
    cfg.use_attr_boost = False
    cfg.use_pca_mahal = False
    return cfg


def _edges_from_adj(adj):
    coo = adj.tocoo()
    mask = coo.row < coo.col
    return np.column_stack([coo.row[mask], coo.col[mask]]).astype(np.int64)


def _sbm(n, num_communities, mu, mean_degree, feature_dim, seed):
    rng = np.random.default_rng(seed)
    sizes = [n // num_communities] * num_communities
    sizes[0] += n - sum(sizes)
    p_in  = (1 - mu) * mean_degree / max(1, (n / num_communities))
    p_out = mu * mean_degree / max(1, n)
    p_in = min(p_in, 1.0); p_out = min(p_out, 1.0)
    rows, cols = [], []
    block_id = np.repeat(np.arange(num_communities), sizes)[:n]
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if block_id[i] == block_id[j] else p_out
            if rng.random() < p:
                rows.append(i); cols.append(j)
    if not rows:
        rows = [0]; cols = [min(1, n - 1)]
    A = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(n, n))
    A = A.maximum(A.T); A.setdiag(0); A.eliminate_zeros()
    centroids = rng.standard_normal((num_communities, feature_dim)).astype(np.float32)
    X = centroids[block_id] + 0.5 * rng.standard_normal((n, feature_dim)).astype(np.float32)
    return A, sp.csr_matrix(X)


def _run_method(name, n, adj, features, edges, cfg):
    set_seed(42)
    if name == "SAG":
        m = create_sag_model(num_nodes=n, config=cfg,
                             features=features, adj=adj)
        m.process_stream_batch(edges, passes=2, shuffle=True,
                               rng=np.random.default_rng(42))
        return m.get_scores()
    cls = BASELINES.get(name)
    if cls is None: return None
    try:
        m = cls(num_nodes=n, config=Config, features=features, seed=42)
        m.fit(adj, edges, None)
        return m.score()
    except Exception as e:
        print(f"    {name} FAILED: {type(e).__name__}: {e}")
        return None


def run_mode_a(methods, n, num_anomalies, exp_dir, logger):
    settings = [
        ("strong_communities", {"num_communities": 5, "mu": 0.05}),
        ("medium_communities", {"num_communities": 5, "mu": 0.20}),
        ("weak_communities",   {"num_communities": 5, "mu": 0.40}),
        ("many_small",         {"num_communities": 20, "mu": 0.20}),
        ("few_large",          {"num_communities": 2, "mu": 0.20}),
    ]
    cfg = _setup_cfg(get_dataset_config("Cora"))
    out = {}
    for name, params in settings:
        logger.info(f"  setting={name} {params}")
        A, X = _sbm(n=n, mean_degree=10, feature_dim=16, seed=7,
                    **params)
        A_inj, X_inj, labels = inject_clique(A, X,
                                             num_anomalies=num_anomalies,
                                             clique_size=15, seed=42)
        edges = _edges_from_adj(A_inj)
        per = {"setting_params": params, "num_nodes": n,
               "num_anomalies": int(labels.sum()), "methods": {}}
        for m in ["SAG"] + methods:
            t0 = time.time()
            scores = _run_method(m, n, A_inj, X_inj, edges, cfg)
            if scores is None:
                per["methods"][m] = {"error": "fit_failed"}; continue
            mt = compute_all_metrics(labels, scores)
            per["methods"][m] = {
                "auroc": float(mt["auroc"]),
                "auprc_bal": float(mt["auprc_bal"]),
                "f1_bal": float(mt["macro_f1_bal"]),
                "elapsed_s": float(time.time() - t0),
            }
            logger.info(f"    {m:12} AUROC={mt['auroc']*100:.1f}")
        out[name] = per
        save_json(per, os.path.join(exp_dir, f"mode_a_{name}.json"))
    return out


def _stratified_subsample(adj, features, labels, frac, seed):
    rng = np.random.default_rng(seed)
    n = adj.shape[0]
    keep_n = int(frac * n)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    pos_keep = rng.choice(pos_idx, size=min(int(keep_n * 0.06), len(pos_idx)),
                          replace=False)
    neg_keep = rng.choice(neg_idx, size=keep_n - len(pos_keep), replace=False)
    keep = np.concatenate([pos_keep, neg_keep])
    keep.sort()
    A_sub = adj[keep][:, keep]
    if features is not None and sp.issparse(features):
        X_sub = features[keep]
    elif features is not None:
        X_sub = features[keep]
    else:
        X_sub = None
    lab_sub = labels[keep]
    return A_sub, X_sub, lab_sub


def run_mode_b(methods, dataset, fracs, exp_dir, logger):
    dc = get_dataset_config(dataset)
    cfg = _setup_cfg(dc)
    loader = DataLoader(data_dir="./data", dataset_name=dataset,
                        clique_size=dc.get("clique_size", 15),
                        num_cliques=dc.get("num_cliques", 5), seed=42)
    loader.load()
    out = {}
    for f in fracs:
        if f == 1.0:
            A, X, lab = loader.adj, loader.features, loader.labels
        else:
            A, X, lab = _stratified_subsample(loader.adj, loader.features,
                                              loader.labels, f, seed=42)
        n = A.shape[0]
        edges = _edges_from_adj(A)
        per = {"frac": f, "num_nodes": n, "num_anomalies": int(lab.sum()),
               "methods": {}}
        logger.info(f"  frac={f}  N={n}  anom={int(lab.sum())}")
        for m in ["SAG"] + methods:
            t0 = time.time()
            scores = _run_method(m, n, A, X, edges, cfg)
            if scores is None:
                per["methods"][m] = {"error": "fit_failed"}; continue
            mt = compute_all_metrics(lab, scores)
            per["methods"][m] = {
                "auroc": float(mt["auroc"]),
                "auprc_bal": float(mt["auprc_bal"]),
                "f1_bal": float(mt["macro_f1_bal"]),
                "elapsed_s": float(time.time() - t0),
            }
            logger.info(f"    {m:12} AUROC={mt['auroc']*100:.1f}")
        out[f"frac_{f}"] = per
        save_json(per, os.path.join(exp_dir, f"mode_b_frac{int(f*100):03d}.json"))
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    p.add_argument("--mode_a_n", type=int, default=2000)
    p.add_argument("--mode_a_anom", type=int, default=60)
    p.add_argument("--mode_b_dataset", default="BlogCatalog")
    p.add_argument("--mode_b_fracs", default="0.2,0.4,0.7,1.0")
    p.add_argument("--output_dir", default="./outputs")
    return p.parse_args()


def main():
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    fracs = [float(x) for x in args.mode_b_fracs.split(",") if x.strip()]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.output_dir, "results", f"comm_scale_{ts}")
    log_dir = os.path.join(args.output_dir, "logs", f"comm_scale_{ts}")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger("COMM_SCALE", log_dir)
    Config.data_dir = "./data"
    Config.output_dir = args.output_dir

    logger.info(f"Methods (+SAG): {methods}")

    logger.info("=== MODE A: fixed scale, varied community structure ===")
    a = run_mode_a(methods, args.mode_a_n, args.mode_a_anom, exp_dir, logger)
    logger.info("=== MODE B: fixed dataset, varied scale ===")
    b = run_mode_b(methods, args.mode_b_dataset, fracs, exp_dir, logger)

    save_json({"mode_a": a, "mode_b": b},
              os.path.join(exp_dir, "comm_scale_summary.json"))
    logger.info(f"Saved to {exp_dir}")


if __name__ == "__main__":
    main()
