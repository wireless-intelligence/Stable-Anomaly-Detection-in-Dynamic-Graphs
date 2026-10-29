"""SAG hyperparameter sensitivity analysis.

For each dataset x hyperparameter, vary the parameter around its default and
record AUROC / AUPRC_bal / F1_bal. Save to JSON.

Bug fix notes:
  * The Config class is a singleton. Per-run state must be a fresh shim or
    failures contaminate later iterations.
  * Default kernel_lambdas is a length-4 list; sweeping kernel_dim != 4 needs
    kernel_lambdas reset to None so StableStateEvolution auto-builds them.
  * Errors must show the full traceback to be debuggable.
  * For datasets that route through fusion_mode in {attr_only, iforest_only},
    alpha/gamma/rho/kernel_dim never reach the score. We override fusion_mode
    to "auto" for the *sweep* of those four params, so the curve is non-trivial.
"""

import argparse
import copy
import os
import sys
import json
import time
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from configs import Config, get_dataset_config
from src.data import DataLoader
from src.evaluation import compute_all_metrics
from src.models import create_sag_model
from src.utils import set_seed, save_json, setup_logger


PARAM_SWEEPS = {
    "alpha":              [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0],
    "gamma":              [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99],
    "rho":                [1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    "kernel_dim":         [1, 2, 4, 6, 8, 12, 16],
    "prior_anomaly_rate": [0.02, 0.05, 0.10, 0.15, 0.20, 0.30],
    "pm_blend_weight":    [0.0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
}

# Params that only affect the score when SAG state/kernel weights are non-zero.
# For datasets whose default fusion_mode bypasses state/kernel, force "auto"
# during these sweeps so the parameter actually has an effect.
SAG_STATE_PARAMS = {"alpha", "gamma", "rho", "kernel_dim"}


class CfgShim:
    """Per-run config that copies all attrs from Config + dataset overrides."""
    pass


def build_cfg(dc, sweep_param=None):
    cfg = CfgShim()
    # Inherit every public attribute from Config first.
    for k in dir(Config):
        if k.startswith("_") or callable(getattr(Config, k, None)):
            continue
        try:
            setattr(cfg, k, copy.deepcopy(getattr(Config, k)))
        except Exception:
            pass
    # Apply dataset overrides.
    for k, v in dc.items():
        try:
            setattr(cfg, k, copy.deepcopy(v))
        except Exception:
            pass
    cfg.has_ground_truth = bool(dc.get("has_ground_truth", False))
    cfg.use_continuous_dev = False
    cfg.use_real_time_decay = False
    # If sweeping kernel_dim, the default length-4 lambdas list won't match.
    if sweep_param == "kernel_dim":
        cfg.kernel_lambdas = None
    # Honest sensitivity: keep the dataset's native fusion_mode. On
    # attr_only/iforest_only datasets the SAG-state params are bypassed by
    # design — flat curves there ARE the result.
    return cfg


def run_one(loader, cfg, seed=42):
    set_seed(seed)
    s = loader.get_stats()
    cfg.window_size = max(128, min(s["num_edges"] // 32, 4096))
    model = create_sag_model(num_nodes=s["num_nodes"], config=cfg,
                             features=loader.features, adj=loader.adj)
    order = getattr(cfg, "edge_order", "auto")
    if order == "auto":
        order = "shuffle" if bool(getattr(cfg, "has_ground_truth", False)) else "bfs"
    do_shuffle = (order == "shuffle")
    passes = (2 if s["num_edges"] > 50000 else 3) if do_shuffle else 1
    model.process_stream_batch(loader.get_edge_stream(mode=order),
                               passes=passes, shuffle=do_shuffle,
                               rng=np.random.default_rng(seed))
    scores = model.get_scores()
    return compute_all_metrics(loader.labels, scores)


def sweep_dataset(name, params_to_sweep, output_path, logger, fusion_mode_override=None):
    dc = get_dataset_config(name)
    loader = DataLoader(data_dir="./data", dataset_name=name,
                        clique_size=dc.get("clique_size", 15),
                        num_cliques=dc.get("num_cliques", 10), seed=42)
    loader.load()
    logger.info(f"=== {name} stats: {loader.get_stats()}")

    # Pre-train AE ONCE per dataset (else each fresh cfg retrains it).
    ae_cache = None
    if dc.get("use_trained_ae", False):
        from src.models.trained_attr import UnsupervisedGCNAutoencoder
        ae_cfg = dc.get("trained_ae_cfg", {}) or {}
        cache_key = (id(loader.features), id(loader.adj),
                     tuple(sorted(ae_cfg.items())), 42)
        logger.info(f"  pre-training AE for {name} (one-time, ~3 min)...")
        t0 = time.time()
        ae = UnsupervisedGCNAutoencoder(seed=42, **ae_cfg)
        ae_score = ae.fit_score(loader.adj, loader.features)
        if np.all(np.isfinite(ae_score)) and ae_score.std() > 1e-12:
            ae_cache = {"key": cache_key, "score": ae_score}
            logger.info(f"  AE trained in {time.time()-t0:.1f}s, will be reused")
        else:
            logger.info(f"  AE produced degenerate output; skipping cache")

    results = {"dataset": name, "params": {}}

    for param in params_to_sweep:
        values = PARAM_SWEEPS[param]
        # Determine the default value: prefer dataset config, then Config class.
        default_val = dc.get(param)
        if default_val is None and hasattr(Config, param):
            default_val = getattr(Config, param)
        logger.info(f"\n  >>> sweep {param} (default={default_val}) values={values}")
        results["params"][param] = {"default": default_val, "values": []}
        for v in values:
            cfg = build_cfg(dc, sweep_param=param)
            if fusion_mode_override:
                cfg.fusion_mode = fusion_mode_override
            setattr(cfg, param, v)
            if ae_cache is not None:
                cfg._trained_ae_score_cache = ae_cache
            t0 = time.time()
            try:
                m = run_one(loader, cfg)
                row = {
                    "value": v,
                    "auroc": float(m["auroc"]),
                    "auprc": float(m["auprc"]),
                    "auprc_bal": float(m["auprc_bal"]),
                    "macro_f1": float(m["macro_f1"]),
                    "macro_f1_bal": float(m["macro_f1_bal"]),
                    "elapsed_s": time.time() - t0,
                }
                results["params"][param]["values"].append(row)
                logger.info(
                    f"    {param}={v}: AUROC={m['auroc']*100:.2f} "
                    f"AUPRC_bal={m['auprc_bal']*100:.2f} "
                    f"F1_bal={m['macro_f1_bal']*100:.2f} "
                    f"({time.time()-t0:.1f}s)"
                )
            except Exception as e:
                tb = traceback.format_exc()
                row = {"value": v, "error": str(e), "traceback": tb}
                results["params"][param]["values"].append(row)
                logger.error(f"    {param}={v}: FAILED {type(e).__name__}: {e}")
                logger.error(tb)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="Cora,BlogCatalog,Amazon")
    p.add_argument("--params", default="alpha,gamma,rho,kernel_dim,prior_anomaly_rate,pm_blend_weight")
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--fusion_mode", default=None,
                   help="Override dataset's fusion_mode for the sweep (e.g. 'h_only').")
    p.add_argument("--tag", default=None,
                   help="Tag appended to result/log dir (e.g. 'h_only').")
    return p.parse_args()


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    params_to_sweep = [p.strip() for p in args.params.split(",") if p.strip()]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_suffix = f"_{args.tag}" if args.tag else ""
    exp_dir = os.path.join(args.output_dir, "results", f"sensitivity{tag_suffix}_{ts}")
    log_dir = os.path.join(args.output_dir, "logs", f"sensitivity{tag_suffix}_{ts}")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger("SAG_SENS", log_dir)
    logger.info(f"Datasets: {datasets} | Params: {params_to_sweep} | "
                f"fusion_mode_override={args.fusion_mode}")

    Config.data_dir = "./data"
    Config.output_dir = args.output_dir
    if args.fusion_mode:
        Config.fusion_mode = args.fusion_mode

    all_results = {}
    for ds in datasets:
        out_path = os.path.join(exp_dir, f"{ds}.json")
        all_results[ds] = sweep_dataset(ds, params_to_sweep, out_path, logger,
                                        fusion_mode_override=args.fusion_mode)

    save_json(all_results, os.path.join(exp_dir, "all_sensitivity.json"))
    logger.info(f"\nSaved to {exp_dir}")


if __name__ == "__main__":
    main()
