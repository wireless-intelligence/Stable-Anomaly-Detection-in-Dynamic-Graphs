"""Joint (parameter, seed) sensitivity sweep on a ground-truth dataset.

For each (param, value, seed) it runs SAG once and records AUROC /
AUPRC_bal / F1_bal. The output JSON has shape:

  {
    "dataset": "Enron",
    "seeds": [42, 100, 200],
    "params": {
      "alpha": {
        "default": 1.0,
        "values": [{"value": 0.5, "per_seed": [{"seed": 42, "auroc": ...}, ...]}, ...]
      },
      ...
    }
  }

Designed for local execution on small GT datasets where one SAG run
takes seconds.

Usage:
  python experiments/run_param_seed_sensitivity.py --dataset Enron \
        --seeds 42,100,200 --params alpha,gamma,rho,kernel_dim
"""

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from configs import Config, get_dataset_config
from src.data import DataLoader
from src.evaluation import compute_all_metrics
from src.models import create_sag_model
from src.utils import set_seed


PARAM_SWEEPS = {
    "alpha":              [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0],
    "gamma":              [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99],
    "rho":                [1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    "kernel_dim":         [1, 2, 4, 6, 8, 12, 16],
    "prior_anomaly_rate": [0.02, 0.05, 0.10, 0.15, 0.20, 0.30],
    "pm_blend_weight":    [0.0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
}


class CfgShim:
    pass


def base_cfg(dc, sweep_param=None):
    cfg = CfgShim()
    for k in dir(Config):
        if k.startswith("_") or callable(getattr(Config, k, None)):
            continue
        try:
            setattr(cfg, k, getattr(Config, k))
        except (AttributeError, TypeError):
            pass
    for k in ("alpha", "gamma", "rho"):
        if k in dc:
            setattr(cfg, k, dc[k])
    # Use the dataset's production fusion mode so the result reflects the
    # configuration the paper actually reports. State-only params (alpha,
    # gamma, kernel_dim) will be flat for datasets whose production fusion
    # bypasses the state branch — which is itself a valid robustness
    # statement: those params don't perturb the deployed scores.
    cfg.fusion_mode = dc.get("fusion_mode", "auto")
    cfg.attr_mode = dc.get("attr_mode", "auto")
    cfg.use_trained_ae = bool(dc.get("use_trained_ae", False))
    cfg.trained_ae_cfg = dc.get("trained_ae_cfg", {})
    cfg.use_attr_boost = bool(dc.get("use_attr_boost", False))
    cfg.use_pca_mahal = bool(dc.get("use_pca_mahal", False))
    cfg.iforest_blend_h_weight = float(dc.get("iforest_blend_h_weight", 0.3))
    if sweep_param == "kernel_dim":
        cfg.kernel_lambdas = None
    return cfg


def run_one(dc, param, value, seed, ld):
    cfg = base_cfg(dc, sweep_param=param)
    setattr(cfg, param, value)
    set_seed(seed)
    m = create_sag_model(num_nodes=ld.stats["num_nodes"], config=cfg,
                         features=ld.features, adj=ld.adj)
    m.process_stream_batch(ld.get_edge_stream(mode="bfs"),
                           passes=1, shuffle=False,
                           rng=np.random.default_rng(seed))
    sc = m.get_scores()
    return compute_all_metrics(ld.labels, sc)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seeds", default="42,100,200")
    p.add_argument("--params", default=",".join(PARAM_SWEEPS.keys()))
    p.add_argument("--output_dir", default="./outputs/param_seed_sens")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir,
                            f"param_seed_{args.dataset}_{ts}.json")

    seeds = [int(s) for s in args.seeds.split(",")]
    params = [p.strip() for p in args.params.split(",") if p.strip()]
    print(f"Dataset={args.dataset}  seeds={seeds}  params={params}")

    set_seed(seeds[0])
    ld = DataLoader("data", args.dataset)
    ld.load()
    dc = get_dataset_config(args.dataset)
    print(f"  N={ld.stats['num_nodes']:,}  anomalies={int(ld.labels.sum())}")

    out = {
        "dataset": args.dataset,
        "seeds": seeds,
        "params": {},
        "stats": {
            "num_nodes": int(ld.stats["num_nodes"]),
            "num_features": int(ld.features.shape[1]),
            "num_anomalies": int(ld.labels.sum()),
        },
    }

    total_start = time.time()
    for param in params:
        if param not in PARAM_SWEEPS:
            print(f"  skip unknown param {param}")
            continue
        values = PARAM_SWEEPS[param]
        default_val = getattr(base_cfg(dc), param, None)
        print(f"\n  >>> {param} (default={default_val}) values={values}")
        out["params"][param] = {"default": default_val, "values": []}
        for v in values:
            row = {"value": v, "per_seed": []}
            t0 = time.time()
            for sd in seeds:
                try:
                    mt = run_one(dc, param, v, sd, ld)
                    row["per_seed"].append({
                        "seed": sd,
                        "auroc": float(mt["auroc"]),
                        "auprc_bal": float(mt["auprc_bal"]),
                        "macro_f1_bal": float(mt["macro_f1_bal"]),
                    })
                except Exception as e:
                    row["per_seed"].append({"seed": sd, "error": str(e)})
            t = time.time() - t0
            mean_auroc = np.mean([r["auroc"] for r in row["per_seed"]
                                  if "auroc" in r]) * 100
            std_auroc = np.std([r["auroc"] for r in row["per_seed"]
                                if "auroc" in r]) * 100
            print(f"    {param}={v:<8}  AUROC={mean_auroc:5.2f}+-{std_auroc:.2f}  ({t:.1f}s)")
            out["params"][param]["values"].append(row)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

    elapsed = time.time() - total_start
    print(f"\n>>> wrote {out_path}  ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
