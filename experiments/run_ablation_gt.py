"""Cross-dataset ablation table on the ground-truth datasets.

For each (dataset, ablation) it runs SAG once with seed=42 and writes a
JSON of metrics + a per-dataset CSV. Designed for local execution: small
GT datasets first (Enron), heaviest last (Amazon).

Usage:
  python experiments/run_ablation_gt.py --datasets Enron,CTU-13,Amazon \
        --output_dir outputs/ablation_gt
"""

import argparse
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
from src.models.ablations import ABLATION_TYPES, ABLATION_NAMES
from src.utils import set_seed


def apply_dc(dc):
    for k in ("alpha", "gamma", "rho"):
        if k in dc:
            setattr(Config, k, dc[k])
    Config.fusion_mode = dc.get("fusion_mode", "auto")
    Config.attr_mode = dc.get("attr_mode", "auto")
    Config.use_trained_ae = bool(dc.get("use_trained_ae", False))
    Config.trained_ae_cfg = dc.get("trained_ae_cfg", {})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="Enron,CTU-13,Amazon")
    p.add_argument("--output_dir", default="./outputs/ablation_gt")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"ablation_gt_{ts}.json")

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    all_results = {}

    for ds in datasets:
        print(f"\n=== {ds} ===")
        set_seed(args.seed)
        ld = DataLoader("data", ds)
        ld.load()
        dc = get_dataset_config(ds)
        apply_dc(dc)
        n = ld.stats["num_nodes"]
        n_anom = int(ld.labels.sum())
        print(f"  N={n:,}  F={ld.features.shape[1]}  anomalies={n_anom}")

        ds_results = {}
        for abl_key, abl_id in ABLATION_TYPES.items():
            set_seed(args.seed)
            t0 = time.time()
            try:
                m = create_sag_model(num_nodes=n, config=Config,
                                     features=ld.features, adj=ld.adj,
                                     ablation=abl_id)
                m.process_stream_batch(ld.get_edge_stream(mode="bfs"),
                                       passes=1, shuffle=False,
                                       rng=np.random.default_rng(args.seed))
                sc = m.get_scores()
                mt = compute_all_metrics(ld.labels, sc)
                t = time.time() - t0
                row = {
                    "ablation": abl_key,
                    "name": ABLATION_NAMES[abl_key],
                    "auroc": float(mt["auroc"]),
                    "auprc_bal": float(mt["auprc_bal"]),
                    "macro_f1_bal": float(mt["macro_f1_bal"]),
                    "elapsed_s": t,
                }
                print(f"  {abl_key:14}  AUROC={mt['auroc']*100:5.2f}  "
                      f"AUPRC_bal={mt['auprc_bal']*100:5.2f}  "
                      f"F1_bal={mt['macro_f1_bal']*100:5.2f}  ({t:.1f}s)")
                ds_results[abl_key] = row
            except Exception as e:
                print(f"  {abl_key:14}  FAILED: {e}")
                ds_results[abl_key] = {"ablation": abl_key, "error": str(e)}

        all_results[ds] = {
            "stats": {
                "num_nodes": n,
                "num_features": int(ld.features.shape[1]),
                "num_anomalies": n_anom,
            },
            "ablations": ds_results,
        }
        with open(out_path, "w") as f:
            json.dump({"seed": args.seed, "results": all_results}, f, indent=2)

    print(f"\n>>> wrote {out_path}")


if __name__ == "__main__":
    main()
