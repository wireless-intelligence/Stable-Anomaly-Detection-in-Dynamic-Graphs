"""H-only ablation pass on ground-truth datasets.

The production fusion mode for Enron/Amazon/CTU-13 is attr_only or
iforest_only — these bypass the streaming-state branch entirely, so
ablating no_stable / no_decay / no_exposure / no_kernel under production
fusion leaves AUROC unchanged. To make those ablations measurable, we
re-run them with fusion_mode='h_only' (state-branch isolated).

This is the same methodology as the existing 5-dataset ablation figure.
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


# Only state-evolution / kernel ablations need the h_only isolation pass.
# Evidence-source ablations (no_attribute / no_structure / no_selection)
# are already meaningful under production fusion.
H_ONLY_ABLATIONS = ["full", "no_stable", "no_decay", "no_exposure", "no_kernel"]


def apply_dc_h_only(dc):
    for k in ("alpha", "gamma", "rho"):
        if k in dc:
            setattr(Config, k, dc[k])
    Config.fusion_mode = "h_only"           # the key override
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
    out_path = os.path.join(args.output_dir, f"ablation_gt_h_only_{ts}.json")

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    all_results = {}

    for ds in datasets:
        print(f"\n=== {ds} (h_only) ===")
        set_seed(args.seed)
        ld = DataLoader("data", ds)
        ld.load()
        dc = get_dataset_config(ds)
        apply_dc_h_only(dc)
        n = ld.stats["num_nodes"]
        n_anom = int(ld.labels.sum())
        print(f"  N={n:,}  F={ld.features.shape[1]}  anomalies={n_anom}  fusion_mode=h_only")

        ds_results = {}
        for abl_key in H_ONLY_ABLATIONS:
            abl_id = ABLATION_TYPES[abl_key]
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
                      f"AUPRC_bal={mt['auprc_bal']*100:5.2f}  ({t:.1f}s)")
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
            json.dump({"seed": args.seed, "fusion_mode": "h_only",
                       "results": all_results}, f, indent=2)

    print(f"\n>>> wrote {out_path}")


if __name__ == "__main__":
    main()
