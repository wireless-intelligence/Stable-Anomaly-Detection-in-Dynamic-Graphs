import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from configs import Config, get_dataset_config
from src.trainer import DataParallelSAGTrainer
from src.utils import save_json, set_seed, setup_logger


def parse_args():
    p = argparse.ArgumentParser(description="SAG parallel multi-run training")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument("--n_runs", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--beta_upd", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=1.0)
    p.add_argument("--rho", type=float, default=2.0)
    return p.parse_args()


def main():
    args = parse_args()
    Config.data_dir = args.data_dir
    Config.output_dir = args.output_dir
    Config.seed = args.seed
    Config.n_runs = args.n_runs
    Config.gamma = args.gamma
    Config.beta_upd = args.beta_upd
    Config.alpha = args.alpha
    Config.delta = args.delta
    Config.rho = args.rho

    dc = get_dataset_config(args.dataset)
    Config.num_epochs = args.num_epochs or dc.get("num_epochs", 20)
    for k in ("alpha", "gamma", "rho"):
        if k in dc:
            setattr(Config, k, dc[k])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp = f"{args.dataset}_multi_{ts}"
    Config.log_dir = os.path.join(args.output_dir, "logs", exp)
    Config.result_dir = os.path.join(args.output_dir, "results", exp)
    os.makedirs(Config.log_dir, exist_ok=True)
    os.makedirs(Config.result_dir, exist_ok=True)

    logger = setup_logger("SAG_MULTI", Config.log_dir)
    logger.info(f"Dataset={args.dataset} n_runs={Config.n_runs} workers={args.num_workers}")
    set_seed(Config.seed)

    runner = DataParallelSAGTrainer(
        config=Config, dataset_name=args.dataset, num_workers=args.num_workers
    )
    agg = runner.train(num_epochs=Config.num_epochs, n_runs=Config.n_runs)

    logger.info(
        "AUROC %.1f±%.1f | AUPRC %.1f±%.1f | F1 %.1f±%.1f (n=%d)"
        % (
            agg["auroc"]["mean"] * 100, agg["auroc"]["std"] * 100,
            agg["auprc"]["mean"] * 100, agg["auprc"]["std"] * 100,
            agg["macro_f1"]["mean"] * 100, agg["macro_f1"]["std"] * 100,
            agg["n_runs"],
        )
    )
    save_json(
        {"dataset": args.dataset, "aggregated_results": {k: agg[k] for k in ("auroc", "auprc", "macro_f1")}, "n_runs": agg["n_runs"], "timestamp": ts},
        os.path.join(Config.result_dir, "results.json"),
    )


if __name__ == "__main__":
    main()
