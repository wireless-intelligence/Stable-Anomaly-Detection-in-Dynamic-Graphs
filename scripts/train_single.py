import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from configs import Config, get_dataset_config
from src.data import DataLoader
from src.trainer import MultiRunSAGTrainer, SAG_FACTORY
from src.utils import save_json, set_seed, setup_logger


def parse_args():
    p = argparse.ArgumentParser(description="SAG single-machine training")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument("--n_runs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--beta_upd", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=1.0)
    p.add_argument("--rho", type=float, default=2.0)
    p.add_argument("--eval_freq", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    Config.data_dir = args.data_dir
    Config.output_dir = args.output_dir
    Config.seed = args.seed
    Config.n_runs = args.n_runs
    Config.eval_freq = args.eval_freq
    Config.gamma = args.gamma
    Config.beta_upd = args.beta_upd
    Config.alpha = args.alpha
    Config.delta = args.delta
    Config.rho = args.rho

    dc = get_dataset_config(args.dataset)
    Config.num_epochs = args.num_epochs or dc.get("num_epochs", 20)
    Config.has_ground_truth = bool(dc.get("has_ground_truth", False))
    for k in ("alpha", "gamma", "rho"):
        if k in dc and getattr(Config, k) == dc.get(k, None) or getattr(Config, k) == (1.0 if k == "alpha" else 0.95 if k == "gamma" else 2.0):
            setattr(Config, k, dc.get(k, getattr(Config, k)))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp = f"{args.dataset}_{ts}"
    Config.log_dir = os.path.join(args.output_dir, "logs", exp)
    Config.checkpoint_dir = os.path.join(args.output_dir, "checkpoints", exp)
    Config.result_dir = os.path.join(args.output_dir, "results", exp)
    Config.figure_dir = os.path.join(args.output_dir, "figures", exp)
    Config.vis_dir = os.path.join(args.output_dir, "visualizations", exp)
    Config.create_dirs()

    logger = setup_logger("SAG_TRAIN", Config.log_dir)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Config: {Config.to_dict()}")
    set_seed(Config.seed)

    loader = DataLoader(
        data_dir=Config.data_dir,
        dataset_name=args.dataset,
        clique_size=dc.get("clique_size", 15),
        num_cliques=dc.get("num_cliques", 10),
        seed=Config.seed,
    )
    loader.load()
    logger.info(f"Dataset stats: {loader.get_stats()}")

    runner = MultiRunSAGTrainer(
        model_factory=SAG_FACTORY,
        data_loader=loader,
        config=Config,
        logger=logger,
        n_runs=Config.n_runs,
    )
    agg = runner.run(num_epochs=Config.num_epochs)
    fmt = runner.get_formatted_results()

    logger.info("=" * 50)
    logger.info(f"Final Results for {args.dataset}:")
    logger.info(f"  AUROC:    {fmt['auroc']}")
    logger.info(f"  AUPRC:    {fmt['auprc']}")
    logger.info(f"  Macro-F1: {fmt['macro_f1']}")
    logger.info(f"  Latency:  {fmt['latency_ms']} ms")
    logger.info(f"  Memory:   {fmt['memory_gb']} GB")
    logger.info("=" * 50)

    out = {
        "dataset": args.dataset,
        "config": Config.to_dict(),
        "dataset_stats": loader.get_stats(),
        "aggregated_results": {k: agg[k] for k in ("auroc", "auprc", "macro_f1", "latency_ms", "memory_gb")},
        "formatted_results": fmt,
        "n_runs": Config.n_runs,
        "timestamp": ts,
    }
    save_json(out, os.path.join(Config.result_dir, "results.json"))
    logger.info(f"Saved to {Config.result_dir}")


if __name__ == "__main__":
    main()
