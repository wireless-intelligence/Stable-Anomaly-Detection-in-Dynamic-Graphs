import argparse
import os
import sys
from datetime import datetime

import numpy as np

from configs import Config, DATASET_CONFIGS, get_dataset_config
from src.data import DataLoader
from src.models import ABLATION_NAMES, create_sag_model
from src.trainer import MultiRunSAGTrainer, SAG_FACTORY
from src.utils import save_json, set_seed, setup_logger


def parse_args():
    p = argparse.ArgumentParser(description="SAG - Stable Anomaly Detection in Dynamic Graphs")
    p.add_argument("--mode", type=str, default="train",
                   choices=["train", "eval", "ablation", "all_datasets", "benchmark"])
    p.add_argument("--dataset", type=str, default="Amazon",
                   choices=list(DATASET_CONFIGS.keys()))
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
    return p.parse_args()


def apply_config(args):
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
    Config.has_ground_truth = bool(dc.get("has_ground_truth", False))
    if "alpha" in dc and args.alpha == 1.0:
        Config.alpha = dc["alpha"]
    if "gamma" in dc and args.gamma == 0.95:
        Config.gamma = dc["gamma"]
    if "rho" in dc and args.rho == 2.0:
        Config.rho = dc["rho"]
    return dc


def train_single(args, logger):
    dc = apply_config(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp = f"{args.dataset}_{timestamp}"
    Config.log_dir = os.path.join(args.output_dir, "logs", exp)
    Config.checkpoint_dir = os.path.join(args.output_dir, "checkpoints", exp)
    Config.result_dir = os.path.join(args.output_dir, "results", exp)
    Config.figure_dir = os.path.join(args.output_dir, "figures", exp)
    Config.vis_dir = os.path.join(args.output_dir, "visualizations", exp)
    Config.create_dirs()

    logger.info(f"Training on {args.dataset}")
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

    logger.info("=" * 60)
    logger.info(f"Results for {args.dataset}:")
    logger.info(f"  AUROC:    {fmt['auroc']}")
    logger.info(f"  AUPRC:    {fmt['auprc']}")
    logger.info(f"  Macro-F1: {fmt['macro_f1']}")
    logger.info(f"  Latency:  {fmt['latency_ms']} ms")
    logger.info(f"  Memory:   {fmt['memory_gb']} GB")
    logger.info("=" * 60)

    out = {
        "dataset": args.dataset,
        "config": Config.to_dict(),
        "dataset_stats": loader.get_stats(),
        "aggregated_results": {k: agg[k] for k in ("auroc", "auprc", "macro_f1", "latency_ms", "memory_gb")},
        "formatted_results": fmt,
        "n_runs": Config.n_runs,
        "timestamp": timestamp,
    }
    save_json(out, os.path.join(Config.result_dir, "results.json"))
    return agg


def train_all(args, logger):
    all_res = {}
    for ds in ["Amazon", "YelpChi", "Enron", "BlogCatalog", "Cora", "Citeseer", "Pubmed", "CTU-13"]:
        args.dataset = ds
        logger.info(f"\n{'='*60}\nProcessing: {ds}\n{'='*60}")
        try:
            all_res[ds] = train_single(args, logger)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {ds}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error on {ds}: {e}")
            continue

    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    for ds, res in all_res.items():
        a, u, f1 = res["auroc"], res["auprc"], res["macro_f1"]
        logger.info(
            f"{ds:<15} AUROC={a['mean']*100:.1f}±{a['std']*100:.1f} | "
            f"AUPRC={u['mean']*100:.1f}±{u['std']*100:.1f} | "
            f"F1={f1['mean']*100:.1f}±{f1['std']*100:.1f}"
        )
    summary = {
        ds: {k: {"mean": res[k]["mean"], "std": res[k]["std"]} for k in ("auroc", "auprc", "macro_f1")}
        for ds, res in all_res.items()
    }
    save_json(summary, os.path.join(args.output_dir, "results", "all_datasets_summary.json"))
    return all_res


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    logger = setup_logger("SAG", os.path.join(args.output_dir, "logs"))
    logger.info(f"SAG -- mode={args.mode}  dataset={args.dataset}")

    if args.mode == "train":
        train_single(args, logger)
    elif args.mode == "all_datasets":
        train_all(args, logger)
    elif args.mode == "ablation":
        from scripts.ablation_study import main as abl_main
        sys.argv = ["ablation.py",
                    "--dataset", args.dataset,
                    "--data_dir", args.data_dir,
                    "--output_dir", args.output_dir,
                    "--n_runs", str(args.n_runs),
                    "--seed", str(args.seed)]
        abl_main()
    elif args.mode == "benchmark":
        from experiments.neo4j_benchmark import main as bench_main
        sys.argv = ["neo4j_benchmark.py",
                    "--dataset", args.dataset,
                    "--data_dir", args.data_dir,
                    "--output_dir", args.output_dir]
        bench_main()
    logger.info("Done.")


if __name__ == "__main__":
    main()
