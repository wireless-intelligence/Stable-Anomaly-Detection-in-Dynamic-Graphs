import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from configs import Config, get_dataset_config
from src.data import DataLoader
from src.evaluation import compute_all_metrics
from src.models import create_sag_model
from src.utils import MemoryTracker, Timer, save_json, set_seed, setup_logger


def parse_args():
    p = argparse.ArgumentParser(description="SAG inference / score export")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_scores", action="store_true")
    p.add_argument("--visualize", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    Config.data_dir = args.data_dir
    Config.output_dir = args.output_dir
    Config.seed = args.seed

    dc = get_dataset_config(args.dataset)
    for k in ("alpha", "gamma", "rho"):
        if k in dc:
            setattr(Config, k, dc[k])
    Config.num_epochs = 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp = f"{args.dataset}_inference_{ts}"
    Config.result_dir = os.path.join(args.output_dir, "inference", exp)
    Config.figure_dir = os.path.join(Config.result_dir, "figures")
    Config.vis_dir = os.path.join(Config.result_dir, "visualizations")
    Config.log_dir = Config.result_dir
    Config.checkpoint_dir = Config.result_dir
    Config.create_dirs()

    logger = setup_logger("SAG_INF", Config.result_dir)
    set_seed(Config.seed)

    loader = DataLoader(
        data_dir=Config.data_dir,
        dataset_name=args.dataset,
        clique_size=dc.get("clique_size", 15),
        num_cliques=dc.get("num_cliques", 10),
        seed=Config.seed,
    )
    loader.load()
    logger.info(f"Stats: {loader.get_stats()}")

    model = create_sag_model(
        num_nodes=loader.stats["num_nodes"], config=Config,
        features=loader.features, adj=loader.adj,
    )
    edges = loader.get_edge_stream(shuffle=True)
    timer = Timer(); mt = MemoryTracker()
    timer.start(); mt.reset()
    model.process_stream_batch(edges, passes=1, shuffle=False, rng=np.random.default_rng(Config.seed))
    timer.stop(); mt.update()
    scores = model.get_scores()
    labels = loader.labels
    metrics = compute_all_metrics(labels, scores)
    logger.info("AUROC=%.4f AUPRC=%.4f F1=%.4f latency=%.1fms mem=%.2fGB" % (
        metrics["auroc"], metrics["auprc"], metrics["macro_f1"],
        timer.get_elapsed_ms(), mt.get_peak()["peak_total_mem_gb"]
    ))

    save_json(
        {"dataset": args.dataset, "metrics": metrics,
         "latency_ms": timer.get_elapsed_ms(),
         "memory_gb": mt.get_peak()["peak_total_mem_gb"], "timestamp": ts},
        os.path.join(Config.result_dir, "inference_results.json"),
    )

    if args.save_scores:
        np.save(os.path.join(Config.result_dir, "anomaly_scores.npy"), scores)
        save_json(
            {"nodes": [{"node_id": i, "score": float(scores[i]),
                        "label": int(labels[i])} for i in range(len(scores))]},
            os.path.join(Config.result_dir, "node_details.json"),
        )

    if args.visualize:
        from src.visualization import create_d3_visualization, plot_score_distribution
        plot_score_distribution(
            scores, labels,
            os.path.join(Config.figure_dir, "score_distribution.png"),
            dataset_name=args.dataset,
        )
        create_d3_visualization(
            loader.raw_data["adj"], scores, labels,
            os.path.join(Config.vis_dir, "graph_visualization.html"),
        )


if __name__ == "__main__":
    main()
