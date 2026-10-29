"""SAG throughput scaling: edges/sec as |E| grows.

Strategy: take a single large dataset (CTU-13 by default) and run SAG on
prefixes of size 50K, 100K, 200K, 500K, 1M, 2M, 4M edges. For each prefix,
record wall-clock time, edges/sec, and peak memory.

Companion baseline: MIDAS (the only true streaming graph anomaly detector).
For pygod baselines we just record OOM/time-out at large sizes.

Output: outputs/results/throughput_<ts>/{ds}.json + summary.
"""

import argparse
import os
import sys
import json
import time
import tracemalloc
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.sparse as sp

from configs import Config, get_dataset_config
from src.data import DataLoader
from src.models import create_sag_model
from src.utils import set_seed, save_json, setup_logger


SIZES_DEFAULT = {
    "CTU-13":      [50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000],
    "YelpReviews": [100_000, 250_000, 500_000, 1_000_000, 2_000_000],
    "Flickr":      [50_000, 100_000, 200_000, 400_000, 700_000],
}


def setup_cfg(dc):
    cfg = Config
    for k in ("alpha", "gamma", "rho"):
        if k in dc:
            setattr(cfg, k, dc[k])
    cfg.num_epochs = dc.get("num_epochs", 20)
    cfg.has_ground_truth = bool(dc.get("has_ground_truth", False))
    cfg.fusion_mode = dc.get("fusion_mode", "auto")
    cfg.edge_order = dc.get("edge_order", "auto")
    cfg.attr_mode = dc.get("attr_mode", "auto")
    cfg.use_trained_ae = bool(dc.get("use_trained_ae", False))
    cfg.trained_ae_cfg = dc.get("trained_ae_cfg", {})
    cfg.use_attr_boost = bool(dc.get("use_attr_boost", False))
    cfg.use_pca_mahal = bool(dc.get("use_pca_mahal", False))
    cfg.use_continuous_dev = False
    cfg.use_real_time_decay = False
    return cfg


def measure_one(loader, prefix_size, dc, logger):
    """Run SAG on the first `prefix_size` edges of the loader. Returns timing dict."""
    set_seed(42)
    s = loader.get_stats()
    cfg = setup_cfg(dc)
    cfg.window_size = max(128, min(prefix_size // 32, 4096))

    edges_iter = list(loader.get_edge_stream(mode=dc.get("edge_order", "auto")
                                              if dc.get("edge_order", "auto") != "auto"
                                              else "bfs"))
    edges_arr = np.asarray(edges_iter[:prefix_size], dtype=np.int64)
    actual_n = int(edges_arr.max()) + 1 if edges_arr.size else 1
    actual_n = max(actual_n, 1)

    model = create_sag_model(num_nodes=actual_n, config=cfg,
                             features=None, adj=None,
                             ablation="no_attribute")  # pure structure for fair scaling

    tracemalloc.start()
    t0 = time.time()
    model.process_stream_batch(edges_arr, passes=1, shuffle=False,
                               rng=np.random.default_rng(42))
    elapsed = time.time() - t0
    cur_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    eps = len(edges_arr) / max(elapsed, 1e-9)
    logger.info(f"  |E|={len(edges_arr):>10,} | {elapsed:7.2f}s | "
                f"{eps:>10,.0f} edges/sec | "
                f"peak {peak_mem/1024/1024:6.1f} MB")
    return {
        "edges_processed": int(len(edges_arr)),
        "wall_seconds": elapsed,
        "edges_per_sec": eps,
        "peak_mem_bytes": int(peak_mem),
        "active_nodes": int(actual_n),
    }


def measure_dataset(name, sizes, output_path, logger):
    dc = get_dataset_config(name)
    loader = DataLoader(data_dir="./data", dataset_name=name,
                        clique_size=dc.get("clique_size", 15),
                        num_cliques=dc.get("num_cliques", 10), seed=42)
    loader.load()
    s = loader.get_stats()
    logger.info(f"=== {name} stats: {s}")
    sizes = [k for k in sizes if k <= s["num_edges"]]

    results = {"dataset": name, "stats": s, "runs": []}
    for k in sizes:
        try:
            row = measure_one(loader, k, dc, logger)
            results["runs"].append(row)
        except Exception as e:
            logger.error(f"  |E|={k:,}: FAILED {type(e).__name__}: {e}")
            results["runs"].append({"edges_processed": k, "error": str(e)})
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="CTU-13,YelpReviews,Flickr")
    p.add_argument("--output_dir", default="./outputs")
    return p.parse_args()


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.output_dir, "results", f"throughput_{ts}")
    log_dir = os.path.join(args.output_dir, "logs", f"throughput_{ts}")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger("SAG_THRU", log_dir)
    logger.info(f"Datasets: {datasets}")

    Config.data_dir = "./data"
    Config.output_dir = args.output_dir

    all_results = {}
    for ds in datasets:
        out_path = os.path.join(exp_dir, f"{ds}.json")
        all_results[ds] = measure_dataset(ds, SIZES_DEFAULT.get(ds, [50_000, 200_000]),
                                          out_path, logger)
    save_json(all_results, os.path.join(exp_dir, "all_throughput.json"))
    logger.info(f"Saved to {exp_dir}")


if __name__ == "__main__":
    main()
