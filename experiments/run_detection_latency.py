"""SAG detection latency: how many edges after an anomaly is injected
does SAG put its node into the top-K predictions?

Setup:
  1. Take a clean dataset (Cora, Pubmed) WITHOUT injected cliques.
  2. Pre-fill SAG with the first 80% of clean edges (warm-up).
  3. Inject K anomaly cliques at known timesteps T_1, ..., T_K spaced over
     the remaining 20% of the stream.
  4. Process the rest of the stream edge-by-edge; after every M edges, snapshot
     per-node scores and recall@K-injected over the injected anomaly set.
  5. Plot recall vs. (edges since injection).

Companion baseline: MIDAS (true streaming). Offline methods (DOMINANT,
CONAD, AnomalyDAE) appear as a horizontal line at their final-batch recall —
i.e., they only "detect" after the entire stream finishes & full retrain.
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from configs import Config, get_dataset_config
from src.data import DataLoader
from src.models import create_sag_model
from src.utils import set_seed, save_json, setup_logger


def setup_cfg(dc):
    cfg = Config
    for k in ("alpha", "gamma", "rho"):
        if k in dc:
            setattr(cfg, k, dc[k])
    cfg.num_epochs = dc.get("num_epochs", 20)
    cfg.has_ground_truth = bool(dc.get("has_ground_truth", False))
    cfg.fusion_mode = dc.get("fusion_mode", "auto")
    cfg.edge_order = "shuffle"  # streaming-style
    cfg.attr_mode = dc.get("attr_mode", "auto")
    cfg.use_trained_ae = False  # disable AE so SAG state is the dominant signal
    cfg.use_attr_boost = False
    cfg.use_pca_mahal = False
    cfg.use_continuous_dev = False
    cfg.use_real_time_decay = False
    return cfg


def inject_clique_at(node_pool, clique_size, rng):
    """Sample `clique_size` distinct nodes from node_pool and return the
    full set of pairs (clique edges)."""
    members = rng.choice(node_pool, size=clique_size, replace=False)
    edges = []
    for i, u in enumerate(members):
        for v in members[i + 1:]:
            edges.append((int(u), int(v)))
    return list(members), edges


def stream_with_injections(loader, num_inject=5, clique_size=15,
                           warmup_frac=0.6, seed=42):
    """Returns:
        edge_stream: list of (u, v) edges
        injection_index: dict mapping anomaly_node -> edge_index where injected
    """
    rng = np.random.default_rng(seed)
    s = loader.get_stats()
    base_edges = list(loader.get_edge_stream(mode="bfs"))
    n_warm = int(len(base_edges) * warmup_frac)
    warmup = base_edges[:n_warm]
    rest = base_edges[n_warm:]

    # space injections evenly through `rest`
    inject_positions = np.linspace(0, len(rest), num_inject + 2)[1:-1].astype(int)

    # pick anomaly nodes from the LOW-degree pool so the injected clique edges
    # are a strong relative spike to SAG's state h_v.
    deg = np.asarray(loader.adj.sum(axis=1)).ravel()
    low_pool = np.argsort(deg)
    # avoid degree-0 / disconnected nodes; take 30%-60% percentile
    top_pool = low_pool[int(0.3 * s["num_nodes"]): int(0.6 * s["num_nodes"])]

    injected = {}
    final_edges = list(warmup)
    pos_so_far = 0
    chunks = np.array_split(rest, num_inject)
    for i, chunk in enumerate(chunks):
        # before this chunk, inject a clique
        members, edges = inject_clique_at(top_pool, clique_size, rng)
        edge_index_at_injection = len(final_edges)
        for u, v in edges:
            final_edges.append((u, v))
        for m in members:
            injected.setdefault(int(m), edge_index_at_injection)
        for u, v in chunk:
            final_edges.append((int(u), int(v)))
    return final_edges, injected, n_warm


def evaluate_detection_latency(name, num_inject, clique_size, snap_every,
                               output_path, logger):
    dc = get_dataset_config(name)
    loader = DataLoader(data_dir="./data", dataset_name=name,
                        clique_size=dc.get("clique_size", 15),
                        num_cliques=dc.get("num_cliques", 10), seed=42)
    loader.load()
    s = loader.get_stats()
    logger.info(f"=== {name} stats: {s}")

    edges, injected, n_warm = stream_with_injections(
        loader, num_inject=num_inject, clique_size=clique_size,
        warmup_frac=0.6, seed=42)
    inj_nodes = sorted(injected.keys())
    inj_step = {n: injected[n] for n in inj_nodes}
    K = len(inj_nodes)
    logger.info(f"  edges total: {len(edges):,}, warmup: {n_warm:,}, "
                f"injected anomalies: {K} (pool size: {len(inj_nodes)})")

    # Initialize SAG model
    set_seed(42)
    cfg = setup_cfg(dc)
    cfg.window_size = max(128, min(len(edges) // 32, 4096))
    n_nodes_max = max((max(u, v) for u, v in edges)) + 1
    model = create_sag_model(num_nodes=n_nodes_max, config=cfg,
                             features=None, adj=None,
                             ablation="no_attribute")  # pure SAG state

    # Process edges in micro-batches; snapshot per-injection-node score
    # trajectories + rank percentile over time.
    snapshots = []
    t0 = time.time()
    BATCH = max(snap_every // 4, 1)
    N = n_nodes_max

    for i in range(0, len(edges), BATCH):
        chunk = np.asarray(edges[i:i + BATCH], dtype=np.int64)
        model.process_stream_batch(chunk, passes=1, shuffle=False,
                                   rng=np.random.default_rng(42 + i))
        edge_idx = i + len(chunk)

        if edge_idx % snap_every < BATCH:
            scores = model.get_scores()
            # rank percentile of each node (1.0 = highest score)
            ranks = np.argsort(np.argsort(scores))
            inj_ranks = {int(n): float(ranks[n]) / max(N - 1, 1)
                         for n in inj_nodes if n < N}
            inj_scores = {int(n): float(scores[n])
                          for n in inj_nodes if n < N}
            snap = {
                "edge_index": edge_idx,
                "inj_node_rank_pct": inj_ranks,
                "inj_node_score": inj_scores,
            }
            snapshots.append(snap)
            if edge_idx % (snap_every * 4) < BATCH:
                avg_rank = np.mean(list(inj_ranks.values())) if inj_ranks else 0
                logger.info(f"  edges={edge_idx:,} | "
                            f"injected avg rank-pct = {avg_rank:.3f} "
                            f"({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    logger.info(f"  total stream processed in {elapsed:.1f}s "
                f"({len(edges)/elapsed:,.0f} edges/sec)")

    out = {
        "dataset": name,
        "num_injected": K,
        "warmup_edges": n_warm,
        "total_edges": len(edges),
        "stream_seconds": elapsed,
        "injections": [{"node": int(n), "edge_index": int(s)}
                       for n, s in inj_step.items()],
        "snapshots": snapshots,
    }
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="Cora,Pubmed")
    p.add_argument("--num_inject", type=int, default=5)
    p.add_argument("--clique_size", type=int, default=10)
    p.add_argument("--snap_every", type=int, default=200)
    p.add_argument("--output_dir", default="./outputs")
    return p.parse_args()


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.output_dir, "results", f"latency_{ts}")
    log_dir = os.path.join(args.output_dir, "logs", f"latency_{ts}")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger("SAG_LAT", log_dir)
    logger.info(f"Datasets: {datasets}")

    Config.data_dir = "./data"
    Config.output_dir = args.output_dir

    all_results = {}
    for ds in datasets:
        out_path = os.path.join(exp_dir, f"{ds}.json")
        all_results[ds] = evaluate_detection_latency(
            ds, args.num_inject, args.clique_size, args.snap_every,
            out_path, logger)
    save_json(all_results, os.path.join(exp_dir, "all_latency.json"))
    logger.info(f"Saved to {exp_dir}")


if __name__ == "__main__":
    main()
