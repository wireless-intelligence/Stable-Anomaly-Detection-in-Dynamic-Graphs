import argparse
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from configs import Config, get_dataset_config
from src.baselines import BASELINES
from src.data import DataLoader
from src.evaluation import compute_all_metrics
from src.models import create_sag_model
from src.utils import MemoryTracker, save_json, set_seed, setup_logger


def _driver(uri: str, user: str, password: str):
    from neo4j import GraphDatabase

    return GraphDatabase.driver(uri, auth=(user, password))


def _ingest_graph(driver, loader, logger, database: str = None, batch: int = 5000):
    name = loader.dataset_name
    with driver.session(database=database) if database else driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        session.run("CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.id)")

        stats = loader.get_stats()
        logger.info(f"Creating {stats['num_nodes']} nodes...")
        labels = loader.labels.tolist()
        for start in range(0, stats["num_nodes"], batch):
            end = min(start + batch, stats["num_nodes"])
            chunk = [{"id": i, "label": int(labels[i])} for i in range(start, end)]
            session.run(
                "UNWIND $batch AS row "
                "CREATE (n:Node {id: row.id, label: row.label, name: 'node_' + toString(row.id), type: $type})",
                batch=chunk,
                type=name,
            )

        logger.info(f"Creating edges (batched)...")
        edges = loader.get_edge_stream(shuffle=True)
        for start in range(0, edges.shape[0], batch):
            end = min(start + batch, edges.shape[0])
            chunk = [{"u": int(edges[i, 0]), "v": int(edges[i, 1]), "t": i}
                     for i in range(start, end)]
            session.run(
                "UNWIND $batch AS row "
                "MATCH (a:Node {id: row.u}), (b:Node {id: row.v}) "
                "CREATE (a)-[:INTERACT {t: row.t}]->(b)",
                batch=chunk,
            )


def _query_stream(driver, dataset: str, database: str = None, limit: int = None):
    with driver.session(database=database) if database else driver.session() as session:
        q = (
            "MATCH (a:Node {type: $type})-[r:INTERACT]->(b:Node {type: $type}) "
            "RETURN a.id AS u, b.id AS v "
            "ORDER BY r.t"
        )
        if limit is not None:
            q += f" LIMIT {int(limit)}"
        result = session.run(q, type=dataset)
        return np.asarray([(r["u"], r["v"]) for r in result], dtype=np.int64)


def _benchmark_method(name, cls_or_factory, loader, cfg, edges_ndarray, logger, seed: int, is_sag: bool):
    set_seed(seed)
    mt = MemoryTracker()
    mt.reset()
    if is_sag:
        model = create_sag_model(
            num_nodes=loader.stats["num_nodes"], config=cfg,
            features=loader.features, adj=loader.adj,
        )
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        order = getattr(cfg, "edge_order", "auto")
        if order == "auto":
            order = "shuffle" if bool(getattr(cfg, "has_ground_truth", False)) else "bfs"
        do_shuffle = (order == "shuffle")
        passes = (2 if edges_ndarray.shape[0] > 50000 else 3) if do_shuffle else 1
        model.process_stream_batch(edges_ndarray, passes=passes, shuffle=do_shuffle, rng=rng)
        elapsed = time.perf_counter() - t0
        scores = model.get_scores()
    else:
        model = cls_or_factory(
            num_nodes=loader.stats["num_nodes"],
            config=cfg,
            features=loader.features,
            seed=seed,
        )
        t0 = time.perf_counter()
        model.fit(loader.adj, edges_ndarray, loader.labels)
        scores = model.score()
        elapsed = time.perf_counter() - t0
    mt.update()
    m = compute_all_metrics(loader.labels, scores)
    per_edge_ms = elapsed * 1000.0 / max(1, edges_ndarray.shape[0])
    return {
        "auroc": m["auroc"],
        "auprc": m["auprc"],
        "macro_f1": m["macro_f1"],
        "query_latency_ms": elapsed * 1000.0,
        "per_edge_ms": per_edge_ms,
        "memory_gb": mt.get_peak()["peak_total_mem_gb"],
    }


def parse_args():
    p = argparse.ArgumentParser(description="Neo4j database testing for SAG")
    p.add_argument("--dataset", type=str, required=True,
                   choices=["YelpChi", "YelpReviews", "Enron", "BlogCatalog", "Amazon"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--uri", type=str, default=os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687"))
    p.add_argument("--user", type=str, default=os.environ.get("NEO4J_USER", "neo4j"))
    p.add_argument("--password", type=str, default=os.environ.get("NEO4J_PASSWORD", "neo4j"))
    p.add_argument("--database", type=str, default=os.environ.get("NEO4J_DATABASE", None))
    p.add_argument("--methods", type=str, default="SAG,GraphSAGE,DOMINANT,DySAT")
    p.add_argument("--edge_limit", type=int, default=200000,
                   help="limit edges to make ingest practical")
    p.add_argument("--skip_ingest", action="store_true")
    p.add_argument("--seed", type=int, default=42)
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
    Config.num_epochs = dc.get("num_epochs", 20)
    Config.has_ground_truth = bool(dc.get("has_ground_truth", False))
    Config.fusion_mode = dc.get("fusion_mode", "auto")
    Config.edge_order = dc.get("edge_order", "auto")
    Config.attr_mode = dc.get("attr_mode", "auto")
    Config.use_trained_ae = bool(dc.get("use_trained_ae", False))
    Config.trained_ae_cfg = dc.get("trained_ae_cfg", {})

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp = f"neo4j_{args.dataset}_{ts}"
    log_dir = os.path.join(args.output_dir, "logs", exp)
    os.makedirs(log_dir, exist_ok=True)
    result_dir = os.path.join(args.output_dir, "results", exp)
    os.makedirs(result_dir, exist_ok=True)
    logger = setup_logger("SAG_NEO4J", log_dir)

    loader = DataLoader(
        data_dir=Config.data_dir,
        dataset_name=args.dataset,
        clique_size=dc.get("clique_size", 15),
        num_cliques=dc.get("num_cliques", 10),
        seed=Config.seed,
    )
    loader.load()
    logger.info(f"Loaded {args.dataset}: {loader.get_stats()}")

    logger.info(f"Connecting to Neo4j at {args.uri}")
    driver = _driver(args.uri, args.user, args.password)

    try:
        if not args.skip_ingest:
            logger.info("Ingesting graph into Neo4j...")
            t0 = time.perf_counter()
            full_edges = loader.get_edge_stream(shuffle=True)
            if args.edge_limit and full_edges.shape[0] > args.edge_limit:
                full_edges = full_edges[: args.edge_limit]
            logger.info(f"Using {full_edges.shape[0]} edges for ingest")

            class _TmpLoader:
                dataset_name = args.dataset
                labels = loader.labels
                def get_stats(self):
                    return loader.get_stats()
                def get_edge_stream(self, shuffle=True):
                    return full_edges
            _ingest_graph(driver, _TmpLoader(), logger, database=args.database)
            logger.info(f"Ingest finished in {time.perf_counter()-t0:.1f}s")

        logger.info("Streaming edges from Neo4j...")
        t0 = time.perf_counter()
        edges = _query_stream(driver, args.dataset, database=args.database, limit=args.edge_limit)
        logger.info(f"Read {edges.shape[0]} edges in {(time.perf_counter()-t0)*1000:.1f}ms")

        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        results = {}
        for name in methods:
            logger.info(f"\n=== Benchmark: {name} ===")
            is_sag = name == "SAG"
            if is_sag:
                cls_or_factory = None
            elif name in BASELINES:
                cls_or_factory = BASELINES[name]
            else:
                logger.warning(f"Unknown method {name}")
                continue
            per_run = []
            for i in range(3):
                seed = Config.seed + i
                r = _benchmark_method(
                    name,
                    cls_or_factory,
                    loader,
                    Config,
                    edges,
                    logger,
                    seed=seed,
                    is_sag=is_sag,
                )
                per_run.append(r)
                logger.info(
                    f"  run seed={seed}: AUROC={r['auroc']:.4f} latency={r['query_latency_ms']:.1f}ms mem={r['memory_gb']:.2f}GB"
                )
            agg = {}
            for k in ["auroc", "auprc", "macro_f1", "query_latency_ms", "per_edge_ms", "memory_gb"]:
                v = np.asarray([r[k] for r in per_run])
                agg[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
            results[name] = {"per_run": per_run, "aggregated": agg}
            logger.info(
                "[%s] AUROC=%.1f  latency=%.1fms  mem=%.2fGB"
                % (name, agg["auroc"]["mean"] * 100, agg["query_latency_ms"]["mean"], agg["memory_gb"]["mean"])
            )
    finally:
        driver.close()

    save_json(
        {
            "dataset": args.dataset,
            "results": results,
            "edge_limit": args.edge_limit,
            "timestamp": ts,
        },
        os.path.join(result_dir, "neo4j_results.json"),
    )
    logger.info(f"Saved to {result_dir}")


if __name__ == "__main__":
    main()
