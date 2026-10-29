"""Loader wrapper for real dynamic-graph datasets.

Mirrors the surface of ``DataLoader`` (``.load()``, ``.stats``, ``.features``,
``.adj``, ``.labels``, ``.get_stats()``, ``.get_edge_stream()``) so existing
evaluation/baseline code works unchanged, and additionally exposes the genuine
timestamped stream via ``.edges``, ``.edge_times``, ``.eval_mask`` and
``.get_temporal_stream()``.
"""

import numpy as np
import scipy.sparse as sp

from .temporal_datasets import (
    TEMPORAL_DATASETS,
    canonical_name,
    is_temporal_dataset,
    load_temporal_dataset,
)


class TemporalDataLoader:
    def __init__(self, data_dir: str, dataset_name: str, seed: int = 42,
                 inject_anomalies: bool = True):
        self.data_dir = data_dir
        self.dataset_name = canonical_name(dataset_name)
        self.seed = seed
        self.inject_anomalies = inject_anomalies

        self.tg = None
        self.adj = None
        self.features = None
        self.labels = None
        self.eval_mask = None
        self.edges = None
        self.edge_times = None
        self.stats = None

    def load(self):
        tg = load_temporal_dataset(
            self.data_dir, self.dataset_name, seed=self.seed,
            inject_anomalies=self.inject_anomalies,
        )
        self.tg = tg
        self.edges = tg.edges
        self.edge_times = tg.edge_times
        self.labels = tg.labels
        self.eval_mask = tg.eval_mask

        feats = tg.features
        if not sp.issparse(feats):
            feats = sp.csr_matrix(feats)
        self.features = feats
        self.adj = tg.build_adj()

        n = tg.num_nodes
        self.stats = {
            "dataset": tg.name,
            "num_nodes": int(n),
            "num_edges": int(tg.edges.shape[0]),          # stream length (with multiplicity)
            "num_unique_edges": int(self.adj.nnz // 2),
            "num_features": int(self.features.shape[1]),
            "num_anomalies": int(tg.labels[tg.eval_mask].sum()),
            "num_eval_nodes": int(tg.eval_mask.sum()),
            "anomaly_ratio": (float(tg.labels[tg.eval_mask].sum()) /
                              float(max(1, tg.eval_mask.sum()))),
            "time_span": (float(tg.edge_times.min()), float(tg.edge_times.max()))
            if tg.edge_times.shape[0] else (0.0, 0.0),
            "synthetic_labels": bool(tg.meta.get("synthetic_labels", False)),
            "meta": tg.meta,
        }
        return self

    def get_temporal_stream(self):
        """Real time-ordered single-pass stream: (edges (E,2), edge_times (E,))."""
        return self.tg.sorted_by_time()

    def get_edge_stream(self, shuffle: bool = False, mode: str = None):
        """Edges only, for baselines that ignore timestamps. Defaults to true
        temporal order (single pass), NOT shuffled replay."""
        edges, _ = self.tg.sorted_by_time()
        if mode == "shuffle" or (mode is None and shuffle):
            rng = np.random.default_rng(self.seed)
            e = edges.copy()
            rng.shuffle(e)
            return e
        return edges

    def get_edge_times(self, edges=None):
        _, t = self.tg.sorted_by_time()
        return t

    def get_stats(self):
        return self.stats
