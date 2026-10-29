import numpy as np
import scipy.sparse as sp

from .base import BaselineModel


class SedanSpotBaseline(BaselineModel):
    name = "SedanSpot"

    def __init__(
        self,
        num_nodes: int,
        config,
        features=None,
        walk_length: int = 4,
        num_walks: int = 4,
        seed: int = 42,
        alpha: float = 0.15,
    ):
        super().__init__(num_nodes, config, features)
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.seed = seed
        self.alpha = alpha
        self.scores_ = None

    def fit(self, adj, edges: np.ndarray, labels=None):
        rng = np.random.default_rng(self.seed)
        n = self.num_nodes
        node_scores = np.zeros(n, dtype=np.float64)
        adj_d = {i: [] for i in range(n)}
        deg = np.zeros(n, dtype=np.int64)

        for i, (u, v) in enumerate(edges):
            u, v = int(u), int(v)
            adj_d[u].append(v)
            adj_d[v].append(u)
            deg[u] += 1
            deg[v] += 1
            if deg[u] == 0 or deg[v] == 0:
                continue
            sim_sum = 0.0
            for _ in range(self.num_walks):
                cur = u
                walk_score = 0.0
                for step in range(self.walk_length):
                    ngb = adj_d[cur]
                    if not ngb:
                        break
                    cur = ngb[rng.integers(0, len(ngb))]
                    walk_score += 1.0 / (1 + deg[cur])
                sim_sum += walk_score
            edge_score = sim_sum / self.num_walks if self.num_walks > 0 else 0.0
            freshness = self.alpha ** (i / max(1, edges.shape[0]))
            contrib = edge_score * freshness
            node_scores[u] += contrib
            node_scores[v] += contrib

        self.scores_ = -node_scores

    def score(self) -> np.ndarray:
        s = self.scores_
        return (s - s.min()) / (s.max() - s.min() + 1e-12)
