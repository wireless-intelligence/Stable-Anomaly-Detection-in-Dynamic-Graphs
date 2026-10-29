import math

import numpy as np

from .base import BaselineModel


class MIDASBaseline(BaselineModel):
    name = "MIDAS"

    def __init__(self, num_nodes: int, config, features=None, num_hash: int = 2, buckets: int = 2 ** 14, seed: int = 42):
        super().__init__(num_nodes, config, features)
        self.num_hash = num_hash
        self.buckets = buckets
        self.seed = seed
        self.scores_ = None

    def _cms_hash_params(self, rng):
        a = rng.integers(1, 2 ** 31 - 1, size=self.num_hash)
        b = rng.integers(0, 2 ** 31 - 1, size=self.num_hash)
        return a, b

    def _hash(self, x: np.ndarray, a, b):
        return (a[:, None] * x[None, :] + b[:, None]) % self.buckets

    def fit(self, adj, edges: np.ndarray, labels=None):
        rng = np.random.default_rng(self.seed)
        a1, b1 = self._cms_hash_params(rng)
        a2, b2 = self._cms_hash_params(rng)

        n = self.num_nodes
        node_scores = np.zeros(n, dtype=np.float64)
        window_size = max(1, edges.shape[0] // 128)
        cur = np.zeros((self.num_hash, self.buckets), dtype=np.float64)
        tot = np.zeros((self.num_hash, self.buckets), dtype=np.float64)
        wcount = 0
        t = 1

        u_arr = edges[:, 0].astype(np.int64)
        v_arr = edges[:, 1].astype(np.int64)
        ekey = u_arr * n + v_arr
        h_e = (a1[:, None] * ekey[None, :] + b1[:, None]) % self.buckets

        for i in range(edges.shape[0]):
            u, v = int(u_arr[i]), int(v_arr[i])
            for hi in range(self.num_hash):
                cur[hi, h_e[hi, i]] += 1.0
                tot[hi, h_e[hi, i]] += 1.0
            wcount += 1
            if wcount >= window_size:
                t += 1
                scores_edge = np.zeros(self.num_hash)
                for hi in range(self.num_hash):
                    idx = h_e[hi, i]
                    s_cur = cur[hi, idx]
                    s_tot = tot[hi, idx]
                    expected = s_tot / t
                    if expected > 0:
                        scores_edge[hi] = ((s_cur - expected) ** 2) / (expected * max(1, t - 1) / t)
                    else:
                        scores_edge[hi] = 0.0
                edge_score = float(np.min(scores_edge))
                node_scores[u] += edge_score
                node_scores[v] += edge_score
                cur.fill(0.0)
                wcount = 0

        self.scores_ = node_scores

    def score(self) -> np.ndarray:
        s = self.scores_
        return (s - s.min()) / (s.max() - s.min() + 1e-12)
