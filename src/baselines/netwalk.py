import numpy as np
import scipy.sparse as sp

from .base import BaselineModel


class NetWalkBaseline(BaselineModel):
    name = "NetWalk"

    def __init__(
        self,
        num_nodes: int,
        config,
        features=None,
        walk_length: int = 10,
        num_walks: int = 20,
        embed_dim: int = 64,
        seed: int = 42,
    ):
        super().__init__(num_nodes, config, features)
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.embed_dim = embed_dim
        self.seed = seed
        self.scores_ = None

    def _random_walk(self, adj, start, rng):
        walk = [start]
        cur = start
        for _ in range(self.walk_length - 1):
            neighbors = adj[cur].indices if sp.issparse(adj) else np.nonzero(adj[cur])[0]
            if len(neighbors) == 0:
                break
            cur = int(rng.choice(neighbors))
            walk.append(cur)
        return walk

    def fit(self, adj, edges: np.ndarray, labels=None):
        rng = np.random.default_rng(self.seed)
        adj = sp.csr_matrix(adj)
        n = self.num_nodes

        co_occur = np.zeros((n,), dtype=np.float64)
        coocc_sum = np.zeros(n, dtype=np.float64)
        for node in range(n):
            for _ in range(self.num_walks):
                walk = self._random_walk(adj, node, rng)
                for w in walk:
                    coocc_sum[w] += 1.0

        mat = sp.lil_matrix((n, n), dtype=np.float32)
        for node in range(n):
            for _ in range(self.num_walks):
                walk = self._random_walk(adj, node, rng)
                for i, u in enumerate(walk):
                    for v in walk[max(0, i - 2): i + 3]:
                        if u != v:
                            mat[u, v] += 1.0
        mat = mat.tocsr()

        rows = mat.sum(axis=1)
        rows[rows == 0] = 1.0
        P = mat.multiply(1.0 / rows)
        U, S, _ = sp.linalg.svds(P.astype(np.float32), k=min(self.embed_dim, n - 1))
        emb = U * np.sqrt(S)

        mu = emb.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(emb - mu, axis=1)
        self.scores_ = dists

    def score(self) -> np.ndarray:
        s = self.scores_
        return (s - s.min()) / (s.max() - s.min() + 1e-12)
