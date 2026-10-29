from collections import deque

import numpy as np
import scipy.sparse as sp


def random_shuffle(adj, seed: int = 42) -> np.ndarray:
    coo = adj.tocoo()
    mask = coo.row < coo.col
    edges = np.stack([coo.row[mask], coo.col[mask]], axis=1).astype(np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(edges)
    return edges


def bfs_order(adj, seed: int = 42) -> np.ndarray:
    A = sp.csr_matrix(adj)
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    visited_nodes = np.zeros(n, dtype=bool)
    seen_edge = set()
    out = []

    starts = rng.permutation(n)
    for start in starts:
        if visited_nodes[start]:
            continue
        q = deque([int(start)])
        visited_nodes[start] = True
        while q:
            u = q.popleft()
            nbrs = A.indices[A.indptr[u]: A.indptr[u + 1]].copy()
            rng.shuffle(nbrs)
            for v in nbrs:
                v = int(v)
                if u == v:
                    continue
                k = (u, v) if u < v else (v, u)
                if k in seen_edge:
                    continue
                seen_edge.add(k)
                out.append(k)
                if not visited_nodes[v]:
                    visited_nodes[v] = True
                    q.append(v)
    return np.asarray(out, dtype=np.int64)


def dfs_order(adj, seed: int = 42) -> np.ndarray:
    A = sp.csr_matrix(adj)
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    visited_nodes = np.zeros(n, dtype=bool)
    seen_edge = set()
    out = []

    starts = rng.permutation(n)
    for start in starts:
        if visited_nodes[start]:
            continue
        stack = [int(start)]
        visited_nodes[start] = True
        while stack:
            u = stack.pop()
            nbrs = A.indices[A.indptr[u]: A.indptr[u + 1]].copy()
            rng.shuffle(nbrs)
            for v in nbrs:
                v = int(v)
                if u == v:
                    continue
                k = (u, v) if u < v else (v, u)
                if k in seen_edge:
                    continue
                seen_edge.add(k)
                out.append(k)
                if not visited_nodes[v]:
                    visited_nodes[v] = True
                    stack.append(v)
    return np.asarray(out, dtype=np.int64)


def random_walk_order(adj, seed: int = 42, walk_length: int = 50, num_walks_per_node: int = 1) -> np.ndarray:
    A = sp.csr_matrix(adj)
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    seen_edge = set()
    out = []

    starts = np.repeat(rng.permutation(n), num_walks_per_node)
    rng.shuffle(starts)
    for start in starts:
        u = int(start)
        for _ in range(walk_length):
            row_start = A.indptr[u]
            row_end = A.indptr[u + 1]
            if row_start == row_end:
                break
            nbrs = A.indices[row_start: row_end]
            v = int(nbrs[rng.integers(0, nbrs.shape[0])])
            if u != v:
                k = (u, v) if u < v else (v, u)
                if k not in seen_edge:
                    seen_edge.add(k)
                    out.append(k)
            u = v

    coo = A.tocoo()
    mask = coo.row < coo.col
    rows = coo.row[mask]
    cols = coo.col[mask]
    leftover = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        if (int(r), int(c)) not in seen_edge:
            leftover.append((int(r), int(c)))
    rng.shuffle(leftover)
    out.extend(leftover)
    return np.asarray(out, dtype=np.int64)


def multi_relation_stream(rel_adjs, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = []
    for _, A in rel_adjs:
        coo = A.tocoo()
        mask = coo.row < coo.col
        rows = coo.row[mask]
        cols = coo.col[mask]
        e = np.stack([rows, cols], axis=1).astype(np.int64)
        rng.shuffle(e)
        parts.append(e)
    interleaved = []
    iters = [iter(p) for p in parts]
    active = [True] * len(parts)
    while any(active):
        for i, it in enumerate(iters):
            if not active[i]:
                continue
            try:
                interleaved.append(next(it))
            except StopIteration:
                active[i] = False
    if not interleaved:
        return np.zeros((0, 2), dtype=np.int64)
    return np.stack(interleaved, axis=0)


def temporal_order(adj, node_times: np.ndarray, seed: int = 42) -> np.ndarray:
    coo = adj.tocoo()
    mask = coo.row < coo.col
    rows = coo.row[mask]
    cols = coo.col[mask]
    edges = np.stack([rows, cols], axis=1).astype(np.int64)
    edge_t = np.minimum(node_times[rows], node_times[cols])
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(0.0, 1e-6, size=edges.shape[0])
    order = np.argsort(edge_t + jitter, kind="mergesort")
    return edges[order]


def get_edge_stream(adj, mode: str = "shuffle", seed: int = 42, node_times=None, **kwargs) -> np.ndarray:
    if mode == "shuffle":
        return random_shuffle(adj, seed=seed)
    if mode == "bfs":
        return bfs_order(adj, seed=seed)
    if mode == "dfs":
        return dfs_order(adj, seed=seed)
    if mode in ("rw", "random_walk"):
        return random_walk_order(adj, seed=seed, **kwargs)
    if mode == "temporal":
        if node_times is None:
            raise ValueError("temporal ordering requires node_times")
        return temporal_order(adj, node_times=np.asarray(node_times), seed=seed)
    raise ValueError(f"unknown edge ordering mode: {mode}")
