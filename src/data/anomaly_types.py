"""Multi-type synthetic anomaly injection.

Anomaly injectors beyond small-clique injection.
Each injector returns ``(adj_new, features_new, labels)`` where ``labels``
is a binary mask aligned to ``num_nodes``.

Types covered:
  structural:      clique, path, star, bridge, community_bridge
  attribute:       attr_swap (default), attr_gaussian, attr_outlier
  mixed:           clique_attr, heterophily

The runner :func:`inject_by_name` dispatches by name string, used by
``experiments/run_anomaly_type_breakdown.py``.
"""

import numpy as np
import scipy.sparse as sp


def _ensure_dense_adj(adj):
    A = sp.lil_matrix(adj, copy=True).astype(np.float32)
    return A


def _ensure_dense_feat(features):
    if features is None:
        return None
    if sp.issparse(features):
        return np.asarray(features.todense(), dtype=np.float32)
    return np.asarray(features, dtype=np.float32)


def _finalize(A_lil, X, n):
    A = A_lil.tocsr()
    A = A.maximum(A.T)
    A.setdiag(0); A.eliminate_zeros()
    return A, (sp.csr_matrix(X) if X is not None else None)


def inject_clique(adj, features, num_anomalies=50, clique_size=15, seed=42):
    rng = np.random.default_rng(seed)
    A = _ensure_dense_adj(adj)
    X = _ensure_dense_feat(features)
    n = A.shape[0]
    labels = np.zeros(n, dtype=np.int64)
    n_groups = max(1, num_anomalies // clique_size)
    used = set()
    for _ in range(n_groups):
        cand = [i for i in range(n) if i not in used]
        if len(cand) < clique_size: break
        grp = rng.choice(cand, size=clique_size, replace=False)
        for u in grp:
            for v in grp:
                if u != v: A[int(u), int(v)] = 1.0
        for u in grp: used.add(int(u)); labels[int(u)] = 1
    A, X = _finalize(A, X, n)
    return A, X, labels


def inject_path(adj, features, num_anomalies=50, path_length=8, seed=42):
    rng = np.random.default_rng(seed)
    A = _ensure_dense_adj(adj)
    X = _ensure_dense_feat(features)
    n = A.shape[0]
    labels = np.zeros(n, dtype=np.int64)
    n_paths = max(1, num_anomalies // path_length)
    used = set()
    for _ in range(n_paths):
        cand = [i for i in range(n) if i not in used]
        if len(cand) < path_length: break
        nodes = rng.choice(cand, size=path_length, replace=False).tolist()
        for i in range(path_length - 1):
            A[int(nodes[i]), int(nodes[i + 1])] = 1.0
        for u in nodes:
            used.add(int(u)); labels[int(u)] = 1
    return _finalize(A, X, n) + (labels,)


def inject_star(adj, features, num_anomalies=50, leaves_per_star=10, seed=42):
    rng = np.random.default_rng(seed)
    A = _ensure_dense_adj(adj)
    X = _ensure_dense_feat(features)
    n = A.shape[0]
    labels = np.zeros(n, dtype=np.int64)
    n_stars = max(1, num_anomalies // (leaves_per_star + 1))
    used = set()
    for _ in range(n_stars):
        cand = [i for i in range(n) if i not in used]
        if len(cand) < (leaves_per_star + 1): break
        nodes = rng.choice(cand, size=leaves_per_star + 1, replace=False).tolist()
        center = int(nodes[0])
        for v in nodes[1:]:
            A[center, int(v)] = 1.0
        for u in nodes:
            used.add(int(u)); labels[int(u)] = 1
    return _finalize(A, X, n) + (labels,)


def inject_bridge(adj, features, num_anomalies=30, seed=42):
    rng = np.random.default_rng(seed)
    A = _ensure_dense_adj(adj)
    X = _ensure_dense_feat(features)
    n = A.shape[0]
    labels = np.zeros(n, dtype=np.int64)
    A_csr = A.tocsr()
    deg = np.asarray(A_csr.sum(axis=1)).flatten()
    candidates = np.where(deg <= 2)[0]
    if len(candidates) < num_anomalies:
        candidates = np.argsort(deg)[:num_anomalies * 3]
    pool = rng.choice(candidates, size=min(num_anomalies, len(candidates)), replace=False)
    high_deg = np.argsort(deg)[::-1][: 2 * num_anomalies]
    for v in pool:
        u1, u2 = rng.choice(high_deg, size=2, replace=False)
        A[int(v), int(u1)] = 1.0
        A[int(v), int(u2)] = 1.0
        labels[int(v)] = 1
    return _finalize(A, X, n) + (labels,)


def inject_community_bridge(adj, features, num_anomalies=30, seed=42):
    rng = np.random.default_rng(seed)
    A = _ensure_dense_adj(adj)
    X = _ensure_dense_feat(features)
    n = A.shape[0]
    labels = np.zeros(n, dtype=np.int64)
    A_csr = A.tocsr()
    deg = np.asarray(A_csr.sum(axis=1)).flatten()
    if X is not None:
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
    else:
        Xn = None
    candidates = rng.choice(n, size=min(num_anomalies * 3, n), replace=False)
    chosen = []
    for v in candidates:
        if len(chosen) >= num_anomalies: break
        far = []
        if Xn is not None:
            sims = Xn @ Xn[int(v)]
            far = np.argsort(sims)[:30].tolist()
        else:
            far = rng.choice(n, size=30, replace=False).tolist()
        far = [int(x) for x in far if int(x) != int(v)][:6]
        if len(far) >= 2:
            for u in far:
                A[int(v), int(u)] = 1.0
            labels[int(v)] = 1
            chosen.append(int(v))
    return _finalize(A, X, n) + (labels,)


def inject_attr_swap(adj, features, num_anomalies=50, k_candidates=50, seed=42):
    rng = np.random.default_rng(seed)
    A = _ensure_dense_adj(adj)
    X = _ensure_dense_feat(features)
    n = A.shape[0]
    if X is None:
        X = np.zeros((n, 1), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    pool = rng.choice(n, size=num_anomalies, replace=False)
    for v in pool:
        cand = rng.choice(n, size=min(k_candidates, n), replace=False)
        diffs = np.linalg.norm(X[cand] - X[int(v)], axis=1)
        far = int(cand[diffs.argmax()])
        X[int(v)] = X[far]
        labels[int(v)] = 1
    return _finalize(A, X, n) + (labels,)


def inject_attr_gaussian(adj, features, num_anomalies=50, sigma=2.0, seed=42):
    rng = np.random.default_rng(seed)
    A = _ensure_dense_adj(adj)
    X = _ensure_dense_feat(features)
    n = A.shape[0]
    if X is None:
        X = np.zeros((n, 1), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    pool = rng.choice(n, size=num_anomalies, replace=False)
    feat_std = X.std(axis=0) + 1e-6
    for v in pool:
        noise = rng.standard_normal(X.shape[1]).astype(np.float32) * (sigma * feat_std)
        X[int(v)] = X[int(v)] + noise
        labels[int(v)] = 1
    return _finalize(A, X, n) + (labels,)


def inject_attr_outlier(adj, features, num_anomalies=50, q=0.99, seed=42):
    rng = np.random.default_rng(seed)
    A = _ensure_dense_adj(adj)
    X = _ensure_dense_feat(features)
    n = A.shape[0]
    if X is None:
        X = np.zeros((n, 1), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    high_q = np.quantile(X, q, axis=0)
    low_q = np.quantile(X, 1 - q, axis=0)
    pool = rng.choice(n, size=num_anomalies, replace=False)
    for v in pool:
        flip = rng.integers(0, 2, size=X.shape[1]).astype(bool)
        X[int(v)] = np.where(flip, high_q, low_q)
        labels[int(v)] = 1
    return _finalize(A, X, n) + (labels,)


def inject_clique_attr(adj, features, num_anomalies=50, clique_size=15, seed=42):
    A1, X1, lab_struct = inject_clique(adj, features, num_anomalies, clique_size, seed)
    A2, X2, lab_attr = inject_attr_swap(A1, X1, num_anomalies, seed=seed + 1)
    labels = np.maximum(lab_struct, lab_attr)
    return A2, X2, labels


def inject_heterophily(adj, features, num_anomalies=50, seed=42):
    rng = np.random.default_rng(seed)
    A = _ensure_dense_adj(adj)
    X = _ensure_dense_feat(features)
    n = A.shape[0]
    A_csr = A.tocsr()
    deg = np.asarray(A_csr.sum(axis=1)).flatten()
    if X is None:
        X = rng.standard_normal((n, 8)).astype(np.float32)
    labels = np.zeros(n, dtype=np.int64)
    high_deg = np.argsort(deg)[::-1][: max(num_anomalies * 2, 100)]
    pool = rng.choice(high_deg, size=min(num_anomalies, len(high_deg)), replace=False)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
    for v in pool:
        sims = Xn @ Xn[int(v)]
        far = np.argsort(sims)[:8]
        for u in far:
            if int(u) != int(v):
                A[int(v), int(u)] = 1.0
        labels[int(v)] = 1
    return _finalize(A, X, n) + (labels,)


INJECTORS = {
    "clique":            inject_clique,
    "path":              inject_path,
    "star":              inject_star,
    "bridge":            inject_bridge,
    "community_bridge":  inject_community_bridge,
    "attr_swap":         inject_attr_swap,
    "attr_gaussian":     inject_attr_gaussian,
    "attr_outlier":      inject_attr_outlier,
    "clique_attr":       inject_clique_attr,
    "heterophily":       inject_heterophily,
}


def inject_by_name(name, adj, features, num_anomalies=50, seed=42, **kwargs):
    if name not in INJECTORS:
        raise ValueError(f"Unknown anomaly type: {name} (have {list(INJECTORS)})")
    return INJECTORS[name](adj, features, num_anomalies=num_anomalies, seed=seed, **kwargs)
