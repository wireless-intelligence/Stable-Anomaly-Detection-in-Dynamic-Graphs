import numpy as np
import scipy.sparse as sp


def _clone_csr(adj):
    src = sp.csr_matrix(adj)
    return sp.csr_matrix(
        (src.data.astype(np.float64, copy=True),
         src.indices.astype(np.int32, copy=True),
         src.indptr.astype(np.int32, copy=True)),
        shape=src.shape,
        copy=False,
    )


def _sym_normalize(adj: sp.csr_matrix) -> sp.csr_matrix:
    n = adj.shape[0]
    A = adj + sp.eye(n, dtype=np.float64)
    deg = np.asarray(A.sum(axis=1)).flatten()
    ds = np.power(np.maximum(deg, 1.0), -0.5)
    D = sp.diags(ds)
    return (D @ A @ D).tocsr()


def _safe_minmax(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(), v.max()
    if hi <= lo:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


class AttributeEvidenceEnhanced:
    def __init__(
        self,
        features=None,
        adj=None,
        num_propagation: int = 2,
        nonzero_weight: float = 3.0,
        epsilon: float = 1e-8,
        compute_iforest: bool = True,
        compute_pca_mahal: bool = True,
        pca_mahal_dim: int = 64,
    ):
        self.features = None
        self.attribute_evidence = None
        self.iforest_evidence = None
        self.pca_mahal_evidence = None
        self.epsilon = epsilon
        self.num_propagation = max(1, int(num_propagation))
        self.nonzero_weight = nonzero_weight
        self.compute_iforest = compute_iforest
        self.compute_pca_mahal = compute_pca_mahal
        self.pca_mahal_dim = pca_mahal_dim
        if features is not None:
            self.set_features(features, adj=adj)

    def set_features(self, features, adj=None) -> None:
        X = features.toarray() if sp.issparse(features) else np.asarray(features)
        X = X.astype(np.float64)
        n = X.shape[0]

        if self.compute_iforest:
            try:
                from sklearn.ensemble import IsolationForest
                rng = np.random.default_rng(42)
                if X.shape[1] > 256:
                    cols = rng.choice(X.shape[1], 256, replace=False)
                    Xr = X[:, cols]
                else:
                    Xr = X
                Xl = np.sign(Xr) * np.log1p(np.abs(Xr))
                if adj is not None and X.shape[1] <= 64:
                    A_loc = _clone_csr(adj)
                    A_loc.setdiag(0); A_loc.eliminate_zeros()
                    deg_loc = np.asarray(A_loc.sum(axis=1)).flatten().astype(np.float64)
                    deg_safe = np.maximum(deg_loc, 1.0)
                    A_norm_loc = sp.diags(1.0 / deg_safe) @ A_loc
                    X_neigh = np.asarray(A_norm_loc @ Xr, dtype=np.float64)
                    Xl_aug = np.concatenate([Xl, X_neigh], axis=1).astype(np.float64)
                else:
                    Xl_aug = Xl
                fit_X = Xl_aug if Xl_aug.shape[0] < 100000 else Xl_aug[:50000]
                clf = IsolationForest(n_estimators=200, contamination="auto",
                                       random_state=42, n_jobs=-1)
                clf.fit(fit_X)
                if_score = -clf.score_samples(Xl_aug)
                self.iforest_evidence = _safe_minmax(if_score).astype(np.float64)
            except Exception:
                self.iforest_evidence = None

        if self.compute_pca_mahal and X.shape[0] > 8:
            try:
                from sklearn.decomposition import PCA
                k = min(int(self.pca_mahal_dim), X.shape[1] - 1, X.shape[0] - 1)
                if k >= 4:
                    pca = PCA(n_components=k, random_state=42)
                    Z = pca.fit_transform(X)
                    cov_z = np.cov(Z, rowvar=False)
                    inv_cov = np.linalg.pinv(cov_z + 1e-6 * np.eye(k))
                    diff = Z - Z.mean(axis=0)
                    mahal = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', diff, inv_cov, diff), 0.0))
                    self.pca_mahal_evidence = _safe_minmax(mahal).astype(np.float64)
            except Exception:
                self.pca_mahal_evidence = None

        if adj is not None:
            A = _clone_csr(adj)
            A.setdiag(0)
            A.eliminate_zeros()
            A_hat = _sym_normalize(A)

            sparsity = 1.0 - float((X != 0).mean())
            eta = self.nonzero_weight if sparsity > 0.5 else 1.0
            B_attr = (X != 0).astype(np.float64) * (eta - 1.0) + 1.0

            H_shallow = A_hat @ X
            rec_err_shallow = np.linalg.norm((X - H_shallow) * B_attr, axis=1)

            H_deep = H_shallow
            for _ in range(max(0, self.num_propagation - 1)):
                H_deep = A_hat @ H_deep
            rec_err_deep = np.linalg.norm((X - H_deep) * B_attr, axis=1)

            evidence = 0.6 * _safe_minmax(rec_err_deep) + 0.4 * _safe_minmax(rec_err_shallow)
        else:
            mu = X.mean(axis=0, keepdims=True)
            centered = X - mu
            sigma = centered.std(axis=0, keepdims=True) + self.epsilon
            z = centered / sigma
            mahal = np.linalg.norm(z, axis=1)
            evidence = _safe_minmax(mahal)

        self.attribute_evidence = evidence.astype(np.float64)

    def boost_vec(self, nodes) -> np.ndarray:
        if self.attribute_evidence is None:
            return np.zeros(len(nodes), dtype=np.float64)
        return self.attribute_evidence[nodes]


class StructuralEvidence:
    def __init__(self, adj=None, epsilon: float = 1e-8):
        self.adj = None
        self.epsilon = epsilon
        self.structural_score = None
        if adj is not None:
            self.set_adj(adj)

    def set_adj(self, adj) -> None:
        A = _clone_csr(adj)
        A.setdiag(0.0)
        A.eliminate_zeros()
        self.adj = A
        n = A.shape[0]

        deg = np.asarray(A.sum(axis=1)).flatten()
        log_deg = np.log1p(deg)

        clust = self._clustering_coefficient(A, deg)

        deg_rec = self._degree_reconstruction_error(A, deg)

        two_hop_excess = self._two_hop_closure(A)

        def _norm(v):
            return _safe_minmax(v)

        score = (
            0.40 * _norm(clust)
            + 0.25 * _norm(log_deg)
            + 0.20 * _norm(deg_rec)
            + 0.15 * _norm(two_hop_excess)
        )
        self.structural_score = score.astype(np.float64)

    def _clustering_coefficient(self, A: sp.csr_matrix, deg: np.ndarray) -> np.ndarray:
        A_bin = sp.csr_matrix(
            (np.ones(A.nnz, dtype=np.float64), A.indices.copy(), A.indptr.copy()),
            shape=A.shape,
        )
        A2 = (A_bin @ A_bin).astype(np.float64)
        prod = A2.multiply(A_bin)
        triangles = np.asarray(prod.sum(axis=1)).flatten()
        denom = deg * (deg - 1)
        denom = np.where(denom > 0, denom, 1.0)
        return triangles / denom

    def _degree_reconstruction_error(self, A: sp.csr_matrix, deg: np.ndarray) -> np.ndarray:
        A_hat = _sym_normalize(A)
        pred_deg = np.asarray(A_hat @ deg).flatten()
        return np.abs(deg - pred_deg)

    def _two_hop_closure(self, A: sp.csr_matrix) -> np.ndarray:
        A_bin = sp.csr_matrix(
            (np.ones(A.nnz, dtype=np.float64), A.indices.copy(), A.indptr.copy()),
            shape=A.shape,
        )
        A2 = (A_bin @ A_bin).astype(np.float64)
        A2.setdiag(0)
        A2.eliminate_zeros()
        two_hop_deg = np.asarray(A2.sum(axis=1)).flatten()
        one_hop_deg = np.asarray(A_bin.sum(axis=1)).flatten()
        return two_hop_deg / (one_hop_deg + self.epsilon)
