import os
import numpy as np
import scipy.io as sio
import scipy.sparse as sp


ADJ_KEYS = ["Network", "A", "adj", "homo", "graph"]
FEATURE_KEYS = ["Attributes", "X", "features", "attributes", "attr"]
LABEL_KEYS = ["Label", "gnd", "label", "labels", "y"]


class MatDataLoader:
    def __init__(self, data_dir: str, dataset_name: str, mat_path: str = None):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        if mat_path is None:
            mat_path = os.path.join(data_dir, dataset_name, f"{dataset_name}.mat")
        self.data_path = mat_path

        self.adj = None
        self.features = None
        self.labels = None

    def _pick(self, data, keys):
        for k in keys:
            if k in data:
                return data[k]
        return None

    def load(self):
        if not os.path.exists(self.data_path):
            alt = os.path.join(os.path.dirname(self.data_path), f"{self.dataset_name}.mat")
            if os.path.exists(alt):
                self.data_path = alt
            else:
                raise FileNotFoundError(f"Data file not found: {self.data_path}")

        data = sio.loadmat(self.data_path)
        adj_raw = self._pick(data, ADJ_KEYS)
        if adj_raw is None:
            if "edge_u" in data and "edge_v" in data:
                u = np.asarray(data["edge_u"]).flatten().astype(np.int64)
                v = np.asarray(data["edge_v"]).flatten().astype(np.int64)
                feat_raw_for_n = self._pick(data, FEATURE_KEYS)
                if feat_raw_for_n is not None:
                    if sp.issparse(feat_raw_for_n):
                        n_inferred = feat_raw_for_n.shape[0]
                    else:
                        n_inferred = np.asarray(feat_raw_for_n).shape[0]
                else:
                    n_inferred = int(max(u.max(), v.max()) + 1)
                vals = np.ones(u.shape[0], dtype=np.float32)
                adj_raw = sp.csr_matrix((vals, (u, v)), shape=(n_inferred, n_inferred))
            else:
                raise KeyError(f"Adjacency key not found in {self.data_path} (keys: {list(data.keys())})")
        feat_raw = self._pick(data, FEATURE_KEYS)
        if feat_raw is None:
            raise KeyError(f"Feature key not found in {self.data_path}")
        label_raw = self._pick(data, LABEL_KEYS)

        self.adj = sp.csr_matrix(adj_raw)
        if self.adj.shape[0] != self.adj.shape[1]:
            raise ValueError(f"Adjacency not square: {self.adj.shape}")
        self.adj.eliminate_zeros()
        self.adj = self.adj.maximum(self.adj.T)

        if sp.issparse(feat_raw):
            self.features = sp.csr_matrix(feat_raw)
        else:
            self.features = sp.csr_matrix(np.asarray(feat_raw))

        n = self.adj.shape[0]
        if label_raw is not None:
            labels = np.asarray(label_raw)
            if sp.issparse(labels):
                labels = labels.toarray()
            labels = labels.flatten().astype(np.int64)
            if labels.shape[0] != n:
                labels = labels[:n] if labels.shape[0] > n else np.concatenate(
                    [labels, np.zeros(n - labels.shape[0], dtype=np.int64)]
                )
            self.labels = labels
        else:
            self.labels = np.zeros(n, dtype=np.int64)

        return self

    def get_edge_list(self):
        coo = self.adj.tocoo()
        mask = coo.row < coo.col
        return list(zip(coo.row[mask].tolist(), coo.col[mask].tolist()))

    def get_stats(self):
        return {
            "dataset": self.dataset_name,
            "num_nodes": int(self.adj.shape[0]),
            "num_edges": int(self.adj.nnz // 2),
            "num_features": int(self.features.shape[1]),
            "num_anomalies": int(np.sum(self.labels)),
            "anomaly_ratio": float(np.sum(self.labels)) / float(self.adj.shape[0]) if self.adj.shape[0] else 0.0,
        }
