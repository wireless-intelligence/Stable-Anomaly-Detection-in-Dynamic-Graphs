import numpy as np
import scipy.sparse as sp


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    return (d_mat @ adj_ @ d_mat).tocsr()


def normalize_features(features):
    if sp.issparse(features):
        features = features.toarray()
    rowsum = np.array(features.sum(1)).flatten()
    r_inv = np.power(rowsum, -1)
    r_inv[np.isinf(r_inv)] = 0.0
    return features * r_inv[:, None]


def sparse_mx_to_torch_sparse_tensor(mx):
    import torch
    mx = mx.tocoo().astype(np.float32)
    indices = np.vstack([mx.row, mx.col]).astype(np.int64)
    values = mx.data
    shape = mx.shape
    return torch.sparse_coo_tensor(torch.from_numpy(indices), torch.from_numpy(values), torch.Size(shape))


class GraphPreprocessor:
    def __init__(self, adj, features, labels):
        self.adj = adj
        self.features = features
        self.labels = labels

    def process(self):
        adj_norm = normalize_adj(self.adj)
        feat = normalize_features(self.features)
        degree = np.array(self.adj.sum(axis=1)).flatten()
        return {
            "adj_norm": adj_norm,
            "features": feat,
            "labels": np.asarray(self.labels).astype(np.int64),
            "degree": degree,
            "num_nodes": int(self.adj.shape[0]),
        }
