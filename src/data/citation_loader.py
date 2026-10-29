import os
import pickle
import numpy as np
import scipy.sparse as sp


CITATION_ALIASES = {
    "Cora": "cora",
    "Citeseer": "citeseer",
    "Pubmed": "pubmed",
}


def _load_planetoid_raw(data_dir: str, name: str):
    name_low = CITATION_ALIASES.get(name, name.lower())
    names = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]
    paths = [os.path.join(data_dir, f"ind.{name_low}.{s}") for s in names]
    if not all(os.path.exists(p) for p in paths):
        return None

    def _read(path, is_index):
        if is_index:
            with open(path, "r") as f:
                return np.asarray([int(l.strip()) for l in f if l.strip()], dtype=np.int64)
        with open(path, "rb") as f:
            return pickle.load(f, encoding="latin1")

    objs = []
    for p, s in zip(paths, names):
        objs.append(_read(p, s == "test.index"))
    x, tx, allx, y, ty, ally, graph, test_idx = objs

    allx = sp.csr_matrix(allx)
    tx = sp.csr_matrix(tx)
    ally = np.asarray(ally)
    ty = np.asarray(ty)

    test_idx_range_full = np.arange(test_idx.min(), test_idx.max() + 1)
    tx_extended = sp.lil_matrix((len(test_idx_range_full), tx.shape[1]), dtype=tx.dtype)
    tx_extended[test_idx - test_idx.min(), :] = tx
    tx_extended = sp.csr_matrix(tx_extended)

    ty_extended = np.zeros((len(test_idx_range_full), ty.shape[1]), dtype=ty.dtype)
    ty_extended[test_idx - test_idx.min(), :] = ty

    features = sp.vstack([allx, tx_extended]).tolil()
    test_idx_sorted = np.sort(test_idx)
    features[test_idx, :] = features[test_idx_sorted, :]
    features = sp.csr_matrix(features)

    labels = np.vstack([ally, ty_extended])
    labels[test_idx, :] = labels[test_idx_sorted, :]
    y_labels = np.argmax(labels, axis=1)

    num_nodes = features.shape[0]
    rows, cols = [], []
    for src, dsts in graph.items():
        for dst in dsts:
            rows.append(src)
            cols.append(dst)
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj, features, y_labels.astype(np.int64)


def load_citation(data_dir: str, name: str):
    raw = _load_planetoid_raw(data_dir, name)
    if raw is not None:
        return raw
    path = os.path.join(data_dir, name, f"{name}.mat")
    if os.path.exists(path):
        import scipy.io as sio
        d = sio.loadmat(path)
        for ka in ("A", "Network", "adj"):
            if ka in d:
                adj = sp.csr_matrix(d[ka])
                break
        else:
            raise KeyError("adj not found")
        for kf in ("X", "Attributes", "features"):
            if kf in d:
                feat = d[kf]
                feat = sp.csr_matrix(feat) if not sp.issparse(feat) else feat.tocsr()
                break
        else:
            raise KeyError("features not found")
        for kl in ("gnd", "Label", "labels"):
            if kl in d:
                y = np.asarray(d[kl]).flatten().astype(np.int64)
                break
        else:
            y = np.zeros(adj.shape[0], dtype=np.int64)
        return adj, feat, y
    raise FileNotFoundError(f"No Planetoid or mat data for {name} in {data_dir}")
