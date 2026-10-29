import os
import pickle

import numpy as np
import scipy.sparse as sp


def _adj_from_neighbor_dict(neigh: dict, num_nodes: int) -> sp.csr_matrix:
    rows, cols = [], []
    for u, ns in neigh.items():
        ui = int(u)
        if ui >= num_nodes:
            continue
        for v in ns:
            vi = int(v)
            if vi >= num_nodes:
                continue
            rows.append(ui)
            cols.append(vi)
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def _to_numpy(x):
    if hasattr(x, "numpy"):
        try:
            return x.numpy()
        except Exception:
            return x.detach().cpu().numpy()
    return np.asarray(x)


def load_llm_graph_dataset(base_dir: str, prefix: str):
    cand_subdirs = [base_dir, os.path.join(base_dir, "my_data")]
    found = None
    for sub in cand_subdirs:
        if not os.path.isdir(sub):
            continue
        files = os.listdir(sub)
        adj_p = next((os.path.join(sub, f) for f in files if f.endswith("L1.pkl") and "Llama" not in f), None)
        feat_p = next(
            (os.path.join(sub, f) for f in files if f.endswith("real_feature.pkl") and "Llama" not in f and "norm" not in f),
            None,
        )
        if feat_p is None:
            feat_p = next((os.path.join(sub, f) for f in files if f.endswith("real_feature.pkl") and "Llama" not in f), None)
        cls_p = next((os.path.join(sub, f) for f in files if f.endswith("classification.pkl")), None)
        if adj_p and feat_p:
            found = (adj_p, feat_p, cls_p)
            break
    if not found:
        raise FileNotFoundError(f"No LLM-graph format files in {base_dir}")
    adj_p, feat_p, cls_p = found

    with open(adj_p, "rb") as f:
        neigh = pickle.load(f)
    with open(feat_p, "rb") as f:
        feat_obj = pickle.load(f)
    cls_list = None
    if cls_p:
        with open(cls_p, "rb") as f:
            cls_list = pickle.load(f)

    feat = _to_numpy(feat_obj).astype(np.float32)
    num_nodes = max(int(max(neigh.keys())) + 1, feat.shape[0])
    if feat.shape[0] < num_nodes:
        pad = np.zeros((num_nodes - feat.shape[0], feat.shape[1]), dtype=np.float32)
        feat = np.vstack([feat, pad])
    feat = feat[:num_nodes]

    adj = _adj_from_neighbor_dict(neigh, num_nodes)

    labels = np.zeros(num_nodes, dtype=np.int64)
    if cls_list is not None:
        cls_arr = np.asarray(cls_list).flatten()[:num_nodes]
        labels[: cls_arr.shape[0]] = 0

    return adj, sp.csr_matrix(feat), labels


def load_arxiv(data_dir: str):
    base = os.path.join(data_dir, "Arxiv", "Arxiv")
    return load_llm_graph_dataset(base, prefix="arxiv")


def load_cora_llm(data_dir: str):
    base = os.path.join(data_dir, "Cora", "Cora")
    return load_llm_graph_dataset(base, prefix="cora")


def load_pubmed_llm(data_dir: str):
    base = os.path.join(data_dir, "Pubmed")
    return load_llm_graph_dataset(base, prefix="pub")


def load_citeseer_raw(data_dir: str):
    base = os.path.join(data_dir, "Citeseer", "citeseer")
    cites = os.path.join(base, "citeseer.cites")
    content = os.path.join(base, "citeseer.content")
    if not (os.path.exists(cites) and os.path.exists(content)):
        raise FileNotFoundError(f"Missing citeseer raw files at {base}")
    rows_feat, ids = [], {}
    feature_rows = []
    labels_list = []
    with open(content, "r", encoding="latin1") as f:
        for line in f:
            tok = line.strip().split()
            if len(tok) < 3:
                continue
            node_id = tok[0]
            feats = [float(x) for x in tok[1:-1]]
            label = tok[-1]
            ids[node_id] = len(ids)
            feature_rows.append(feats)
            labels_list.append(label)
    n = len(ids)
    feat = np.asarray(feature_rows, dtype=np.float32)
    rows, cols = [], []
    with open(cites, "r", encoding="latin1") as f:
        for line in f:
            tok = line.strip().split()
            if len(tok) < 2:
                continue
            a, b = tok[0], tok[1]
            if a in ids and b in ids:
                rows.append(ids[a])
                cols.append(ids[b])
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    labels = np.zeros(n, dtype=np.int64)
    return A, sp.csr_matrix(feat), labels


def load_cora_raw(data_dir: str):
    base = os.path.join(data_dir, "cora")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Missing cora dir {base}")
    cites = os.path.join(base, "cora.cites")
    content = os.path.join(base, "cora.content")
    if not (os.path.exists(cites) and os.path.exists(content)):
        raise FileNotFoundError(f"Missing cora raw files at {base}")
    ids = {}
    feature_rows = []
    with open(content, "r", encoding="latin1") as f:
        for line in f:
            tok = line.strip().split()
            if len(tok) < 3:
                continue
            ids[tok[0]] = len(ids)
            feature_rows.append([float(x) for x in tok[1:-1]])
    n = len(ids)
    feat = np.asarray(feature_rows, dtype=np.float32)
    rows, cols = [], []
    with open(cites, "r", encoding="latin1") as f:
        for line in f:
            tok = line.strip().split()
            if len(tok) < 2:
                continue
            a, b = tok[0], tok[1]
            if a in ids and b in ids:
                rows.append(ids[a])
                cols.append(ids[b])
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    return A, sp.csr_matrix(feat), np.zeros(n, dtype=np.int64)
