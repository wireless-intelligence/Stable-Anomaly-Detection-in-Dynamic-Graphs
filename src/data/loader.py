import os
import numpy as np
import scipy.sparse as sp

from .mat_loader import MatDataLoader
from .anomaly_injection import AnomalyInjector
from .citation_loader import load_citation
from .llm_graph_loader import (
    load_arxiv,
    load_cora_llm,
    load_cora_raw,
    load_citeseer_raw,
    load_pubmed_llm,
)


INJECTED_DATASETS = {"BlogCatalog", "Flickr", "ACM", "Cora", "Citeseer", "Pubmed", "Arxiv"}
GROUNDTRUTH_DATASETS = {"Amazon", "YelpChi", "YelpReviews", "Enron", "CTU-13"}


class DataLoader:
    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        inject_anomaly: bool = None,
        clique_size: int = 15,
        num_cliques: int = 10,
        seed: int = 42,
        k_attr: int = 50,
    ):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.clique_size = clique_size
        self.num_cliques = num_cliques
        self.k_attr = k_attr
        self.seed = seed
        if inject_anomaly is None:
            inject_anomaly = dataset_name in INJECTED_DATASETS
        self.inject_anomaly = inject_anomaly

        self.adj = None
        self.features = None
        self.labels = None
        self.stats = None
        self.raw_data = None

    def _resolve_mat(self, *cands):
        for p in cands:
            if p and os.path.exists(p):
                return p
        return None

    def _load_with_fallbacks(self):
        name = self.dataset_name
        d = self.data_dir
        if name == "Cora":
            try:
                a, f, l = load_citation(d, "Cora")
                return a, f, np.zeros(a.shape[0], dtype=np.int64)
            except FileNotFoundError:
                pass
            try:
                return load_cora_raw(d)
            except FileNotFoundError:
                pass
            return load_cora_llm(d)
        if name == "Citeseer":
            try:
                a, f, l = load_citation(d, "Citeseer")
                return a, f, np.zeros(a.shape[0], dtype=np.int64)
            except FileNotFoundError:
                pass
            return load_citeseer_raw(d)
        if name == "Pubmed":
            try:
                a, f, l = load_citation(d, "Pubmed")
                return a, f, np.zeros(a.shape[0], dtype=np.int64)
            except FileNotFoundError:
                pass
            return load_pubmed_llm(d)
        if name == "Arxiv":
            return load_arxiv(d)
        if name == "Enron":
            mat = self._resolve_mat(
                os.path.join(d, "Enron", "Enron.mat"),
                os.path.join(d, "Enron", "enron.mat"),
                os.path.join(d, "Enron.mat"),
            )
            if mat:
                ld = MatDataLoader(d, "Enron", mat_path=mat)
                ld.load()
                return ld.adj, ld.features, ld.labels
            txt = self._resolve_mat(
                os.path.join(d, "Enron", "email-Enron.txt"),
                os.path.join(d, "email-Enron.txt"),
            )
            if txt:
                return _build_graph_from_edgelist(txt, d, "Enron")
            raise FileNotFoundError(f"Enron data missing in {d}/Enron/")
        if name == "CTU-13":
            mat = self._resolve_mat(
                os.path.join(d, "CTU-13", "CTU-13.mat"),
                os.path.join(d, "CTU13", "CTU13.mat"),
                os.path.join(d, "CTU-13", "ctu13.mat"),
                os.path.join(d, "CTU-13", "ctu-13.mat"),
            )
            if mat:
                import scipy.io as _sio
                raw = _sio.loadmat(mat)
                if "starttime" in raw:
                    self.node_times = np.asarray(raw["starttime"]).flatten().astype(np.float64)
                ld = MatDataLoader(d, "CTU-13", mat_path=mat)
                ld.load()
                return ld.adj, ld.features, ld.labels
            return _ctu13_from_directory(d)
        if name == "BlogCatalog":
            mat = self._resolve_mat(
                os.path.join(d, "BlogCatalog", "BlogCatalog.mat"),
                os.path.join(d, "BlogCatalog.mat"),
            )
            if mat:
                ld = MatDataLoader(d, "BlogCatalog", mat_path=mat)
                ld.load()
                return ld.adj, ld.features, ld.labels
            edge = self._resolve_mat(
                os.path.join(d, "BlogCatalog", "BlogCatalog.edgelist"),
                os.path.join(d, "BlogCatalog.edgelist"),
            )
            if edge:
                return _build_graph_from_edgelist(edge, d, "BlogCatalog", delim=" ")
            raise FileNotFoundError(f"BlogCatalog data missing in {d}")
        if name == "YelpChi":
            mat = self._resolve_mat(
                os.path.join(d, "yelp", "YelpChi.mat"),
                os.path.join(d, "YelpChi", "YelpChi.mat"),
                os.path.join(d, "YelpChi.mat"),
            )
            if mat:
                self._stash_multirelation(mat, ["net_rur", "net_rtr", "net_rsr"])
                ld = MatDataLoader(d, name, mat_path=mat)
                ld.load()
                return ld.adj, ld.features, ld.labels
            raise FileNotFoundError(f"YelpChi data missing in {d}")
        if name == "YelpReviews":
            mat = self._resolve_mat(
                os.path.join(d, "yelp", "yelp_review_graph.mat"),
                os.path.join(d, "YelpReviews", "yelp_review_graph.mat"),
                os.path.join(d, "yelp_review_graph.mat"),
            )
            if mat:
                ld = MatDataLoader(d, name, mat_path=mat)
                ld.load()
                return ld.adj, ld.features, ld.labels
            raise FileNotFoundError(f"YelpReviews data missing in {d}")
        if name == "Amazon":
            mat = self._resolve_mat(
                os.path.join(d, "Amazon", "Amazon.mat"),
                os.path.join(d, "Amazon.mat"),
            )
            if mat:
                self._stash_multirelation(mat, ["net_upu", "net_usu", "net_uvu"])
                ld = MatDataLoader(d, "Amazon", mat_path=mat)
                ld.load()
                return ld.adj, ld.features, ld.labels
            raise FileNotFoundError(f"Amazon data missing in {d}")
        mat = self._resolve_mat(
            os.path.join(d, name, f"{name}.mat"),
            os.path.join(d, name.lower(), f"{name}.mat"),
            os.path.join(d, f"{name}.mat"),
        )
        if mat:
            ld = MatDataLoader(d, name, mat_path=mat)
            ld.load()
            return ld.adj, ld.features, ld.labels
        raise FileNotFoundError(f"No data for {name} in {d}")

    def load(self):
        adj, features, labels = self._load_with_fallbacks()

        if self.inject_anomaly and int(np.sum(labels)) == 0:
            inj = AnomalyInjector(
                adj=adj,
                features=features,
                clique_size=self.clique_size,
                num_cliques=self.num_cliques,
                k_attr=self.k_attr,
                seed=self.seed,
            )
            inj.inject_all()
            data = inj.get_data()
            adj, features, labels = data["adj"], data["features"], data["labels"]

        adj = sp.csr_matrix(adj)
        adj.eliminate_zeros()
        adj = adj.maximum(adj.T)
        if not sp.issparse(features):
            features = sp.csr_matrix(features)

        self.adj = adj
        self.features = features
        self.labels = labels
        self.raw_data = {"adj": adj, "features": features, "labels": labels}
        self.stats = {
            "dataset": self.dataset_name,
            "num_nodes": int(adj.shape[0]),
            "num_edges": int(adj.nnz // 2),
            "num_features": int(features.shape[1]),
            "num_anomalies": int(np.sum(labels)),
            "anomaly_ratio": float(np.sum(labels)) / float(adj.shape[0]) if adj.shape[0] else 0.0,
        }
        return self

    def _stash_multirelation(self, mat_path: str, keys):
        import scipy.io as _sio
        raw = _sio.loadmat(mat_path)
        rels = []
        for k in keys:
            if k in raw:
                A = sp.csr_matrix(raw[k]).astype(np.float32)
                A.eliminate_zeros()
                A = A.maximum(A.T)
                A.setdiag(0)
                A.eliminate_zeros()
                rels.append((k, A))
        if rels:
            self.relation_adjs = rels

    def get_edge_times(self, edges: np.ndarray):
        node_times = getattr(self, "node_times", None)
        if node_times is None or edges is None or edges.shape[0] == 0:
            return None
        node_times = np.asarray(node_times, dtype=np.float64)
        u = edges[:, 0]
        v = edges[:, 1]
        return np.minimum(node_times[u], node_times[v])

    def get_edge_stream(self, shuffle: bool = True, mode: str = None):
        if mode is None:
            mode = "shuffle" if shuffle else "raw"
        if mode == "raw":
            coo = self.adj.tocoo()
            mask = coo.row < coo.col
            return np.stack([coo.row[mask], coo.col[mask]], axis=1).astype(np.int64)
        if mode == "multi_relation":
            rels = getattr(self, "relation_adjs", None)
            if not rels:
                mode = "shuffle"
            else:
                from .edge_ordering import multi_relation_stream
                return multi_relation_stream(rels, seed=self.seed)
        from .edge_ordering import get_edge_stream as _get
        kw = {}
        if mode == "temporal":
            kw["node_times"] = getattr(self, "node_times", None)
            if kw["node_times"] is None:
                mode = "shuffle"
                kw = {}
        return _get(self.adj, mode=mode, seed=self.seed, **kw)

    def get_stats(self):
        return self.stats


def _build_graph_from_edgelist(path: str, data_dir: str, name: str, delim: str = None):
    edges = []
    nodes = set()
    with open(path, "r", encoding="latin1") as f:
        for line in f:
            ln = line.strip()
            if not ln or ln.startswith("#") or ln.startswith("%"):
                continue
            tok = ln.split(delim) if delim else ln.split()
            if len(tok) < 2:
                continue
            try:
                a = int(tok[0])
                b = int(tok[1])
            except ValueError:
                continue
            if a == b:
                continue
            edges.append((a, b))
            nodes.add(a)
            nodes.add(b)
    nodes = sorted(nodes)
    remap = {nid: i for i, nid in enumerate(nodes)}
    n = len(nodes)
    rows = np.fromiter((remap[a] for a, _ in edges), dtype=np.int64, count=len(edges))
    cols = np.fromiter((remap[b] for _, b in edges), dtype=np.int64, count=len(edges))
    data = np.ones(rows.shape[0], dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    deg = np.asarray(A.sum(axis=1)).flatten()
    log_deg = np.log1p(deg)
    feats = np.stack(
        [
            deg, log_deg, np.sqrt(deg),
            deg / (deg.max() + 1e-6),
            log_deg - log_deg.mean(),
            np.clip(deg - deg.mean(), 0, None),
            np.sign(deg - deg.mean()) * log_deg,
            np.ones_like(deg),
        ],
        axis=1,
    ).astype(np.float32)
    return A, sp.csr_matrix(feats), np.zeros(n, dtype=np.int64)


def _ctu13_from_directory(data_dir: str):
    base = os.path.join(data_dir, "CTU-13")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"CTU-13 missing at {base}")
    flow_files = []
    for root, _, files in os.walk(base):
        for fn in files:
            if fn.endswith(".binetflow") or fn.endswith(".biargus") or fn.endswith(".csv"):
                flow_files.append(os.path.join(root, fn))
    if not flow_files:
        raise FileNotFoundError(f"No flow files (.binetflow/.csv) in {base}")
    return _parse_ctu13_flows(flow_files)


def _parse_ctu13_flows(flow_files):
    import csv as _csv
    nodes = {}
    edges = []
    labels_set = set()
    for path in flow_files:
        with open(path, "r", encoding="latin1") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                src = row.get("SrcAddr") or row.get("srcaddr")
                dst = row.get("DstAddr") or row.get("dstaddr")
                lab = (row.get("Label") or "").lower()
                if not src or not dst:
                    continue
                if src not in nodes:
                    nodes[src] = len(nodes)
                if dst not in nodes:
                    nodes[dst] = len(nodes)
                u = nodes[src]
                v = nodes[dst]
                edges.append((u, v))
                if "botnet" in lab or "from-botnet" in lab or "cc" in lab or "command" in lab:
                    labels_set.add(u)
                    labels_set.add(v)
    n = len(nodes)
    rows = np.fromiter((u for u, _ in edges), dtype=np.int64, count=len(edges))
    cols = np.fromiter((v for _, v in edges), dtype=np.int64, count=len(edges))
    data = np.ones(rows.shape[0], dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    deg = np.asarray(A.sum(axis=1)).flatten()
    log_deg = np.log1p(deg)
    feats = np.stack(
        [
            deg, log_deg, np.sqrt(deg),
            deg / (deg.max() + 1e-6),
            log_deg - log_deg.mean(),
            np.clip(deg - deg.mean(), 0, None),
            np.sign(deg - deg.mean()) * log_deg,
            np.ones_like(deg),
        ],
        axis=1,
    ).astype(np.float32)
    labels = np.zeros(n, dtype=np.int64)
    for nid in labels_set:
        labels[nid] = 1
    return A, sp.csr_matrix(feats), labels
