import os
import numpy as np
import scipy.sparse as sp


def _resolve_gpu_id(config):
    explicit = os.environ.get("SAG_DEVICE")
    cfg_dev = explicit if explicit else str(getattr(config, "device", "cpu") or "cpu")
    if cfg_dev.startswith("cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                if ":" in cfg_dev:
                    try:
                        return int(cfg_dev.split(":")[1])
                    except Exception:
                        return 0
                return 0
        except Exception:
            pass
    return -1


def _to_pyg_data(adj, features, labels):
    import torch
    from torch_geometric.data import Data
    try:
        from torch_geometric.utils import add_self_loops as _add_self_loops
    except Exception:
        _add_self_loops = None

    A = sp.csr_matrix(adj).tocoo()
    edge_index = torch.from_numpy(np.vstack([A.row, A.col]).astype(np.int64))
    X = features.toarray() if sp.issparse(features) else np.asarray(features)
    n = X.shape[0]
    if _add_self_loops is not None:
        edge_index, _ = _add_self_loops(edge_index, num_nodes=n)
    x = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(np.asarray(labels).astype(np.int64))
    return Data(x=x, edge_index=edge_index, y=y)


class _PyGODBaselineBase:
    pygod_cls = None
    name = None
    default_kwargs = {}

    def __init__(self, num_nodes, config, features=None, seed: int = 42, **kwargs):
        self.num_nodes = num_nodes
        self.config = config
        self.features = features
        self.seed = seed
        self.kwargs = dict(self.default_kwargs)
        self.kwargs.update(kwargs)
        self.scores_ = None

    def fit(self, adj, edges, labels=None):
        import torch
        import gc

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        data = _to_pyg_data(adj, self.features, labels if labels is not None else np.zeros(self.num_nodes))
        gpu = _resolve_gpu_id(self.config)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        hid = min(32, int(self.kwargs.get("hid_dim", 64)))
        big = int(self.num_nodes) > 200000
        if big:
            # Full-batch structure reconstruction is O(N^2) and hopeless at this
            # scale; go straight to mini-batch (NeighborLoader). Try GPU first,
            # then CPU (this box has ~754 GB RAM). Env SAG_STATIC_CPU forces CPU.
            import os as _os
            force_cpu = _os.environ.get("SAG_STATIC_CPU", "") not in ("", "0", "false")
            bs = int(_os.environ.get("SAG_STATIC_BATCH", "4096"))
            attempts = []
            if not force_cpu:
                attempts.append(("big_gpu_batch",
                                 {**self.kwargs, "batch_size": bs, "hid_dim": hid, "num_neigh": 10}, gpu))
            attempts += [
                ("big_cpu_batch",
                 {**self.kwargs, "batch_size": bs, "hid_dim": hid, "num_neigh": 10}, -1),
                ("big_cpu_batch_small",
                 {**self.kwargs, "batch_size": 1024, "hid_dim": hid, "num_neigh": 5}, -1),
            ]
        else:
            attempts = [
                ("default", dict(self.kwargs), gpu),
                ("retry_smaller_hid", {**self.kwargs, "hid_dim": hid}, gpu),
                ("retry_smaller_batch", {**self.kwargs, "batch_size": 1024, "hid_dim": hid}, gpu),
                ("retry_cpu_batch", {**self.kwargs, "batch_size": 1024, "hid_dim": hid}, -1),
                ("retry_cpu", dict(self.kwargs), -1),
            ]
        last_err = None
        for tag, kw, g in attempts:
            try:
                kw = dict(kw)
                kw.setdefault("gpu", g)
                if g != -1 and not (kw.get("gpu") == g):
                    kw["gpu"] = g
                if g == -1:
                    kw["gpu"] = -1
                det = None
                for drop in ([], ["gpu"], ["gpu", "num_neigh"],
                             ["gpu", "num_neigh", "batch_size"]):
                    try:
                        det = self.pygod_cls(**{k: v for k, v in kw.items() if k not in drop})
                        break
                    except TypeError:
                        continue
                if det is None:
                    det = self.pygod_cls(**{k: v for k, v in kw.items()
                                            if k not in ("gpu", "num_neigh", "batch_size", "hid_dim")})
                det.fit(data)
                scores = det.decision_score_
                if hasattr(scores, "detach"):
                    scores = scores.detach().cpu().numpy()
                self.scores_ = np.asarray(scores).flatten()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                return
            except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as e:
                msg = str(e).lower()
                oom = ("out of memory" in msg or "oom" in msg or "cuda" in msg
                       or "cudnn" in msg or "can't allocate" in msg
                       or "cannot allocate" in msg or isinstance(e, MemoryError))
                if not oom:
                    raise
                last_err = e
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                continue
        raise last_err if last_err is not None else RuntimeError("PyGOD fit failed")

    def score(self) -> np.ndarray:
        s = np.asarray(self.scores_, dtype=np.float64)
        finite = np.isfinite(s)
        if not finite.any():
            return np.zeros_like(s)
        med = float(np.median(s[finite]))
        s = np.where(finite, s, med)
        if s.max() > s.min():
            return (s - s.min()) / (s.max() - s.min())
        return np.zeros_like(s)


def _make(cls_name, name, **defaults):
    def _factory():
        from pygod.detector import (
            DOMINANT, AnomalyDAE, CoLA, CONAD, GUIDE, GAE, GAAN, ANOMALOUS,
            AdONE, DONE, OCGNN, Radar, ONE, SCAN, DMGD, GADNR,
        )

        mapping = {
            "DOMINANT": DOMINANT,
            "AnomalyDAE": AnomalyDAE,
            "CoLA": CoLA,
            "CONAD": CONAD,
            "GUIDE": GUIDE,
            "GAE": GAE,
            "GAAN": GAAN,
            "ANOMALOUS": ANOMALOUS,
            "AdONE": AdONE,
            "DONE": DONE,
            "OCGNN": OCGNN,
            "Radar": Radar,
            "ONE": ONE,
            "SCAN": SCAN,
            "DMGD": DMGD,
            "GADNR": GADNR,
        }
        return mapping[cls_name]

    cls = type(
        f"Official{cls_name}",
        (_PyGODBaselineBase,),
        {"name": name, "default_kwargs": defaults},
    )

    def _init(self, num_nodes, config, features=None, seed=42, **kwargs):
        _PyGODBaselineBase.__init__(self, num_nodes, config, features, seed, **kwargs)
        self.pygod_cls = _factory()

    cls.__init__ = _init
    return cls


OfficialDOMINANT = _make("DOMINANT", "DOMINANT", epoch=20, hid_dim=64, num_layers=2, verbose=0)
OfficialAnomalyDAE = _make("AnomalyDAE", "AnomalyDAE", epoch=20, hid_dim=64, verbose=0)
OfficialCoLA = _make("CoLA", "CoLA", epoch=20, hid_dim=64, verbose=0)
OfficialCONAD = _make("CONAD", "CONAD", epoch=20, hid_dim=64, verbose=0)
OfficialGUIDE = _make("GUIDE", "GUIDE", epoch=20, verbose=0)
OfficialGAE = _make("GAE", "GAE", epoch=20, hid_dim=64, verbose=0)
OfficialGAAN = _make("GAAN", "GAAN", epoch=20, hid_dim=64, verbose=0)
OfficialANOMALOUS = _make("ANOMALOUS", "ANOMALOUS")
OfficialAdONE = _make("AdONE", "AdONE", epoch=20, hid_dim=64, verbose=0)
OfficialDONE = _make("DONE", "DONE", epoch=20, hid_dim=64, verbose=0)
OfficialOCGNN = _make("OCGNN", "OCGNN", epoch=20, hid_dim=64, verbose=0)
OfficialRadar = _make("Radar", "Radar")
OfficialONE = _make("ONE", "ONE")
OfficialSCAN = _make("SCAN", "SCAN")
OfficialDMGD = _make("DMGD", "DMGD", epoch=20, hid_dim=64, verbose=0)
OfficialGADNR = _make("GADNR", "GADNR", epoch=20, hid_dim=64, verbose=0)


class _PyGGraphSAGE:
    name = "GraphSAGE"

    def __init__(self, num_nodes, config, features=None, seed: int = 42, hidden_dim: int = 64, num_layers: int = 2):
        self.num_nodes = num_nodes
        self.config = config
        self.features = features
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.scores_ = None

    def fit(self, adj, edges, labels=None):
        import torch
        from torch_geometric.nn import SAGEConv

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        gpu = _resolve_gpu_id(self.config)
        device = torch.device(f"cuda:{gpu}" if gpu >= 0 else "cpu")
        data = _to_pyg_data(adj, self.features, labels if labels is not None else np.zeros(self.num_nodes))
        data = data.to(device)
        in_dim = data.x.shape[1]
        layers = []
        prev = in_dim
        for _ in range(self.num_layers):
            layers.append(SAGEConv(prev, self.hidden_dim).to(device))
            prev = self.hidden_dim
        h = data.x
        with torch.no_grad():
            for layer in layers:
                h = torch.relu(layer(h, data.edge_index))
        H = h.detach().cpu().numpy()
        H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-6)
        mu = H.mean(axis=0, keepdims=True)
        self.scores_ = np.linalg.norm(H - mu, axis=1)

    def score(self) -> np.ndarray:
        s = np.asarray(self.scores_, dtype=np.float64)
        finite = np.isfinite(s)
        if not finite.any():
            return np.zeros_like(s)
        med = float(np.median(s[finite]))
        s = np.where(finite, s, med)
        if s.max() > s.min():
            return (s - s.min()) / (s.max() - s.min())
        return np.zeros_like(s)


class _PyGHGT:
    name = "HGT"

    def __init__(self, num_nodes, config, features=None, seed: int = 42, hidden_dim: int = 64, num_heads: int = 2):
        self.num_nodes = num_nodes
        self.config = config
        self.features = features
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.scores_ = None

    def fit(self, adj, edges, labels=None):
        import torch
        from torch_geometric.nn import GATConv

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        gpu = _resolve_gpu_id(self.config)
        device = torch.device(f"cuda:{gpu}" if gpu >= 0 else "cpu")
        data = _to_pyg_data(adj, self.features, labels if labels is not None else np.zeros(self.num_nodes))
        data = data.to(device)
        layer = GATConv(data.x.shape[1], self.hidden_dim, heads=self.num_heads, concat=False).to(device)
        with torch.no_grad():
            h = torch.relu(layer(data.x, data.edge_index))
        H = h.detach().cpu().numpy()
        H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-6)
        mu = H.mean(axis=0, keepdims=True)
        self.scores_ = np.linalg.norm(H - mu, axis=1)

    def score(self) -> np.ndarray:
        s = np.asarray(self.scores_, dtype=np.float64)
        finite = np.isfinite(s)
        if not finite.any():
            return np.zeros_like(s)
        med = float(np.median(s[finite]))
        s = np.where(finite, s, med)
        if s.max() > s.min():
            return (s - s.min()) / (s.max() - s.min())
        return np.zeros_like(s)


OFFICIAL_BASELINES = {
    "DOMINANT": OfficialDOMINANT,
    "AnomalyDAE": OfficialAnomalyDAE,
    "CoLA": OfficialCoLA,
    "CONAD": OfficialCONAD,
    "GUIDE": OfficialGUIDE,
    "GAE": OfficialGAE,
    "GAAN": OfficialGAAN,
    "ANOMALOUS": OfficialANOMALOUS,
    "AdONE": OfficialAdONE,
    "DONE": OfficialDONE,
    "OCGNN": OfficialOCGNN,
    "Radar": OfficialRadar,
    "ONE": OfficialONE,
    "SCAN": OfficialSCAN,
    "DMGD": OfficialDMGD,
    "GADNR": OfficialGADNR,
    "GraphSAGE": _PyGGraphSAGE,
    "HGT": _PyGHGT,
}
