"""OFFICIAL temporal-graph baselines (the canonical paper baselines).

These wrap the **official** implementations from DyGLib
(https://github.com/yule-BUAA/DyGLib), the standard unified library that ships
the original authors' JODIE / DyRep / TGAT / TGN (and CAWN / TCL / GraphMixer /
DyGFormer) code. We do not reimplement the models: we write the temporal graph
into DyGLib's processed format, then run a subprocess that imports DyGLib's own
``MemoryModel`` / ``TGAT`` classes, neighbor sampler, negative sampler and
``MergeLayer`` link predictor, and runs DyGLib's official self-supervised
temporal link-prediction training loop. The per-node anomaly score is the mean
link-prediction surprise (-log p of the node's true incident edges, computed
online before memory updates), matching SAD/SLADE.

This mirrors the ``_SubprocessVendorBase`` convention already used for ComGA /
DySAT / Graphormer (repos with their own pipeline): the model code is official,
only the thin driver around it is ours.

Requires ``vendor/DyGLib/`` to be a clone of the official repo (see
``setup_vendor_baselines.sh``). Without it, ``fit_temporal`` raises a clear
error and the runner reports the baseline as unavailable -- it never fabricates
scores. For a torch-only fallback that runs without the clone, use the
``*-reimpl`` variants in ``dynamic.py``.

Interface:
    m = TGNOfficial(num_nodes, config, features=..., seed=...)
    m.fit_temporal(edges, edge_times, labels=None)
    scores = m.score()   # (num_nodes,) higher = anomalous
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import scipy.sparse as sp


_VENDOR_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "vendor",
)


def _vendor_path(name):
    return os.path.join(_VENDOR_ROOT, name)


def build_dyglib_arrays(edges, edge_times, features, num_nodes):
    """Map a (0-based) temporal graph to DyGLib's processed convention.

    Reference for what the subprocess runner writes; also used by
    ``scripts/check_dyglib_pipeline.py`` to prove equivalence with DyGLib's
    official ``preprocess_data.py``. For NON-bipartite graphs DyGLib does
    ``u += 1; i += 1`` (1-based, no item offset), prepends a zero padding row to
    edge features, and sizes node features as ``(max_idx + 1, dim)``.
    Returns the arrays DyGLib's models consume.
    """
    edges = np.ascontiguousarray(edges, dtype=np.int64).reshape(-1, 2)
    times = np.ascontiguousarray(edge_times, dtype=np.float64).reshape(-1)
    order = np.argsort(times, kind="mergesort")   # DyGLib asserts ascending ts
    edges, times = edges[order], times[order]
    E = edges.shape[0]

    X = features
    X = X.toarray() if sp.issparse(X) else (np.asarray(X) if X is not None
                                            else np.ones((num_nodes, 1), np.float32))
    X = X.astype(np.float32)

    src = (edges[:, 0] + 1).astype(np.longlong)          # 1-based
    dst = (edges[:, 1] + 1).astype(np.longlong)
    node_interact_times = times.astype(np.float64)
    edge_ids = np.arange(1, E + 1, dtype=np.longlong)    # 1-based, idx col
    labels = np.zeros(E, dtype=np.float64)

    node_raw_features = np.zeros((num_nodes + 1, X.shape[1]), dtype=np.float32)
    node_raw_features[1:] = X                            # row 0 = padding
    edge_raw_features = np.zeros((E + 1, 1), dtype=np.float32)  # row 0 = padding
    return {
        "src_node_ids": src, "dst_node_ids": dst,
        "node_interact_times": node_interact_times,
        "edge_ids": edge_ids, "labels": labels,
        "node_raw_features": node_raw_features,
        "edge_raw_features": edge_raw_features,
    }


class _DyGLibOfficialBase:
    """Subprocess wrapper around DyGLib's official temporal models."""

    name = "DyGLibOfficial"
    model_name = None          # 'JODIE' | 'DyRep' | 'TGN' | 'TGAT'
    vendor_name = "DyGLib"
    requirements_note = (
        "requires vendor/DyGLib/ (clone of https://github.com/yule-BUAA/DyGLib) "
        "plus torch + torch_geometric. Run setup_vendor_baselines.sh."
    )

    def __init__(self, num_nodes, config, features=None, seed=42,
                 num_epochs=None, num_neighbors=None, num_layers=1, num_heads=2,
                 dropout=0.1, time_feat_dim=100, batch_size=200, lr=1e-4,
                 timeout=86400, verbose=False, **kw):
        # DyGLib's neighbor sampler is pure-Python; keep the fan-out modest so
        # large graphs stay tractable (override via config.dyn_num_neighbors).
        if num_neighbors is None:
            num_neighbors = int(getattr(config, "dyn_num_neighbors", 10))
        self.num_nodes = int(num_nodes)
        self.config = config
        self.features = features
        self.seed = int(seed)
        self.num_epochs = int(num_epochs if num_epochs is not None
                              else getattr(config, "dyn_epochs", 5))
        self.num_neighbors = int(num_neighbors)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.time_feat_dim = int(time_feat_dim)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.timeout = int(timeout)
        self.verbose = bool(verbose)
        self.scores_ = None

    def _check_vendor(self):
        p = _vendor_path(self.vendor_name)
        if not os.path.isdir(p):
            raise FileNotFoundError(
                f"vendor/{self.vendor_name} not found at {p}. {self.requirements_note}"
            )
        # sanity: expect DyGLib's models/ package
        if not os.path.isdir(os.path.join(p, "models")):
            raise FileNotFoundError(
                f"{p} does not look like DyGLib (no models/ dir). {self.requirements_note}"
            )
        return p

    def _device_str(self):
        dev = str(getattr(self.config, "device", "cpu") or "cpu")
        if dev.startswith("cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    return dev
            except Exception:
                pass
        return "cpu"

    def fit_temporal(self, edges, edge_times, labels=None):
        vendor_dir = self._check_vendor()
        edges = np.ascontiguousarray(edges, dtype=np.int64).reshape(-1, 2)
        t = np.ascontiguousarray(edge_times, dtype=np.float64).reshape(-1)
        order = np.argsort(t, kind="mergesort")
        edges, t = edges[order], t[order]

        X = self.features
        X = X.toarray() if sp.issparse(X) else (np.asarray(X) if X is not None else None)
        if X is None:
            X = np.ones((self.num_nodes, 1), dtype=np.float32)
        X = X.astype(np.float32)

        with tempfile.TemporaryDirectory(prefix="sag_dyglib_") as tmp:
            npz = os.path.join(tmp, "graph.npz")
            np.savez_compressed(npz, edges=edges, times=t, X=X,
                                num_nodes=np.int64(self.num_nodes))
            runner = os.path.join(tmp, "_runner.py")
            with open(runner, "w", encoding="utf-8") as f:
                f.write(_DYGLIB_RUNNER.format(
                    vendor=vendor_dir.replace("\\", "/"),
                    npz=npz.replace("\\", "/"),
                    tmp=tmp.replace("\\", "/"),
                    model_name=self.model_name,
                    seed=self.seed,
                    epochs=self.num_epochs,
                    num_neighbors=self.num_neighbors,
                    num_layers=self.num_layers,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    time_feat_dim=self.time_feat_dim,
                    batch_size=self.batch_size,
                    lr=self.lr,
                    device=self._device_str(),
                ))
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            try:
                subprocess.run([sys.executable, runner], cwd=vendor_dir, env=env,
                               capture_output=True, timeout=self.timeout, check=True)
            except subprocess.CalledProcessError as e:
                tail = (e.stdout or b"").decode("utf-8", "replace")[-2500:]
                err = (e.stderr or b"").decode("utf-8", "replace")[-2500:]
                raise RuntimeError(
                    f"{self.model_name} (DyGLib) failed rc={e.returncode}. "
                    f"{self.requirements_note}\nSTDOUT:\n{tail}\nSTDERR:\n{err}") from e
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(f"{self.model_name} (DyGLib) timed out "
                                   f"after {self.timeout}s") from e
            out = os.path.join(tmp, "scores.npy")
            if not os.path.exists(out):
                raise RuntimeError(f"{self.model_name} (DyGLib) produced no scores.npy")
            self.scores_ = np.load(out).astype(np.float64)
        return self

    def score(self):
        if self.scores_ is None:
            raise RuntimeError(f"{self.model_name}.fit_temporal must run before score")
        s = self.scores_.astype(np.float64)
        if s.max() > s.min():
            return (s - s.min()) / (s.max() - s.min())
        return np.zeros_like(s)


class TGNOfficial(_DyGLibOfficialBase):
    name = "TGN"
    model_name = "TGN"


class JODIEOfficial(_DyGLibOfficialBase):
    name = "JODIE"
    model_name = "JODIE"


class DyRepOfficial(_DyGLibOfficialBase):
    name = "DyRep"
    model_name = "DyRep"


class TGATOfficial(_DyGLibOfficialBase):
    name = "TGAT"
    model_name = "TGAT"


# --------------------------------------------------------------------------- #
# The subprocess runner: imports DyGLib's OFFICIAL model + sampler + link
# predictor and reproduces its official self-supervised link-prediction loop,
# then extracts per-node surprise. Executed with cwd = vendor/DyGLib.
# --------------------------------------------------------------------------- #

_DYGLIB_RUNNER = r'''
import os, sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "{vendor}")

MODEL_NAME = "{model_name}"
SEED = {seed}
EPOCHS = {epochs}
NUM_NEIGHBORS = {num_neighbors}
NUM_LAYERS = {num_layers}
NUM_HEADS = {num_heads}
DROPOUT = {dropout}
TIME_FEAT_DIM = {time_feat_dim}
BATCH = {batch_size}
LR = {lr}
DEVICE = torch.device("{device}")

torch.manual_seed(SEED); np.random.seed(SEED)

from utils.DataLoader import Data
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from models.modules import MergeLayer
try:
    from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
except Exception:
    from models.MemoryModel import MemoryModel
    from utils.utils import compute_src_dst_node_time_shifts
from models.TGAT import TGAT

d = np.load("{npz}")
edges = d["edges"].astype(np.int64); times = d["times"].astype(np.float64)
X = d["X"].astype(np.float32); N = int(d["num_nodes"])
E = edges.shape[0]

# DyGLib convention: node ids are 1-based with a padded zero row at index 0;
# edge ids 1-based with a padded zero row in edge features.
src = (edges[:, 0] + 1).astype(np.longlong)
dst = (edges[:, 1] + 1).astype(np.longlong)
node_interact_times = times.astype(np.float64)
edge_ids = np.arange(1, E + 1, dtype=np.longlong)
edge_labels = np.zeros(E, dtype=np.float64)

node_raw_features = np.zeros((N + 1, X.shape[1]), dtype=np.float32)
node_raw_features[1:] = X
edge_raw_features = np.zeros((E + 1, 1), dtype=np.float32)  # no native edge feats


def _pad_to_multiple(mat, h):
    d = mat.shape[1]; p = (-d) % h
    if p == 0:
        return mat
    return np.concatenate([mat, np.zeros((mat.shape[0], p), dtype=mat.dtype)], axis=1)


# DyGLib's multi-head attention asserts (node_feat_dim + time_feat_dim) and
# (node+edge+time) are divisible by num_heads. Pad feature dims to a multiple of
# num_heads (time_feat_dim is kept a multiple too) so any raw feature count works.
if TIME_FEAT_DIM % NUM_HEADS != 0:
    TIME_FEAT_DIM = ((TIME_FEAT_DIM + NUM_HEADS - 1) // NUM_HEADS) * NUM_HEADS
node_raw_features = _pad_to_multiple(node_raw_features, NUM_HEADS)
edge_raw_features = _pad_to_multiple(edge_raw_features, NUM_HEADS)
node_feat_dim = node_raw_features.shape[1]

full_data = Data(src_node_ids=src, dst_node_ids=dst,
                 node_interact_times=node_interact_times,
                 edge_ids=edge_ids, labels=edge_labels)
# time-based train split (train the self-supervised model on the earlier stream)
split_t = np.quantile(node_interact_times, 0.85) if E > 5 else node_interact_times.max()
tr = node_interact_times <= split_t
train_data = Data(src_node_ids=src[tr], dst_node_ids=dst[tr],
                  node_interact_times=node_interact_times[tr],
                  edge_ids=edge_ids[tr], labels=edge_labels[tr])

train_ns = get_neighbor_sampler(train_data, sample_neighbor_strategy="recent", seed=SEED)
full_ns = get_neighbor_sampler(full_data, sample_neighbor_strategy="recent", seed=SEED + 1)
train_neg = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids,
                                dst_node_ids=train_data.dst_node_ids, seed=SEED)

is_memory = MODEL_NAME in ("TGN", "DyRep", "JODIE")
if is_memory:
    sm, ss, dm, ds = compute_src_dst_node_time_shifts(
        train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
    backbone = MemoryModel(
        node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
        neighbor_sampler=train_ns, time_feat_dim=TIME_FEAT_DIM, model_name=MODEL_NAME,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=DROPOUT,
        src_node_mean_time_shift=sm, src_node_std_time_shift=ss,
        dst_node_mean_time_shift_dst=dm, dst_node_std_time_shift=ds, device=DEVICE)
else:
    backbone = TGAT(
        node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
        neighbor_sampler=train_ns, time_feat_dim=TIME_FEAT_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=DROPOUT, device=DEVICE)

link = MergeLayer(input_dim1=node_feat_dim, input_dim2=node_feat_dim,
                  hidden_dim=node_feat_dim, output_dim=1)
model = nn.Sequential(backbone, link).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
crit = nn.BCELoss()


def embed(bk, s_ids, d_ids, t_arr, e_ids, positive):
    if is_memory:
        return bk.compute_src_dst_node_temporal_embeddings(
            src_node_ids=s_ids, dst_node_ids=d_ids, node_interact_times=t_arr,
            edge_ids=e_ids, edges_are_positive=positive, num_neighbors=NUM_NEIGHBORS)
    return bk.compute_src_dst_node_temporal_embeddings(
        src_node_ids=s_ids, dst_node_ids=d_ids, node_interact_times=t_arr,
        num_neighbors=NUM_NEIGHBORS)


def run_epoch(data, ns, train=True, record=None):
    # JODIE uses a time-projection embedding with NO neighbor sampler; DyGLib's
    # MemoryModel.set_neighbor_sampler asserts against JODIE, so skip it there.
    if MODEL_NAME != "JODIE":
        model[0].set_neighbor_sampler(ns)
    if is_memory:
        model[0].memory_bank.__init_memory_bank__()
    model.train(train)
    M = data.src_node_ids.shape[0]
    for s in range(0, M, BATCH):
        e = min(M, s + BATCH)
        b_src = data.src_node_ids[s:e]; b_dst = data.dst_node_ids[s:e]
        b_t = data.node_interact_times[s:e]; b_eid = data.edge_ids[s:e]
        _, neg_dst = train_neg.sample(size=len(b_src))
        neg_src = b_src
        p_src, p_dst = embed(model[0], b_src, b_dst, b_t, b_eid, True)
        pos_prob = model[1](input_1=p_src, input_2=p_dst).squeeze(dim=-1).sigmoid()
        if train:
            n_src, n_dst = embed(model[0], neg_src, neg_dst, b_t, b_eid, False)
            neg_prob = model[1](input_1=n_src, input_2=n_dst).squeeze(dim=-1).sigmoid()
            loss = crit(pos_prob, torch.ones_like(pos_prob)) + \
                   crit(neg_prob, torch.zeros_like(neg_prob))
            opt.zero_grad(); loss.backward(); opt.step()
            if is_memory:
                model[0].memory_bank.detach_memory_bank()
        if record is not None:
            with torch.no_grad():
                p = pos_prob.detach().clamp(1e-6, 1 - 1e-6).cpu().numpy()
            surp = -np.log(p)
            np.add.at(record[0], b_src - 1, surp)
            np.add.at(record[0], b_dst - 1, surp)
            np.add.at(record[1], b_src - 1, 1.0)
            np.add.at(record[1], b_dst - 1, 1.0)


for ep in range(max(1, EPOCHS)):
    run_epoch(train_data, train_ns, train=True)

# scoring pass: whole stream, chronological, online surprise before updates
surp_sum = np.zeros(N, dtype=np.float64); surp_cnt = np.zeros(N, dtype=np.float64)
with torch.no_grad():
    run_epoch(full_data, full_ns, train=False, record=(surp_sum, surp_cnt))

seen = surp_cnt > 0
scores = np.zeros(N, dtype=np.float64)
scores[seen] = surp_sum[seen] / surp_cnt[seen]
if seen.any():
    scores[~seen] = scores[seen].min()
np.save("{tmp}/scores.npy", scores)
print("DYGLIB_DONE", MODEL_NAME, int(seen.sum()))
'''


# --------------------------------------------------------------------------- #
# EvolveGCN -- official model from the maintained PyTorch Geometric Temporal
# library (torch_geometric_temporal.nn.recurrent.EvolveGCNO), the standard
# reference implementation of Pareja+2020. Discrete snapshots; the GCN weights
# evolve via a recurrent unit across time steps. Self-supervised link
# prediction per snapshot -> per-node surprise. In-process (no vendor clone);
# requires `pip install torch-geometric-temporal`.
# --------------------------------------------------------------------------- #

class EvolveGCNOfficial:
    name = "EvolveGCN"
    requirements_note = ("requires torch-geometric-temporal "
                         "(`pip install torch-geometric-temporal`); uses the "
                         "official EvolveGCNO layer.")

    def __init__(self, num_nodes, config, features=None, seed=42,
                 num_snapshots=None, epochs=None, lr=1e-2, **kw):
        self.num_nodes = int(num_nodes)
        self.config = config
        self.features = features
        self.seed = int(seed)
        self.num_snapshots = num_snapshots
        self.epochs = int(epochs if epochs is not None else getattr(config, "dyn_epochs", 3))
        self.lr = float(lr)
        self.scores_ = None

    def fit_temporal(self, edges, edge_times, labels=None):
        import torch
        from torch import nn
        try:
            from torch_geometric_temporal.nn.recurrent import EvolveGCNO
        except Exception as e:
            raise ImportError(
                f"EvolveGCN official requires torch_geometric_temporal: {e}. "
                f"{self.requirements_note} (or use EvolveGCN-reimpl).") from e

        torch.manual_seed(self.seed); np.random.seed(self.seed)
        device = torch.device(str(getattr(self.config, "device", "cpu") or "cpu"))
        edges = np.ascontiguousarray(edges, dtype=np.int64).reshape(-1, 2)
        times = np.ascontiguousarray(edge_times, dtype=np.float64).reshape(-1)
        n, E = self.num_nodes, edges.shape[0]
        if E == 0:
            self.scores_ = np.zeros(n); return self

        X = self.features
        X = X.toarray() if sp.issparse(X) else (np.asarray(X) if X is not None
                                                else np.ones((n, 1), np.float32))
        Xt = torch.from_numpy(X.astype(np.float32)).to(device)
        in_dim = Xt.shape[1]

        T = self.num_snapshots or max(2, min(20, E // 500))
        lo, hi = times.min(), times.max()
        binw = (hi - lo) or 1.0
        bins = np.clip(((times - lo) / binw * T).astype(int), 0, T - 1)
        snaps = [edges[bins == k] for k in range(T)]

        model = EvolveGCNO(in_dim).to(device)
        dec = nn.Linear(in_dim, in_dim).to(device)
        opt = torch.optim.Adam(list(model.parameters()) + list(dec.parameters()), lr=self.lr)

        err_sum = np.zeros(n); err_cnt = np.zeros(n)
        for ep in range(max(1, self.epochs)):
            model.reset_parameters() if hasattr(model, "reset_parameters") else None
            record = (ep == max(1, self.epochs) - 1)
            for k, se in enumerate(snaps):
                if se.shape[0] == 0:
                    continue
                ei = torch.from_numpy(np.concatenate([se.T, se.T[::-1]], axis=1).copy()
                                      ).long().to(device)
                H = model(Xt, ei)
                Z = dec(H)
                s_i = torch.from_numpy(se[:, 0]).long().to(device)
                d_i = torch.from_numpy(se[:, 1]).long().to(device)
                neg = torch.randint(0, n, (s_i.numel(),), device=device)
                pos = (Z[s_i] * H[d_i]).sum(-1)
                ng = (Z[s_i] * H[neg]).sum(-1)
                loss = nn.functional.softplus(-pos).mean() + nn.functional.softplus(ng).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                if record:
                    with torch.no_grad():
                        p = torch.sigmoid(pos).clamp(1e-6, 1 - 1e-6).cpu().numpy()
                    surp = -np.log(p)
                    np.add.at(err_sum, se[:, 0], surp); np.add.at(err_sum, se[:, 1], surp)
                    np.add.at(err_cnt, se[:, 0], 1.0); np.add.at(err_cnt, se[:, 1], 1.0)

        seen = err_cnt > 0
        out = np.zeros(n)
        out[seen] = err_sum[seen] / err_cnt[seen]
        if seen.any():
            out[~seen] = out[seen].min()
        self.scores_ = out
        return self

    def score(self):
        if self.scores_ is None:
            raise RuntimeError("EvolveGCN.fit_temporal must run before score")
        s = self.scores_.astype(np.float64)
        return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else np.zeros_like(s)


OFFICIAL_DYNAMIC_BASELINES = {
    "TGN": TGNOfficial,
    "JODIE": JODIEOfficial,
    "DyRep": DyRepOfficial,
    "TGAT": TGATOfficial,
    "EvolveGCN": EvolveGCNOfficial,
}

OFFICIAL_DYNAMIC_REQUIREMENTS = {
    k: v.requirements_note for k, v in OFFICIAL_DYNAMIC_BASELINES.items()
}


def list_official_dynamic_baselines():
    return sorted(OFFICIAL_DYNAMIC_BASELINES.keys())
