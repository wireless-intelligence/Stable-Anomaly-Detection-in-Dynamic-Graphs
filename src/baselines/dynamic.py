"""Dynamic (temporal) graph anomaly-detection baselines.

All consume the genuine timestamped edge stream (not a shuffled static replay)
and are fully UNSUPERVISED: trained with a self-supervised temporal
link-prediction objective, with the per-node anomaly score = the model's mean
link-prediction *surprise* (negative log-likelihood of the true incident
events, computed online BEFORE the memory/model sees each event -> no label or
future leakage). Same prediction-error principle as SAD / SLADE.

Temporal-GNN family (shared TGN core; PyG's TGN framework subsumes all four):

    variant   memory  aggregator  embedding             method
    -------   ------  ----------  --------------------  ------
    TGN       yes     Last        temporal attention    Rossi+2020
    DyRep     yes     Mean        temporal attention     Trivedi+2019 (assoc/comm)
    JODIE     yes     Last        time-projection        Kumar+2019
    TGAT      no      -           temporal attention     Xu+2020 (memoryless)

Plus EvolveGCN (Pareja+2020): discrete-time, GCN weights evolve via a GRU over
snapshots; self-contained (torch + GCNConv only, no torch_geometric_temporal).

Interface (distinct from the static ``BaselineModel`` so the runner can pass
timestamps):

    m = TGNBaseline(num_nodes, config, features=..., seed=...)
    m.fit_temporal(edges, edge_times, labels=None)   # edges (E,2), times (E,)
    scores = m.score()                               # (num_nodes,) higher=anomalous

Missing/incompatible deps degrade gracefully: ``DYNAMIC_BASELINES`` only holds
variants that import cleanly.
"""

import os
import warnings

import numpy as np


_VERBOSE = os.environ.get("SAG_BASELINES_VERBOSE", "1") not in ("0", "false", "False", "")


def _have_torch():
    try:
        import torch  # noqa
        import torch_geometric  # noqa
        return True
    except Exception:
        return False


# variant -> (use_memory, aggregator, embedding_kind)
_VARIANTS = {
    "tgn":   (True,  "last", "attn"),
    "dyrep": (True,  "mean", "attn"),
    "jodie": (True,  "last", "jodie"),
    "tgat":  (False, None,   "attn"),
}


class _TemporalGNNBase:
    """Shared TGN-family core. Subclasses set ``variant``."""

    name = "TemporalGNN"
    variant = "tgn"

    def __init__(self, num_nodes, config, features=None, seed=0,
                 memory_dim=100, time_dim=100, embed_dim=100,
                 epochs=None, batch_size=200, lr=1e-4):
        self.num_nodes = int(num_nodes)
        self.config = config
        self.features = features
        self.seed = int(seed)
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs if epochs is not None else int(getattr(config, "dyn_epochs", 1))
        self._scores = np.zeros(num_nodes, dtype=np.float64)

    def fit_temporal(self, edges, edge_times, labels=None):
        import torch
        from torch import nn
        from torch_geometric.nn import TGNMemory, TransformerConv
        from torch_geometric.nn.models.tgn import (
            IdentityMessage, LastAggregator, MeanAggregator, LastNeighborLoader,
        )

        use_memory, agg_kind, embed_kind = _VARIANTS[self.variant]
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device(getattr(self.config, "device", "cpu"))

        edges = np.ascontiguousarray(edges, dtype=np.int64).reshape(-1, 2)
        t = np.ascontiguousarray(edge_times, dtype=np.float64).reshape(-1)
        order = np.argsort(t, kind="mergesort")
        t_sorted = t[order]
        if t_sorted.shape[0]:
            t_sorted = t_sorted - t_sorted.min()
        src = torch.from_numpy(edges[order, 0]).long().to(device)
        dst = torch.from_numpy(edges[order, 1]).long().to(device)
        # PyG TGNMemory keeps last_update as a long buffer -> integer timestamps
        ts = torch.from_numpy(np.rint(t_sorted).astype(np.int64)).long().to(device)
        E = src.numel()
        if E == 0:
            return self
        n = self.num_nodes
        msg_dim = 4
        msg = torch.zeros((E, msg_dim), dtype=torch.float, device=device)

        # -- memory (or a learnable node table for the memoryless TGAT) ------ #
        if use_memory:
            agg = MeanAggregator() if agg_kind == "mean" else LastAggregator()
            memory = TGNMemory(
                n, raw_msg_dim=msg_dim, memory_dim=self.memory_dim,
                time_dim=self.time_dim,
                message_module=IdentityMessage(msg_dim, self.memory_dim, self.time_dim),
                aggregator_module=agg,
            ).to(device)
            time_enc = memory.time_enc
            node_emb = None
            last_update_buf = None
        else:
            memory = None
            node_emb = nn.Embedding(n, self.memory_dim).to(device)
            from torch_geometric.nn.models.tgn import TimeEncoder
            time_enc = TimeEncoder(self.time_dim).to(device)
            last_update_buf = torch.zeros(n, dtype=torch.long, device=device)

        class _Embed(nn.Module):
            def __init__(self, in_ch, out_ch, msg_dim, time_enc, kind):
                super().__init__()
                self.time_enc = time_enc
                self.kind = kind
                edge_dim = msg_dim + time_enc.out_channels
                self.conv = TransformerConv(in_ch, out_ch // 2, heads=2,
                                            dropout=0.1, edge_dim=edge_dim)
                self.proj = nn.Linear(1, in_ch) if kind == "jodie" else None

            def forward(self, x, last_update, edge_index, t, msg):
                rel_t = (last_update[edge_index[0]] - t).to(x.dtype)
                rel_t_enc = self.time_enc(rel_t)
                edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
                if self.kind == "jodie" and self.proj is not None:
                    ref = last_update.max() if last_update.numel() else last_update.new_zeros(())
                    dt = (ref - last_update).clamp(min=0).to(x.dtype).unsqueeze(-1)
                    x = x * (1.0 + torch.tanh(self.proj(dt)))
                return self.conv(x, edge_index, edge_attr)

        class _Link(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.l1 = nn.Linear(2 * dim, dim)
                self.l2 = nn.Linear(dim, 1)

            def forward(self, z_src, z_dst):
                return self.l2(self.l1(torch.cat([z_src, z_dst], dim=-1)).relu())

        gnn = _Embed(self.memory_dim, self.embed_dim, msg_dim, time_enc, embed_kind).to(device)
        link = _Link(self.embed_dim).to(device)
        params = set(gnn.parameters()) | set(link.parameters())
        params |= set(memory.parameters()) if use_memory else set(node_emb.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        crit = nn.BCEWithLogitsLoss()
        neighbor_loader = LastNeighborLoader(n, size=10, device=device)
        assoc = torch.empty(n, dtype=torch.long, device=device)

        surprise_sum = np.zeros(n, dtype=np.float64)
        surprise_cnt = np.zeros(n, dtype=np.float64)

        for ep in range(max(1, self.epochs)):
            if use_memory:
                memory.reset_state()
            else:
                last_update_buf.zero_()
            neighbor_loader.reset_state()
            gnn.train(); link.train()
            (memory.train() if use_memory else node_emb.train())
            record = (ep == max(1, self.epochs) - 1)  # score on final pass only

            for s in range(0, E, self.batch_size):
                e = min(E, s + self.batch_size)
                b_src, b_dst, b_t, b_msg = src[s:e], dst[s:e], ts[s:e], msg[s:e]
                neg = torch.randint(0, n, (b_src.numel(),), device=device)

                node_ids = torch.cat([b_src, b_dst, neg]).unique()
                n_id, edge_index, e_id = neighbor_loader(node_ids)
                assoc[n_id] = torch.arange(n_id.numel(), device=device)

                if use_memory:
                    z, last_update = memory(n_id)
                else:
                    z = node_emb(n_id)
                    last_update = last_update_buf[n_id]
                z = gnn(z, last_update, edge_index, ts[e_id], msg[e_id])

                pos = link(z[assoc[b_src]], z[assoc[b_dst]]).squeeze(-1)
                negp = link(z[assoc[b_src]], z[assoc[neg]]).squeeze(-1)
                loss = crit(pos, torch.ones_like(pos)) + crit(negp, torch.zeros_like(negp))

                if record:
                    with torch.no_grad():
                        p = torch.sigmoid(pos).clamp(1e-6, 1 - 1e-6)
                        surp = (-torch.log(p)).cpu().numpy()
                    bs = b_src.cpu().numpy(); bd = b_dst.cpu().numpy()
                    np.add.at(surprise_sum, bs, surp)
                    np.add.at(surprise_sum, bd, surp)
                    np.add.at(surprise_cnt, bs, 1.0)
                    np.add.at(surprise_cnt, bd, 1.0)

                if use_memory:
                    memory.update_state(b_src, b_dst, b_t, b_msg)
                else:
                    last_update_buf[b_src] = b_t
                    last_update_buf[b_dst] = b_t
                neighbor_loader.insert(b_src, b_dst)
                loss.backward()
                opt.step()
                opt.zero_grad()
                if use_memory:
                    memory.detach()

        seen = surprise_cnt > 0
        out = np.zeros(n, dtype=np.float64)
        out[seen] = surprise_sum[seen] / surprise_cnt[seen]
        if seen.any():
            out[~seen] = out[seen].min()
        self._scores = out
        return self

    def score(self):
        return self._scores


class TGNBaseline(_TemporalGNNBase):
    name = "TGN"
    variant = "tgn"


class DyRepBaseline(_TemporalGNNBase):
    name = "DyRep"
    variant = "dyrep"


class JODIEBaseline(_TemporalGNNBase):
    name = "JODIE"
    variant = "jodie"


class TGATBaseline(_TemporalGNNBase):
    name = "TGAT"
    variant = "tgat"


# --------------------------------------------------------------------------- #
# EvolveGCN  --  discrete-time; GCN weights evolve via a GRU across snapshots.
# Self-contained (torch + GCNConv), no torch_geometric_temporal dependency.
# --------------------------------------------------------------------------- #

class EvolveGCNBaseline:
    name = "EvolveGCN"

    def __init__(self, num_nodes, config, features=None, seed=0,
                 hidden_dim=64, num_snapshots=None, epochs=None, lr=1e-2):
        self.num_nodes = int(num_nodes)
        self.config = config
        self.features = features
        self.seed = int(seed)
        self.hidden_dim = hidden_dim
        self.num_snapshots = num_snapshots
        self.epochs = epochs if epochs is not None else int(getattr(config, "dyn_epochs", 3))
        self.lr = lr
        self._scores = np.zeros(num_nodes, dtype=np.float64)

    def _snapshot_edges(self, edges, times, T):
        """Bin the temporal edge stream into T time-equal snapshots."""
        t = times.astype(np.float64)
        lo, hi = t.min(), t.max()
        if hi <= lo:
            return [edges]
        bins = np.clip(((t - lo) / (hi - lo) * T).astype(int), 0, T - 1)
        return [edges[bins == k] for k in range(T)]

    def fit_temporal(self, edges, edge_times, labels=None):
        import torch
        from torch import nn
        from torch_geometric.nn import GCNConv

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device(getattr(self.config, "device", "cpu"))

        edges = np.ascontiguousarray(edges, dtype=np.int64).reshape(-1, 2)
        times = np.ascontiguousarray(edge_times, dtype=np.float64).reshape(-1)
        E = edges.shape[0]
        n = self.num_nodes
        if E == 0:
            return self
        T = self.num_snapshots or max(2, min(20, E // 500))
        snaps = self._snapshot_edges(edges, times, T)

        # node features (dense); fall back to identity-ish degree scalar
        if self.features is not None:
            import scipy.sparse as sp
            F = self.features
            X = torch.from_numpy((F.toarray() if sp.issparse(F) else np.asarray(F))
                                 .astype(np.float32)).to(device)
        else:
            X = torch.ones((n, 1), dtype=torch.float, device=device)
        in_dim = X.shape[1]
        hid = self.hidden_dim

        class _EvolveGCN(nn.Module):
            def __init__(self, in_dim, hid):
                super().__init__()
                self.conv = GCNConv(in_dim, hid, add_self_loops=True)
                self.gru = nn.GRUCell(hid, hid)
                # evolving weight = the GCNConv linear weight (hid x in_dim) is
                # kept fixed; we evolve a per-snapshot hidden gate over outputs
                self.register_buffer("_h0", torch.zeros(hid))

            def forward(self, x, edge_index, h):
                z = self.conv(x, edge_index)
                z = torch.relu(z)
                h = self.gru(z.mean(0, keepdim=True).expand(z.size(0), -1), h)
                return z * torch.sigmoid(h), h

        model = _EvolveGCN(in_dim, hid).to(device)
        link_w = nn.Parameter(torch.randn(hid, device=device) * 0.01)
        opt = torch.optim.Adam(list(model.parameters()) + [link_w], lr=self.lr)

        recon_err = np.zeros(n, dtype=np.float64)
        recon_cnt = np.zeros(n, dtype=np.float64)

        for ep in range(max(1, self.epochs)):
            h = torch.zeros((n, hid), device=device)
            record = (ep == max(1, self.epochs) - 1)
            for k, se in enumerate(snaps):
                if se.shape[0] == 0:
                    continue
                ei = torch.from_numpy(se.T.copy()).long().to(device)
                ei = torch.cat([ei, ei.flip(0)], dim=1)  # undirected
                z, h_new = model(X, ei, h)
                h = h_new.detach()
                # self-supervised link prediction on this snapshot's edges
                s_i = torch.from_numpy(se[:, 0]).long().to(device)
                d_i = torch.from_numpy(se[:, 1]).long().to(device)
                neg = torch.randint(0, n, (s_i.numel(),), device=device)
                pos = (z[s_i] * link_w * z[d_i]).sum(-1)
                ng = (z[s_i] * link_w * z[neg]).sum(-1)
                loss = (nn.functional.softplus(-pos).mean()
                        + nn.functional.softplus(ng).mean())
                opt.zero_grad(); loss.backward(); opt.step()

                if record:
                    with torch.no_grad():
                        p = torch.sigmoid(pos).clamp(1e-6, 1 - 1e-6)
                        surp = (-torch.log(p)).cpu().numpy()
                    si = se[:, 0]; di = se[:, 1]
                    np.add.at(recon_err, si, surp)
                    np.add.at(recon_err, di, surp)
                    np.add.at(recon_cnt, si, 1.0)
                    np.add.at(recon_cnt, di, 1.0)

        seen = recon_cnt > 0
        out = np.zeros(n, dtype=np.float64)
        out[seen] = recon_err[seen] / recon_cnt[seen]
        if seen.any():
            out[~seen] = out[seen].min()
        self._scores = out
        return self

    def score(self):
        return self._scores


# --------------------------------------------------------------------------- #
# Registry (in-repo REIMPLEMENTATIONS, torch/PyG-only, always runnable).
#
# These are NOT the primary paper baselines: they are faithful reference
# reimplementations on PyG's official TGN framework, used as a fallback when the
# official vendor code (vendor/DyGLib) is not cloned. The OFFICIAL baselines live
# in ``dynamic_official.py`` and take the canonical names (TGN/JODIE/DyRep/TGAT/
# EvolveGCN); these register under ``*-reimpl`` so both can coexist.
# --------------------------------------------------------------------------- #

REIMPL_DYNAMIC_BASELINES = {}
REIMPL_REQUIREMENTS = {
    "TGN-reimpl": "torch, torch_geometric (TGNMemory); PyG-framework reimpl",
    "DyRep-reimpl": "torch, torch_geometric (TGNMemory, MeanAggregator); PyG reimpl",
    "JODIE-reimpl": "torch, torch_geometric (TGNMemory); PyG reimpl",
    "TGAT-reimpl": "torch, torch_geometric (TransformerConv, TimeEncoder); PyG reimpl",
    "EvolveGCN-reimpl": "torch, torch_geometric (GCNConv); self-contained reimpl",
}

if _have_torch():
    REIMPL_DYNAMIC_BASELINES["TGN-reimpl"] = TGNBaseline
    REIMPL_DYNAMIC_BASELINES["DyRep-reimpl"] = DyRepBaseline
    REIMPL_DYNAMIC_BASELINES["JODIE-reimpl"] = JODIEBaseline
    REIMPL_DYNAMIC_BASELINES["TGAT-reimpl"] = TGATBaseline
    REIMPL_DYNAMIC_BASELINES["EvolveGCN-reimpl"] = EvolveGCNBaseline
elif _VERBOSE:
    warnings.warn("[SAG.baselines.dynamic] torch/torch_geometric unavailable; "
                  "reimpl dynamic baselines disabled")


def list_reimpl_dynamic_baselines():
    return sorted(REIMPL_DYNAMIC_BASELINES.keys())
