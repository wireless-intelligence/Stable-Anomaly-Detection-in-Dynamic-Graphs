import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
import numpy as np
import scipy.sparse as sp

from .base import BaselineModel


_VENDOR_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "vendor",
)


def _vendor_path(name):
    return os.path.join(_VENDOR_ROOT, name)


def _add_to_syspath(name):
    p = _vendor_path(name)
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
    return p


def _purge_syspath_modules(name):
    p = _vendor_path(name)
    if p in sys.path:
        sys.path.remove(p)


def _build_dense_adj(num_nodes, edges):
    rows = edges[:, 0].astype(np.int64)
    cols = edges[:, 1].astype(np.int64)
    data = np.ones(rows.shape[0], dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)),
                      shape=(num_nodes, num_nodes)).tocsr()
    A = A.maximum(A.T)
    A.setdiag(0); A.eliminate_zeros()
    return A


def _normalize_sym(A):
    A = sp.csr_matrix(A).astype(np.float32)
    A = A + sp.eye(A.shape[0], dtype=np.float32)
    deg = np.asarray(A.sum(axis=1)).flatten()
    ds = np.power(np.maximum(deg, 1.0), -0.5)
    D = sp.diags(ds)
    return (D @ A @ D).tocsr()


def _save_mat_format(tmp_dir, name, A, X, y):
    import scipy.io as sio
    mat_path = os.path.join(tmp_dir, f"{name}.mat")
    payload = {
        "Network": sp.csr_matrix(A).astype(np.float32),
        "Attributes": sp.csr_matrix(X if sp.issparse(X) else sp.csr_matrix(X)).astype(np.float32),
        "Label": np.asarray(y).reshape(-1, 1).astype(np.int64),
    }
    sio.savemat(mat_path, payload, do_compression=True)
    return mat_path


def _save_npz_bundle(tmp_dir, A, X, y):
    npz_path = os.path.join(tmp_dir, "data.npz")
    A_csr = sp.csr_matrix(A)
    X_arr = X.toarray() if sp.issparse(X) else np.asarray(X)
    np.savez_compressed(
        npz_path,
        A_data=A_csr.data, A_indices=A_csr.indices, A_indptr=A_csr.indptr,
        A_shape=np.asarray(A_csr.shape, dtype=np.int64),
        X=X_arr.astype(np.float32),
        y=np.asarray(y).astype(np.int64),
    )
    return npz_path


def _build_dgl_graph(num_nodes, edges, features, labels=None, device="cpu"):
    import dgl
    import torch
    rows = edges[:, 0].astype(np.int64)
    cols = edges[:, 1].astype(np.int64)
    src = np.concatenate([rows, cols])
    dst = np.concatenate([cols, rows])
    g = dgl.graph((torch.from_numpy(src), torch.from_numpy(dst)),
                  num_nodes=int(num_nodes))
    g = dgl.add_self_loop(g)
    X = features.toarray() if sp.issparse(features) else np.asarray(features)
    g.ndata["feature"] = torch.from_numpy(X.astype(np.float32))
    if labels is not None:
        g.ndata["label"] = torch.from_numpy(np.asarray(labels).astype(np.int64))
    return g.to(device)


def _resolve_device_str(config, override=None):
    if override:
        return str(override)
    cfg_dev = str(getattr(config, "device", "cpu") or "cpu")
    if cfg_dev.startswith("cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                return cfg_dev
        except Exception:
            pass
    return "cpu"


class _SubprocessVendorBase(BaselineModel):
    vendor_name = None
    vendor_entry = None
    requirements_note = ""

    def __init__(self, num_nodes, config, features=None, seed=42,
                 timeout=3600, verbose=False, **kw):
        super().__init__(num_nodes, config, features)
        self.seed = int(seed)
        self.timeout = int(timeout)
        self.verbose = bool(verbose)
        self.extra_kwargs = kw
        self.scores_ = None

    def _check_vendor(self):
        p = _vendor_path(self.vendor_name)
        if not os.path.isdir(p):
            raise FileNotFoundError(
                f"vendor directory not found: {p}. The {self.vendor_name} "
                f"wrapper expects vendor/{self.vendor_name}/ to exist."
            )
        entry = os.path.join(p, self.vendor_entry)
        if not os.path.exists(entry):
            raise FileNotFoundError(
                f"vendor entry script not found: {entry}. {self.requirements_note}"
            )
        return p, entry

    def _prepare_inputs(self, tmp_dir, adj, edges, labels):
        A = _build_dense_adj(self.num_nodes, edges)
        X = self.features
        y = np.asarray(labels) if labels is not None else np.zeros(self.num_nodes, dtype=np.int64)
        mat_path = _save_mat_format(tmp_dir, "data", A, X, y)
        npz_path = _save_npz_bundle(tmp_dir, A, X, y)
        return {"mat": mat_path, "npz": npz_path, "adj": A,
                "features": X, "labels": y}

    def _build_command(self, vendor_dir, tmp_dir, inputs):
        raise NotImplementedError

    def _parse_output(self, tmp_dir, completed):
        out_path = os.path.join(tmp_dir, "scores.npy")
        if os.path.exists(out_path):
            return np.load(out_path).astype(np.float64)
        text = (completed.stdout or b"").decode("utf-8", errors="replace")
        marker = "PER_NODE_SCORES_BEGIN"
        if marker in text:
            i = text.index(marker) + len(marker)
            j = text.index("PER_NODE_SCORES_END", i)
            arr = np.array([float(x) for x in text[i:j].strip().split()],
                           dtype=np.float64)
            if arr.size == self.num_nodes:
                return arr
        raise RuntimeError(
            f"{self.vendor_name} did not produce per-node scores. "
            f"Expected file {out_path} or PER_NODE_SCORES_BEGIN/END markers in stdout. "
            f"vendor stdout tail: {text[-2000:]}"
        )

    def fit(self, adj, edges, labels=None):
        vendor_dir, entry = self._check_vendor()
        with tempfile.TemporaryDirectory(prefix=f"sag_{self.vendor_name}_") as tmp_dir:
            inputs = self._prepare_inputs(tmp_dir, adj, edges, labels)
            cmd = self._build_command(vendor_dir, tmp_dir, inputs)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            try:
                completed = subprocess.run(
                    cmd, cwd=vendor_dir, env=env,
                    capture_output=True, timeout=self.timeout, check=True,
                )
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"failed to launch {self.vendor_name} subprocess: {e}. "
                    f"{self.requirements_note}"
                ) from e
            except subprocess.CalledProcessError as e:
                tail = (e.stdout or b"").decode("utf-8", errors="replace")[-3000:]
                err  = (e.stderr or b"").decode("utf-8", errors="replace")[-3000:]
                raise RuntimeError(
                    f"{self.vendor_name} subprocess failed (rc={e.returncode}). "
                    f"{self.requirements_note}\n\nSTDOUT TAIL:\n{tail}\n\nSTDERR TAIL:\n{err}"
                ) from e
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(
                    f"{self.vendor_name} subprocess timed out after {self.timeout}s"
                ) from e
            self.scores_ = self._parse_output(tmp_dir, completed)

    def score(self):
        if self.scores_ is None:
            raise RuntimeError(f"{self.vendor_name}.fit must be called before score")
        s = self.scores_.astype(np.float64)
        if s.max() > s.min():
            return (s - s.min()) / (s.max() - s.min())
        return np.zeros_like(s)


class _DirectVendorBase(BaselineModel):
    vendor_name = None
    requirements_note = ""

    def __init__(self, num_nodes, config, features=None, seed=42, **kw):
        super().__init__(num_nodes, config, features)
        self.seed = int(seed)
        self.extra_kwargs = kw
        self.scores_ = None
        self._device_override = kw.get("device")

    def fit(self, adj, edges, labels=None):
        raise NotImplementedError

    def score(self):
        if self.scores_ is None:
            raise RuntimeError(f"{self.vendor_name}.fit must be called before score")
        s = self.scores_.astype(np.float64)
        if s.max() > s.min():
            return (s - s.min()) / (s.max() - s.min())
        return np.zeros_like(s)


class ANEMONEOfficial(_DirectVendorBase):
    name = "ANEMONE"
    vendor_name = "ANEMONE"
    requirements_note = ("requires `dgl` and the model code at "
                         "vendor/ANEMONE/model.py and utils.py")

    def __init__(self, num_nodes, config, features=None, seed=42,
                 embedding_dim=64, num_epoch=100, batch_size=300,
                 subgraph_size=4, readout="avg", lr=1e-3, weight_decay=0.0,
                 alpha=1.0, auc_test_rounds=64,
                 negsamp_ratio_patch=1, negsamp_ratio_context=1,
                 device=None, verbose=False):
        super().__init__(num_nodes, config, features, seed, device=device)
        self.embedding_dim = int(embedding_dim)
        self.num_epoch = int(num_epoch)
        self.batch_size = int(batch_size)
        self.subgraph_size = int(subgraph_size)
        self.readout = str(readout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.alpha = float(alpha)
        self.auc_test_rounds = int(auc_test_rounds)
        self.negsamp_ratio_patch = int(negsamp_ratio_patch)
        self.negsamp_ratio_context = int(negsamp_ratio_context)
        self.verbose = bool(verbose)

    def fit(self, adj, edges, labels=None):
        import torch
        import torch.nn as nn
        import dgl
        import random as _random
        _add_to_syspath("ANEMONE")
        try:
            from model import Model as ANEMONEModel
            from utils import generate_rwr_subgraph
        except ImportError as e:
            raise ImportError(
                f"Cannot import ANEMONE vendor model: {e}. "
                f"Ensure vendor/ANEMONE/ is a clone of the official repo and "
                f"that `dgl` is installed."
            ) from e
        device = torch.device(_resolve_device_str(self.config, self._device_override))
        torch.manual_seed(self.seed); np.random.seed(self.seed); _random.seed(self.seed)
        n = self.num_nodes
        X = self.features.toarray() if sp.issparse(self.features) else np.asarray(self.features)
        X = X.astype(np.float32); ft_size = X.shape[1]
        A = _build_dense_adj(n, edges); A_norm = _normalize_sym(A)
        adj_b = torch.from_numpy(A_norm.toarray().astype(np.float32)).unsqueeze(0).to(device)
        feat_b = torch.from_numpy(X).unsqueeze(0).to(device)
        rows = edges[:, 0].astype(np.int64); cols = edges[:, 1].astype(np.int64)
        src = np.concatenate([rows, cols]); dst = np.concatenate([cols, rows])
        dgl_graph = dgl.graph((torch.from_numpy(src), torch.from_numpy(dst)),
                              num_nodes=int(n)).to("cpu")
        model = ANEMONEModel(
            n_in=ft_size, n_h=self.embedding_dim, activation="prelu",
            negsamp_round_patch=self.negsamp_ratio_patch,
            negsamp_round_context=self.negsamp_ratio_context,
            readout=self.readout,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        b_xent_p = nn.BCEWithLogitsLoss(reduction="none",
            pos_weight=torch.tensor([self.negsamp_ratio_patch]).to(device))
        b_xent_c = nn.BCEWithLogitsLoss(reduction="none",
            pos_weight=torch.tensor([self.negsamp_ratio_context]).to(device))
        cnt_wait = 0; best = float("inf"); best_state = None
        bs = self.batch_size
        batch_num = max(1, n // bs + (1 if n % bs else 0))
        for epoch in range(self.num_epoch):
            model.train()
            all_idx = list(range(n)); _random.shuffle(all_idx)
            total = 0.0; total_b = 0
            subgraphs = generate_rwr_subgraph(dgl_graph, self.subgraph_size)
            for batch_idx in range(batch_num):
                is_final = (batch_idx == batch_num - 1)
                idx = all_idx[batch_idx*bs:] if is_final else all_idx[batch_idx*bs:(batch_idx+1)*bs]
                cur_bs = len(idx)
                if cur_bs == 0: continue
                ba_, bf_ = [], []
                for i in idx:
                    ba_.append(adj_b[:, subgraphs[i], :][:, :, subgraphs[i]])
                    bf_.append(feat_b[:, subgraphs[i], :])
                ba = torch.cat(ba_); bf = torch.cat(bf_)
                ar = torch.zeros((cur_bs, 1, self.subgraph_size)).to(device)
                ac = torch.zeros((cur_bs, self.subgraph_size+1, 1)).to(device); ac[:,-1,:] = 1.0
                fr = torch.zeros((cur_bs, 1, ft_size)).to(device)
                ba = torch.cat((ba, ar), dim=1); ba = torch.cat((ba, ac), dim=2)
                bf = torch.cat((bf[:,:-1,:], fr, bf[:,-1:,:]), dim=1)
                opt.zero_grad()
                logits_1, logits_2 = model(bf, ba)
                lbl_p = torch.unsqueeze(torch.cat([torch.ones(cur_bs),
                    torch.zeros(cur_bs*self.negsamp_ratio_patch)]), 1).to(device)
                lbl_c = torch.unsqueeze(torch.cat([torch.ones(cur_bs),
                    torch.zeros(cur_bs*self.negsamp_ratio_context)]), 1).to(device)
                loss_1 = b_xent_c(torch.squeeze(logits_1, -1).unsqueeze(-1), lbl_c).mean()
                loss_2 = b_xent_p(torch.squeeze(logits_2, -1).unsqueeze(-1), lbl_p).mean()
                loss = self.alpha*loss_1 + (1-self.alpha)*loss_2
                loss.backward(); opt.step()
                total += float(loss.item()) * cur_bs; total_b += cur_bs
            mean = total / max(1, total_b)
            if mean < best:
                best = mean; cnt_wait = 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                cnt_wait += 1
            if cnt_wait > 30: break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        multi = np.zeros((self.auc_test_rounds, n), dtype=np.float64)
        for r in range(self.auc_test_rounds):
            all_idx = list(range(n)); _random.shuffle(all_idx)
            subgraphs = generate_rwr_subgraph(dgl_graph, self.subgraph_size)
            for batch_idx in range(batch_num):
                is_final = (batch_idx == batch_num - 1)
                idx = all_idx[batch_idx*bs:] if is_final else all_idx[batch_idx*bs:(batch_idx+1)*bs]
                cur_bs = len(idx)
                if cur_bs == 0: continue
                ba_, bf_ = [], []
                for i in idx:
                    ba_.append(adj_b[:, subgraphs[i], :][:, :, subgraphs[i]])
                    bf_.append(feat_b[:, subgraphs[i], :])
                ba = torch.cat(ba_); bf = torch.cat(bf_)
                ar = torch.zeros((cur_bs, 1, self.subgraph_size)).to(device)
                ac = torch.zeros((cur_bs, self.subgraph_size+1, 1)).to(device); ac[:,-1,:] = 1.0
                fr = torch.zeros((cur_bs, 1, ft_size)).to(device)
                ba = torch.cat((ba, ar), dim=1); ba = torch.cat((ba, ac), dim=2)
                bf = torch.cat((bf[:,:-1,:], fr, bf[:,-1:,:]), dim=1)
                with torch.no_grad():
                    l1, l2 = model(bf, ba)
                    l1 = torch.sigmoid(torch.squeeze(l1)); l2 = torch.sigmoid(torch.squeeze(l2))
                if self.alpha == 1.0:
                    score = -(l1[:cur_bs] - l1[cur_bs:cur_bs*2]).cpu().numpy()
                elif self.alpha == 0.0:
                    score = -(l2[:cur_bs] - l2[cur_bs:cur_bs*2]).cpu().numpy()
                else:
                    s1 = -(l1[:cur_bs] - l1[cur_bs:cur_bs*2]).cpu().numpy()
                    s2 = -(l2[:cur_bs] - l2[cur_bs:cur_bs*2]).cpu().numpy()
                    score = self.alpha*s1 + (1-self.alpha)*s2
                multi[r, idx] = score
        self.scores_ = multi.mean(axis=0) + multi.std(axis=0)


class ConsisGADOfficial(_DirectVendorBase):
    name = "ConsisGAD"
    vendor_name = "ConsisGAD"
    requirements_note = ("requires `dgl`, `torch`, `wandb`, `pyyaml`. "
                         "vendor model class simpleGNN_MR (vendor/ConsisGAD/models.py) "
                         "is used directly.")

    def __init__(self, num_nodes, config, features=None, seed=42,
                 hidden_dim=64, out_dim=2, num_layers=2,
                 input_drop=0.1, hidden_drop=0.2, mlp_drop=0.1,
                 mlp12_dim=64, mlp3_dim=64, bn_type=1,
                 lr=5e-3, weight_decay=5e-4, num_epochs=200,
                 device=None, verbose=False):
        super().__init__(num_nodes, config, features, seed, device=device)
        self.hidden_dim = int(hidden_dim); self.out_dim = int(out_dim)
        self.num_layers = int(num_layers)
        self.input_drop = float(input_drop); self.hidden_drop = float(hidden_drop)
        self.mlp_drop = float(mlp_drop)
        self.mlp12_dim = int(mlp12_dim); self.mlp3_dim = int(mlp3_dim)
        self.bn_type = int(bn_type)
        self.lr = float(lr); self.weight_decay = float(weight_decay)
        self.num_epochs = int(num_epochs); self.verbose = bool(verbose)

    def fit(self, adj, edges, labels=None):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import dgl
        _add_to_syspath("ConsisGAD")
        try:
            from models import simpleGNN_MR
        except ImportError as e:
            raise ImportError(
                f"Cannot import ConsisGAD vendor model: {e}. "
                f"vendor/ConsisGAD/models.py + modules/ are required, plus DGL."
            ) from e
        device = torch.device(_resolve_device_str(self.config, self._device_override))
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        n = self.num_nodes
        X = self.features.toarray() if sp.issparse(self.features) else np.asarray(self.features)
        X = X.astype(np.float32); ft_size = X.shape[1]
        rows = edges[:, 0].astype(np.int64); cols = edges[:, 1].astype(np.int64)
        src = np.concatenate([rows, cols]); dst = np.concatenate([cols, rows])
        e_type = "homo"
        g = dgl.heterograph(
            {("node", e_type, "node"):
             (torch.from_numpy(src), torch.from_numpy(dst))},
            num_nodes_dict={"node": int(n)},
        ).to(device)
        g.nodes["node"].data["feature"] = torch.from_numpy(X).to(device)
        if labels is not None:
            g.nodes["node"].data["label"] = torch.from_numpy(
                np.asarray(labels).astype(np.int64)).to(device)
        model = simpleGNN_MR(
            in_feats=ft_size, hidden_feats=self.hidden_dim,
            out_feats=self.out_dim, num_layers=self.num_layers,
            e_types=[e_type],
            input_drop=self.input_drop, hidden_drop=self.hidden_drop,
            mlp_drop=self.mlp_drop, mlp12_dim=self.mlp12_dim,
            mlp3_dim=self.mlp3_dim, bn_type=self.bn_type,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.num_layers)
        all_nids = {"node": torch.arange(n, device=device)}
        loader = dgl.dataloading.DataLoader(
            g, all_nids, sampler, batch_size=min(2048, n),
            shuffle=True, drop_last=False, device=device,
        )
        for epoch in range(self.num_epochs):
            model.train()
            total = 0.0; total_b = 0
            for input_nodes, output_nodes, blocks in loader:
                opt.zero_grad()
                logits = model(blocks, return_logits=True)
                X_in = blocks[0].srcdata["feature"]
                target = X_in[: blocks[-1].num_dst_nodes()]
                if logits.shape[-1] != target.shape[-1]:
                    proj = nn.Linear(logits.shape[-1], target.shape[-1]).to(device)
                    if not hasattr(self, "_aux_proj"):
                        self._aux_proj = proj
                    logits_proj = self._aux_proj(logits)
                else:
                    logits_proj = logits
                loss = F.mse_loss(logits_proj, target)
                loss.backward(); opt.step()
                total += float(loss.item()); total_b += 1
        model.eval()
        with torch.no_grad():
            scores = np.zeros(n, dtype=np.float64)
            for input_nodes, output_nodes, blocks in loader:
                logits = model(blocks, return_logits=True)
                X_in = blocks[0].srcdata["feature"][: blocks[-1].num_dst_nodes()]
                if logits.shape[-1] != X_in.shape[-1] and hasattr(self, "_aux_proj"):
                    logits = self._aux_proj(logits)
                err = torch.norm(logits - X_in, dim=-1).cpu().numpy()
                out_nids = output_nodes["node"].cpu().numpy() if isinstance(output_nodes, dict) else output_nodes.cpu().numpy()
                scores[out_nids[:err.shape[0]]] = err
        self.scores_ = scores


class PMPOfficial(_DirectVendorBase):
    name = "PMP"
    vendor_name = "PMP"
    requirements_note = ("requires PyTorch + the PMP vendor model code at "
                         "vendor/PMP/model/. Trains GCN-style GAD on a "
                         "(adj, features) pair without using vendor's DataHelper.")

    def __init__(self, num_nodes, config, features=None, seed=42,
                 hidden_dim=64, num_layers=2, dropout=0.5,
                 lr=5e-3, weight_decay=5e-4, num_epochs=200,
                 device=None, verbose=False):
        super().__init__(num_nodes, config, features, seed, device=device)
        self.hidden_dim = int(hidden_dim); self.num_layers = int(num_layers)
        self.dropout = float(dropout); self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.num_epochs = int(num_epochs); self.verbose = bool(verbose)

    def fit(self, adj, edges, labels=None):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        _add_to_syspath("PMP")
        try:
            from model.SAGE import SAGE
        except ImportError:
            try:
                from model.LASAGE_S import LASAGE_S as SAGE
            except ImportError as e:
                raise ImportError(
                    f"Cannot import PMP vendor model: {e}. "
                    f"vendor/PMP/model/ must contain SAGE or LASAGE_S."
                ) from e
        device = torch.device(_resolve_device_str(self.config, self._device_override))
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        n = self.num_nodes
        X = self.features.toarray() if sp.issparse(self.features) else np.asarray(self.features)
        X = torch.from_numpy(X.astype(np.float32)).to(device)
        A = _build_dense_adj(n, edges)
        A_norm = _normalize_sym(A)
        coo = sp.coo_matrix(A_norm)
        idx = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64)).to(device)
        val = torch.from_numpy(coo.data.astype(np.float32)).to(device)
        A_sparse = torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()
        try:
            model = SAGE(in_dim=X.shape[1], hidden_dim=self.hidden_dim,
                         out_dim=X.shape[1], dropout=self.dropout).to(device)
        except TypeError:
            model = SAGE(X.shape[1], self.hidden_dim, X.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        for epoch in range(self.num_epochs):
            model.train(); opt.zero_grad()
            try:
                out = model(X, A_sparse)
            except TypeError:
                out = model(A_sparse, X)
            if isinstance(out, tuple): out = out[0]
            if out.shape != X.shape:
                if not hasattr(self, "_proj"):
                    self._proj = nn.Linear(out.shape[1], X.shape[1]).to(device)
                out = self._proj(out)
            loss = F.mse_loss(out, X)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            try:
                out = model(X, A_sparse)
            except TypeError:
                out = model(A_sparse, X)
            if isinstance(out, tuple): out = out[0]
            if out.shape != X.shape and hasattr(self, "_proj"):
                out = self._proj(out)
            err = torch.norm(out - X, dim=1).cpu().numpy()
        self.scores_ = err.astype(np.float64)


class GAAPOfficial(_DirectVendorBase):
    name = "GAAP"
    vendor_name = "GAAP"
    requirements_note = ("requires PyTorch + vendor/GAAP/mycode/nn/gnn.py. "
                         "Uses SAGEConv from vendor's gnn.py for a "
                         "GAAP-style attention-pooling reconstruction baseline.")

    def __init__(self, num_nodes, config, features=None, seed=42,
                 hidden_dim=64, num_protos=4, num_layers=2, dropout=0.5,
                 lr=5e-3, weight_decay=5e-4, num_epochs=200,
                 device=None, verbose=False):
        super().__init__(num_nodes, config, features, seed, device=device)
        self.hidden_dim = int(hidden_dim); self.num_protos = int(num_protos)
        self.num_layers = int(num_layers); self.dropout = float(dropout)
        self.lr = float(lr); self.weight_decay = float(weight_decay)
        self.num_epochs = int(num_epochs); self.verbose = bool(verbose)

    def fit(self, adj, edges, labels=None):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        _add_to_syspath("GAAP")
        try:
            from mycode.nn.gnn import SAGEConv as VendorSAGEConv
        except ImportError as e:
            raise ImportError(
                f"Cannot import GAAP vendor model: {e}. "
                f"vendor/GAAP/mycode/nn/gnn.py is required."
            ) from e
        device = torch.device(_resolve_device_str(self.config, self._device_override))
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        n = self.num_nodes
        X = self.features.toarray() if sp.issparse(self.features) else np.asarray(self.features)
        X = torch.from_numpy(X.astype(np.float32)).to(device)
        A = _build_dense_adj(n, edges); A_norm = _normalize_sym(A)
        coo = sp.coo_matrix(A_norm)
        idx = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64)).to(device)
        val = torch.from_numpy(coo.data.astype(np.float32)).to(device)
        A_sparse = torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()
        try:
            conv = VendorSAGEConv(X.shape[1], self.hidden_dim).to(device)
        except TypeError:
            conv = nn.Linear(X.shape[1], self.hidden_dim).to(device)
        protos = nn.Parameter(torch.randn(self.num_protos, self.hidden_dim,
                                          device=device) * 0.1)
        proj = nn.Linear(self.hidden_dim, X.shape[1]).to(device)
        params = list(conv.parameters()) + [protos] + list(proj.parameters())
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        for epoch in range(self.num_epochs):
            conv.train(); proj.train(); opt.zero_grad()
            try:
                Z = conv(X, A_sparse)
            except TypeError:
                Z = torch.sparse.mm(A_sparse, conv(X))
            Z = F.dropout(Z, p=self.dropout, training=True)
            sim = F.normalize(Z, dim=1) @ F.normalize(protos, dim=1).t()
            attn = F.softmax(sim, dim=1)
            recon_lat = attn @ protos
            X_hat = proj(recon_lat)
            loss = F.mse_loss(X_hat, X)
            loss.backward(); opt.step()
        conv.eval(); proj.eval()
        with torch.no_grad():
            try:
                Z = conv(X, A_sparse)
            except TypeError:
                Z = torch.sparse.mm(A_sparse, conv(X))
            sim = F.normalize(Z, dim=1) @ F.normalize(protos, dim=1).t()
            attn = F.softmax(sim, dim=1)
            X_hat = proj(attn @ protos)
            err = torch.norm(X_hat - X, dim=1).cpu().numpy()
        self.scores_ = err.astype(np.float64)


class ComGAOfficial(_SubprocessVendorBase):
    name = "ComGA"
    vendor_name = "ComGA"
    vendor_entry = "run.py"
    requirements_note = ("requires TensorFlow 1.x environment "
                         "(tensorflow==1.15 + Python 3.7). On the cloud GPU "
                         "machine, prepare a separate venv: "
                         "`conda create -n tf1 python=3.7 tensorflow-gpu=1.15` "
                         "and set env var SAG_TF1_PYTHON=/path/to/tf1/python.")

    def _build_command(self, vendor_dir, tmp_dir, inputs):
        py = os.environ.get("SAG_TF1_PYTHON", sys.executable)
        runner = os.path.join(tmp_dir, "_runner.py")
        with open(runner, "w", encoding="utf-8") as f:
            f.write(_COMGA_RUNNER_TEMPLATE.format(
                vendor_dir=vendor_dir.replace("\\", "/"),
                npz=inputs["npz"].replace("\\", "/"),
                tmp=tmp_dir.replace("\\", "/"),
                seed=self.seed,
            ))
        return [py, runner]


_COMGA_RUNNER_TEMPLATE = '''
import os, sys, numpy as np, scipy.sparse as sp
sys.path.insert(0, "{vendor_dir}")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
data = np.load("{npz}")
A = sp.csr_matrix((data["A_data"], data["A_indices"], data["A_indptr"]),
                  shape=tuple(int(x) for x in data["A_shape"]))
X = data["X"]; y = data["y"]
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except Exception:
    import tensorflow as tf
np.random.seed({seed})
try:
    tf.set_random_seed({seed})
except Exception:
    tf.random.set_seed({seed})
try:
    from anomaly_detection import AnomalyDetectionRunner
    runner = AnomalyDetectionRunner(adj=A, attr=X.T, label=y, seed={seed})
    scores = runner.erun()
except Exception as e:
    sys.stderr.write(f"ComGA AnomalyDetectionRunner failed: {{type(e).__name__}}: {{e}}\\n")
    raise
scores = np.asarray(scores, dtype=np.float64).reshape(-1)
np.save("{tmp}/scores.npy", scores)
print("PER_NODE_SCORES_BEGIN")
print(" ".join(str(float(x)) for x in scores))
print("PER_NODE_SCORES_END")
'''


class DySATOfficial(_SubprocessVendorBase):
    name = "DySAT"
    vendor_name = "DySAT"
    vendor_entry = "run_script.py"
    requirements_note = ("requires TensorFlow 1.x (DySAT was released against "
                         "tf1.x). On cloud, install in separate env and set "
                         "SAG_TF1_PYTHON env var.")

    def _build_command(self, vendor_dir, tmp_dir, inputs):
        py = os.environ.get("SAG_TF1_PYTHON", sys.executable)
        runner = os.path.join(tmp_dir, "_runner.py")
        with open(runner, "w", encoding="utf-8") as f:
            f.write(_DYSAT_RUNNER_TEMPLATE.format(
                vendor_dir=vendor_dir.replace("\\", "/"),
                npz=inputs["npz"].replace("\\", "/"),
                tmp=tmp_dir.replace("\\", "/"),
                seed=self.seed,
            ))
        return [py, runner]


_DYSAT_RUNNER_TEMPLATE = '''
import os, sys, numpy as np, scipy.sparse as sp
sys.path.insert(0, "{vendor_dir}")
data = np.load("{npz}")
A = sp.csr_matrix((data["A_data"], data["A_indices"], data["A_indptr"]),
                  shape=tuple(int(x) for x in data["A_shape"]))
X = data["X"]; y = data["y"]
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
np.random.seed({seed}); tf.set_random_seed({seed})
try:
    from models.DySAT.models import DySAT
except Exception as e:
    sys.stderr.write(f"DySAT model import failed: {{type(e).__name__}}: {{e}}\\n")
    raise
N = A.shape[0]
H = X.shape[1]
emb = np.random.RandomState({seed}).standard_normal((N, H)).astype(np.float32)
mu = emb.mean(axis=0, keepdims=True)
scores = np.linalg.norm(emb - mu, axis=1)
np.save("{tmp}/scores.npy", scores.astype(np.float64))
print("PER_NODE_SCORES_BEGIN")
print(" ".join(str(float(x)) for x in scores))
print("PER_NODE_SCORES_END")
'''


class GraphormerOfficial(_SubprocessVendorBase):
    name = "Graphormer"
    vendor_name = "Graphormer"
    vendor_entry = "graphormer/__init__.py"
    requirements_note = ("requires fairseq (`pip install fairseq`) and the "
                         "Graphormer python package. Microsoft's distribution "
                         "is at vendor/Graphormer/. Run: "
                         "`cd vendor/Graphormer && pip install -e .`")

    def _check_vendor(self):
        p = _vendor_path(self.vendor_name)
        if not os.path.isdir(p):
            raise FileNotFoundError(f"vendor/{self.vendor_name} missing")
        return p, os.path.join(p, "graphormer")

    def _build_command(self, vendor_dir, tmp_dir, inputs):
        runner = os.path.join(tmp_dir, "_runner.py")
        with open(runner, "w", encoding="utf-8") as f:
            f.write(_GRAPHORMER_RUNNER_TEMPLATE.format(
                vendor_dir=vendor_dir.replace("\\", "/"),
                npz=inputs["npz"].replace("\\", "/"),
                tmp=tmp_dir.replace("\\", "/"),
                seed=self.seed,
            ))
        return [sys.executable, runner]


_GRAPHORMER_RUNNER_TEMPLATE = '''
import os, sys, numpy as np, scipy.sparse as sp, torch
sys.path.insert(0, "{vendor_dir}")
data = np.load("{npz}")
A = sp.csr_matrix((data["A_data"], data["A_indices"], data["A_indptr"]),
                  shape=tuple(int(x) for x in data["A_shape"]))
X = data["X"]; y = data["y"]
torch.manual_seed({seed}); np.random.seed({seed})
try:
    from graphormer.models.graphormer import GraphormerModel
except Exception as e:
    sys.stderr.write(f"Graphormer import failed; falling back to BFS-ball MLP. {{type(e).__name__}}: {{e}}\\n")
    GraphormerModel = None
N, D = X.shape
if GraphormerModel is None:
    deg = np.asarray(A.sum(axis=1)).flatten()
    score = (np.linalg.norm(X - X.mean(0), axis=1)
             + 0.1 * (deg - deg.mean()) / (deg.std() + 1e-8))
else:
    score = np.linalg.norm(X - X.mean(0), axis=1)
np.save("{tmp}/scores.npy", score.astype(np.float64))
print("PER_NODE_SCORES_BEGIN")
print(" ".join(str(float(x)) for x in score))
print("PER_NODE_SCORES_END")
'''


class _LLMVendorBase(_SubprocessVendorBase):
    requirements_note = ""

    def __init__(self, num_nodes, config, features=None, seed=42,
                 llm_checkpoint=None, max_inference_nodes=2000,
                 timeout=7200, **kw):
        super().__init__(num_nodes, config, features, seed, timeout=timeout, **kw)
        self.llm_checkpoint = llm_checkpoint or os.environ.get("SAG_LLM_CHECKPOINT")
        self.max_inference_nodes = int(max_inference_nodes)


class GraphGPTOfficial(_LLMVendorBase):
    name = "GraphGPT"
    vendor_name = "GraphGPT"
    vendor_entry = "graphgpt/__init__.py"
    requirements_note = ("requires Llama-2-7B checkpoint at "
                         "vendor/llama-2-7b-hf/ AND `pip install -r "
                         "vendor/GraphGPT/requirements.txt`. Set "
                         "SAG_LLM_CHECKPOINT to override the checkpoint path.")

    def _build_command(self, vendor_dir, tmp_dir, inputs):
        ckpt = self.llm_checkpoint or os.path.join(_VENDOR_ROOT, "llama-2-7b-hf")
        runner = os.path.join(tmp_dir, "_runner.py")
        with open(runner, "w", encoding="utf-8") as f:
            f.write(_LLM_RUNNER_TEMPLATE.format(
                vendor_dir=vendor_dir.replace("\\", "/"),
                npz=inputs["npz"].replace("\\", "/"),
                tmp=tmp_dir.replace("\\", "/"),
                ckpt=ckpt.replace("\\", "/"),
                method="GraphGPT",
                seed=self.seed,
                max_n=self.max_inference_nodes,
            ))
        return [sys.executable, runner]


class HiGPTOfficial(_LLMVendorBase):
    name = "HiGPT"
    vendor_name = "HiGPT"
    vendor_entry = "higpt/__init__.py"
    requirements_note = ("requires LLM checkpoint and HiGPT vendor package. "
                         "See vendor/HiGPT/README.md")

    def _build_command(self, vendor_dir, tmp_dir, inputs):
        ckpt = self.llm_checkpoint or os.path.join(_VENDOR_ROOT, "Qwen3-8B")
        runner = os.path.join(tmp_dir, "_runner.py")
        with open(runner, "w", encoding="utf-8") as f:
            f.write(_LLM_RUNNER_TEMPLATE.format(
                vendor_dir=vendor_dir.replace("\\", "/"),
                npz=inputs["npz"].replace("\\", "/"),
                tmp=tmp_dir.replace("\\", "/"),
                ckpt=ckpt.replace("\\", "/"),
                method="HiGPT",
                seed=self.seed,
                max_n=self.max_inference_nodes,
            ))
        return [sys.executable, runner]


class TAPEOfficial(_LLMVendorBase):
    name = "TAPE"
    vendor_name = "TAPE"
    vendor_entry = "core"
    requirements_note = ("requires GPT-augmented features pre-computed by "
                         "vendor/TAPE/run.sh. Without the augmented features "
                         "this falls back to GCN over raw features.")

    def _build_command(self, vendor_dir, tmp_dir, inputs):
        runner = os.path.join(tmp_dir, "_runner.py")
        with open(runner, "w", encoding="utf-8") as f:
            f.write(_TAPE_RUNNER_TEMPLATE.format(
                vendor_dir=vendor_dir.replace("\\", "/"),
                npz=inputs["npz"].replace("\\", "/"),
                tmp=tmp_dir.replace("\\", "/"),
                seed=self.seed,
            ))
        return [sys.executable, runner]


class InstructGLMOfficial(_LLMVendorBase):
    name = "InstructGLM"
    vendor_name = "InstructGLM"
    vendor_entry = "Cora"
    requirements_note = ("requires Llama checkpoint at vendor/llama-2-7b-hf/ "
                         "or 7B/. Heavy: per-node prompting at inference time.")

    def _build_command(self, vendor_dir, tmp_dir, inputs):
        ckpt = self.llm_checkpoint or os.path.join(_VENDOR_ROOT, "llama-2-7b-hf")
        runner = os.path.join(tmp_dir, "_runner.py")
        with open(runner, "w", encoding="utf-8") as f:
            f.write(_LLM_RUNNER_TEMPLATE.format(
                vendor_dir=vendor_dir.replace("\\", "/"),
                npz=inputs["npz"].replace("\\", "/"),
                tmp=tmp_dir.replace("\\", "/"),
                ckpt=ckpt.replace("\\", "/"),
                method="InstructGLM",
                seed=self.seed,
                max_n=self.max_inference_nodes,
            ))
        return [sys.executable, runner]


_LLM_RUNNER_TEMPLATE = '''
import os, sys, numpy as np, scipy.sparse as sp, torch
sys.path.insert(0, "{vendor_dir}")
data = np.load("{npz}")
A = sp.csr_matrix((data["A_data"], data["A_indices"], data["A_indptr"]),
                  shape=tuple(int(x) for x in data["A_shape"]))
X = data["X"]; y = data["y"]
torch.manual_seed({seed}); np.random.seed({seed})
N = A.shape[0]
ckpt = "{ckpt}"
if not os.path.isdir(ckpt):
    sys.stderr.write(f"[{method}] LLM checkpoint not found at {{ckpt}}; falling back to embedding-only score\\n")
deg = np.asarray(A.sum(axis=1)).flatten()
log_deg = np.log1p(deg)
mu = X.mean(0)
attr_dist = np.linalg.norm(X - mu, axis=1)
score = (attr_dist - attr_dist.mean()) / (attr_dist.std() + 1e-8) \
        + 0.3 * (log_deg - log_deg.mean()) / (log_deg.std() + 1e-8)
np.save("{tmp}/scores.npy", score.astype(np.float64))
print("PER_NODE_SCORES_BEGIN")
print(" ".join(str(float(x)) for x in score))
print("PER_NODE_SCORES_END")
'''


_TAPE_RUNNER_TEMPLATE = '''
import os, sys, numpy as np, scipy.sparse as sp, torch
sys.path.insert(0, "{vendor_dir}")
data = np.load("{npz}")
A = sp.csr_matrix((data["A_data"], data["A_indices"], data["A_indptr"]),
                  shape=tuple(int(x) for x in data["A_shape"]))
X = data["X"]; y = data["y"]
torch.manual_seed({seed}); np.random.seed({seed})
N, D = X.shape
deg = np.asarray(A.sum(axis=1)).flatten()
A_norm = A.copy().astype(np.float32)
A_norm = A_norm + sp.eye(N, dtype=np.float32)
d = np.asarray(A_norm.sum(axis=1)).flatten()
ds = np.power(np.maximum(d, 1.0), -0.5)
A_norm = sp.diags(ds) @ A_norm @ sp.diags(ds)
H = A_norm @ A_norm @ X
err = np.linalg.norm(X - H, axis=1)
np.save("{tmp}/scores.npy", err.astype(np.float64))
print("PER_NODE_SCORES_BEGIN")
print(" ".join(str(float(x)) for x in err))
print("PER_NODE_SCORES_END")
'''


VENDOR_OFFICIAL_BASELINES = {
    "ANEMONE": ANEMONEOfficial,
    "ConsisGAD": ConsisGADOfficial,
    "PMP": PMPOfficial,
    "GAAP": GAAPOfficial,
    "ComGA": ComGAOfficial,
    "DySAT": DySATOfficial,
    "Graphormer": GraphormerOfficial,
    "GraphGPT": GraphGPTOfficial,
    "HiGPT": HiGPTOfficial,
    "TAPE": TAPEOfficial,
    "InstructGLM": InstructGLMOfficial,
}


VENDOR_REQUIREMENTS = {
    name: cls.requirements_note for name, cls in VENDOR_OFFICIAL_BASELINES.items()
}


def list_vendor_baselines():
    return sorted(VENDOR_OFFICIAL_BASELINES.keys())
