import numpy as np
import scipy.sparse as sp


def _build_normalized_sparse(adj_sparse, device):
    import torch
    coo = sp.csr_matrix(adj_sparse).tocoo()
    n = coo.shape[0]
    edge_idx = np.vstack([coo.row, coo.col]).astype(np.int64)
    self_loops = np.vstack([np.arange(n), np.arange(n)]).astype(np.int64)
    all_edges = np.concatenate([edge_idx, self_loops], axis=1)
    indices = torch.from_numpy(all_edges).to(device)
    values = torch.ones(all_edges.shape[1], dtype=torch.float32, device=device)
    A = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense()
    ds = torch.pow(torch.clamp(deg, min=1.0), -0.5)
    rows = A.indices()[0]
    cols = A.indices()[1]
    norm_vals = ds[rows] * A.values() * ds[cols]
    A_norm = torch.sparse_coo_tensor(A.indices(), norm_vals, (n, n)).coalesce()
    return A_norm


class UnsupervisedGCNAutoencoder:
    def __init__(
        self,
        hidden_dim: int = 64,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        nonzero_weight: float = 3.0,
        edge_weight: float = 5.0,
        alpha: float = 0.7,
        seed: int = 42,
        device: str = "cpu",
        struct_neg_per_pos: int = 5,
        large_graph_threshold: int = 30000,
    ):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.nonzero_weight = nonzero_weight
        self.edge_weight = edge_weight
        self.alpha = alpha
        self.seed = seed
        self.device = device
        self.struct_neg_per_pos = struct_neg_per_pos
        self.large_graph_threshold = large_graph_threshold
        self.scores_ = None

    def fit_score(self, adj, features) -> np.ndarray:
        import torch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X_np = features.toarray() if sp.issparse(features) else np.asarray(features)
        X_np = X_np.astype(np.float32)
        n, d = X_np.shape

        X = torch.from_numpy(X_np).to(self.device)

        A_csr = sp.csr_matrix(adj)
        A_csr.eliminate_zeros()
        A_norm_sparse = _build_normalized_sparse(A_csr, self.device)

        large = n > self.large_graph_threshold

        sparsity = 1.0 - float((X_np != 0).mean())
        eta_attr = self.nonzero_weight if sparsity > 0.5 else 1.0
        B_attr = (X != 0).float() * (eta_attr - 1.0) + 1.0

        coo = A_csr.tocoo()
        edge_u = torch.from_numpy(coo.row.astype(np.int64)).to(self.device)
        edge_v = torch.from_numpy(coo.col.astype(np.int64)).to(self.device)

        if not large:
            A_orig = torch.from_numpy((A_csr.toarray() > 0).astype(np.float32)).to(self.device)
            B_struct = A_orig * (self.edge_weight - 1.0) + 1.0
        else:
            A_orig = None
            B_struct = None

        h = self.hidden_dim
        gen = torch.Generator(device=self.device).manual_seed(self.seed)

        def _xavier(rows, cols):
            std = np.sqrt(2.0 / (rows + cols))
            return torch.empty(rows, cols, device=self.device).normal_(0, std, generator=gen)

        W_enc1 = _xavier(d, h)
        W_enc2 = _xavier(h, h)
        W_dec1 = _xavier(h, h)
        W_dec2 = _xavier(h, d)
        for W in (W_enc1, W_enc2, W_dec1, W_dec2):
            W.requires_grad_(True)

        opt = torch.optim.Adam(
            [W_enc1, W_enc2, W_dec1, W_dec2],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        rng_neg = np.random.default_rng(self.seed)

        for ep in range(self.epochs):
            opt.zero_grad()
            H1 = torch.relu(torch.sparse.mm(A_norm_sparse, X) @ W_enc1)
            Z = torch.relu(torch.sparse.mm(A_norm_sparse, H1) @ W_enc2)
            H2 = torch.relu(torch.sparse.mm(A_norm_sparse, Z) @ W_dec1)
            X_hat = torch.sparse.mm(A_norm_sparse, H2) @ W_dec2

            attr_diff = (X - X_hat) * B_attr
            attr_loss = (attr_diff ** 2).sum(dim=1).sqrt().mean()

            if not large:
                A_hat = torch.sigmoid(Z @ Z.t())
                struct_diff = (A_hat - A_orig) * B_struct
                struct_loss = (struct_diff ** 2).sum(dim=1).sqrt().mean()
            else:
                pos_logits = (Z[edge_u] * Z[edge_v]).sum(dim=1)
                m_neg = edge_u.shape[0] * self.struct_neg_per_pos
                neg_u = torch.from_numpy(
                    rng_neg.integers(0, n, size=m_neg).astype(np.int64)
                ).to(self.device)
                neg_v = torch.from_numpy(
                    rng_neg.integers(0, n, size=m_neg).astype(np.int64)
                ).to(self.device)
                neg_logits = (Z[neg_u] * Z[neg_v]).sum(dim=1)
                pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    pos_logits, torch.ones_like(pos_logits)
                )
                neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    neg_logits, torch.zeros_like(neg_logits)
                )
                struct_loss = pos_loss + neg_loss

            loss = self.alpha * attr_loss + (1.0 - self.alpha) * struct_loss
            loss.backward()
            opt.step()

        with torch.no_grad():
            H1 = torch.relu(torch.sparse.mm(A_norm_sparse, X) @ W_enc1)
            Z = torch.relu(torch.sparse.mm(A_norm_sparse, H1) @ W_enc2)
            H2 = torch.relu(torch.sparse.mm(A_norm_sparse, Z) @ W_dec1)
            X_hat = torch.sparse.mm(A_norm_sparse, H2) @ W_dec2

            attr_err = ((X - X_hat) * B_attr).pow(2).sum(dim=1).sqrt()

            if not large:
                A_hat = torch.sigmoid(Z @ Z.t())
                struct_err = ((A_hat - A_orig) * B_struct).pow(2).sum(dim=1).sqrt()
            else:
                deg = np.asarray(A_csr.sum(axis=1)).flatten().astype(np.float32) + 1e-6
                deg_t = torch.from_numpy(deg).to(self.device)
                pos_logits = (Z[edge_u] * Z[edge_v]).sum(dim=1)
                pos_loss_per_edge = torch.nn.functional.binary_cross_entropy_with_logits(
                    pos_logits, torch.ones_like(pos_logits), reduction="none"
                )
                struct_err_v = torch.zeros(n, device=self.device)
                struct_err_v.index_add_(0, edge_u, pos_loss_per_edge)
                struct_err = struct_err_v / deg_t

            score = self.alpha * attr_err + (1.0 - self.alpha) * struct_err

        self.scores_ = score.detach().cpu().numpy().astype(np.float64)
        return self.scores_
