"""PyTorch GPU backend for the SAG streaming state evolution.

Mirrors src/models/state_evolution.py exactly (Algorithm 1 + Eq. state_evolution
of the paper) but with all state stored as torch tensors. Lets the entire
SAG pipeline run on GPU when ``Config.device='cuda'``.

Per-edge ops are inherently sequential (streaming semantics), so per-edge GPU
kernel-launch overhead can make small graphs slower than the NumPy backend.
This backend's purpose is GPU-pipeline compatibility (uniform evaluation with
GNN baselines, large-scale streams) rather than raw speedup on small graphs.
"""

import math

import numpy as np
import torch


class StableStateEvolutionTorch:
    def __init__(
        self,
        num_nodes: int,
        gamma: float = 0.95,
        beta_upd: float = 1.0,
        alpha: float = 1.0,
        delta: float = 1.0,
        h_max: float = None,
        use_decay: bool = True,
        use_stable: bool = True,
        use_exposure: bool = True,
        kernel_dim: int = 4,
        decay_lambdas=None,
        readout_weights=None,
        time_mode: str = "global",
        device: str = "cuda",
    ):
        self.num_nodes = int(num_nodes)
        self.gamma = float(gamma)
        self.beta_upd = float(beta_upd)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.use_decay = bool(use_decay)
        self.use_stable = bool(use_stable)
        self.use_exposure = bool(use_exposure)
        self.R = max(1, int(kernel_dim))
        self.time_mode = time_mode
        self.device = torch.device(device)

        if decay_lambdas is None:
            if self.R == 1:
                decay_lambdas = [-np.log(max(self.gamma, 1e-9))]
            else:
                gammas_default = np.array([0.9, 0.95, 0.99, 0.999])[: self.R]
                decay_lambdas = list(-np.log(gammas_default))
        lambdas_np = np.asarray(decay_lambdas, dtype=np.float64).reshape(-1)
        self.lambdas = torch.tensor(lambdas_np, dtype=torch.float64, device=self.device)
        self.gammas = torch.exp(-self.lambdas)
        self.basis = torch.ones(self.R, dtype=torch.float64, device=self.device)

        if readout_weights is None:
            readout_weights = np.ones(self.R, dtype=np.float64) / self.R
        self.readout = torch.tensor(np.asarray(readout_weights, dtype=np.float64).reshape(-1),
                                    dtype=torch.float64, device=self.device)

        if h_max is None:
            h_max = beta_upd / max(1.0 - float(self.gammas.max().item()), 1e-6)
        self.h_max = float(h_max)

        self.s = torch.zeros((num_nodes, self.R), dtype=torch.float64, device=self.device)
        self.exposure = torch.zeros(num_nodes, dtype=torch.int64, device=self.device)
        self.last_time = torch.zeros(num_nodes, dtype=torch.float64, device=self.device)

    def reset(self) -> None:
        self.s.zero_()
        self.exposure.zero_()
        self.last_time.zero_()

    def scores(self) -> np.ndarray:
        with torch.no_grad():
            sc = (self.s @ self.readout).clamp(max=self.h_max)
        return sc.detach().cpu().numpy()

    def per_timescale_scores(self) -> np.ndarray:
        return self.s.detach().cpu().numpy()


class StateEvolutionTorchNoStable(StableStateEvolutionTorch):
    def __init__(self, *args, **kwargs):
        kwargs["use_stable"] = False
        super().__init__(*args, **kwargs)


class StateEvolutionTorchNoDecay(StableStateEvolutionTorch):
    def __init__(self, *args, **kwargs):
        kwargs["use_decay"] = False
        super().__init__(*args, **kwargs)


class StateEvolutionTorchNoExposure(StableStateEvolutionTorch):
    def __init__(self, *args, **kwargs):
        kwargs["use_exposure"] = False
        super().__init__(*args, **kwargs)


def process_stream_torch(state: StableStateEvolutionTorch,
                         scorer,
                         edges: np.ndarray,
                         alpha: float, beta: float, delta: float,
                         rho: float, epsilon: float,
                         attr_vec_np=None,
                         passes: int = 1, shuffle: bool = True,
                         rng=None, edge_times: np.ndarray = None,
                         use_continuous_dev: bool = False,
                         dev_smoothing_alpha: float = 2.0,
                         window_size: int = 4096) -> int:
    """GPU torch port of SAG.process_stream_batch's inner loop.

    Returns total processed edge count.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    device = state.device
    n = state.num_nodes
    R = state.R

    inv_rho = 1.0 / rho
    log_rho = math.log(rho)

    p_dev = float(scorer._posterior_dev)
    p_stb = float(scorer._posterior_stable)
    p_span = p_dev - p_stb

    use_decay = state.use_decay
    use_stable = state.use_stable
    use_exp = state.use_exposure
    time_mode = state.time_mode
    h_max = state.h_max

    gammas = state.gammas
    lambdas = state.lambdas
    basis = state.basis

    s = state.s
    exposure = state.exposure
    last_time = state.last_time

    attr_vec = None
    if attr_vec_np is not None:
        attr_vec = torch.as_tensor(attr_vec_np, dtype=torch.float64, device=device)

    use_real_time = edge_times is not None and len(edge_times) == edges.shape[0]
    if use_real_time:
        edge_times_local = (np.asarray(edge_times, dtype=np.float64) -
                            float(np.min(edge_times)))
    else:
        edge_times_local = None

    total_processed = 0
    t_global = 0
    edges_np = np.asarray(edges, dtype=np.int64)

    for _ in range(passes):
        if shuffle:
            perm = np.arange(edges_np.shape[0])
            rng.shuffle(perm)
            edges_np = edges_np[perm]
            if use_real_time:
                edge_times_local = edge_times_local[perm]

        edge_prev = {}
        edge_curr = {}
        node_prev = np.zeros(n, dtype=np.float64)
        node_curr = np.zeros(n, dtype=np.float64)
        wcount = 0

        for i in range(edges_np.shape[0]):
            u = int(edges_np[i, 0]); v = int(edges_np[i, 1])
            k = (u, v) if u <= v else (v, u)
            edge_curr[k] = edge_curr.get(k, 0) + 1
            node_curr[u] += 1.0; node_curr[v] += 1.0
            wcount += 1

            cp_e = edge_prev.get(k, 0) + epsilon
            cc_e = edge_curr[k] + epsilon
            cp_u = node_prev[u] + epsilon; cc_u = node_curr[u] + epsilon
            cp_v = node_prev[v] + epsilon; cc_v = node_curr[v] + epsilon

            if use_continuous_dev:
                a_edge = abs(math.tanh(math.log((cc_e + dev_smoothing_alpha) /
                                                (cp_e + dev_smoothing_alpha)) / log_rho))
                a_node_u = abs(math.tanh(math.log((cc_u + dev_smoothing_alpha) /
                                                  (cp_u + dev_smoothing_alpha)) / log_rho))
                a_node_v = abs(math.tanh(math.log((cc_v + dev_smoothing_alpha) /
                                                  (cp_v + dev_smoothing_alpha)) / log_rho))
                a_e_u = max(a_edge, a_node_u)
                a_e_v = max(a_edge, a_node_v)
                p_base_u = p_stb + a_e_u * p_span
                p_base_v = p_stb + a_e_v * p_span
            else:
                a_e = 1 if (cc_e > rho * cp_e or cc_e < inv_rho * cp_e) else 0
                if a_e == 0:
                    if (cc_u > rho * cp_u or cc_u < inv_rho * cp_u
                            or cc_v > rho * cp_v or cc_v < inv_rho * cp_v):
                        a_e = 1
                p_base_u = p_dev if a_e == 1 else p_stb
                p_base_v = p_base_u

            for node, p_base in ((u, p_base_u), (v, p_base_v)):
                if attr_vec is not None:
                    boost = float(attr_vec[node].item())
                    p_node = p_base + (1.0 - p_base) * boost
                    if p_node > 1.0: p_node = 1.0
                else:
                    p_node = p_base
                if use_stable:
                    b_node = math.tanh(alpha * p_node)
                else:
                    x = alpha * p_node
                    b_node = x if -1.0 <= x <= 1.0 else (1.0 if x > 0 else -1.0)
                exposure[node] += 1
                if use_decay:
                    if time_mode == "per_update" or R == 1:
                        s[node].mul_(gammas)
                    else:
                        now_t = float(edge_times_local[i]) if use_real_time else float(t_global)
                        dt = now_t - float(last_time[node].item())
                        if dt > 0:
                            s[node].mul_(torch.exp(-lambdas * dt))
                        last_time[node] = now_t
                lam = int(exposure[node].item())
                g = (beta / (math.log1p(lam) + delta)) if use_exp else beta
                s[node].add_(basis * (g * b_node))

            t_global += 1
            if wcount >= window_size:
                edge_prev = edge_curr; edge_curr = {}
                node_prev = node_curr; node_curr = np.zeros(n, dtype=np.float64)
                wcount = 0

        total_processed += int(edges_np.shape[0])

    return total_processed
