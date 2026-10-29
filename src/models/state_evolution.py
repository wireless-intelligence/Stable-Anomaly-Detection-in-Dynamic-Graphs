import numpy as np


class StableStateEvolution:
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
    ):
        self.num_nodes = num_nodes
        self.gamma = float(gamma)
        self.beta_upd = float(beta_upd)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.use_decay = use_decay
        self.use_stable = use_stable
        self.use_exposure = use_exposure
        self.R = max(1, int(kernel_dim))
        self.time_mode = time_mode

        if decay_lambdas is None:
            if self.R == 1:
                decay_lambdas = [-np.log(max(self.gamma, 1e-9))]
            else:
                # log-spaced gammas in [0.9, 0.999], extended for any R
                base4 = np.array([0.9, 0.95, 0.99, 0.999])
                if self.R <= 4:
                    gammas_default = base4[: self.R]
                else:
                    gammas_default = np.exp(np.linspace(
                        np.log(0.9), np.log(0.999), self.R))
                decay_lambdas = list(-np.log(gammas_default))
        self.lambdas = np.asarray(decay_lambdas, dtype=np.float64).reshape(-1)
        assert self.lambdas.shape[0] == self.R

        self.gammas = np.exp(-self.lambdas)
        self.basis = np.ones(self.R, dtype=np.float64)
        self.per_component_max = beta_upd / np.maximum(1.0 - self.gammas, 1e-6)

        if readout_weights is None:
            readout_weights = np.ones(self.R, dtype=np.float64) / self.R
        self.readout = np.asarray(readout_weights, dtype=np.float64).reshape(-1)
        assert self.readout.shape[0] == self.R

        if h_max is None:
            h_max = beta_upd / max(1.0 - max(self.gammas), 1e-6)
        self.h_max = float(h_max)

        self.s = np.zeros((num_nodes, self.R), dtype=np.float64)
        self.exposure = np.zeros(num_nodes, dtype=np.int64)
        self.last_time = np.zeros(num_nodes, dtype=np.float64)

    @property
    def h(self) -> np.ndarray:
        return self.scores()

    def _max_pool_per_timescale(self) -> np.ndarray:
        if self.R == 1:
            return self.s[:, 0].copy()
        ranks = np.empty_like(self.s)
        for i in range(self.R):
            col = self.s[:, i]
            order = np.argsort(col, kind="mergesort")
            r = np.empty_like(order, dtype=np.float64)
            r[order] = np.arange(col.shape[0], dtype=np.float64)
            if col.shape[0] > 1:
                r /= (col.shape[0] - 1)
            ranks[:, i] = r
        return ranks.max(axis=1)

    def reset(self) -> None:
        self.s.fill(0.0)
        self.exposure.fill(0)
        self.last_time.fill(0.0)

    def _bounded(self, delta_v: float) -> float:
        if self.use_stable:
            return float(np.tanh(self.alpha * delta_v))
        x = self.alpha * delta_v
        return float(max(-1.0, min(1.0, x)) if abs(x) > 1.0 else x)

    def _gain(self, node: int) -> float:
        if not self.use_exposure:
            return self.beta_upd
        lam = self.exposure[node]
        return self.beta_upd / (np.log1p(lam) + self.delta)

    def update_single(self, node: int, delta_v: float, t: float = None) -> float:
        self.exposure[node] += 1
        if self.use_decay:
            if t is None or self.time_mode == "per_update":
                self.s[node] *= self.gammas
            else:
                dt = float(t) - float(self.last_time[node])
                if dt > 0:
                    self.s[node] *= np.exp(-self.lambdas * dt)
                self.last_time[node] = float(t)
        g = self._gain(node)
        b = self._bounded(delta_v)
        self.s[node] += g * self.basis * b
        return float(self.s[node] @ self.readout)

    def update(self, nodes, delta_v) -> None:
        if np.isscalar(delta_v):
            for n in nodes:
                self.update_single(int(n), float(delta_v))
        else:
            for n, d in zip(nodes, delta_v):
                self.update_single(int(n), float(d))

    def exposure_gain(self, node: int) -> float:
        return self._gain(node)

    def scores(self) -> np.ndarray:
        return np.minimum(self.s @ self.readout, self.h_max)

    def per_timescale_scores(self) -> np.ndarray:
        return self.s.copy()


class StateEvolutionNoStable(StableStateEvolution):
    def __init__(self, *args, **kwargs):
        kwargs["use_stable"] = False
        super().__init__(*args, **kwargs)


class StateEvolutionNoDecay(StableStateEvolution):
    def __init__(self, *args, **kwargs):
        kwargs["use_decay"] = False
        super().__init__(*args, **kwargs)


class StateEvolutionNoExposure(StableStateEvolution):
    def __init__(self, *args, **kwargs):
        kwargs["use_exposure"] = False
        super().__init__(*args, **kwargs)
