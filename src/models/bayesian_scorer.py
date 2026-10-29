import math
import numpy as np


class BayesianAnomalyScorer:
    def __init__(
        self,
        prior_anomaly_rate: float = 0.1,
        likelihood_anomaly: float = 0.9,
        likelihood_normal: float = 0.1,
        rho: float = 2.0,
        epsilon: float = 1e-6,
        window_size: int = 4096,
        num_nodes: int = None,
    ):
        self.prior_anomaly = prior_anomaly_rate
        self.prior_normal = 1.0 - prior_anomaly_rate
        self.pa = likelihood_anomaly
        self.pn = likelihood_normal
        self.rho = float(rho)
        self.log_rho = math.log(self.rho)
        self.epsilon = epsilon
        self.window_size = window_size
        self.num_nodes = num_nodes

        self._posterior_dev = self._bayes(1)
        self._posterior_stable = self._bayes(0)
        self.use_node_deviation = True

        self.reset()

    def _bayes(self, deviation_indicator: int) -> float:
        if deviation_indicator == 1:
            num = self.pa * self.prior_anomaly
            den = num + self.pn * self.prior_normal
        else:
            num = (1.0 - self.pa) * self.prior_anomaly
            den = num + (1.0 - self.pn) * self.prior_normal
        return num / (den + self.epsilon)

    def reset(self) -> None:
        self.edge_prev = {}
        self.edge_curr = {}
        if self.num_nodes is not None:
            self.node_prev = np.zeros(self.num_nodes, dtype=np.float64)
            self.node_curr = np.zeros(self.num_nodes, dtype=np.float64)
        else:
            self.node_prev = None
            self.node_curr = None
        self._window_count = 0
        self._global_mean_curr = 0.0

    def _key(self, u, v):
        return (u, v) if u <= v else (v, u)

    def record(self, u: int, v: int) -> None:
        k = self._key(u, v)
        self.edge_curr[k] = self.edge_curr.get(k, 0) + 1
        if self.node_curr is not None:
            self.node_curr[u] += 1.0
            self.node_curr[v] += 1.0
        self._window_count += 1
        if self._window_count >= self.window_size:
            self.advance_window()

    def advance_window(self) -> None:
        self.edge_prev = self.edge_curr
        self.edge_curr = {}
        if self.node_curr is not None:
            self.node_prev = self.node_curr
            self.node_curr = np.zeros(self.num_nodes, dtype=np.float64)
        self._window_count = 0

    def _edge_deviation(self, u: int, v: int) -> int:
        k = self._key(u, v)
        cp = self.edge_prev.get(k, 0) + self.epsilon
        cc = self.edge_curr.get(k, 0) + self.epsilon
        if cc > self.rho * cp or cc * self.rho < cp:
            return 1
        return 0

    def _node_deviation(self, node: int) -> int:
        if self.node_prev is None:
            return 0
        cp = self.node_prev[node] + self.epsilon
        cc = self.node_curr[node] + self.epsilon
        if cc > self.rho * cp or cc * self.rho < cp:
            return 1
        return 0

    def deviation_indicator(self, u: int, v: int) -> int:
        a_e = self._edge_deviation(u, v)
        if a_e == 1:
            return 1
        if self.use_node_deviation and (
            self._node_deviation(u) == 1 or self._node_deviation(v) == 1
        ):
            return 1
        return 0

    def posterior(self, deviation_indicator: int) -> float:
        return self._posterior_dev if deviation_indicator == 1 else self._posterior_stable

    def score_edge(self, u: int, v: int, external_boost: float = 0.0) -> float:
        self.record(u, v)
        a = self.deviation_indicator(u, v)
        p = self.posterior(a)
        if external_boost > 0.0:
            p = p + (1.0 - p) * float(external_boost)
        return float(min(max(p, 0.0), 1.0))


class NodeDegreeDeviation:
    def __init__(self, num_nodes: int, rho: float = 2.0, epsilon: float = 1e-6):
        self.num_nodes = num_nodes
        self.rho = rho
        self.epsilon = epsilon
        self.prev = np.zeros(num_nodes, dtype=np.float64)
        self.curr = np.zeros(num_nodes, dtype=np.float64)

    def reset(self) -> None:
        self.prev.fill(0.0)
        self.curr.fill(0.0)

    def record(self, node: int) -> None:
        self.curr[node] += 1.0

    def advance(self) -> None:
        self.prev = self.curr
        self.curr = np.zeros(self.num_nodes, dtype=np.float64)

    def deviation(self, node: int) -> int:
        cp = self.prev[node] + self.epsilon
        cc = self.curr[node] + self.epsilon
        ratio = cc / cp
        return int(ratio > self.rho or ratio < 1.0 / self.rho)
