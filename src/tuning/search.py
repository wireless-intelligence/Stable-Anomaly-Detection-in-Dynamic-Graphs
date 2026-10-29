import itertools
from copy import deepcopy

import numpy as np

from ..data import DataLoader
from ..evaluation import compute_all_metrics
from ..models import create_sag_model
from ..utils import save_json, set_seed


PARAM_GRID = {
    "gamma": [0.9, 0.95, 0.99],
    "beta_upd": [0.5, 1.0, 2.0],
    "alpha": [0.5, 1.0, 2.0],
    "delta": [0.5, 1.0, 2.0],
    "rho": [1.5, 2.0, 3.0],
}

PARAM_GRID_SMALL = {
    "gamma": [0.9, 0.95],
    "beta_upd": [0.5, 1.0],
    "alpha": [1.0, 2.0],
}

PARAM_GRID_FINE = {
    "gamma": [0.93, 0.94, 0.95, 0.96, 0.97],
    "beta_upd": [0.8, 0.9, 1.0, 1.1, 1.2],
    "alpha": [0.8, 0.9, 1.0, 1.1, 1.2],
    "delta": [0.8, 0.9, 1.0, 1.1, 1.2],
}


def _run_once(data_loader, config, seed):
    set_seed(seed)
    model = create_sag_model(
        num_nodes=data_loader.stats["num_nodes"],
        config=config,
        features=data_loader.features,
        adj=data_loader.adj,
    )
    edges = data_loader.get_edge_stream(shuffle=True)
    passes = 2 if data_loader.stats["num_edges"] > 50000 else 3
    model.process_stream_batch(edges, passes=passes, shuffle=True, rng=np.random.default_rng(seed))
    scores = model.get_scores()
    return compute_all_metrics(data_loader.labels, scores)


class GridSearchCV:
    def __init__(self, param_grid, data_loader, config, n_runs=3, num_epochs=None,
                 metric="auroc", logger=None):
        self.param_grid = param_grid
        self.data_loader = data_loader
        self.base_config = config
        self.n_runs = n_runs
        self.metric = metric
        self.logger = logger
        self.results = []
        self.best_params = None
        self.best_score = -np.inf
        self.best_result = None

    def _combos(self):
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        return [dict(zip(keys, c)) for c in itertools.product(*values)]

    def _evaluate(self, params):
        cfg = deepcopy(self.base_config)
        for k, v in params.items():
            setattr(cfg, k, v)
        scores = []
        for i in range(self.n_runs):
            m = _run_once(self.data_loader, cfg, cfg.seed + i)
            scores.append(m[self.metric])
        return {
            "params": params,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "all_scores": scores,
        }

    def search(self):
        for idx, params in enumerate(self._combos()):
            if self.logger:
                self.logger.info(f"[{idx+1}] {params}")
            r = self._evaluate(params)
            self.results.append(r)
            if r["mean_score"] > self.best_score:
                self.best_score = r["mean_score"]
                self.best_params = params
                self.best_result = r
                if self.logger:
                    self.logger.info(f"  * best {self.metric}={r['mean_score']:.4f}")
        return self.best_params, self.best_score

    def save_results(self, filepath):
        save_json(
            {"best_params": self.best_params, "best_score": self.best_score,
             "metric": self.metric, "n_runs": self.n_runs, "param_grid": self.param_grid,
             "all_results": [{"params": r["params"], "mean_score": r["mean_score"],
                              "std_score": r["std_score"], "all_scores": r["all_scores"]}
                             for r in self.results]},
            filepath,
        )


class RandomSearchCV:
    def __init__(self, param_distributions, data_loader, config, n_iter=20,
                 n_runs=3, num_epochs=None, metric="auroc", logger=None, seed=42):
        self.param_distributions = param_distributions
        self.data_loader = data_loader
        self.base_config = config
        self.n_iter = n_iter
        self.n_runs = n_runs
        self.metric = metric
        self.logger = logger
        self.seed = seed
        self.results = []
        self.best_params = None
        self.best_score = -np.inf

    def _sample(self, rng):
        p = {}
        for k, d in self.param_distributions.items():
            if isinstance(d, list):
                p[k] = d[rng.integers(0, len(d))]
            elif isinstance(d, tuple) and len(d) == 2:
                lo, hi = d
                if isinstance(lo, int) and isinstance(hi, int):
                    p[k] = int(rng.integers(lo, hi + 1))
                else:
                    p[k] = float(rng.uniform(lo, hi))
            elif callable(d):
                p[k] = d()
            else:
                p[k] = d
        return p

    def _evaluate(self, params):
        cfg = deepcopy(self.base_config)
        for k, v in params.items():
            setattr(cfg, k, v)
        scores = []
        for i in range(self.n_runs):
            m = _run_once(self.data_loader, cfg, cfg.seed + i)
            scores.append(m[self.metric])
        return {
            "params": params,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "all_scores": scores,
        }

    def search(self):
        rng = np.random.default_rng(self.seed)
        for i in range(self.n_iter):
            params = self._sample(rng)
            if self.logger:
                self.logger.info(f"[{i+1}/{self.n_iter}] {params}")
            r = self._evaluate(params)
            self.results.append(r)
            if r["mean_score"] > self.best_score:
                self.best_score = r["mean_score"]
                self.best_params = params
                if self.logger:
                    self.logger.info(f"  * best {self.metric}={r['mean_score']:.4f}")
        return self.best_params, self.best_score

    def save_results(self, filepath):
        save_json(
            {"best_params": self.best_params, "best_score": self.best_score,
             "metric": self.metric, "n_iter": self.n_iter, "n_runs": self.n_runs,
             "all_results": [{"params": r["params"], "mean_score": r["mean_score"],
                              "std_score": r["std_score"], "all_scores": r["all_scores"]}
                             for r in self.results]},
            filepath,
        )


class BayesianOptimization:
    def __init__(self, param_bounds, data_loader, config, n_iter=30,
                 n_runs=3, num_epochs=None, metric="auroc", logger=None, seed=42):
        self.param_bounds = param_bounds
        self.data_loader = data_loader
        self.base_config = config
        self.n_iter = n_iter
        self.n_runs = n_runs
        self.metric = metric
        self.logger = logger
        self.seed = seed
        self.X = []
        self.y = []
        self.results = []
        self.best_params = None
        self.best_score = -np.inf

    def _keys(self):
        return sorted(self.param_bounds.keys())

    def _params_to_arr(self, p):
        return np.asarray([p[k] for k in self._keys()])

    def _arr_to_params(self, a):
        return {k: float(a[i]) for i, k in enumerate(self._keys())}

    def _evaluate(self, params):
        cfg = deepcopy(self.base_config)
        for k, v in params.items():
            setattr(cfg, k, float(v))
        scores = []
        for i in range(self.n_runs):
            m = _run_once(self.data_loader, cfg, cfg.seed + i)
            scores.append(m[self.metric])
        return float(np.mean(scores))

    def _ei(self, candidates, xi=0.01):
        if len(self.X) < 2:
            return np.random.rand(len(candidates))
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        from scipy.stats import norm

        gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True)
        gp.fit(np.asarray(self.X), np.asarray(self.y))
        mu, sigma = gp.predict(candidates, return_std=True)
        best = np.max(self.y)
        imp = mu - best - xi
        z = imp / (sigma + 1e-9)
        ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma == 0.0] = 0.0
        return ei

    def search(self):
        rng = np.random.default_rng(self.seed)
        n_init = min(5, self.n_iter // 3)
        for i in range(n_init):
            p = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in self.param_bounds.items()}
            s = self._evaluate(p)
            self.X.append(self._params_to_arr(p))
            self.y.append(s)
            self.results.append({"params": p, "score": s})
            if s > self.best_score:
                self.best_score = s
                self.best_params = p
        for i in range(n_init, self.n_iter):
            cands = []
            for _ in range(1000):
                p = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in self.param_bounds.items()}
                cands.append(self._params_to_arr(p))
            cands = np.asarray(cands)
            ei = self._ei(cands)
            pick = cands[int(np.argmax(ei))]
            p = self._arr_to_params(pick)
            s = self._evaluate(p)
            self.X.append(pick)
            self.y.append(s)
            self.results.append({"params": p, "score": s})
            if s > self.best_score:
                self.best_score = s
                self.best_params = p
        return self.best_params, self.best_score

    def save_results(self, filepath):
        save_json(
            {"best_params": {k: float(v) for k, v in self.best_params.items()},
             "best_score": float(self.best_score), "metric": self.metric,
             "n_iter": self.n_iter, "param_bounds": self.param_bounds,
             "all_results": [{"params": {k: float(v) for k, v in r["params"].items()},
                              "score": float(r["score"])} for r in self.results]},
            filepath,
        )
