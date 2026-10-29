import numpy as np

from .base_trainer import BaseTrainer
from ..evaluation import compute_all_metrics
from ..models import create_sag_model


class SAGTrainer(BaseTrainer):
    def __init__(self, model, data_loader, config, logger=None):
        super().__init__(model, data_loader, config, logger)
        order = getattr(config, "edge_order", "auto")
        if order == "auto":
            order = "shuffle" if bool(getattr(config, "has_ground_truth", False)) else "bfs"
        self.edges = data_loader.get_edge_stream(mode=order)
        self.labels = np.asarray(data_loader.labels)
        self.num_nodes = int(data_loader.stats["num_nodes"])
        self.rng = np.random.default_rng(config.seed)
        self._edge_order = order

    def train_epoch(self, epoch: int):
        if epoch == 1:
            self.model.reset()
        do_shuffle = (self._edge_order == "shuffle")
        if hasattr(self.model, "process_stream_batch"):
            self.model.process_stream_batch(self.edges, passes=1, shuffle=do_shuffle, rng=self.rng)
        else:
            if do_shuffle:
                self.rng.shuffle(self.edges)
            for u, v in self.edges:
                self.model.process_edge(int(u), int(v))
            if hasattr(self.model, "scorer"):
                self.model.scorer.advance_window()
        return {"loss": 0.0, "processed_edges": int(self.edges.shape[0])}

    def evaluate(self):
        scores = self.model.get_scores()
        return compute_all_metrics(self.labels, scores)

    def get_scores(self) -> np.ndarray:
        return self.model.get_scores()


class MultiRunSAGTrainer:
    def __init__(self, model_factory, data_loader, config, logger=None, n_runs: int = 10):
        self.model_factory = model_factory
        self.data_loader = data_loader
        self.config = config
        self.logger = logger
        self.n_runs = n_runs

        self.run_results = []
        self.all_scores = []

    def _build_model(self, seed: int):
        kwargs = {
            "num_nodes": self.data_loader.stats["num_nodes"],
            "config": self.config,
            "features": self.data_loader.features,
            "adj": self.data_loader.adj,
        }
        try:
            return self.model_factory(**kwargs)
        except TypeError:
            kwargs.pop("adj", None)
            try:
                return self.model_factory(**kwargs)
            except TypeError:
                kwargs.pop("features", None)
                return self.model_factory(**kwargs)

    def run(self, num_epochs: int):
        cfg = self.config
        for run_idx in range(self.n_runs):
            seed = cfg.seed + run_idx
            np.random.seed(seed)
            if self.logger:
                self.logger.info("=== Run %d/%d (seed=%d) ===" % (run_idx + 1, self.n_runs, seed))

            model = self._build_model(seed)
            trainer = SAGTrainer(model, self.data_loader, cfg, self.logger)
            trainer.rng = np.random.default_rng(seed)
            trainer.train(num_epochs=num_epochs, eval_freq=cfg.eval_freq, save_freq=num_epochs + 1)
            results = trainer.get_results()
            results["run_idx"] = run_idx
            results["seed"] = seed
            self.run_results.append(results)
            self.all_scores.append(trainer.get_scores())

            if self.logger:
                b = results["best_metrics"]
                self.logger.info(
                    "Run %d Best -> AUROC: %.4f | AUPRC: %.4f | F1: %.4f"
                    % (run_idx + 1, b["auroc"], b["auprc"], b["macro_f1"])
                )

        return self.aggregate_results()

    def aggregate_results(self):
        keys = ["auroc", "auprc", "macro_f1"]
        agg = {}
        for k in keys:
            vals = [r["best_metrics"][k] for r in self.run_results]
            agg[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "values": vals,
            }
        latency = [r["avg_latency_ms"] for r in self.run_results]
        memory = [r["max_memory_gb"] for r in self.run_results]
        agg["latency_ms"] = {
            "mean": float(np.mean(latency)),
            "std": float(np.std(latency, ddof=1)) if len(latency) > 1 else 0.0,
            "values": latency,
        }
        agg["memory_gb"] = {
            "mean": float(np.mean(memory)),
            "std": float(np.std(memory, ddof=1)) if len(memory) > 1 else 0.0,
            "values": memory,
        }
        agg["n_runs"] = self.n_runs
        agg["run_results"] = self.run_results
        return agg

    def get_formatted_results(self):
        a = self.aggregate_results()
        return {
            "auroc": "%.1f±%.1f" % (a["auroc"]["mean"] * 100, a["auroc"]["std"] * 100),
            "auprc": "%.1f±%.1f" % (a["auprc"]["mean"] * 100, a["auprc"]["std"] * 100),
            "macro_f1": "%.1f±%.1f" % (a["macro_f1"]["mean"] * 100, a["macro_f1"]["std"] * 100),
            "latency_ms": "%.1f±%.1f" % (a["latency_ms"]["mean"], a["latency_ms"]["std"]),
            "memory_gb": "%.2f±%.2f" % (a["memory_gb"]["mean"], a["memory_gb"]["std"]),
        }


def _sag_factory_full(num_nodes, config, features=None, adj=None):
    return create_sag_model(num_nodes, config, features=features, adj=adj, ablation="full")


SAG_FACTORY = _sag_factory_full
