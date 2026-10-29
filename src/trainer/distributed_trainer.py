import concurrent.futures
import os

import numpy as np

from .sag_trainer import SAGTrainer
from ..data import DataLoader
from ..models import create_sag_model
from ..utils import set_seed


class DataParallelSAGTrainer:
    def __init__(self, config, dataset_name: str, num_workers: int = None):
        self.config = config
        self.dataset_name = dataset_name
        self.num_workers = num_workers or max(1, os.cpu_count() // 2)

    def _run_once(self, seed: int, num_epochs: int):
        set_seed(seed)
        loader = DataLoader(
            data_dir=self.config.data_dir,
            dataset_name=self.dataset_name,
            seed=seed,
        )
        loader.load()
        model = create_sag_model(
            num_nodes=loader.stats["num_nodes"],
            config=self.config,
            features=loader.features,
            adj=loader.adj,
        )
        trainer = SAGTrainer(model, loader, self.config)
        trainer.rng = np.random.default_rng(seed)
        trainer.train(num_epochs=num_epochs, eval_freq=self.config.eval_freq, save_freq=num_epochs + 1)
        return trainer.get_results()

    def train(self, num_epochs: int, n_runs: int = 10):
        seeds = [self.config.seed + i for i in range(n_runs)]
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = [pool.submit(self._run_once, s, num_epochs) for s in seeds]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())
        return self._aggregate(results)

    def _aggregate(self, results):
        keys = ["auroc", "auprc", "macro_f1"]
        out = {}
        for k in keys:
            vals = [r["best_metrics"][k] for r in results]
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
        out["n_runs"] = len(results)
        out["all_results"] = results
        return out


DistributedSAGTrainer = DataParallelSAGTrainer
