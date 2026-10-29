import os
from abc import ABC, abstractmethod

import numpy as np

from ..utils import Timer, MemoryTracker, save_json


class BaseTrainer(ABC):
    def __init__(self, model, data_loader, config, logger=None):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.logger = logger

        self.timer = Timer()
        self.memory_tracker = MemoryTracker()
        self.current_epoch = 0
        self.history = {
            "train_loss": [],
            "metrics": [],
            "latency_ms": [],
            "memory_gb": [],
        }

    @abstractmethod
    def train_epoch(self, epoch: int):
        ...

    @abstractmethod
    def evaluate(self):
        ...

    def train(self, num_epochs: int, eval_freq: int = 1, save_freq: int = 10):
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            self.timer.reset()
            self.timer.start()
            self.memory_tracker.reset()

            train_result = self.train_epoch(epoch)

            self.timer.stop()
            self.memory_tracker.update()

            latency = self.timer.get_elapsed_ms()
            memory = self.memory_tracker.get_peak()

            self.history["train_loss"].append(train_result.get("loss", 0.0))
            self.history["latency_ms"].append(latency)
            self.history["memory_gb"].append(memory["peak_total_mem_gb"])

            if epoch % eval_freq == 0:
                metrics = self.evaluate()
                self.history["metrics"].append({"epoch": epoch, **metrics})
                if self.logger:
                    self.logger.info(
                        "Epoch %04d | AUROC: %.4f | AUPRC: %.4f | F1: %.4f | "
                        "Latency: %.2fms | Mem: %.2fGB"
                        % (
                            epoch,
                            metrics.get("auroc", 0.0),
                            metrics.get("auprc", 0.0),
                            metrics.get("macro_f1", 0.0),
                            latency,
                            memory["peak_total_mem_gb"],
                        )
                    )

            if epoch % save_freq == 0:
                self.save_checkpoint(epoch)

        return self.history

    def save_checkpoint(self, epoch: int):
        path = os.path.join(self.config.checkpoint_dir, f"checkpoint_epoch_{epoch}.json")
        state = {"epoch": epoch, "history": self.history, "config": self.config.to_dict()}
        save_json(state, path)

    def get_results(self):
        if not self.history["metrics"]:
            return {}
        best_idx = int(np.argmax([m["auroc"] for m in self.history["metrics"]]))
        return {
            "best_metrics": self.history["metrics"][best_idx],
            "final_metrics": self.history["metrics"][-1],
            "avg_latency_ms": float(np.mean(self.history["latency_ms"])),
            "avg_memory_gb": float(np.mean(self.history["memory_gb"])),
            "max_memory_gb": float(np.max(self.history["memory_gb"])),
            "history": self.history,
        }
