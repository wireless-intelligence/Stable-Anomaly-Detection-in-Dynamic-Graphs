from abc import ABC, abstractmethod

import numpy as np


class BaselineModel(ABC):
    name = "Base"

    def __init__(self, num_nodes: int, config, features=None):
        self.num_nodes = num_nodes
        self.config = config
        self.features = features

    @abstractmethod
    def fit(self, adj, edges: np.ndarray, labels=None):
        ...

    @abstractmethod
    def score(self) -> np.ndarray:
        ...

    def fit_score(self, adj, edges: np.ndarray, labels=None) -> np.ndarray:
        self.fit(adj, edges, labels)
        return self.score()
