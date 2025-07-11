import numpy as np
from typing import Sequence, Optional, Any
from sklearn.neighbors import NearestNeighbors
from topobench.nn.wrappers.graph.tabpfn.samplers.base_sampler import (
    BaseSampler,
)


class KNNSampler(BaseSampler):
    """
    Sampler that returns k nearest neighbors using sklearn's NearestNeighbors.
    """

    def __init__(self, k: int = 5, **kwargs) -> None:
        self.k = k
        self._nn: Optional[NearestNeighbors] = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        train_mask = kwargs.pop("train_mask", None)
        self._nn = NearestNeighbors(n_neighbors=self.k)
        self._nn.fit(X[train_mask])

    def sample(self, x: np.ndarray, idx: int = -1) -> Sequence[int]:
        if self._nn is None:
            raise RuntimeError("KNNSampler must be fitted before sampling.")
        # Query the k nearest neighbors for the feature vector x
        return self._nn.kneighbors(x.reshape(1, -1), return_distance=False)[0]
