import numpy as np
from typing import Sequence, Optional, Any
from sklearn.neighbors import NearestNeighbors
from topobench.nn.wrappers.graph.sklearn.sklearn_wrappers.samplers.base_sampler import (
    BaseSampler,
)


class KNNSampler(BaseSampler):
    """
    Sampler that returns k nearest neighbors using sklearn's NearestNeighbors.
    """

    def __init__(self, k: int = 5, **kwargs) -> None:
        self.k = k
        self._nn: Optional[NearestNeighbors] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        edge_index: Optional[np.ndarray] = None,
        train_mask: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> None:
        if train_mask is None:
            raise ValueError("KNNSampler requires train_mask to be provided in fit().")
        train_arr = np.asarray(train_mask)
        self.train_mask = np.where(train_arr)[0] if train_arr.dtype == bool else train_arr.reshape(-1)
        self._nn = NearestNeighbors(n_neighbors=self.k)
        self._nn.fit(X[self.train_mask])

    def _sample(self, x: np.ndarray, ids: list = -1) -> Sequence[int]:
        if self._nn is None:
            raise RuntimeError("KNNSampler must be fitted before sampling.")
        # Query the k nearest neighbors for the feature vector x
        # obtaining the indices of the k nearest neighbors considering only the training set
        train_indexed = self._nn.kneighbors(
            x.reshape(len(ids), -1), return_distance=False
        )[0]
        # Converting the indexes to the original indices  of the graph
        graph_indexes = self.train_mask[train_indexed]
        return graph_indexes
