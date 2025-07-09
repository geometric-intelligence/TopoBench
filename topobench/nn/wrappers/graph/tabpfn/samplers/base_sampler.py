from abc import ABC, abstractmethod
import numpy as np
from typing import Sequence, Optional
import networkx as nx

class BaseSampler(ABC):
    """
    Abstract base class for sampling neighbor indices from training data.
    """

    @abstractmethod
    def fit(
        self, X: np.ndarray, y: np.ndarray, graph: Optional[nx.Graph] = None
    ) -> None:
        """
        Fit the sampler on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Training labels.
        graph : networkx.Graph, optional
            Optional graph structure for graph-based samplers.
        """
        pass

    @abstractmethod
    def sample(self, x: np.ndarray, idx: int) -> Sequence[int]:
        """
        Return indices of neighbors for the sample at index `idx` or features `x`.

        Parameters
        ----------
        x : array-like of shape (n_features,)
            Feature vector of the sample.
        idx : int
            Index of the sample in the training set (used by graph samplers).

        Returns
        -------
        neighbors : Sequence[int]
            Indices of sampled neighbors.
        """
        pass