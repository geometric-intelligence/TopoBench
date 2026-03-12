from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

import numpy as np


class BaseSampler(ABC):
    """
    Abstract base class for sampling neighbor indices from training data.
    """

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        edge_index: Optional[np.ndarray] = None,
        train_mask: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> None:
        """
        Fit the sampler on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Training labels.
        edge_index : np.ndarray, optional
            Edge indices of shape (2, num_edges) for graph-based samplers.
        train_mask : np.ndarray, optional
            Boolean mask or indices of training nodes. Required by most samplers.
        **kwargs : Any
            Additional sampler-specific arguments.
        """
        pass

    def sample(self, x: np.ndarray, ids: list, **kwargs: Any) -> Sequence[int]:
        sampled_nodes = self._sample(x, ids, **kwargs)
        if type(sampled_nodes) != list:
            sampled_nodes = list(
                sampled_nodes.reshape(-1)
            )  # create a 1-D list of indexes, if it isn't a list
        seen = set()
        return [n for n in sampled_nodes if n not in seen and not seen.add(n)]

    @abstractmethod
    def _sample(
        self, x: np.ndarray, ids: list, **kwargs: Any
    ) -> Sequence[int]:
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
