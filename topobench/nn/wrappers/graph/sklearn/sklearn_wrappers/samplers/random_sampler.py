import numpy as np
from typing import Sequence, Optional, Any
from topobench.nn.wrappers.graph.sklearn.sklearn_wrappers.samplers.base_sampler import (
    BaseSampler,
)


class RandomSampler(BaseSampler):
    """
    Sampler that returns k random nodes from the training set.

    Notes
    -----
    - `fit()` expects `train_mask` in kwargs.
    - `_sample(x, ids)` returns an array of shape (len(ids), k).
      (If you pass a single id, it returns shape (1, k).)
    """

    def __init__(
        self,
        k: int = 5,
        replace: bool = False,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.k = int(k)
        self.replace = bool(replace)
        self.random_state = random_state

        self.train_mask: Optional[np.ndarray] = None
        self._rng: Optional[np.random.Generator] = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        train_mask = kwargs.pop("train_mask", None)
        if train_mask is None:
            raise RuntimeError(
                "'train_mask' must be provided in fit() as a keyword argument."
            )

        train_mask = np.asarray(train_mask)
        if train_mask.ndim != 1:
            raise ValueError(f"train_mask must be 1D, got shape {train_mask.shape}")

        if self.k <= 0:
            raise ValueError(f"k must be > 0, got {self.k}")

        if (not self.replace) and self.k > len(train_mask):
            raise ValueError(
                f"Cannot sample k={self.k} without replacement from "
                f"{len(train_mask)} training nodes."
            )

        self.train_mask = train_mask
        self._rng = np.random.default_rng(self.random_state)

    def _sample(self, x: np.ndarray, ids: list = -1) -> Sequence[int]:
        if self.train_mask is None or self._rng is None:
            raise RuntimeError("RandomSampler must be fitted before sampling.")

        # Support both a single id and a list of ids
        n_targets = len(ids)

        sampled = self._rng.choice(
            self.train_mask,
            size=(n_targets, self.k),
            replace=self.replace,
        )

        return sampled
