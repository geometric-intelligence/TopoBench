from abc import ABC, abstractmethod
from typing import Any, Dict, Union
import torch
import numpy as np
import math


class BaseWrapper(torch.nn.Module, ABC):
    def __init__(self, backbone: Any, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.use_embeddings = kwargs.get("use_embeddings", True)
        self.use_node_features = kwargs.get("use_node_features", True)
        self.sampler = kwargs.get("sampler", None)
        self.num_test_nodes = kwargs.get("num_test_nodes", 1)

        assert self.use_embeddings or self.use_node_features, (
            "Either use_embeddings or use_node_features could be False, not both."
        )
        self.logger = kwargs.get("logger", None)
        # Initialize the counters
        self.num_no_neighbors = 0
        self.num_one_neighbor = 0
        self.num_all_same_label = 0
        self.num_all_feat_constant = 0
        self.num_model_trained = 0

    @abstractmethod
    def _init_targets(self, y_train: np.ndarray):
        """Called after you know your train targets; e.g. compute self.global_default."""
        ...

    @abstractmethod
    def _handle_no_neighbors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a default prediction when len(nbr_idx) == 0."""
        ...

    @abstractmethod
    def _get_predictions(self, model, X) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a the prediction of the model."""
        ...

    @abstractmethod
    def _calculate_metric_pbar(
        self, preds: list, y_nb: list
    ) -> tuple[str, float]:
        """
        Given prediction and true labels, calculate a metric and return its name and value.
        This is used to update the tqdm progress bar.
        """
        ...

    def fit(self, x: np.ndarray, y: np.ndarray):
        # common: store target dtype, do nothing else
        return self

    @abstractmethod
    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Implement forward pass"""

    def log_model_stat(self, num_test_points):
        total_ratio = (
            self.num_no_neighbors / num_test_points
            + self.num_one_neighbor / num_test_points
            + self.num_all_feat_constant / num_test_points
            + self.num_model_trained / num_test_points
            + self.num_all_same_label / num_test_points
        )

        assert math.isclose(total_ratio, 1.0, rel_tol=1e-9, abs_tol=1e-6), (
            f"The sum of the ratios should be 1 (within tolerance), but got {total_ratio:.10f}"
        )
        self.logger(
            "test/no_neighbors",
            np.round((100 * self.num_no_neighbors / num_test_points), 2),
            prog_bar=True,
            on_step=False,
        )
        self.logger(
            "test/one_neighbor",
            np.round((100 * self.num_one_neighbor / num_test_points), 2),
            prog_bar=True,
            on_step=False,
        )
        self.logger(
            "test/all_features_constant",
            np.round((100 * self.num_all_feat_constant / num_test_points), 2),
            prog_bar=True,
            on_step=False,
        )
        self.logger(
            "test/num_all_same_label",
            np.round((100 * self.num_all_same_label / num_test_points), 2),
            prog_bar=True,
            on_step=False,
        )
        self.logger(
            "test/model_trained",
            np.round((100 * self.num_model_trained / num_test_points), 2),
            prog_bar=True,
            on_step=False,
        )

    def get_constant_columns(self, X: np.ndarray) -> np.ndarray:
        """
        Returns a boolean array indicating which columns are constant in X.
        If X has only one row, all columns are considered constant.

        Parameters:
        - X (np.ndarray): 2D input array (n_samples, n_features)

        Returns:
        - np.ndarray: Boolean array of shape (n_features,), True for constant columns
        """
        if X.shape[0] == 1:
            return (
                np.ones(X.shape[1], dtype=bool) == 0
            )  # Returns a list of False
        return np.ptp(X, axis=0) == 0  # range == 0 ⇒ constant
