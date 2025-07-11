from abc import ABC, abstractmethod
from typing import Any, Dict, Union
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from sklearn.base import is_classifier, is_regressor


class BaseWrapper(torch.nn.Module, ABC):
    def __init__(self, backbone: Any, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.use_embeddings = kwargs.get("use_embeddings", True)
        self.use_node_features = kwargs.get("use_node_features", True)
        self.sampler = kwargs.get("sampler", None)

        assert self.use_embeddings or self.use_node_features, (
            "Either use_embeddings or use_node_features could be False, not both."
        )

        self.n_no_neighbors = 0
        self.n_model_trained = 0
        self.n_features_constant = 0

    @abstractmethod
    def _init_targets(self, y_train: np.ndarray):
        """Called after you know your train targets; e.g. compute self.global_default."""
        ...

    @abstractmethod
    def _handle_no_neighbors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a default prediction when len(nbr_idx) == 0."""
        ...

    @abstractmethod
    def _get_prediction(self, model, X) -> tuple[torch.Tensor, torch.Tensor]:
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
