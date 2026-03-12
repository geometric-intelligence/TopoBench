"""Regression wrappers: BaseRegressorWrapper and RegressorWrapper."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import mean_squared_error

from topobench.nn.wrappers.graph.sklearn.sklearn_wrappers.base import BaseWrapper


class BaseRegressorWrapper(BaseWrapper):
    """
    Thin base for regression: implements all abstract methods with
    regression semantics (predictions only, global_mean_ fallback).
    """

    def __init__(self, backbone: Any, sampler: Optional[Any] = None, **kwargs: Any):
        super().__init__(backbone, sampler=sampler, **kwargs)
        self.global_mean_: float = 0.0

    def _init_targets(self, y_train: np.ndarray) -> None:
        self.global_mean_ = float(np.mean(y_train))

    def _no_neighbors(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        preds = np.full(batch_size, self.global_mean_)
        return preds, preds

    def _one_neighbor(
        self, labels: np.ndarray, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        label = labels[0]
        preds = np.full(batch_size, label)
        return preds, preds

    def _all_features_constant(
        self, labels: np.ndarray, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        preds = np.full(batch_size, np.mean(labels))
        return preds, preds

    def _get_predictions(self, model: Any, X_test: np.ndarray) -> Tuple[list, list]:
        preds = model.predict(X_test)
        return list(preds), list(preds)

    def _process_output(
        self, output_tensor: torch.Tensor, num_dataset_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_tensor = output_tensor.view(-1, 1)
        empty_tensor = torch.zeros(
            (num_dataset_points, 1), device=output_tensor.device
        )
        return empty_tensor, output_tensor

    def _full_graph_training(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        self.backbone.fit(X_train, y_train)
        out = self.backbone.predict(X_test)
        return torch.from_numpy(out).float().to(device).view(-1, 1)

    def _update_progress_and_results(
        self,
        probs: list,
        predictions: list,
        true_labels: torch.Tensor,
        outputs: list,
        preds: list,
        trues: list,
        pbar_or_callback: Any,
    ) -> None:
        outputs.extend(predictions)
        trues.extend(list(true_labels.cpu().numpy()))
        preds.extend(predictions)
        pbar_or_callback.update(len(predictions))
        mse = mean_squared_error(outputs, trues)
        pbar_or_callback.set_postfix({"MSE": f"{mse:.2f}"})


class RegressorWrapper(BaseRegressorWrapper):
    """
    Wrapper for sklearn regression models in graph node prediction.

    Use this class directly. It extends BaseRegressorWrapper with no
    additional behavior; the name is the public API.
    """

    pass
