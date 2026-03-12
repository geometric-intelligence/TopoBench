"""Classification wrappers: BaseClassifierWrapper and ClassifierWrapper."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from topobench.nn.wrappers.graph.sklearn.sklearn_wrappers.base import BaseWrapper


class BaseClassifierWrapper(BaseWrapper):
    """
    Thin base for classification: implements all abstract methods with
    classification semantics (probs + preds, num_classes_, fallbacks).
    """

    def _init_targets(self, y_train: np.ndarray) -> None:
        self.classes_, counts = np.unique(y_train, return_counts=True)
        self.num_classes_ = len(self.classes_)
        self.uniform_ = np.ones(self.num_classes_) / self.num_classes_
        self.class_distribution_ = counts / counts.sum()
        most_common_idx = np.argmax(counts)
        self.most_common_class_ = self.classes_[most_common_idx]

    def _no_neighbors(self, batch_size: int) -> Tuple[np.ndarray, list]:
        probs = np.tile(self.class_distribution_, (batch_size, 1))
        preds = np.full(batch_size, self.most_common_class_)
        return probs, list(preds)

    def _one_neighbor(
        self, labels: np.ndarray, batch_size: int
    ) -> Tuple[np.ndarray, list]:
        label = labels[0]
        one_hot = np.zeros(self.num_classes_, dtype=float)
        one_hot[label] = 1.0
        probs = np.tile(one_hot, (batch_size, 1))
        preds = np.full(batch_size, label)
        return probs, list(preds)

    def _all_features_constant(
        self, labels: np.ndarray, batch_size: int
    ) -> Tuple[np.ndarray, list]:
        counts = np.bincount(labels, minlength=self.num_classes_).astype(np.float64)
        total = counts.sum()
        probs_np = (
            np.ones(self.num_classes_, dtype=np.float64) / self.num_classes_
            if total == 0
            else counts / total
        )
        pred_idx = int(np.argmax(counts))
        probs = np.tile(probs_np.astype(np.float32), (batch_size, 1))
        preds = np.full(batch_size, pred_idx)
        return probs, list(preds)

    def _get_predictions(self, model: Any, X_test: np.ndarray) -> Tuple[list, list]:
        try:
            raw_proba = model.predict_proba(X_test)
        except AttributeError:
            y_pred = model.predict(X_test)
            raw_proba = np.eye(self.num_classes_, dtype=int)[y_pred]

        n_samples = raw_proba.shape[0]
        num_classes = self.num_classes_
        global_pos = {label: i for i, label in enumerate(self.classes_)}
        full_proba = np.zeros((n_samples, num_classes), dtype=float)
        for j, local_label in enumerate(model.classes_):
            gj = global_pos[local_label]
            full_proba[:, gj] = raw_proba[:, j]
        preds = [np.argmax(row) for row in full_proba]
        return list(full_proba), preds

    def _process_output(
        self, output_tensor: torch.Tensor, num_dataset_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        empty_tensor = torch.zeros(
            (num_dataset_points, self.num_classes_), device=output_tensor.device
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
        try:
            raw_proba = self.backbone.predict_proba(X_test)
        except AttributeError:
            y_pred = self.backbone.predict(X_test)
            raw_proba = np.eye(self.num_classes_, dtype=int)[y_pred]
        out = torch.from_numpy(raw_proba).float().to(device)
        if out.dim() == 1:
            out = out.view(-1, self.num_classes_)
        else:
            out = out.view(-1, self.num_classes_)
        return out

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
        y_true = list(true_labels.cpu().numpy())
        outputs.extend(probs)
        preds.extend(predictions)
        trues.extend(y_true)
        pbar_or_callback.update(len(predictions))
        acc = accuracy_score(trues, preds)
        pbar_or_callback.set_postfix({"accuracy": f"{acc:.2%}"})


class ClassifierWrapper(BaseClassifierWrapper):
    """
    Wrapper for sklearn classification models in graph node prediction.

    Use this class directly. It extends BaseClassifierWrapper with no
    additional behavior; the name is the public API.
    """

    pass
