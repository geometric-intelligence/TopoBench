from typing import Optional, Any
import numpy as np
import torch
from topobench.nn.wrappers.graph.sklearn.sklearn_wrappers.base import BaseWrapper
from typing import Any
from sklearn.metrics import mean_squared_error


class RegressorWrapper(BaseWrapper):
    """
    Regression version of ClassificationWrapper:
    - always returns a scalar prediction per node
    - falls back to the global mean if no neighbours
    - trains backbone per‐node on its sampled neighbours
    """

    def __init__(self, backbone: Any, sampler: Optional[Any] = None, **kwargs):
        super().__init__(backbone, sampler=sampler, **kwargs)
        self.global_mean_: float = 0.0

    def _init_targets(self, y_train: np.ndarray) -> None:
        self.global_mean_ = float(np.mean(y_train))

    def _no_neighborns(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        # Fill with global mean for all test nodes
        return np.full((batch_size), self.global_mean_), np.full(
            (batch_size,), self.global_mean_
        )

    def _one_neighborn(
        self, labels: np.ndarray, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        # Repeat the neighbor label for all test nodes
        label = labels[0]
        return np.full((batch_size,), label), np.full((batch_size,), label)

    def _all_features_constant(
        self, labels: np.ndarray, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        # Use the average of labels as constant prediction
        labels_mean = np.mean(labels)
        return np.full((batch_size,), labels_mean), np.full(
            (batch_size,), labels_mean
        )

    def _get_predictions(self, model, X) -> tuple[np.ndarray, np.ndarray]:
        preds = model.predict(X)
        return list(preds), list(preds)

    def _process_output(self, output_tensor, num_dataset_points):
        output_tensor = output_tensor.view(-1, 1)
        empty_tensor = torch.zeros((num_dataset_points, 1)).to(
            output_tensor.device
        )
        return empty_tensor, output_tensor

    def _update_progress_and_results(
        self, probs, predictions, true_labels, outputs, preds, trues, pbar
    ):
        """Helper method to update results and progress bar"""
        outputs.extend(predictions)
        true_labels = list(
            true_labels.cpu().numpy()
        )  # converting true label as a list
        trues.extend(true_labels)

        pbar.update(len(predictions))
        mse = mean_squared_error(outputs, trues)
        pbar.set_postfix({"MSE": f"{mse:.2f}"})

    def _full_graph_training(self, X_train, y_train, X_test, device):
        # If sample is None training the network on the whole dataset
        self.backbone.fit(X_train, y_train)

        # Predict probabilities for the whole dataset (to allow compatibility with the rest of the code)
        output = self.backbone.predict(X_test)

        prob_tensor = torch.from_numpy(output).float().to(device).view(-1, 1)

        return prob_tensor