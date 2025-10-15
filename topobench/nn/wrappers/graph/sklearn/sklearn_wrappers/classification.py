from topobench.nn.wrappers.graph.sklearn.sklearn_wrappers.base import BaseWrapper
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from typing import Tuple


class ClassifierWrapper(BaseWrapper):
    def _init_targets(self, y_train):
        # Unique classes and number of classes
        self.classes_, counts = np.unique(y_train, return_counts=True)
        self.num_classes_ = len(self.classes_)

        # Uniform distribution over classes
        self.uniform_ = np.ones(self.num_classes_) / self.num_classes_

        # Class distribution (normalized counts)
        self.class_distribution_ = counts / counts.sum()

        # Most common class (mode)
        most_common_idx = np.argmax(counts)
        self.most_common_class_ = self.classes_[most_common_idx]

    def _no_neighborns(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle case with no neighbors by repeating the class distribution and most common class.

        Returns
        -------
        probs : np.ndarray
            Shape (batch_size, self.num_classes_),
            each row is self.class_distribution_.
        preds : np.ndarray
            Shape (batch_size,),
            each element is self.most_common_class_.
        """
        probs = np.tile(self.class_distribution_, (batch_size, 1))
        preds = np.full(batch_size, self.most_common_class_)
        return probs, list(preds)

    def _one_neighborn(
        self, labels: np.ndarray, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle case with exactly one neighbor.

        Parameters
        ----------
        labels : np.ndarray or torch.Tensor
            Label(s) of the neighbor(s). Should contain exactly one element.

        Returns
        -------
        probs : np.ndarray
            Shape (batch_size, self.num_classes_),
            each row is a one-hot vector for the neighbor's label.
        preds : np.ndarray
            Shape (batch_size,),
            each element is the neighbor's label.
        """
        # taking the label of the node
        label = labels[0]

        # One-hot vector for this label
        one_hot = np.zeros(self.num_classes_, dtype=float)
        one_hot[label] = 1.0

        # Repeat for all test nodes
        probs = np.tile(one_hot, (batch_size, 1))
        preds = np.full(batch_size, label)

        return probs, list(preds)

    def _all_features_constant(
        self, labels: np.ndarray, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        counts = np.bincount(labels, minlength=self.num_classes_).astype(
            np.float64
        )
        total = counts.sum()
        if total == 0:
            probs_np = (
                np.ones(self.num_classes_, dtype=np.float64)
                / self.num_classes_
            )
        else:
            probs_np = counts / total

        probs = probs_np.astype(np.float32)
        pred_idx = int(np.argmax(counts))

        # Repeat for all test nodes
        probs = np.tile(probs, (batch_size, 1))
        preds = np.full(batch_size, pred_idx)

        return probs, list(preds)

    def _get_predictions(self, model, X_test) -> torch.Tensor:
        """
        Map model's local class ordering to the wrapper's global class order (self.classes_)
        and support batch inputs X with shape (n_samples, n_features).

        Returns
        -------
        probs : torch.FloatTensor, shape (n_samples, num_classes)
            Class probabilities in global class order.
        preds : torch.Tensor, shape (n_samples,)
            Predicted class labels (assumes numeric labels).
        """

        # raw_proba: (n_samples, n_local_classes)
        raw_proba = model.predict_proba(X_test)

        n_samples = raw_proba.shape[0]
        num_classes = self.num_classes_

        # Build global index: label -> position in self.classes_
        # This avoids assuming labels are 0..K-1 or contiguous
        global_pos = {label: i for i, label in enumerate(self.classes_)}

        # Allocate and fill probs in global order
        full_proba = np.zeros((n_samples, num_classes), dtype=float)
        for j, local_label in enumerate(model.classes_):
            gj = global_pos[local_label]
            full_proba[:, gj] = raw_proba[:, j]

        # indices of the max in each inner list
        preds = [np.argmax(test_point) for test_point in full_proba]

        return list(full_proba), preds

    def _process_output(self, output_tensor, num_dataset_points):
        empty_tensor = torch.zeros((num_dataset_points, self.num_classes_)).to(
            output_tensor.device
        )
        return empty_tensor, output_tensor

    def _update_progress_and_results(
        self, probs, predictions, true_labels, outputs, preds, trues, pbar
    ):
        y_true = list(
            true_labels.cpu().numpy()
        )  # making sure that everything is a list

        """Helper method to update results and progress bar"""
        # update array containg prediction and label for infered nodes
        outputs.extend(probs)
        preds.extend(predictions)
        trues.extend(y_true)

        acc = accuracy_score(trues, preds)

        # update progress bar
        pbar.update(len(predictions))
        pbar.set_postfix({"accuracy": f"{acc:.2%}"})

    def _full_graph_training(self, X_train, y_train, X_test, device):
        # If sample is None training the network on the whole dataset
        self.backbone.fit(X_train, y_train)

        # Predict probabilities for the whole dataset (to allow compatibility with the rest of the code)
        output = self.backbone.predict_proba(X_test)

        prob_tensor = torch.from_numpy(output).float().to(device).view(-1, self.num_classes_)

        return prob_tensor

