from topobench.nn.wrappers.graph.tabpfn.tabpfn_wrappers.base import BaseWrapper
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from typing import Any, Dict, Union
from tqdm import tqdm


class TabPFNClassifierWrapper(BaseWrapper):
    def _init_targets(self, y_train):
        self.classes_ = np.unique(y_train)
        self.num_classes_ = len(self.classes_)
        self.uniform_ = np.ones(len(self.classes_)) / len(self.classes_)

    def _handle_no_neighbors(self):
        # return a uniform probability vector
        return torch.from_numpy(self.uniform_).float(), torch.zeros(
            self.num_classes_
        )

    def _get_predictions(self, model, X) -> torch.Tensor:
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
        raw_proba = model.predict_proba(X)

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

    def _calculate_metric_pbar(
        self, preds: list, trues: list
    ) -> tuple[str, float]:
        acc = accuracy_score(trues, preds)
        return "accuracy", acc

    def _update_progress_and_results(
        self, prob, prediction, true_label, outputs, preds, trues, pbar
    ):
        """Helper method to update results and progress bar"""
        outputs.append(prob.float())
        preds.append(
            prediction
            if isinstance(prediction, (int, float))
            else prediction.item()
        )
        trues.append(
            true_label.item() if hasattr(true_label, "item") else true_label
        )

        pbar.update(1)
        metric_name, metric = self._calculate_metric_pbar(preds, trues)
        pbar.set_postfix({metric_name: f"{metric:.2%}"})

    def _process_test_nodes(
        self, ids_test, node_features, labels, batch, val_mask, test_mask
    ):
        """Process test nodes and return probabilities and predictions"""
        x_np = node_features[ids_test]

        # Sample neighbours
        nbr_idx = self.sampler.sample(x=x_np, ids=ids_test)

        # Check if there's any overlap
        assert set(nbr_idx).isdisjoint(set(val_mask)), (
            "nbr_idx contains indices in val_mask"
        )
        assert set(nbr_idx).isdisjoint(set(test_mask)), (
            "nbr_idx contains indices in test_mask"
        )

        # Case 1: No neighbors
        if len(nbr_idx) == 0:
            prob, _ = self._handle_no_neighbors()
            prediction = 0

            # Update the counter and return
            self.num_no_neighbors += 1
            return prob, prediction

        X_nb = node_features[nbr_idx]
        y_nb = labels[nbr_idx]

        # Case 2: All features are constant
        const_cols = self.get_constant_columns(X_nb)
        if np.all(const_cols):
            prob, _ = self._handle_no_neighbors()
            prob[y_nb] = 1
            prediction = 0

            # Update the counter and return
            self.num_all_feat_constant += 1
            return prob, prediction

        # Case 3: Only one neighbor
        if X_nb.shape[0] == 1:
            prob = torch.zeros(self.classes_.shape[0])
            prob[y_nb] = 1
            prediction = y_nb.item()

            # Update the counter and return
            self.num_one_neighbor += 1
            return prob, prediction

        # Case 4: Normal case - fit backbone on neighbors
        if np.any(const_cols):
            # Remove constant columns for train and test
            X_nb = X_nb[:, ~const_cols]
            x_filtered = x_np[:, ~const_cols]
        else:
            x_filtered = x_np

        self.backbone.fit(X_nb, y_nb)

        probs, predictions = self._get_predictions(self.backbone, x_filtered)

        # Update the counter and return
        self.num_model_trained += len(ids_test)
        return probs, predictions

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        train_mask = batch.get("train_mask", None).cpu().numpy().copy()
        val_mask = batch.get("val_mask", None).cpu().numpy().copy()
        test_mask = batch.get("test_mask", None).cpu().numpy().copy()

        # encoded node features
        rank0_features = []

        if self.use_embeddings:
            all_keys = batch.keys()
            # adding all the rank0 features
            rank0_features.extend([s for s in all_keys if s.startswith("x0")])
        else:
            rank0_features = ["x0_0"]

        if self.use_node_features is False:
            rank0_features.remove("x0_0")  # remove x0_0 if using embeddings

        # Concatenate tensors along the column dimension (dim=1)
        tensors_to_concat = [batch[k] for k in rank0_features]
        tensor_features = torch.cat(tensors_to_concat, dim=1)

        node_features = tensor_features.cpu().numpy().copy()
        labels = batch["y"].cpu().numpy()

        X_train = node_features[train_mask].copy()
        y_train = batch["y"][train_mask].cpu().numpy().copy()

        # Record unique labels handling edge cases
        self._init_targets(y_train)

        edge_index = batch["edge_index"].cpu().numpy()

        if self.sampler is None:
            # If sample is None training the network on the whole dataset
            self.backbone.fit(X_train, y_train)

            # Predict probabilities for the whole dataset (to allow compatibility with the rest of the code)
            output = self.backbone.predict_proba(node_features[test_mask])

            # To ensure compatibility with the rest of the code
            self.num_model_trained += len(test_mask)

            prob_tensor = (
                torch.from_numpy(output).float().to(batch["x_0"].device)
            )

        else:
            # Fit sampler
            self.sampler.fit(
                node_features,
                labels,
                edge_index=edge_index,
                train_mask=train_mask,
            )

            outputs = []
            trues = []
            preds = []
            out_indices = []

            # create the progress bar once
            pbar = tqdm(
                total=len(test_mask),
                desc="Sampling and predicting",
                dynamic_ncols=True,
            )

            batch_size = max(1, self.num_test_nodes)

            for start in range(0, len(test_mask), batch_size):
                idx_batch = test_mask[start : start + batch_size]

                # keep one-model-per-test-node logic; iterate within the batch
                probs, predictions = self._process_test_nodes(
                    idx_batch,
                    node_features,
                    labels,
                    batch,
                    val_mask,
                    test_mask,
                )

                outputs.extend(probs)

                # progress/metric
                y_true = list(labels[idx_batch])
                preds.extend(predictions)
                trues.extend(y_true)
                metric_name, metric = self._calculate_metric_pbar(preds, trues)
                pbar.update(len(idx_batch))
                pbar.set_postfix({metric_name: f"{metric:.2%}"})

            # close bar (optional when loop ends)
            pbar.close()

        # stack & return exactly like before
        prob_tensor = torch.FloatTensor(outputs).to(batch["x_0"].device)

        # Prepare the output
        prob_logits = torch.zeros(batch["y"].shape[0], self.num_classes_).to(
            prob_tensor.device
        )

        # Making sure that test mask is on the same device
        test_mask = torch.from_numpy(test_mask).to(prob_tensor.device)
        prob_logits[test_mask] = prob_tensor

        num_test_points = test_mask.shape[0]
        # Log the metrics calculated within wrapper
        self.log_model_stat(num_test_points)

        return {
            "labels": batch["y"],
            "batch_0": batch["batch_0"],
            "x_0": prob_logits,
        }
