from typing import Optional, Any
import numpy as np
from sklearn.metrics import mean_squared_error
from copy import deepcopy
import torch
from topobench.nn.wrappers.graph.tabpfn.tabpfn_wrappers.base import BaseWrapper
from tqdm import tqdm
from typing import Any, Dict, Union


class TabPFNRegressorWrapper(BaseWrapper):
    """
    Regression version of TabPFNWrapper:
    - always returns a scalar prediction per node
    - falls back to the global mean if no neighbours
    - trains backbone per‐node on its sampled neighbours
    """

    def __init__(self, backbone: Any, sampler: Optional[Any] = None, **kwargs):
        super().__init__(backbone, sampler=sampler, **kwargs)
        self.global_mean_: float = 0.0

    def _init_targets(self, y_train: np.ndarray) -> None:
        self.global_mean_ = torch.tensor(float(np.mean(y_train)))

    def _handle_no_neighbors(self):
        return self.global_mean_, self.global_mean_

    def _handle_constant_features(
        self, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Replace constant features with arrays filled with the mean of y.

        Args:
            y (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Two tensors with the same shape as y,
                                            filled with y.mean().
        """
        y_mean = y.mean()
        return torch.full_like(y, y_mean), torch.full_like(y, y_mean)

    def _get_predictions(self, model, X) -> torch.Tensor:
        preds = model.predict(X)
        return list(preds), list(preds)

    def _get_train_prediction(self, y) -> torch.Tensor:
        return y

    def _calculate_metric_pbar(
        self, preds: list, trues: list
    ) -> tuple[str, float]:
        mse = mean_squared_error(trues, preds)
        return "MSE", mse

    def _update_progress_and_results(
        self, prob, prediction, true_label, outputs, preds, trues, pbar
    ):
        """Helper method to update results and progress bar"""
        outputs.extend(prediction)
        true_label = list(
            true_label.cpu().numpy()
        )  # converting true label as a list
        trues.extend(true_label)

        pbar.update(len(prediction))
        metric_name, metric = self._calculate_metric_pbar(outputs, trues)
        pbar.set_postfix({metric_name: f"{metric:.2f}"})

    def _process_test_nodes(
        self, ids_test, node_features, labels, batch, val_mask, test_mask
    ):
        """Process a single test node and return probability and prediction"""
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
            _, preds = self._handle_no_neighbors()

            # Update the counter and return
            self.num_no_neighbors += 1
            return _, preds

        X_nb = node_features[nbr_idx]
        y_nb = labels[nbr_idx]

        # Case 2: All features are constant
        const_cols = self.get_constant_columns(X_nb)
        if np.all(const_cols):
            _, pred = self._handle_constant_features(y_nb)

            # Update the counter and return
            self.num_all_feat_constant += 1
            return None, torch.tensor(pred)

        # Case 3: Only one neighbor
        if X_nb.shape[0] == 1:
            preds = y_nb.item()

            # Update the counter and return
            self.num_one_neighbor += 1
            return None, preds

        # Case 4: All the node have the same label - return the label, to avoid TabPFN error
        if np.unique(y_nb).shape[0] == 1:
            preds = y_nb[0].item()

            # Update the counter and return
            self.num_all_same_label += 1
            return None, torch.tensor(preds)

        # Case 5: Normal case - fit backbone on neighbors
        if np.any(const_cols):
            # Remove constant columns for train and test
            X_nb = X_nb[:, ~const_cols]
            x_filtered = x_np[:, ~const_cols]
        else:
            x_filtered = x_np

        self.backbone.fit(X_nb, y_nb)

        _, preds = self._get_predictions(self.backbone, x_filtered)

        # Update the counter and return
        self.num_model_trained += len(ids_test)
        return _, preds

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
            output = self.backbone.predict(node_features[test_mask])

            # To ensure compatibility with the rest of the code
            self.num_model_trained += len(test_mask)

            prob_tensor = (
                torch.from_numpy(output)
                .float()
                .to(batch["x_0"].device)
                .view(-1, 1)
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

            # create the progress bar once
            pbar = tqdm(
                total=len(test_mask),
                desc="Sampling and predicting",
                dynamic_ncols=True,
            )

            batch_size = max(1, self.num_test_nodes)

            for start in range(0, len(test_mask), batch_size):
                idx_batch = test_mask[start : start + batch_size]

                # Process the test node
                _, predictions = self._process_test_nodes(
                    idx_batch,
                    node_features,
                    labels,
                    batch,
                    val_mask,
                    test_mask,
                )

                # Update results and progress bar
                self._update_progress_and_results(
                    None,
                    predictions,
                    batch.y[idx_batch],
                    outputs,
                    None,
                    trues,
                    pbar,
                )

            # close bar (optional when loop ends)
            pbar.close()

            # stack & return exactly like before
            prob_tensor = (
                torch.FloatTensor(outputs).to(batch["x_0"].device).view(-1, 1)
            )

        # Prepare the output
        prob_logits = torch.zeros(
            batch["y"].shape[0], 1, device=batch["x_0"].device
        ).to(prob_tensor.device)
        num_test_points = test_mask.shape[0]

        # Making sure that test mask is on the same device
        test_mask = torch.from_numpy(test_mask).to(prob_tensor.device)

        prob_logits[test_mask] = prob_tensor

        # Log the metrics calculated within wrapper
        self.log_model_stat(num_test_points)

        return {
            "labels": batch["y"],
            "batch_0": batch["batch_0"],
            "x_0": prob_logits,
        }
