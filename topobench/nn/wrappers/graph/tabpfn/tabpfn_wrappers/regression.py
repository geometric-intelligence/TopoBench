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
    
    def _handle_constant_features(self, y) -> tuple[torch.Tensor, torch.Tensor]:
        y_mean = y.mean()
        return y_mean, y_mean

    def _get_prediction(self, model, X) -> torch.Tensor:
        pred = torch.tensor(model.predict(X))
        return pred, pred

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
        outputs.append(prediction.float())
        trues.append(
            true_label.item() if hasattr(true_label, "item") else true_label
        )

        pbar.update(1)
        metric_name, metric = self._calculate_metric_pbar(preds, trues)
        pbar.set_postfix({metric_name: f"{metric:.2%}"})

    def _process_single_test_node(
        self, idx, node_features, labels, batch, val_mask, test_mask
    ):
        """Process a single test node and return probability and prediction"""
        x_np = node_features[idx]

        # Sample neighbours
        nbr_idx = self.sampler.sample(x=x_np, idx=idx[0])

        # Check if there's any overlap
        assert set(nbr_idx).isdisjoint(set(val_mask)), (
            "nbr_idx contains indices in val_mask"
        )
        assert set(nbr_idx).isdisjoint(set(test_mask)), (
            "nbr_idx contains indices in test_mask"
        )

        # Case 1: No neighbors
        if len(nbr_idx) == 0:
            _, pred = self._handle_no_neighbors()

            # Update the counter and return
            self.num_no_neighbors += 1
            return _, pred

        X_nb = node_features[nbr_idx]
        y_nb = labels[nbr_idx]

        # Case 2: All features are constant
        const_cols = self.get_constant_columns(X_nb)
        if np.all(const_cols):
            _, pred = self._handle_constant_features(y_nb)

            # Update the counter and return
            self.num_all_feat_constant += 1
            return _, pred

        # Case 3: Only one neighbor
        if X_nb.shape[0] == 1:
            pred = y_nb.item()

            # Update the counter and return
            self.num_one_neighbor += 1
            return _, pred

        # Case 4: Normal case - fit backbone on neighbors
        if np.any(const_cols):
            # Remove constant columns for train and test
            X_nb = X_nb[:, ~const_cols]
            x_filtered = x_np[:, ~const_cols]
        else:
            x_filtered = x_np

        self.backbone.fit(X_nb, y_nb)

        _, pred = self._get_prediction(self.backbone, x_filtered)

        # Update the counter and return
        self.num_model_trained += 1
        return _, pred

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
            output = self.backbone.predict(node_features)

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

            # create the progress bar once
            pbar = tqdm(
                total=len(test_mask),
                desc="Sampling and predicting",
                dynamic_ncols=True,
            )

            n_test_nodes_per_sample = 1
            for i in range(0, len(test_mask), n_test_nodes_per_sample):
                idx = test_mask[i : i + n_test_nodes_per_sample]

                # Process the test node
                _, prediction = self._process_single_test_node(
                    idx, node_features, labels, batch, val_mask, test_mask
                )

                # Update results and progress bar
                self._update_progress_and_results(
                    None, prediction, batch.y[idx], outputs, None, trues, pbar
                )

            # close bar (optional when loop ends)
            pbar.close()

            # stack & return exactly like before
            prob_tensor = torch.stack(outputs).to(batch["x_0"].device)

        # Prepare the output
        prob_logits = torch.zeros(batch["y"].shape[0], device=batch["x_0"].device)
        prob_logits[test_mask] = prob_tensor

        num_test_points = test_mask.shape[0]

        # Log the metrics calculated within wrapper
        self.log_model_stat(num_test_points)

        return {
            "labels": batch["y"],
            "batch_0": batch["batch_0"],
            "x_0": prob_logits,
        }
