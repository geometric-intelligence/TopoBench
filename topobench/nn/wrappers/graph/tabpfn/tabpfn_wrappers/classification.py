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

    def _get_prediction(self, model, X) -> torch.Tensor:
        # Reshaping to fit the model's expected input shape
        X_reshaped = X.reshape(1, -1)
        raw_proba = model.predict_proba(X_reshaped)
        # Reshaping raw_proba to 1D
        raw_proba = raw_proba[0]

        # Map local class order → global class order
        full_proba = np.zeros(self.num_classes_, dtype=float)
        for i, cls in enumerate(model.classes_):
            full_proba[cls] = raw_proba[i]

        prob = torch.from_numpy(full_proba).float()
        pred = torch.tensor(self.classes_[full_proba.argmax()])
        return prob, pred

    def _calculate_metric_pbar(
        self, preds: list, trues: list
    ) -> tuple[str, float]:
        acc = accuracy_score(trues, preds)
        return "accuracy", acc

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

            # Predict probabilities for the ehole dataset (to allow compatibility with the rest of the code)
            output = self.backbone.predict_proba(node_features)

            output_tensor = (
                torch.from_numpy(output).float().to(batch["x_0"].device)
            )
            # TODO: output the counters
            return {
                "labels": batch["y"],
                "batch_0": batch["batch_0"],
                "x_0": output_tensor,
            }

        # Fit sampler
        self.sampler.fit(
            node_features, labels, edge_index=edge_index, train_mask=train_mask
        )

        outputs = []
        trues = []
        preds = []

        # create the progress bar once
        pbar = tqdm(
            total=len(test_mask),
            desc="Sampling and predicting",
            dynamic_ncols=True,
        )

        n_test_nodes_per_sample = 1
        for i in range(0, len(test_mask), n_test_nodes_per_sample):
            idx = test_mask[i : i + n_test_nodes_per_sample]
            # ─────────────────────────────────────────
            x_np = node_features[idx]
            # Training-set items → one-hot + skip
            # ─────────────────────────────────────────
            # if idx in train_mask:
            #     train_output = self._get_train_prediction(batch["y"][idx])
            #     outputs.append(torch.tensor(train_output, dtype=torch.float32))
            #     pbar.update(1)  # advance bar
            #     continue  # DO NOT count toward accuracy

            # ─────────────────────────────────────────
            # Sample neighbours
            # ─────────────────────────────────────────
            nbr_idx = self.sampler.sample(x=x_np, idx=idx)
            # Check if there’s any overlap
            assert set(nbr_idx).isdisjoint(set(val_mask)), (
                "nbr_idx contains indices in val_mask"
            )
            assert set(nbr_idx).isdisjoint(set(test_mask)), (
                "nbr_idx contains indices in test_mask"
            )

            if len(nbr_idx) == 0:
                # no neighbours → uniform distribution
                self.n_no_neighbors += 1
                no_neighborn_prob, no_neighborn_prediction = (
                    self._handle_no_neighbors()
                )
                no_neighborn_prediction = 0
                outputs.append(no_neighborn_prob)
                preds.append(no_neighborn_prediction)
                trues.append(batch.y[idx].item())

                # update bar
                pbar.update(1)
                metric_name, metric = self._calculate_metric_pbar(preds, trues)
                pbar.set_postfix({metric_name: f"{metric:.2%}"})
                continue

            X_nb = node_features[nbr_idx]
            y_nb = labels[nbr_idx]

            # ─────────────────────────────────────────
            # Handle constant columns
            # ─────────────────────────────────────────

            const_cols = get_constant_columns(X_nb)  # range = 0 ⇒ constant
            if np.all(const_cols):
                self.n_features_constant += 1
                no_neighborn_prob, no_neighborn_prediction = (
                    self._handle_no_neighbors()
                )

                no_neighborn_prob[y_nb] = 1
                no_neighborn_prediction = 0

                outputs.append(no_neighborn_prob.float())
                preds.append(torch.tensor(no_neighborn_prediction).float())
                trues.append(batch.y[idx].item())

                pbar.update(1)
                metric_name, metric = self._calculate_metric_pbar(preds, trues)
                pbar.set_postfix({metric_name: f"{metric:.2%}"})
                continue

            # ─────────────────────────────────────────
            # Handle the case when there is precisely one neighbour
            # ─────────────────────────────────────────
            if X_nb.shape[0] == 1:
                # TODO: For now just repeated code of above but should come up with smth clever here
                self.n_features_constant += 1

                no_neighborn_prob, no_neighborn_prediction = (
                    torch.zeros(self.classes_.shape[0]),
                    y_nb.item(),
                )
                no_neighborn_prob[y_nb] = 1

                outputs.append(no_neighborn_prob.float())
                preds.append(no_neighborn_prediction)
                trues.append(batch.y[idx].item())

                pbar.update(1)
                metric_name, metric = self._calculate_metric_pbar(preds, trues)
                pbar.set_postfix({metric_name: f"{metric:.2%}"})
                continue

            # ─────────────────────────────────────────
            # Fit backbone on neighbours
            # ─────────────────────────────────────────
            if np.any(const_cols):
                # Get rid of constant columns for train and tes
                X_nb = X_nb[:, ~const_cols]
                x_filtered = x_np[~const_cols]
            else:
                x_filtered = x_np

            self.backbone.fit(X_nb, y_nb)
            self.n_model_trained += 1

            prob, prediction = self._get_prediction(self.backbone, x_filtered)

            outputs.append(prob.float())
            preds.append(prediction.item())
            trues.append(batch.y[idx].item())

            # ─────────────────────────────────────────
            # Update progress bar
            # ─────────────────────────────────────────
            pbar.update(1)
            metric_name, metric = self._calculate_metric_pbar(preds, trues)
            pbar.set_postfix({metric_name: f"{metric:.2%}"})

        # close bar (optional when loop ends)
        pbar.close()

        # stack & return exactly like before
        prob_tensor = torch.stack(outputs).to(batch["x_0"].device)
        return {
            "labels": batch["y"],
            "batch_0": batch["batch_0"],
            "x_0": prob_tensor,
        }


def get_constant_columns(X: np.ndarray) -> np.ndarray:
    """
    Returns a boolean array indicating which columns are constant in X.
    If X has only one row, all columns are considered constant.

    Parameters:
    - X (np.ndarray): 2D input array (n_samples, n_features)

    Returns:
    - np.ndarray: Boolean array of shape (n_features,), True for constant columns
    """
    if X.shape[0] == 1:
        return np.ones(X.shape[1], dtype=bool) == 0  # Returns a list of False
    return np.ptp(X, axis=0) == 0  # range == 0 ⇒ constant
