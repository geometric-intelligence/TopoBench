from abc import ABC, abstractmethod
from typing import Any, Dict
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

        assert self.use_embeddings or self.use_node_features, \
            "Either use_embeddings or use_node_features could be False, not both."
        
        self.n_no_neighbors = 0
        self.n_model_trained = 0
    
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
    def _get_train_prediction(self) -> torch.Tensor:
        """Return a the labels for the training set."""
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
            if is_classifier(self.backbone):
                output = self.backbone.predict_proba(node_features)
            elif is_regressor(self.backbone):
                output = self.backbone.predict(node_features)
            else:
                raise ValueError("Backbone must be a classifier or regressor.")
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
            total=len(node_features),
            desc="Sampling and predicting",
            dynamic_ncols=True,
        )

        for idx, x_np in enumerate(node_features):
            # ─────────────────────────────────────────
            # Training-set items → one-hot + skip
            # ─────────────────────────────────────────
            if idx in train_mask:
                train_output = self._get_train_prediction(batch["y"][idx])
                outputs.append(torch.tensor(train_output, dtype=torch.float32))
                pbar.update(1)  # advance bar
                continue  # DO NOT count toward accuracy

            # ─────────────────────────────────────────
            # Sample neighbours
            # ─────────────────────────────────────────
            nbr_idx = self.sampler.sample(x_np, idx)
             # Check if there’s any overlap
            assert set(nbr_idx).isdisjoint(set(val_mask)), "nbr_idx contains indices in val_mask"
            assert set(nbr_idx).isdisjoint(set(test_mask)), "nbr_idx contains indices in test_mask"

            if len(nbr_idx) == 0:
                # no neighbours → uniform distribution
                self.n_no_neighbors += 1
                no_neighborn_prob, no_neighborn_prediction = self._handle_no_neighbors()

                outputs.append(torch.from_numpy(no_neighborn_prob).float())
                preds.append(torch.tensor(no_neighborn_prediction).float())
                trues.append(batch.y[idx].item())

                # update bar
                pbar.update(1)
                metric_name, metric = self._calculate_metric_pbar(preds, trues)
                pbar.set_postfix({metric_name:f"{metric:.2%}"})
                continue

            X_nb = node_features[nbr_idx]
            y_nb = labels[nbr_idx]

            # ─────────────────────────────────────────
            # Handle constant columns
            # ─────────────────────────────────────────
            const_cols = np.ptp(X_nb, axis=0) == 0  # range = 0 ⇒ constant
            if np.all(const_cols):
                self.n_features_constant += 1
                no_neighborn_prob, no_neighborn_prediction = self._handle_no_neighbors()
               
                outputs.append(torch.from_numpy(no_neighborn_prob).float())
                preds.append(torch.tensor(no_neighborn_prediction).float())
                trues.append(batch.y[idx].item())

                pbar.update(1)
                metric_name, metric = self._calculate_metric_pbar(preds, trues)
                pbar.set_postfix({metric_name:f"{metric:.2%}"})
                continue

            # ─────────────────────────────────────────
            # Fit backbone on neighbours
            # ─────────────────────────────────────────
            if np.any(const_cols):
                X_nb = X_nb[:, ~const_cols]
                x_filtered = x_np[~const_cols]
            else:
                x_filtered = x_np

            model = deepcopy(self.backbone)
            model.fit(X_nb, y_nb)
            self.n_model_trained += 1

            prob, prediction = self._get_prediction(model, x_filtered)
               
            outputs.append(prob.float())
            preds.append(prediction.item())
            trues.append(batch.y[idx].item())

            # ─────────────────────────────────────────
            # Update progress bar
            # ─────────────────────────────────────────
            pbar.update(1)
            metric_name, metric = self._calculate_metric_pbar(preds, trues)
            pbar.set_postfix({metric_name:f"{metric:.2%}"})

        # close bar (optional when loop ends)
        pbar.close()

        # stack & return exactly like before
        prob_tensor = torch.stack(outputs).to(batch["x_0"].device)
        return {
            "labels": batch["y"],
            "batch_0": batch["batch_0"],
            "x_0": prob_tensor,
        }