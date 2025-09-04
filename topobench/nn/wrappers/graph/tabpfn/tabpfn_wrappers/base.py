from abc import ABC, abstractmethod
from typing import Any, Dict, Union, Tuple
import torch
import numpy as np
import math
from tqdm import tqdm
from copy import deepcopy

from topobench.nn.wrappers.graph.tabpfn.tabpfn_wrappers.components import (
    Checker,
    SafePredictor,
    LoggerStats,
)


class BaseWrapper(torch.nn.Module, ABC):
    def __init__(self, backbone: Any, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.use_embeddings = kwargs.get("use_embeddings", True)
        self.use_node_features = kwargs.get("use_node_features", True)
        self.sampler = kwargs.get("sampler", None)
        self.num_test_nodes = kwargs.get("num_test_nodes", 1)

        assert self.use_embeddings or self.use_node_features, (
            "Either use_embeddings or use_node_features could be False, not both."
        )
        self.logger = kwargs.get("logger", None)

        # initializzate components
        self.checker = Checker()
        self.stats = LoggerStats()
        self.safe = SafePredictor(
            checker=self.checker,
            backbone_factory=lambda: deepcopy(self.backbone),
            fallback_no_neighbors=self._no_neighborns,  # must return (B,C), (B,)
            fallback_one_neighbor=self._one_neighborn,  # must accept y_nb
            fallback_all_const=self._all_features_constant,  # must accept y_nb
            get_predictions=self._get_predictions,  # must accept (model, X_test)
        )

    @abstractmethod
    def _init_targets(self, y_train: np.ndarray):
        """Called after you know your train targets; e.g. compute self.global_default."""
        ...

    @abstractmethod
    def _no_neighborns(self, y_train: np.ndarray): ...

    @abstractmethod
    def _one_neighborn(self, y_train: np.ndarray): ...

    @abstractmethod
    def _all_features_constant(self, y_train: np.ndarray): ...

    @abstractmethod
    def _get_predictions(
        self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray
    ): ...

    @abstractmethod
    def _process_output(
        self, output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def _full_graph_training(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a default prediction when len(neighbors_id) == 0."""
        ...

    @abstractmethod
    def _update_progress_and_results(
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
        """Implement forward pass"""
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
            prob_tensor = self._full_graph_training(
                X_train, y_train, node_features[test_mask], batch["x_0"].device
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
                test_ids = test_mask[start : start + batch_size]

                # Sample neighbours
                neighbors_id = self.sampler.sample(
                    x=node_features[test_ids], ids=test_ids
                )

                probs, predictions, case = self.safe.predict_batch(
                    neighbors_id=neighbors_id,
                    test_ids=test_ids,
                    node_features=node_features,
                    labels=labels,
                    val_mask=val_mask,
                    test_mask=test_mask,
                )

                # updating stats
                self.stats.inc(case, n=len(test_ids))

                # Update results and progress bar
                self._update_progress_and_results(
                    probs,
                    predictions,
                    batch.y[test_ids],
                    outputs,
                    preds,
                    trues,
                    pbar,
                )

            # close bar (optional when loop ends)
            pbar.close()

        # stack & return exactly like before
        outputs_tensor = torch.FloatTensor(outputs).to(batch["x_0"].device)

        # Prepare the output
        empty_tensor, outputs_tensor = self._process_output(
            output_tensor=outputs_tensor, num_dataset_points=len(labels)
        )

        # Making sure that test mask is on the same device
        test_mask = torch.from_numpy(test_mask).to(outputs_tensor.device)
        empty_tensor[test_mask] = outputs_tensor

        # logging everything on WandB
        self.stats.log(num_test_points=len(test_mask), logger=self.logger)

        return {
            "labels": batch["y"],
            "batch_0": batch["batch_0"],
            "x_0": empty_tensor,
        }
