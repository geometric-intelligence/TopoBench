"""
Base wrapper for sklearn-based graph node prediction models.

Design: minimal BaseWrapper provides orchestration only. Task-specific behavior
is implemented by BaseClassifierWrapper and BaseRegressorWrapper (two thin bases).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from topobench.nn.wrappers.graph.sklearn.sklearn_wrappers.components import (
    Checker,
    LoggerStats,
    SafePredictor,
)
from topobench.nn.wrappers.graph.sklearn.sklearn_wrappers.types import (
    masks_to_bool,
    validate_batch,
)

# Constants for sampler features
SAMPLER_FEATURES_ALL = "all"
SAMPLER_FEATURES_NODE = "node"
SAMPLER_FEATURES_STRUCTURAL = "structural"
VALID_SAMPLER_FEATURES = {
    SAMPLER_FEATURES_ALL,
    SAMPLER_FEATURES_NODE,
    SAMPLER_FEATURES_STRUCTURAL,
}


def _build_sampler_features(
    batch: Dict[str, torch.Tensor],
    node_features_model: np.ndarray,
    sampler_features: str,
) -> np.ndarray:
    """
    Build sampler features based on the specified feature type.
    Structural feature keys are sorted for deterministic concatenation order.
    """
    if sampler_features not in VALID_SAMPLER_FEATURES:
        raise ValueError(
            f"sampler_features must be one of {VALID_SAMPLER_FEATURES}, "
            f"got '{sampler_features}'"
        )
    if sampler_features == SAMPLER_FEATURES_ALL:
        return node_features_model

    if sampler_features == SAMPLER_FEATURES_NODE:
        if "x_0" not in batch:
            raise KeyError(
                f"sampler_features='{SAMPLER_FEATURES_NODE}' requires batch['x_0']"
            )
        return batch["x_0"].cpu().numpy().copy()
    # structural: all x_* except x_0, in sorted order
    structural_keys = sorted(
        k for k in batch.keys() if k.startswith("x_") and k != "x_0"
    )
    if len(structural_keys) == 0:
        raise RuntimeError(
            f"sampler_features='{SAMPLER_FEATURES_STRUCTURAL}' requested but no "
            "structural features found (expected keys like 'x_1_hop_mean', etc.)."
        )
    structural_tensors = [batch[k] for k in structural_keys]
    return torch.cat(structural_tensors, dim=1).cpu().numpy().copy()


class BaseWrapper(torch.nn.Module, ABC):
    """
    Minimal base wrapper: orchestration only.

    Assumes one graph per forward() call. Target-related state (_init_targets)
    is re-initialized on each forward from the current batch's training labels.

    Parameters
    ----------
    backbone : Any
        Sklearn model (fit + predict; classifiers may also expose predict_proba).
    use_embeddings : bool
        If True, use all keys starting with "x_" as features.
    use_node_features : bool
        If True, include "x_0". At least one of use_embeddings/use_node_features must be True.
    sampler : Optional[Any]
        Neighbor sampler. If None, full-graph training is used.
    num_test_nodes : int
        Batch size for test nodes when using sampler.
    sampler_features : str
        One of "all", "node", "structural".
    logger : Optional[Any]
        Logger for stats (e.g. WandB).
    device : Optional[torch.device]
        Override device for output tensors. If None, use batch["x_0"].device.
    progress_callback : Optional[Callable]
        If provided, called as (current, total, postfix_dict) instead of tqdm.
    max_context_nodes : Optional[int]
        If set, cap the number of nodes used as context for the backbone. In full-graph
        mode this limits training nodes; in sampler mode it limits neighbors per test node.
        None means no limit.
    """

    def __init__(
        self,
        backbone: Any,
        *,
        use_embeddings: bool = True,
        use_node_features: bool = True,
        sampler: Optional[Any] = None,
        num_test_nodes: int = 1,
        sampler_features: str = SAMPLER_FEATURES_ALL,
        logger: Optional[Any] = None,
        device: Optional[torch.device] = None,
        progress_callback: Optional[
            Callable[[int, int, Dict[str, str]], None]
        ] = None,
        max_context_nodes: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.backbone = backbone
        self.use_embeddings = use_embeddings
        self.use_node_features = use_node_features
        self.num_test_nodes = num_test_nodes
        self.logger = logger
        self._device_override = device
        self._progress_callback = progress_callback
        self.max_context_nodes = max_context_nodes

        self.sampler = sampler if sampler != {} else None

        if not (self.use_embeddings or self.use_node_features):
            raise ValueError(
                "Either use_embeddings or use_node_features must be True, not both False."
            )
        if sampler_features not in VALID_SAMPLER_FEATURES:
            raise ValueError(
                f"sampler_features must be one of {VALID_SAMPLER_FEATURES}, "
                f"got '{sampler_features}'"
            )
        self.sampler_features = sampler_features

        self.checker = Checker()
        self.stats = LoggerStats()
        self.safe = SafePredictor(
            checker=self.checker,
            backbone_factory=lambda: deepcopy(self.backbone),
            fallback_no_neighbors=self._no_neighbors,
            fallback_one_neighbor=self._one_neighbor,
            fallback_all_const=self._all_features_constant,
            get_predictions=self._get_predictions,
        )

    def _device(self, batch: Dict[str, torch.Tensor]) -> torch.device:
        if self._device_override is not None:
            return self._device_override
        return batch["x_0"].device

    # ---------- Abstract API (implemented by BaseClassifierWrapper / BaseRegressorWrapper) ----------

    @abstractmethod
    def _init_targets(self, y_train: np.ndarray) -> None:
        """Initialize task-specific state from training targets (called once per forward)."""
        ...

    @abstractmethod
    def _no_neighbors(self, batch_size: int) -> Tuple[Any, Any]:
        """Return (probs_or_vals, preds) for the no-neighbors case."""
        ...

    @abstractmethod
    def _one_neighbor(
        self, labels: np.ndarray, batch_size: int
    ) -> Tuple[Any, Any]:
        """Return (probs_or_vals, preds) for the one-neighbor case."""
        ...

    @abstractmethod
    def _all_features_constant(
        self, labels: np.ndarray, batch_size: int
    ) -> Tuple[Any, Any]:
        """Return (probs_or_vals, preds) when all neighbor features are constant."""
        ...

    @abstractmethod
    def _get_predictions(self, model: Any, X_test: np.ndarray) -> Tuple[list, list]:
        """Return (list of probs/vals, list of preds) from trained model."""
        ...

    @abstractmethod
    def _process_output(
        self, output_tensor: torch.Tensor, num_dataset_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (empty_tensor, output_tensor) with correct shapes and device."""
        ...

    @abstractmethod
    def _full_graph_training(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        """Train on full graph and return predictions tensor for test nodes."""
        ...

    @abstractmethod
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
        """Update progress and accumulate outputs/preds/trues."""
        ...

    # ---------- Public API ----------

    def fit(self, x: np.ndarray, y: np.ndarray) -> BaseWrapper:
        """No-op in base; subclasses may override. Returns self."""
        return self

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Single-graph forward. Validates batch, extracts features, runs prediction, formats output.
        """
        validate_batch(batch)
        num_nodes = batch["x_0"].shape[0]
        train_mask, val_mask, test_mask = masks_to_bool(batch, num_nodes)

        node_features = self._extract_node_features(batch)
        labels = batch["y"].cpu().numpy()

        X_train = node_features[train_mask].copy()
        y_train = batch["y"][train_mask].cpu().numpy().copy()

        if self.max_context_nodes is not None and len(X_train) > self.max_context_nodes:
            idx = np.arange(len(X_train), dtype=np.intp)
            np.random.default_rng(42).shuffle(idx)
            idx = idx[: self.max_context_nodes]
            X_train = X_train[idx]
            y_train = y_train[idx]

        self._init_targets(y_train)

        device = self._device(batch)

        if self.sampler is None:
            outputs_tensor = self._full_graph_training(
                X_train, y_train, node_features[test_mask], device
            )
        else:
            outputs_tensor = self._predict_with_sampler(
                batch, node_features, labels, train_mask, val_mask, test_mask

        return self._format_output(batch, outputs_tensor, test_mask, len(labels))

    # ---------- Shared helpers ----------

    def _extract_node_features(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        """Concatenate node features. Feature keys are sorted for deterministic order."""
        if self.use_embeddings:
            rank0 = sorted(k for k in batch.keys() if k.startswith("x_"))
        else:
            rank0 = ["x_0"]

        if not self.use_node_features and "x_0" in rank0:
            rank0.remove("x_0")

        tensors = [batch[k] for k in rank0]
        out = torch.cat(tensors, dim=1)
        return out.cpu().numpy().copy()

    def _predict_with_sampler(
        self,
        batch: Dict[str, torch.Tensor],
        node_features: np.ndarray,
        labels: np.ndarray,
        train_mask: np.ndarray,
        val_mask: np.ndarray,
        test_mask: np.ndarray,
    ) -> torch.Tensor:
        """Run sampler-based prediction loop."""
        node_features_sampler = _build_sampler_features(
            batch, node_features, self.sampler_features
        )
        edge_index = batch["edge_index"].cpu().numpy()

        self.sampler.fit(
            node_features_sampler,
            labels,
            edge_index=edge_index,
            train_mask=train_mask,
        )

        outputs: list = []
        preds: list = []
        trues: list = []

        total = int(test_mask.sum())
        batch_size = max(1, self.num_test_nodes)
        test_indices = np.where(test_mask)[0]

        if self._progress_callback is not None:
            class _ProgressAdapter:
                def __init__(self, total: int, callback: Callable):
                    self._total = total
                    self._callback = callback
                    self._current = 0
                def update(self, n: int = 1) -> None:
                    self._current = min(self._current + n, self._total)
                    self._callback(self._current, self._total, {})
                def set_postfix(self, *args: Any, **kwargs: Any) -> None:
                    pass
            pbar = _ProgressAdapter(total, self._progress_callback)
        else:
            pbar = tqdm(total=total, desc="Sampling and predicting", dynamic_ncols=True)

        for start in range(0, len(test_indices), batch_size):
            end = start + batch_size
            test_ids = test_indices[start:end]

            neighbors_id = self.sampler.sample(
                x=node_features_sampler[test_ids], ids=test_ids
            )
            if self.max_context_nodes is not None and len(neighbors_id) > self.max_context_nodes:
                neighbors_id = np.asarray(neighbors_id, dtype=np.intp)[: self.max_context_nodes]

            probs, predictions, case = self.safe.predict_batch(
                neighbors_id=neighbors_id,
                test_ids=test_ids,
                node_features=node_features,
                labels=labels,
                val_mask=val_mask,
                test_mask=test_mask,
            )

            self.stats.inc(case, n=len(test_ids))

            self._update_progress_and_results(
                probs,
                predictions,
                batch["y"][test_ids],
                outputs,
                preds,
                trues,
                pbar,
            )

        if hasattr(pbar, "close"):
            pbar.close()

        return torch.FloatTensor(outputs).to(self._device(batch))

    def _format_output(
        self,
        batch: Dict[str, torch.Tensor],
        outputs_tensor: torch.Tensor,
        test_mask: np.ndarray,
        num_dataset_points: int,
    ) -> Dict[str, torch.Tensor]:
        """Build output dict with labels, batch_0, and x_0 (predictions)."""
        empty_tensor, outputs_tensor = self._process_output(
            outputs_tensor, num_dataset_points
        )
        test_mask_t = torch.from_numpy(test_mask).to(outputs_tensor.device)
        empty_tensor[test_mask_t] = outputs_tensor
        self.stats.log(num_test_points=int(test_mask.sum()), logger=self.logger)
        return {
            "labels": batch["y"],
            "batch_0": batch["batch_0"],
            "x_0": empty_tensor,
        }
