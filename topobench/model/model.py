"""This module defines the `TBModel` class."""

from collections.abc import Mapping
from numbers import Integral
from typing import Any

import torch
from lightning import LightningModule
from torch_geometric.data import Batch, Data, HeteroData
from torchmetrics import MeanMetric

from topobench.data.hypergraph import HypergraphData
from topobench.model.supervision import (
    DefaultSupervisionAdapter,
    SupervisionAdapter,
)
from topobench.nn.wrappers.graph.gnn_wrapper import (
    _bind_graph_batch_evidence,
    _prepare_graph_batch_evidence,
)


def _clone_evaluator_state(value: object) -> object:
    """Clone only checkpoint-safe evaluator primitives without retaining graphs."""
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise TypeError("evaluator state mapping keys must be non-empty strings")
        return {
            key: _clone_evaluator_state(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_clone_evaluator_state(item) for item in value)
    if isinstance(value, list):
        return [_clone_evaluator_state(item) for item in value]
    raise TypeError(
        "evaluator state must contain only JSON or tensor checkpoint primitives"
    )


class TBModel(LightningModule):
    r"""A `LightningModule` to define a network.

    Parameters
    ----------
    backbone : torch.nn.Module
        The backbone model to train.
    readout : torch.nn.Module
        The readout class.
    loss : torch.nn.Module
        The loss class.
    backbone_wrapper : torch.nn.Module, optional
        The backbone wrapper class (default: None).
    feature_encoder : torch.nn.Module, optional
        The feature encoder (default: None).
    evaluator : Any, optional
        The evaluator class (default: None).
    optimizer : Any, optional
        The optimizer class (default: None).
    supervision_adapter : SupervisionAdapter, optional
        Strategy selecting the predictions and labels supervised by each
        phase. Defaults to the legacy task-level behavior. The adapter is a
        runtime strategy and is intentionally excluded from checkpoint
        hyperparameters; checkpoint reruns must reconstruct it from the Hydra
        or run configuration.
    **kwargs : Any
        Additional keyword arguments.

    Notes
    -----
    Loss logging weights each epoch reduction by the number of supervised
    examples on the current process. Heterogeneous v1 does not claim globally
    weighted distributed aggregation, and ``sync_dist`` intentionally retains
    Lightning's default value of ``False``.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        readout: torch.nn.Module,
        loss: torch.nn.Module,
        backbone_wrapper: torch.nn.Module | None = None,
        feature_encoder: torch.nn.Module | None = None,
        evaluator: Any = None,
        optimizer: Any = None,
        supervision_adapter: SupervisionAdapter | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # This line allows accessing init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "backbone",
                "readout",
                "feature_encoder",
                "supervision_adapter",
            ],
        )

        self.feature_encoder = (
            feature_encoder
            if feature_encoder is not None
            else torch.nn.Identity()
        )
        if backbone_wrapper is None:
            self.backbone = backbone
        else:
            self.backbone = backbone_wrapper(backbone)
        self.readout = readout

        # Evaluator
        self.evaluator = evaluator
        self.train_metrics_logged = False
        self._dataloader_optimizer_success_token = 0
        self._dataloader_evaluator_sequence = 0
        self._dataloader_evaluator_count = 0

        # Optimizer (it also internally manages Scheduler if provided)
        self.optimizer = optimizer

        # Loss function
        self.loss = loss
        self.task_level = self.readout.task_level
        self.supervision_adapter = (
            DefaultSupervisionAdapter(self.task_level)
            if supervision_adapter is None
            else supervision_adapter
        )

        # Tracking best so far validation accuracy
        self.val_acc_best = MeanMetric()
        self.metric_collector_val = []
        self.metric_collector_val2 = []
        self.metric_collector_test = []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(backbone={self.backbone}, readout={self.readout}, loss={self.loss}, feature_encoder={self.feature_encoder})"

    def transfer_batch_to_device(
        self,
        batch: Any,
        device: torch.device,
        dataloader_idx: int,
    ) -> Any:
        """Transfer homogeneous graph batches with CPU-derived evidence."""
        evidence = None
        if (
            isinstance(batch, Batch)
            and not isinstance(batch, HeteroData)
            and not isinstance(batch, HypergraphData)
        ):
            evidence = _prepare_graph_batch_evidence(batch)
        transferred = super().transfer_batch_to_device(
            batch,
            device,
            dataloader_idx,
        )
        if evidence is not None:
            if not isinstance(transferred, Data):
                raise TypeError(
                    "homogeneous graph transfer must return native Data"
                )
            _bind_graph_batch_evidence(transferred, evidence)
        return transferred

    def forward(self, batch: Data | HeteroData) -> dict[str, Any]:
        r"""Perform a forward pass through the model.

        Parameters
        ----------
        batch : torch_geometric.data.Data or torch_geometric.data.HeteroData
            Homogeneous or heterogeneous batch containing the model inputs.

        Returns
        -------
        dict
            Dictionary containing the model output, which includes the logits and other relevant information.
        """
        # Feature Encoder
        model_out = self.feature_encoder(batch)

        # Domain model
        model_out = self.backbone(model_out)

        # Readout
        model_out = self.readout(model_out=model_out, batch=batch)

        return model_out

    def model_step(self, batch: Data | HeteroData) -> dict[str, Any]:
        r"""Perform a single model step on a batch of data.

        Parameters
        ----------
        batch : torch_geometric.data.Data or torch_geometric.data.HeteroData
            Homogeneous or heterogeneous batch containing the model inputs.

        Returns
        -------
        dict
            Dictionary containing the model output and the loss.
        """
        # Allow batch object to know the phase of the training
        batch["model_state"] = self.state_str

        # Forward pass
        model_out = self.forward(batch)

        # Loss
        model_out = self.process_outputs(model_out=model_out, batch=batch)

        # Metric
        model_out = self.loss(model_out=model_out, batch=batch)

        # Add batch to model_out for evaluator access to target normalizer stats
        model_out["batch"] = batch

        self.evaluator.update(model_out)

        return model_out

    def training_step(
        self, batch: Data | HeteroData, batch_idx: int
    ) -> torch.Tensor:
        r"""Perform a single training step on a batch of data.

        Parameters
        ----------
        batch : torch_geometric.data.Data or torch_geometric.data.HeteroData
            Homogeneous or heterogeneous batch containing the model inputs.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        torch.Tensor
            A tensor of losses between model predictions and targets.
        """
        self.state_str = "Training"
        model_out = self.model_step(batch)
        sequence_id = getattr(batch, "sequence_id", None)
        if sequence_id is not None:
            if isinstance(sequence_id, bool) or not isinstance(
                sequence_id,
                Integral,
            ):
                raise TypeError("training batch sequence_id must be an integer")
            sequence_id = int(sequence_id)
            expected_sequence = self._dataloader_evaluator_sequence + 1
            if sequence_id != expected_sequence:
                raise ValueError(
                    "training evaluator sequence expected "
                    f"{expected_sequence}, received {sequence_id}"
                )
            self._dataloader_evaluator_sequence = sequence_id
            self._dataloader_evaluator_count += 1

        # Update and log metrics
        loss_value = model_out["loss"].item()
        # This reduction weight is local to the current process; see the class
        # note for the intentionally unsupported globally weighted DDP case.
        self.log(
            "train/loss",
            loss_value,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=model_out["num_supervised_examples"],
        )

        # Return loss for backpropagation step
        return model_out["loss"]

    def validation_step(
        self, batch: Data | HeteroData, batch_idx: int
    ) -> None:
        r"""Perform a single validation step on a batch of data.

        Parameters
        ----------
        batch : torch_geometric.data.Data or torch_geometric.data.HeteroData
            Homogeneous or heterogeneous batch containing the model inputs.
        batch_idx : int
            The index of the current batch.
        """
        self.state_str = "Validation"
        model_out = self.model_step(batch)

        # Log Loss
        loss_value = model_out["loss"].item()
        self.log(
            "val/loss",
            loss_value,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=model_out["num_supervised_examples"],
        )

    def test_step(self, batch: Data | HeteroData, batch_idx: int) -> None:
        r"""Perform a single test step on a batch of data.

        Parameters
        ----------
        batch : torch_geometric.data.Data or torch_geometric.data.HeteroData
            Homogeneous or heterogeneous batch containing the model inputs.
        batch_idx : int
            The index of the current batch.
        """
        self.state_str = "Test"
        model_out = self.model_step(batch)

        # Log loss
        loss_value = model_out["loss"].item()
        self.log(
            "test/loss",
            loss_value,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=model_out["num_supervised_examples"],
        )

    def process_outputs(
        self,
        model_out: dict[str, Any],
        batch: Data | HeteroData,
    ) -> dict[str, Any]:
        r"""Handle model outputs.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data or torch_geometric.data.HeteroData
            Homogeneous or heterogeneous batch containing phase supervision.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        supervised = self.supervision_adapter.select(
            model_out=model_out,
            batch=batch,
            phase=self.state_str,
        )
        model_out["logits"] = supervised.logits
        model_out["labels"] = supervised.targets
        model_out["num_supervised_examples"] = supervised.num_examples
        return model_out

    def log_metrics(self, mode=None):
        r"""Log metrics.

        Parameters
        ----------
        mode : str, optional
            The mode of the model, either "train", "val", or "test" (default: None).
        """
        metrics_dict = self.evaluator.compute()

        # Log current metrics
        for key in metrics_dict:
            self.log(
                f"{mode}/{key}",
                metrics_dict[key],
                prog_bar=True,
                on_step=False,
            )

        # Reset evaluator for next epoch
        self.evaluator.reset()

    def on_validation_epoch_start(self) -> None:
        r"""Hook called when a validation epoch begins.

        According pytorch lightning documentation this hook is called at the beginning of the
        validation epoch.

        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks

        Note that the validation step is within the train epoch. Hence here we have to log the train metrics
        before we reset the evaluator to start the validation loop.
        """
        # Log train metrics and reset evaluator
        self.log_metrics(mode="train")
        self.train_metrics_logged = True

    def on_train_epoch_end(self) -> None:
        r"""Lightning hook that is called when a train epoch ends.

        This hook is used to log the train metrics.
        """
        # Log train metrics and reset evaluator
        if not self.train_metrics_logged:
            self.log_metrics(mode="train")
            self.train_metrics_logged = True

    def on_validation_epoch_end(self) -> None:
        r"""Lightning hook that is called when a validation epoch ends.

        This hook is used to log the validation metrics.
        """
        # Log validation metrics and reset evaluator
        self.log_metrics(mode="val")

    def on_test_epoch_end(self) -> None:
        r"""Lightning hook that is called when a test epoch ends.

        This hook is used to log the test metrics.
        """
        self.log_metrics(mode="test")

    def on_train_epoch_start(self) -> None:
        r"""Lightning hook that is called when a train epoch begins.

        This hook is used to reset the train metrics.
        """
        self.evaluator.reset()
        self.train_metrics_logged = False

    def on_val_epoch_start(self) -> None:
        r"""Lightning hook that is called when a validation epoch begins.

        This hook is used to reset the validation metrics.
        """
        self.evaluator.reset()

    def on_test_epoch_start(self) -> None:
        r"""Lightning hook that is called when a test epoch begins.

        This hook is used to reset the test metrics.
        """
        self.evaluator.reset()

    def setup(self, stage: str) -> None:
        r"""Hook to call torch.compile.

        Lightning hook that is called at the beginning of fit (train +
        validate), validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust
        something about them. This hook is called on every process when using
        DDP.

        Parameters
        ----------
        stage : str
            Either "fit", "validate", "test", or "predict".
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    @property
    def dataloader_optimizer_success_token(self) -> int:
        """Monotonic count of raw optimizer steps that returned successfully."""
        return self._dataloader_optimizer_success_token

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Any,
        optimizer_closure: Any = None,
    ) -> None:
        """Advance the success token only when the raw optimizer really ran."""
        raw_optimizer = getattr(optimizer, "optimizer", optimizer)
        register_hook = getattr(raw_optimizer, "register_step_post_hook", None)
        if not callable(register_hook):
            raise RuntimeError(
                "optimizer cannot prove successful completion with a post-step hook"
            )
        raw_step_completed = False

        def mark_completed(*_: object) -> None:
            nonlocal raw_step_completed
            raw_step_completed = True

        handle = register_hook(mark_completed)
        try:
            super().optimizer_step(
                epoch,
                batch_idx,
                optimizer,
                optimizer_closure,
            )
        finally:
            handle.remove()
        if raw_step_completed:
            self._dataloader_optimizer_success_token += 1

    def _dataloader_evaluator_owner(self) -> Any:
        owner = self.evaluator
        if not callable(getattr(owner, "state_dict", None)):
            owner = getattr(owner, "metrics", None)
        if (
            owner is None
            or not callable(getattr(owner, "state_dict", None))
            or not callable(getattr(owner, "load_state_dict", None))
        ):
            raise TypeError(
                "evaluator must expose strict state_dict/load_state_dict participation"
            )
        return owner

    def dataloader_evaluator_snapshot(self) -> dict[str, object]:
        """Capture the current evaluator participant without sampler knowledge."""
        owner = self._dataloader_evaluator_owner()
        return {
            "sequence_id": self._dataloader_evaluator_sequence,
            "count": self._dataloader_evaluator_count,
            "state": _clone_evaluator_state(owner.state_dict()),
        }

    def dataloader_restore_evaluator(
        self,
        snapshot: Mapping[str, object],
    ) -> None:
        """Strictly restore one previously committed evaluator participant."""
        if set(snapshot) != {"sequence_id", "count", "state"}:
            raise ValueError(
                "evaluator snapshot keys must be sequence_id, count, and state"
            )
        sequence_id = snapshot["sequence_id"]
        count = snapshot["count"]
        if (
            isinstance(sequence_id, bool)
            or not isinstance(sequence_id, Integral)
            or isinstance(count, bool)
            or not isinstance(count, Integral)
        ):
            raise TypeError("evaluator sequence and count must be integers")
        sequence_id, count = int(sequence_id), int(count)
        if sequence_id < 0 or count != sequence_id:
            raise ValueError(
                "evaluator sequence/count must be equal non-negative values"
            )
        state = _clone_evaluator_state(snapshot["state"])
        owner = self._dataloader_evaluator_owner()
        owner.load_state_dict(state, strict=True)
        self._dataloader_evaluator_sequence = sequence_id
        self._dataloader_evaluator_count = count

    def configure_optimizers(self) -> dict[str, Any]:
        r"""Configure optimizers and learning-rate schedulers.

        Choose what optimizers and learning-rate schedulers to use in your
        optimization. Normally you'd need one. But in the case of GANs or
        similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        Returns
        -------
        dict:
            A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer_config = self.optimizer.configure_optimizer(
            list(self.backbone.parameters())
            + list(self.readout.parameters())
            + list(self.feature_encoder.parameters())
        )

        return optimizer_config
