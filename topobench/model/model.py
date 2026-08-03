"""This module defines the `TBModel` class."""

from collections.abc import Mapping
from numbers import Integral
from typing import Any

import torch
from lightning import LightningModule
from torch_geometric.data import Batch, Data, HeteroData

from topobench.evaluator import (
    AbstractEvaluator,
    EvaluationBatch,
    EvaluationContext,
    EvaluationPassKind,
    EvaluationResult,
    EvaluationSplit,
)
from topobench.data.hypergraph import HypergraphData
from topobench.dataloader.input_monitor import (
    InputMonitor,
    InputStallError,
    MonitorOverflowError,
    OperationToken,
)
from topobench.profiling.execution_events import (
    ExecutionOperation,
    ExecutionStatus,
)
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
    evaluator : AbstractEvaluator, optional
        Typed evaluator lifecycle owned by this model (default: None).
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
        evaluator: AbstractEvaluator | None = None,
        optimizer: Any = None,
        supervision_adapter: SupervisionAdapter | None = None,
        execution_monitor: InputMonitor | None = None,
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
                "execution_monitor",
            ],
        )
        if execution_monitor is not None and not isinstance(
            execution_monitor,
            InputMonitor,
        ):
            raise TypeError("execution_monitor must be InputMonitor or None")
        self.execution_monitor = execution_monitor

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

        self.evaluator = evaluator
        self._active_evaluation_context: EvaluationContext | None = None
        self._next_validation_pass_kind: EvaluationPassKind = "fit_epoch"
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

    def _execution_context(
        self,
        batch: Data | HeteroData,
    ) -> tuple[str, str, int | None, int | None, int | None, str | None, bool]:
        state = getattr(self, "state_str", "")
        split = {
            "Training": "train",
            "Validation": "val",
            "Test": "test",
        }.get(state, "predict")
        phase = "fit" if split in {"train", "val"} else "test"
        sequence = getattr(batch, "sequence_id", None)
        if isinstance(sequence, bool) or not isinstance(sequence, Integral):
            sequence = None
        else:
            sequence = int(sequence)
        trainer = getattr(self, "_trainer", None)
        epoch = getattr(trainer, "current_epoch", None)
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            epoch = None
        global_step = getattr(trainer, "global_step", None)
        if isinstance(global_step, bool) or not isinstance(global_step, int):
            global_step = None
        digest = getattr(batch, "execution_descriptor_digest", None)
        cuda_timing = (
            getattr(getattr(self, "device", None), "type", None) == "cuda"
        )
        return (
            phase,
            split,
            sequence,
            epoch,
            global_step,
            digest,
            cuda_timing,
        )

    def model_step(self, batch: Data | HeteroData) -> dict[str, Any]:
        r"""Perform a single model and evaluator step on native graph data."""
        monitor = self.execution_monitor
        compute_token: OperationToken | None = None
        context = self._execution_context(batch) if monitor is not None else None
        evaluation_context = self._active_evaluation_context
        if evaluation_context is None:
            raise RuntimeError("model_step requires an active evaluation phase")
        if monitor is not None:
            assert context is not None
            phase, split, sequence, epoch, global_step, digest, cuda_timing = (
                context
            )
            compute_token = monitor.begin_model_compute(
                phase=phase,
                split=split,
                descriptor_sequence=sequence,
                descriptor_digest_value=digest,
                epoch=epoch,
                global_step=global_step,
                cuda_timing=cuda_timing,
            )
        try:
            batch["model_state"] = self.state_str
            model_out = self.forward(batch)
            model_out = self.process_outputs(model_out=model_out, batch=batch)
            evaluation_batch = EvaluationBatch(
                outputs=model_out["logits"],
                targets=model_out["labels"],
                num_examples=model_out["num_supervised_examples"],
                context=evaluation_context,
                sequence_id=getattr(batch, "sequence_id", None),
            )
            model_out = self.loss(model_out=model_out, batch=batch)
        except BaseException:
            if monitor is not None and compute_token is not None:
                try:
                    monitor.finish_model_compute(
                        compute_token,
                        status=ExecutionStatus.ERROR,
                        evidence={"failure_stage": "model_compute"},
                    )
                except Exception:
                    pass
            self._abort_after_failure()
            raise
        if monitor is not None and compute_token is not None:
            examples = model_out.get("num_supervised_examples")
            if isinstance(examples, bool) or not isinstance(examples, Integral):
                examples = None
            monitor.finish_model_compute(
                compute_token,
                example_count=None if examples is None else int(examples),
            )
        model_out["batch"] = batch
        evaluator_token: OperationToken | None = None
        if monitor is not None:
            assert context is not None
            phase, split, sequence, epoch, global_step, digest, cuda_timing = (
                context
            )
            evaluator_token = monitor.begin(
                ExecutionOperation.EVALUATOR,
                phase=phase,
                split=split,
                descriptor_sequence=sequence,
                descriptor_digest_value=digest,
                epoch=epoch,
                global_step=global_step,
                cuda_timing=cuda_timing,
            )
        try:
            self.evaluator.update(evaluation_batch)
        except BaseException:
            if monitor is not None and evaluator_token is not None:
                try:
                    monitor.finish(
                        evaluator_token,
                        status=ExecutionStatus.ERROR,
                        evidence={"failure_stage": "evaluator_update"},
                    )
                except Exception:
                    pass
            self._abort_after_failure()
            raise
        if monitor is not None and evaluator_token is not None:
            monitor.finish(evaluator_token)
        return model_out

    def training_step(
        self, batch: Data | HeteroData, batch_idx: int
    ) -> torch.Tensor:
        """Perform one failure-safe typed training step."""
        del batch_idx
        self.state_str = "Training"
        try:
            model_out = self.model_step(batch)
            loss_value = model_out["loss"].item()
            self.log(
                "train/loss",
                loss_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=model_out["num_supervised_examples"],
            )
        except BaseException:
            self._abort_after_failure()
            raise

        sequence_id = getattr(batch, "sequence_id", None)
        if (
            sequence_id is not None
            and not isinstance(sequence_id, bool)
            and isinstance(sequence_id, Integral)
        ):
            self._dataloader_evaluator_sequence = int(sequence_id)
        self._dataloader_evaluator_count += 1
        return model_out["loss"]

    def validation_step(
        self, batch: Data | HeteroData, batch_idx: int
    ) -> None:
        """Perform one failure-safe typed validation step."""
        del batch_idx
        self.state_str = "Validation"
        try:
            model_out = self.model_step(batch)
            self.log(
                "val/loss",
                model_out["loss"].item(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=model_out["num_supervised_examples"],
            )
        except BaseException:
            self._abort_after_failure()
            raise

    def test_step(self, batch: Data | HeteroData, batch_idx: int) -> None:
        """Perform one failure-safe typed test step."""
        del batch_idx
        self.state_str = "Test"
        try:
            model_out = self.model_step(batch)
            self.log(
                "test/loss",
                model_out["loss"].item(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=model_out["num_supervised_examples"],
            )
        except BaseException:
            self._abort_after_failure()
            raise

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

    def _evaluation_context(
        self,
        split: EvaluationSplit,
        pass_kind: EvaluationPassKind,
    ) -> EvaluationContext:
        """Build the only context used by production steps and probes."""
        task = getattr(self.evaluator, "task", None)
        num_classes = getattr(self.evaluator, "num_classes", None)
        if task not in {"classification", "regression"}:
            raise TypeError("evaluator must expose a supported task")
        if isinstance(num_classes, bool) or not isinstance(num_classes, int):
            raise TypeError("evaluator must expose an integer num_classes")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        configured_policy = getattr(self.evaluator, "policy", None)
        if configured_policy is None:
            selected_policy = "online" if split == "train" else "exact"
        else:
            if not isinstance(configured_policy, Mapping):
                raise TypeError("evaluator.policy must be a mapping")
            if set(configured_policy) != {"train", "val", "test"}:
                raise ValueError(
                    "evaluator.policy must define exactly train, val, and test"
                )
            if any(
                policy not in {"online", "exact", "audit"}
                for policy in configured_policy.values()
            ):
                raise ValueError(
                    "evaluator.policy values must be online, exact, or audit"
                )
            selected_policy = configured_policy[split]
        return EvaluationContext(
            split=split,
            pass_kind=pass_kind,
            policy=selected_policy,
            task=task,
            num_classes=num_classes,
        )

    def _begin_evaluation(
        self,
        split: EvaluationSplit,
        pass_kind: EvaluationPassKind,
    ) -> None:
        """Open one model-owned evaluator context without phase overlap."""
        if self._active_evaluation_context is not None:
            raise RuntimeError(
                "cannot begin an evaluation phase while another is active"
            )
        context = self._evaluation_context(split, pass_kind)
        self._active_evaluation_context = context
        try:
            self.evaluator.begin(context)
        except BaseException:
            self._abort_after_failure()
            raise
        self.state_str = {
            "train": "Training",
            "val": "Validation",
            "test": "Test",
        }[split]

    def _log_evaluation_result(self, result: EvaluationResult) -> None:
        """Publish finalized metrics and the exact phase count once."""
        namespace = result.context.split
        for name, value in result.metrics.items():
            self.log(
                f"{namespace}/{name}",
                value,
                prog_bar=True,
                on_step=False,
            )
        self.log(
            f"{namespace}/num_examples",
            result.num_examples,
            prog_bar=False,
            on_step=False,
        )

    def _finalize_evaluation(self, split: EvaluationSplit) -> None:
        """Finalize and log exactly one matching active phase."""
        context = self._active_evaluation_context
        if context is None:
            return
        if context.split != split:
            raise RuntimeError(
                f"cannot finalize {split!r} while {context.split!r} is active"
            )
        try:
            result = self.evaluator.finalize()
        except BaseException:
            self._abort_after_failure()
            raise
        self._active_evaluation_context = None
        try:
            self._log_evaluation_result(result)
        except BaseException:
            self._abort_after_failure(force=True)
            raise

    def _abort_after_failure(self, *, force: bool = False) -> None:
        """Clear evaluator and model phase state while preserving root errors."""
        context = self._active_evaluation_context
        state = getattr(self.evaluator, "state", None)
        if not force and context is None and state not in {"active", "failed"}:
            return
        try:
            self.evaluator.abort()
        except RuntimeError:
            if not force and context is not None:
                raise
        finally:
            self._active_evaluation_context = None

    def abort_evaluation(self) -> None:
        """Abort an active evaluator phase, including throwaway probes."""
        self._abort_after_failure()

    def set_next_validation_pass_kind(
        self,
        pass_kind: EvaluationPassKind,
    ) -> None:
        """Select one upcoming validation context without replacing hooks."""
        if pass_kind not in {"fit_epoch", "selected_checkpoint"}:
            raise ValueError(
                "validation pass kind must be fit_epoch or selected_checkpoint"
            )
        if self._active_evaluation_context is not None:
            raise RuntimeError(
                "validation pass kind cannot change during an active phase"
            )
        self._next_validation_pass_kind = pass_kind

    def on_train_epoch_start(self) -> None:
        """Begin the online training accumulation window."""
        self._begin_evaluation("train", "fit_epoch")

    def on_validation_epoch_start(self) -> None:
        """Close training once, then begin the requested validation pass."""
        if (
            self._active_evaluation_context is not None
            and self._active_evaluation_context.split == "train"
        ):
            self._finalize_evaluation("train")
        pass_kind = self._next_validation_pass_kind
        self._begin_evaluation("val", pass_kind)
        self._next_validation_pass_kind = "fit_epoch"

    def on_train_epoch_end(self) -> None:
        """Finalize training when no validation loop already closed it."""
        if (
            self._active_evaluation_context is not None
            and self._active_evaluation_context.split == "train"
        ):
            self._finalize_evaluation("train")

    def on_validation_epoch_end(self) -> None:
        """Finalize the exact validation accumulation window."""
        self._finalize_evaluation("val")

    def on_test_epoch_start(self) -> None:
        """Begin exact selected-checkpoint testing."""
        self._begin_evaluation("test", "selected_checkpoint")

    def on_test_epoch_end(self) -> None:
        """Finalize the exact selected-checkpoint test window."""
        self._finalize_evaluation("test")

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
        monitor = self.execution_monitor
        monitor_token = (
            None
            if monitor is None
            else monitor.begin(
                ExecutionOperation.OPTIMIZER,
                phase="optimizer",
                split="train",
                epoch=epoch,
                global_step=(
                    getattr(getattr(self, "_trainer", None), "global_step", None)
                ),
                cuda_timing=(
                    getattr(getattr(self, "device", None), "type", None)
                    == "cuda"
                ),
            )
        )
        try:
            super().optimizer_step(
                epoch,
                batch_idx,
                optimizer,
                optimizer_closure,
            )
        except BaseException:
            if monitor_token is not None:
                try:
                    monitor.finish(
                        monitor_token,
                        status=ExecutionStatus.ERROR,
                        evidence={"failure_stage": "optimizer"},
                    )
                except Exception:
                    pass
            raise
        finally:
            handle.remove()
        if raw_step_completed:
            self._dataloader_optimizer_success_token += 1
        if monitor_token is not None:
            try:
                monitor.finish(
                    monitor_token,
                    status=(
                        ExecutionStatus.SUCCESS
                        if raw_step_completed
                        else ExecutionStatus.WARNING
                    ),
                    evidence={"raw_step_completed": raw_step_completed},
                )
            except (InputStallError, MonitorOverflowError):
                raise
            except Exception:
                pass

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
        if sequence_id < 0 or count < 0:
            raise ValueError(
                "evaluator sequence and count must be non-negative values"
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
