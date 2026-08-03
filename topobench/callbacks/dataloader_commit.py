"""Commit disk-sampler state only after a proven optimizer/global-step advance."""

from __future__ import annotations

import random
from collections.abc import Mapping
from numbers import Integral
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from lightning import Callback

_CALLBACK_FORMAT = "dataloader-commit-callback-v2"
_CALLBACK_KEYS = frozenset(
    {
        "format_version",
        "data_module",
        "committed_optimizer_steps",
        "committed_optimizer_steps_in_epoch",
    }
)
_EVALUATOR_KEYS = frozenset({"sequence_id", "count", "state"})

_RNG_CHECKPOINT_KEY = "topobench_dataloader_commit_rng"
_RNG_FORMAT = "dataloader-commit-rng-v2"
_RNG_RESTORE_BEFORE_DATA_ITERATOR = "before_data_iterator"
_RNG_RESTORE_AFTER_DATA_ITERATOR = "after_data_iterator"
_RNG_RESTORE_TIMINGS = frozenset(
    {
        _RNG_RESTORE_BEFORE_DATA_ITERATOR,
        _RNG_RESTORE_AFTER_DATA_ITERATOR,
    }
)
_RNG_KEYS = frozenset(
    {
        "format_version",
        "restore_timing",
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }
)
_METRICS_CHECKPOINT_KEY = "topobench_dataloader_commit_metrics"
_METRICS_FORMAT = "dataloader-commit-metrics-v1"
_METRICS_KEYS = frozenset({"format_version", "train_loss"})


@runtime_checkable
class _CommitDataModule(Protocol):
    sequence_state: Any

    def consume_sequence(self, sequence_id: int) -> None: ...

    def commit_optimizer_step(
        self,
        *,
        optimizer_succeeded: bool,
        model_global_step: int,
        evaluator_snapshot: Mapping[str, object],
        epoch: int,
    ) -> bool: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None: ...


@runtime_checkable
class _CommitModel(Protocol):
    @property
    def dataloader_optimizer_success_token(self) -> int: ...

    def dataloader_evaluator_snapshot(self) -> dict[str, object]: ...

    def dataloader_restore_evaluator(
        self,
        snapshot: Mapping[str, object],
    ) -> None: ...


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _sequence_id(batch: object) -> int:
    sequence_id = getattr(batch, "sequence_id", None)
    return _integer(sequence_id, "training batch sequence_id", minimum=1)


def _rng_state_dict(
    *,
    restore_timing: str = _RNG_RESTORE_AFTER_DATA_ITERATOR,
) -> dict[str, object]:
    """Capture process RNG state at one explicit iterator boundary."""
    if restore_timing not in _RNG_RESTORE_TIMINGS:
        raise ValueError("RNG restore timing is unsupported")
    numpy_state = np.random.get_state()
    return {
        "format_version": _RNG_FORMAT,
        "restore_timing": restore_timing,
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": (
            tuple(state.clone() for state in torch.cuda.get_rng_state_all())
            if torch.cuda.is_initialized()
            else None
        ),
    }


def _rng_restore_timing(state: object) -> str:
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint RNG state must be a mapping")
    actual = frozenset(state)
    if actual != _RNG_KEYS:
        raise ValueError(
            "checkpoint RNG state keys must match exactly; "
            f"missing={sorted(_RNG_KEYS - actual)!r}, "
            f"extra={sorted(actual - _RNG_KEYS)!r}"
        )
    if state["format_version"] != _RNG_FORMAT:
        raise ValueError("unsupported checkpoint RNG state version")
    restore_timing = state["restore_timing"]
    if restore_timing not in _RNG_RESTORE_TIMINGS:
        raise ValueError("checkpoint RNG restore timing is unsupported")
    return str(restore_timing)


def _restore_rng_state(state: object) -> None:
    """Restore one strict process RNG snapshot without silent fallback."""
    _rng_restore_timing(state)
    assert isinstance(state, Mapping)
    python_state = state["python"]
    numpy_state = state["numpy"]
    torch_cpu = state["torch_cpu"]
    torch_cuda = state["torch_cuda"]
    if not isinstance(python_state, tuple):
        raise TypeError("Python RNG state must be a tuple")
    if (
        not isinstance(numpy_state, tuple)
        or len(numpy_state) != 5
        or not isinstance(numpy_state[1], np.ndarray)
    ):
        raise TypeError("NumPy RNG state must be an exact state tuple")
    if (
        not isinstance(torch_cpu, torch.Tensor)
        or torch_cpu.dtype != torch.uint8
        or torch_cpu.device.type != "cpu"
        or torch_cpu.ndim != 1
    ):
        raise TypeError(
            "CPU torch RNG state must be a one-dimensional byte tensor"
        )
    if torch_cuda is not None and (
        not isinstance(torch_cuda, (tuple, list))
        or not all(
            isinstance(item, torch.Tensor)
            and item.dtype == torch.uint8
            and item.ndim == 1
            for item in torch_cuda
        )
    ):
        raise TypeError("CUDA RNG state must be a sequence of byte tensors")
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_cpu)
    if torch_cuda is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        torch.cuda.set_rng_state_all(list(torch_cuda))


def _checkpoint_values_equal(left: object, right: object) -> bool:
    """Return exact equality for supported evaluator checkpoint values."""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and left.keys() == right.keys()
            and all(
                _checkpoint_values_equal(left[key], right[key]) for key in left
            )
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(
                _checkpoint_values_equal(first, second)
                for first, second in zip(left, right, strict=True)
            )
        )
    return type(left) is type(right) and left == right


class DataloaderCommitCallback(Callback):
    """Coordinate one commit boundary after Lightning finishes a train batch."""

    def __init__(self) -> None:
        super().__init__()
        self._data_module: _CommitDataModule | None = None
        self._optimizer_token: int | None = None
        self._observed_global_step: int | None = None
        self._deferred_state: Mapping[str, object] | None = None
        self._restore_snapshot: dict[str, object] | None = None
        self._trainer: Any | None = None
        self._model: _CommitModel | None = None
        self._resume_rng_state: Mapping[str, object] | None = None
        self._active_batch_rng_state: Mapping[str, object] | None = None
        self._active_batch_evaluator_snapshot: Mapping[str, object] | None = (
            None
        )
        self._resume_train_loss: torch.Tensor | None = None
        self._resumed_committed_epoch_end = False
        self._committed_optimizer_steps = 0
        self._committed_optimizer_steps_in_epoch = 0
        self._committed_optimizer_epoch = 0
        self._non_resumable_reason: str | None = None

    def _bind(self, trainer: Any, pl_module: Any) -> None:
        duplicates = sum(
            isinstance(callback, DataloaderCommitCallback)
            for callback in trainer.callbacks
        )
        if duplicates != 1:
            raise RuntimeError(
                "training requires exactly one DataloaderCommitCallback instance"
            )
        datamodule = trainer.datamodule
        if not isinstance(datamodule, _CommitDataModule):
            raise TypeError(
                "DataloaderCommitCallback requires a commit-capable data module"
            )
        if not isinstance(pl_module, _CommitModel):
            raise TypeError(
                "DataloaderCommitCallback requires optimizer/evaluator model participation"
            )
        self._data_module = datamodule
        self._trainer = trainer
        self._model = pl_module
        self._optimizer_token = _integer(
            pl_module.dataloader_optimizer_success_token,
            "dataloader optimizer success token",
        )
        self._observed_global_step = _integer(
            trainer.global_step,
            "trainer.global_step",
        )
        if self._deferred_state is not None:
            deferred, self._deferred_state = self._deferred_state, None
            self.load_state_dict(deferred)

    def setup(self, trainer: Any, pl_module: Any, stage: str) -> None:
        """Bind exact participants before Lightning restores callback state."""
        if stage == "fit":
            self._bind(trainer, pl_module)

    def on_load_checkpoint(
        self,
        trainer: Any,
        pl_module: Any,
        checkpoint: Mapping[str, Any],
    ) -> None:
        """Stage exact RNG and metric restoration for resumed phases."""
        del pl_module
        state = checkpoint.get(_RNG_CHECKPOINT_KEY)
        if state is None:
            raise ValueError(
                "disk training checkpoint lacks committed process RNG state"
            )
        restore_timing = _rng_restore_timing(state)
        metrics = checkpoint.get(_METRICS_CHECKPOINT_KEY)
        if not isinstance(metrics, Mapping):
            raise TypeError("checkpoint metric state must be a mapping")
        if frozenset(metrics) != _METRICS_KEYS:
            raise ValueError("checkpoint metric state schema is unsupported")
        if metrics["format_version"] != _METRICS_FORMAT:
            raise ValueError("checkpoint metric state version is unsupported")
        train_loss = metrics["train_loss"]
        if train_loss is not None and (
            not isinstance(train_loss, torch.Tensor) or train_loss.numel() != 1
        ):
            raise TypeError("checkpoint train loss must be a scalar tensor")
        self._resume_rng_state = state
        self._resume_train_loss = train_loss
        if (
            restore_timing == _RNG_RESTORE_BEFORE_DATA_ITERATOR
            and trainer.strategy.restore_checkpoint_after_setup
        ):
            self._restore_process_rng_if_needed(
                _RNG_RESTORE_BEFORE_DATA_ITERATOR
            )

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        """Restore epoch-boundary RNG before Lightning creates train iterators."""
        del trainer, pl_module
        self._restore_process_rng_if_needed(_RNG_RESTORE_BEFORE_DATA_ITERATOR)

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        """Restore committed train metrics after Lightning resets its cache."""
        del pl_module
        train_loss = self._resume_train_loss
        if train_loss is not None:
            trainer._logger_connector._callback_metrics["train/loss"] = (
                train_loss.detach().clone()
            )
            self._resume_train_loss = None

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        """Restore pending-validation RNG after the restart train iterator."""
        del trainer, pl_module
        state = self._resume_rng_state
        if (
            state is None
            or _rng_restore_timing(state) != _RNG_RESTORE_AFTER_DATA_ITERATOR
        ):
            return
        self._require_resumable()
        if self._data_module is None:
            raise RuntimeError("callback participants are not bound")
        if self._epoch_completion_pending(self._data_module.sequence_state):
            self._restore_process_rng_if_needed(
                _RNG_RESTORE_AFTER_DATA_ITERATOR
            )

    def _restore_process_rng_if_needed(self, restore_timing: str) -> None:
        state = self._resume_rng_state
        if state is not None and _rng_restore_timing(state) == restore_timing:
            _restore_rng_state(state)
            self._resume_rng_state = None

    def _has_pristine_consumed_boundary(self) -> bool:
        if (
            self._data_module is None
            or self._model is None
            or self._trainer is None
            or self._optimizer_token is None
            or self._active_batch_rng_state is None
            or self._active_batch_evaluator_snapshot is None
        ):
            return False
        sequence_state = self._data_module.sequence_state
        if (
            len(sequence_state.consumed) != 1
            or sequence_state.consumed != sequence_state.pending_group
            or sequence_state.consumed[0]
            != sequence_state.committed_cursor + 1
        ):
            return False
        snapshot = self._model.dataloader_evaluator_snapshot()
        if frozenset(snapshot) != _EVALUATOR_KEYS:
            raise ValueError(
                "model evaluator snapshot keys must be sequence_id, count, and state"
            )
        return (
            _integer(
                self._trainer.global_step,
                "trainer.global_step",
            )
            == sequence_state.committed_global_step
            and _integer(
                self._model.dataloader_optimizer_success_token,
                "dataloader_optimizer_success_token",
            )
            == self._optimizer_token
            and _checkpoint_values_equal(
                snapshot,
                self._active_batch_evaluator_snapshot,
            )
        )

    def _require_resumable(self) -> None:
        reason = self._non_resumable_reason
        if reason is not None:
            raise RuntimeError(f"training is non-resumable: {reason}")

    def _require_optimizer_progress_consistent(self) -> None:
        if self._data_module is None:
            raise RuntimeError("callback participants are not bound")
        sequence_state = self._data_module.sequence_state
        committed_global_step = _integer(
            sequence_state.committed_global_step,
            "committed_global_step",
        )
        if self._committed_optimizer_steps != committed_global_step:
            raise ValueError(
                "committed optimizer step count disagrees with the data boundary"
            )
        if (
            self._committed_optimizer_steps_in_epoch
            > self._committed_optimizer_steps
        ):
            raise ValueError(
                "committed optimizer step count in epoch exceeds total steps"
            )
        cursor = _integer(sequence_state.committed_cursor, "committed_cursor")
        if (cursor == 0) != (self._committed_optimizer_steps_in_epoch == 0):
            raise ValueError(
                "committed optimizer step count in epoch disagrees with "
                "the data boundary"
            )
        committed_epoch = _integer(
            sequence_state.committed_epoch,
            "committed_epoch",
        )
        if cursor > 0 and self._committed_optimizer_epoch != committed_epoch:
            raise ValueError(
                "committed optimizer step epoch disagrees with the data boundary"
            )

    def _require_committed_checkpoint_boundary(self) -> None:
        self._require_resumable()
        if self._data_module is None:
            raise RuntimeError("callback participants are not bound")
        self._require_optimizer_progress_consistent()
        sequence_state = self._data_module.sequence_state
        if (
            sequence_state.consumed or sequence_state.pending_group
        ) and not self._has_pristine_consumed_boundary():
            raise RuntimeError(
                "checkpoint rejected: uncommitted training work may have "
                "mutated model state"
            )

    def _epoch_completion_pending(self, sequence_state: Any) -> bool:
        descriptor_count = len(sequence_state.identity.phase_descriptor_order)
        if descriptor_count <= 0:
            raise ValueError("training descriptor order must not be empty")
        cursor = _integer(sequence_state.committed_cursor, "committed_cursor")
        if cursor == 0 or cursor % descriptor_count != 0:
            return False
        committed_epoch = _integer(
            sequence_state.committed_epoch,
            "committed_epoch",
        )
        if self._trainer is None:
            raise RuntimeError("callback participants are not bound")
        fit_loop = getattr(self._trainer, "fit_loop", None)
        epoch_progress = getattr(fit_loop, "epoch_progress", None)
        current = getattr(epoch_progress, "current", None)
        processed = _integer(
            getattr(current, "processed", None),
            "trainer epoch processed progress",
        )
        completed_epochs = committed_epoch + 1
        if processed not in {committed_epoch, completed_epochs}:
            raise ValueError(
                "trainer epoch progress is inconsistent with the committed "
                "epoch boundary"
            )
        return processed == committed_epoch

    def _checkpoint_rng_state(self) -> Mapping[str, object]:
        self._require_committed_checkpoint_boundary()
        if self._data_module is None:
            raise RuntimeError("callback participants are not bound")
        sequence_state = self._data_module.sequence_state
        if sequence_state.consumed or sequence_state.pending_group:
            assert self._active_batch_rng_state is not None
            return self._active_batch_rng_state
        if (
            sequence_state.issued
            or sequence_state.prepared
            or sequence_state.delivered
        ):
            return _rng_state_dict(
                restore_timing=_RNG_RESTORE_AFTER_DATA_ITERATOR
            )
        descriptor_count = len(sequence_state.identity.phase_descriptor_order)
        if descriptor_count <= 0:
            raise ValueError("training descriptor order must not be empty")
        cursor = _integer(sequence_state.committed_cursor, "committed_cursor")
        epoch_completion_pending = self._epoch_completion_pending(
            sequence_state
        )
        restore_timing = (
            _RNG_RESTORE_AFTER_DATA_ITERATOR
            if epoch_completion_pending
            else (
                _RNG_RESTORE_BEFORE_DATA_ITERATOR
                if cursor > 0 and cursor % descriptor_count == 0
                else _RNG_RESTORE_AFTER_DATA_ITERATOR
            )
        )
        return _rng_state_dict(restore_timing=restore_timing)

    def state_dict(self) -> dict[str, object]:
        """Publish the exact durable data and optimizer progress boundary."""
        if self._data_module is None:
            if self._deferred_state is not None:
                return dict(self._deferred_state)
            return {
                "format_version": _CALLBACK_FORMAT,
                "data_module": {},
                "committed_optimizer_steps": 0,
                "committed_optimizer_steps_in_epoch": 0,
            }
        self._require_committed_checkpoint_boundary()
        return {
            "format_version": _CALLBACK_FORMAT,
            "data_module": self._data_module.state_dict(),
            "committed_optimizer_steps": self._committed_optimizer_steps,
            "committed_optimizer_steps_in_epoch": (
                self._committed_optimizer_steps_in_epoch
            ),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore an exact committed boundary or reject its identity."""
        if not isinstance(state_dict, Mapping):
            raise TypeError("callback state must be a mapping")
        actual = frozenset(state_dict)
        if actual != _CALLBACK_KEYS:
            raise ValueError(
                "callback state keys must match exactly; "
                f"missing={sorted(_CALLBACK_KEYS - actual)!r}, "
                f"extra={sorted(actual - _CALLBACK_KEYS)!r}"
            )
        if state_dict["format_version"] != _CALLBACK_FORMAT:
            raise ValueError(
                f"unsupported callback state version {state_dict['format_version']!r}"
            )
        module_state = state_dict["data_module"]
        if not isinstance(module_state, Mapping):
            raise TypeError("callback data_module state must be a mapping")
        committed_optimizer_steps = _integer(
            state_dict["committed_optimizer_steps"],
            "committed_optimizer_steps",
        )
        committed_optimizer_steps_in_epoch = _integer(
            state_dict["committed_optimizer_steps_in_epoch"],
            "committed_optimizer_steps_in_epoch",
        )
        if self._data_module is None:
            self._deferred_state = dict(state_dict)
            return
        self._data_module.load_state_dict(module_state)
        sequence_state = self._data_module.sequence_state
        self._committed_optimizer_steps = committed_optimizer_steps
        self._committed_optimizer_steps_in_epoch = (
            committed_optimizer_steps_in_epoch
        )
        self._committed_optimizer_epoch = _integer(
            sequence_state.committed_epoch,
            "committed_epoch",
        )
        self._require_optimizer_progress_consistent()
        self._non_resumable_reason = None
        self._observed_global_step = None
        if self._model is not None:
            self._optimizer_token = _integer(
                self._model.dataloader_optimizer_success_token,
                "dataloader optimizer success token",
            )
        self._restore_snapshot = {
            "sequence_id": sequence_state.committed_evaluator_sequence,
            "count": sequence_state.committed_evaluator_count,
            "state": sequence_state.committed_evaluator_state,
        }

    def _restore_evaluator_if_needed(
        self,
        pl_module: _CommitModel,
        *,
        preserve_committed_epoch_state: bool = False,
    ) -> None:
        snapshot = self._restore_snapshot
        if snapshot is None:
            return
        self._restore_snapshot = None
        sequence_state = self._data_module.sequence_state
        descriptor_count = len(sequence_state.identity.phase_descriptor_order)
        if descriptor_count <= 0:
            raise ValueError("training descriptor order must not be empty")
        cursor = _integer(
            sequence_state.committed_cursor,
            "committed_cursor",
        )
        if (
            cursor > 0
            and cursor % descriptor_count == 0
            and not preserve_committed_epoch_state
        ):
            current = pl_module.dataloader_evaluator_snapshot()
            if frozenset(current) != _EVALUATOR_KEYS:
                raise ValueError("model evaluator snapshot keys are invalid")
            snapshot = {
                "sequence_id": snapshot["sequence_id"],
                "count": snapshot["count"],
                "state": current["state"],
            }
        pl_module.dataloader_restore_evaluator(snapshot)

    def _observe_restored_global_step(self, trainer: Any) -> None:
        if self._observed_global_step is not None:
            return
        restored_global_step = _integer(
            trainer.global_step,
            "trainer.global_step",
        )
        if restored_global_step != self._committed_optimizer_steps:
            self._non_resumable_reason = (
                "restored trainer global_step disagrees with committed "
                "optimizer advances"
            )
            self._require_resumable()
        self._observed_global_step = restored_global_step

    def on_train_batch_start(
        self,
        trainer: Any,
        pl_module: Any,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Consume one already delivered sequence before model/evaluator work."""
        if self._data_module is None or not isinstance(
            pl_module, _CommitModel
        ):
            raise RuntimeError("callback participants are not bound")
        self._require_resumable()
        self._observe_restored_global_step(trainer)
        self._restore_process_rng_if_needed(_RNG_RESTORE_AFTER_DATA_ITERATOR)
        self._restore_evaluator_if_needed(pl_module)
        evaluator_snapshot = pl_module.dataloader_evaluator_snapshot()
        if frozenset(evaluator_snapshot) != _EVALUATOR_KEYS:
            raise ValueError(
                "model evaluator snapshot keys must be sequence_id, count, and state"
            )
        self._active_batch_evaluator_snapshot = evaluator_snapshot
        self._active_batch_rng_state = _rng_state_dict()
        self._data_module.consume_sequence(_sequence_id(batch))

    def on_validation_epoch_start(
        self,
        trainer: Any,
        pl_module: Any,
    ) -> None:
        """Restore a committed last-batch evaluator before train finalization."""
        if not isinstance(pl_module, _CommitModel):
            raise TypeError("model no longer satisfies commit participation")
        self._require_resumable()
        self._observe_restored_global_step(trainer)
        self._restore_process_rng_if_needed(_RNG_RESTORE_AFTER_DATA_ITERATOR)
        self._resumed_committed_epoch_end = self._restore_snapshot is not None
        if self._data_module is None:
            raise RuntimeError("callback participants are not bound")
        replay_pending_validation = self._epoch_completion_pending(
            self._data_module.sequence_state
        )
        self._restore_evaluator_if_needed(
            pl_module,
            preserve_committed_epoch_state=replay_pending_validation,
        )

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Dispatch only the pending plateau scheduler step after replay."""
        del pl_module
        if self._resumed_committed_epoch_end:
            fit_loop = trainer.fit_loop
            batch_progress = fit_loop.epoch_loop.batch_progress
            num_training_batches = _integer(
                trainer.num_training_batches,
                "trainer.num_training_batches",
                minimum=1,
            )
            current_ready = _integer(
                batch_progress.current.ready,
                "training batch current.ready",
            )
            if current_ready != num_training_batches:
                raise RuntimeError(
                    "replayed validation requires a fully processed train epoch"
                )
            if not fit_loop.restarting:
                raise RuntimeError(
                    "replayed validation requires Lightning restart state"
                )
            fit_loop.restarting = False
            self._resumed_committed_epoch_end = False

    def _commit_advanced_boundary(
        self,
        trainer: Any,
        pl_module: _CommitModel,
    ) -> None:
        """Commit one proven optimizer advance at the batch-end boundary."""
        self._require_resumable()
        if (
            self._data_module is None
            or self._optimizer_token is None
            or self._observed_global_step is None
        ):
            raise RuntimeError("callback participants are not bound")
        self._require_optimizer_progress_consistent()
        token = _integer(
            pl_module.dataloader_optimizer_success_token,
            "dataloader optimizer success token",
        )
        global_step = _integer(trainer.global_step, "trainer.global_step")
        if token < self._optimizer_token:
            raise ValueError("optimizer success token regressed")
        if global_step < self._observed_global_step:
            raise ValueError("trainer global_step regressed")
        if global_step > self._observed_global_step + 1:
            raise ValueError(
                "trainer global_step jumped by more than one observed boundary"
            )
        step_advanced = global_step == self._observed_global_step + 1
        if token == self._optimizer_token:
            if step_advanced:
                self._non_resumable_reason = (
                    "trainer global_step advanced without a proven raw "
                    "optimizer advance; pending gradients cannot be proven "
                    "to survive for a later commit"
                )
                self._require_resumable()
            return
        if token != self._optimizer_token + 1:
            raise ValueError(
                "optimizer success token must advance exactly once per boundary"
            )
        if not step_advanced:
            self._non_resumable_reason = (
                "optimizer success token advanced without global_step advance"
            )
            self._require_resumable()
        next_optimizer_step = self._committed_optimizer_steps + 1
        if global_step != next_optimizer_step:
            self._non_resumable_reason = (
                "trainer global_step disagrees with committed optimizer "
                "advance count"
            )
            self._require_resumable()
        snapshot = pl_module.dataloader_evaluator_snapshot()
        if frozenset(snapshot) != _EVALUATOR_KEYS:
            raise ValueError(
                "model evaluator snapshot keys must be sequence_id, count, and state"
            )
        epoch = _integer(trainer.current_epoch, "trainer.current_epoch")
        committed = self._data_module.commit_optimizer_step(
            optimizer_succeeded=True,
            model_global_step=next_optimizer_step,
            evaluator_snapshot=snapshot,
            epoch=epoch,
        )
        if not committed:
            raise RuntimeError(
                "successful observed optimizer boundary was not committed"
            )
        if epoch != self._committed_optimizer_epoch:
            self._committed_optimizer_epoch = epoch
            self._committed_optimizer_steps_in_epoch = 0
        self._committed_optimizer_steps = next_optimizer_step
        self._committed_optimizer_steps_in_epoch += 1
        self._require_optimizer_progress_consistent()
        self._optimizer_token = token
        self._active_batch_evaluator_snapshot = None
        self._observed_global_step = global_step
        self._active_batch_rng_state = None

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Commit only after exact transient token and global-step advances."""
        del outputs, batch, batch_idx
        if not isinstance(pl_module, _CommitModel):
            raise TypeError("model no longer satisfies commit participation")
        self._commit_advanced_boundary(trainer, pl_module)

    def on_save_checkpoint(
        self,
        trainer: Any,
        pl_module: Any,
        checkpoint: dict[str, Any],
    ) -> None:
        """Normalize Lightning loop progress to the committed data boundary."""
        if self._data_module is None or not isinstance(
            pl_module, _CommitModel
        ):
            raise RuntimeError("callback participants are not bound")
        self._require_committed_checkpoint_boundary()
        sequence_state = self._data_module.sequence_state
        cursor = _integer(
            sequence_state.committed_cursor,
            "committed_cursor",
        )
        committed_epoch = _integer(
            sequence_state.committed_epoch,
            "committed_epoch",
        )
        optimizer_steps = self._committed_optimizer_steps
        descriptor_count = len(sequence_state.identity.phase_descriptor_order)
        if descriptor_count <= 0:
            raise ValueError("training descriptor order must not be empty")
        current_epoch = _integer(
            trainer.current_epoch,
            "trainer.current_epoch",
        )
        minimum_cursor = committed_epoch * descriptor_count
        maximum_cursor = minimum_cursor + descriptor_count
        if cursor < minimum_cursor or cursor > maximum_cursor:
            raise ValueError(
                "committed cursor is inconsistent with descriptor order and epoch"
            )
        epoch_complete = cursor > 0 and cursor == maximum_cursor
        completed_epochs = committed_epoch + int(epoch_complete)
        active_epochs = completed_epochs + int(not epoch_complete)
        epoch_completion_pending = self._epoch_completion_pending(
            sequence_state
        )
        restored_completed_epochs = (
            committed_epoch if epoch_completion_pending else completed_epochs
        )
        current_batches = (
            descriptor_count
            if epoch_completion_pending
            else cursor % descriptor_count
        )
        current_optimizer_steps = (
            0
            if epoch_complete and not epoch_completion_pending
            else self._committed_optimizer_steps_in_epoch
        )
        if current_epoch not in {
            committed_epoch,
            completed_epochs,
        }:
            raise ValueError(
                "trainer epoch is inconsistent with committed data boundary"
            )
        loops = checkpoint.get("loops")
        fit_loop = (
            loops.get("fit_loop") if isinstance(loops, Mapping) else None
        )
        if not isinstance(fit_loop, dict):
            raise ValueError(
                "Lightning checkpoint lacks mutable fit-loop progress"
            )
        checkpoint["global_step"] = optimizer_steps

        epoch_progress = fit_loop.get("epoch_progress")
        batch_progress = fit_loop.get("epoch_loop.batch_progress")
        optimization = fit_loop.get(
            "epoch_loop.automatic_optimization.optim_progress"
        )
        val_progress = fit_loop.get("epoch_loop.val_loop.batch_progress")
        if not all(
            isinstance(item, dict)
            for item in (
                epoch_progress,
                batch_progress,
                optimization,
                val_progress,
            )
        ):
            raise ValueError(
                "Lightning checkpoint loop progress schema is unsupported"
            )
        for scope in ("total", "current"):
            progress = epoch_progress.get(scope)
            if not isinstance(progress, dict):
                raise ValueError(
                    "Lightning epoch progress schema is unsupported"
                )
            progress.update(
                {
                    "ready": active_epochs,
                    "started": active_epochs,
                    "processed": restored_completed_epochs,
                    "completed": restored_completed_epochs,
                }
            )
        for scope, batches in (
            ("total", cursor),
            ("current", current_batches),
        ):
            progress = batch_progress.get(scope)
            if not isinstance(progress, dict):
                raise ValueError(
                    "Lightning batch progress schema is unsupported"
                )
            progress.update(
                {
                    "ready": batches,
                    "started": batches,
                    "processed": batches,
                    "completed": batches,
                }
            )
        batch_progress["is_last_batch"] = epoch_complete
        optimizer = optimization.get("optimizer")
        if not isinstance(optimizer, dict):
            raise ValueError(
                "Lightning optimizer progress schema is unsupported"
            )
        for operation in ("step", "zero_grad"):
            operation_progress = optimizer.get(operation)
            if not isinstance(operation_progress, dict):
                raise ValueError(
                    "Lightning optimizer progress schema is unsupported"
                )
            for scope, steps in (
                ("total", optimizer_steps),
                ("current", current_optimizer_steps),
            ):
                progress = operation_progress.get(scope)
                if not isinstance(progress, dict):
                    raise ValueError(
                        "Lightning optimizer progress schema is unsupported"
                    )
                for key in tuple(progress):
                    progress[key] = steps
        val_current = val_progress.get("current")
        if not isinstance(val_current, dict):
            raise ValueError(
                "Lightning validation progress schema is unsupported"
            )
        for key in tuple(val_current):
            val_current[key] = 0
        val_progress["is_last_batch"] = False
        state = fit_loop.get("epoch_loop.state_dict")
        if not isinstance(state, dict):
            raise ValueError(
                "Lightning epoch loop state schema is unsupported"
            )
        state["_batches_that_stepped"] = optimizer_steps
        if _RNG_CHECKPOINT_KEY in checkpoint:
            raise ValueError("checkpoint RNG state key is already occupied")
        checkpoint[_RNG_CHECKPOINT_KEY] = self._checkpoint_rng_state()
        if _METRICS_CHECKPOINT_KEY in checkpoint:
            raise ValueError("checkpoint metric state key is already occupied")
        results = trainer._results
        active_train_loss = (
            None
            if results is None
            else results.get("training_step.train/loss")
        )
        if active_train_loss is not None and active_train_loss.update_called:
            train_loss = active_train_loss.compute()
        else:
            train_loss = trainer.callback_metrics.get("train/loss")
        if train_loss is not None and not isinstance(train_loss, torch.Tensor):
            raise TypeError("trainer train loss must be a tensor")
        checkpoint[_METRICS_CHECKPOINT_KEY] = {
            "format_version": _METRICS_FORMAT,
            "train_loss": (
                None
                if train_loss is None
                else train_loss.detach().cpu().clone()
            ),
        }

    def on_exception(
        self,
        trainer: Any,
        pl_module: Any,
        exception: BaseException,
    ) -> None:
        """Intentionally leave all pending sequences transient on failure."""


__all__ = ["DataloaderCommitCallback"]
