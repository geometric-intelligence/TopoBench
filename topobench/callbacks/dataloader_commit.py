"""Commit disk-sampler state only after a proven optimizer/global-step advance."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any, Protocol, runtime_checkable

from lightning import Callback

_CALLBACK_FORMAT = "dataloader-commit-callback-v1"
_CALLBACK_KEYS = frozenset({"format_version", "data_module"})
_EVALUATOR_KEYS = frozenset({"sequence_id", "count", "state"})


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


class DataloaderCommitCallback(Callback):
    """Coordinate one commit boundary after Lightning finishes a train batch."""

    def __init__(self) -> None:
        super().__init__()
        self._data_module: _CommitDataModule | None = None
        self._optimizer_token: int | None = None
        self._observed_global_step: int | None = None
        self._deferred_state: Mapping[str, object] | None = None
        self._restore_snapshot: dict[str, object] | None = None

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

    def state_dict(self) -> dict[str, object]:
        """Publish only the data module's durable committed boundary."""
        if self._data_module is None:
            if self._deferred_state is not None:
                return dict(self._deferred_state)
            return {"format_version": _CALLBACK_FORMAT, "data_module": {}}
        return {
            "format_version": _CALLBACK_FORMAT,
            "data_module": self._data_module.state_dict(),
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
        if self._data_module is None:
            self._deferred_state = dict(state_dict)
            return
        self._data_module.load_state_dict(module_state)
        self._observed_global_step = None
        sequence_state = self._data_module.sequence_state
        self._restore_snapshot = {
            "sequence_id": sequence_state.committed_evaluator_sequence,
            "count": sequence_state.committed_evaluator_count,
            "state": sequence_state.committed_evaluator_state,
        }

    def _restore_evaluator_if_needed(
        self,
        trainer: Any,
        pl_module: _CommitModel,
    ) -> None:
        snapshot = self._restore_snapshot
        if snapshot is None:
            return
        self._restore_snapshot = None
        sequence_state = self._data_module.sequence_state
        current_epoch = _integer(trainer.current_epoch, "trainer.current_epoch")
        if current_epoch > sequence_state.committed_epoch:
            current = pl_module.dataloader_evaluator_snapshot()
            if frozenset(current) != _EVALUATOR_KEYS:
                raise ValueError("model evaluator snapshot keys are invalid")
            snapshot = {
                "sequence_id": snapshot["sequence_id"],
                "count": snapshot["count"],
                "state": current["state"],
            }
        pl_module.dataloader_restore_evaluator(snapshot)

    def on_train_batch_start(
        self,
        trainer: Any,
        pl_module: Any,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Consume one already delivered sequence before model/evaluator work."""
        if self._data_module is None or not isinstance(pl_module, _CommitModel):
            raise RuntimeError("callback participants are not bound")
        if self._observed_global_step is None:
            self._observed_global_step = _integer(
                trainer.global_step,
                "trainer.global_step",
            )
        self._restore_evaluator_if_needed(trainer, pl_module)
        self._data_module.consume_sequence(_sequence_id(batch))

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Commit only after exact transient token and global-step advances."""
        if (
            self._data_module is None
            or self._optimizer_token is None
            or self._observed_global_step is None
        ):
            raise RuntimeError("callback participants are not bound")
        if not isinstance(pl_module, _CommitModel):
            raise TypeError("model no longer satisfies commit participation")
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
                self._observed_global_step = global_step
            return
        if token != self._optimizer_token + 1:
            raise ValueError(
                "optimizer success token must advance exactly once per boundary"
            )
        if not step_advanced:
            raise RuntimeError(
                "optimizer success token advanced without global_step advance"
            )
        snapshot = pl_module.dataloader_evaluator_snapshot()
        if frozenset(snapshot) != _EVALUATOR_KEYS:
            raise ValueError(
                "model evaluator snapshot keys must be sequence_id, count, and state"
            )
        committed = self._data_module.commit_optimizer_step(
            optimizer_succeeded=True,
            model_global_step=global_step,
            evaluator_snapshot=snapshot,
            epoch=_integer(trainer.current_epoch, "trainer.current_epoch"),
        )
        if not committed:
            raise RuntimeError(
                "successful observed optimizer boundary was not committed"
            )
        self._optimizer_token = token
        self._observed_global_step = global_step

    def on_exception(
        self,
        trainer: Any,
        pl_module: Any,
        exception: BaseException,
    ) -> None:
        """Intentionally leave all pending sequences transient on failure."""


__all__ = ["DataloaderCommitCallback"]
