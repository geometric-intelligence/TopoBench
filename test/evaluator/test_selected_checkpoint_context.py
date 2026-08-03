"""Selected-checkpoint context, result-authority, and logger contracts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

import topobench.run as run_module
from topobench.evaluator import (
    EvaluationBatch,
    EvaluationContext,
    EvaluationResult,
    TBEvaluator,
)
from topobench.model import TBModel

_CHECKPOINT_DIGEST = "a" * 64


def _evaluator() -> TBEvaluator:
    return TBEvaluator(
        task="classification",
        num_classes=2,
        metrics=["accuracy", "auroc", "auprc", "somers_d"],
    )


def _finalize(
    evaluator: TBEvaluator,
    *,
    split: str,
    pass_kind: str,
    policy: str = "exact",
    batch_sizes: tuple[int, ...] = (2, 3),
) -> EvaluationResult:
    context = EvaluationContext(
        split=split,
        pass_kind=pass_kind,
        policy=policy,
        task="classification",
        num_classes=2,
        expected_num_examples=sum(batch_sizes),
        checkpoint_id=(
            _CHECKPOINT_DIGEST if pass_kind == "selected_checkpoint" else None
        ),
    )
    evaluator.begin(context)
    offset = 0
    logits = torch.tensor(
        [
            [4.0, -1.0],
            [-1.0, 4.0],
            [3.0, 0.0],
            [0.0, 3.0],
            [0.5, 1.5],
        ]
    )
    targets = torch.tensor([0, 1, 0, 1, 1])
    for batch_size in batch_sizes:
        evaluator.update(
            EvaluationBatch(
                outputs=logits[offset : offset + batch_size],
                targets=targets[offset : offset + batch_size],
                num_examples=batch_size,
                context=context,
            )
        )
        offset += batch_size
    return evaluator.finalize()


def _model(evaluator: TBEvaluator) -> TBModel:
    model = TBModel(
        backbone=MagicMock(),
        readout=MagicMock(task_level="graph"),
        loss=MagicMock(),
        evaluator=evaluator,
    )
    model.log = MagicMock()
    return model


def _logged_scalars(model: TBModel) -> dict[str, object]:
    return {
        str(call.args[0]): call.args[1]
        for call in model.log.call_args_list
        if len(call.args) >= 2
    }


def test_fit_validation_and_selected_reruns_own_independent_exact_contexts() -> (
    None
):
    """Ordinary validation cannot share state with either checkpoint rerun."""
    evaluator = _evaluator()

    ordinary = _finalize(
        evaluator,
        split="val",
        pass_kind="fit_epoch",
        batch_sizes=(5,),
    )
    selected_val = _finalize(
        evaluator,
        split="val",
        pass_kind="selected_checkpoint",
        batch_sizes=(2, 3),
    )
    selected_test = _finalize(
        evaluator,
        split="test",
        pass_kind="selected_checkpoint",
        batch_sizes=(3, 2),
    )

    assert [
        (result.context.split, result.context.pass_kind, result.context.policy)
        for result in (ordinary, selected_val, selected_test)
    ] == [
        ("val", "fit_epoch", "exact"),
        ("val", "selected_checkpoint", "exact"),
        ("test", "selected_checkpoint", "exact"),
    ]
    assert ordinary.context is not selected_val.context
    assert selected_val.context is not selected_test.context
    assert all(
        result.num_examples == 5
        for result in (ordinary, selected_val, selected_test)
    )
    assert ordinary.context.checkpoint_id is None
    assert selected_val.context.checkpoint_id == _CHECKPOINT_DIGEST
    assert selected_test.context.checkpoint_id == _CHECKPOINT_DIGEST
    assert evaluator.state == "idle"


def test_selected_validation_requires_checkpoint_before_evaluator_begin() -> (
    None
):
    """A selected pass cannot begin without an explicit checkpoint digest."""
    evaluator = _evaluator()
    model = _model(evaluator)
    model.set_next_validation_pass_kind("selected_checkpoint")

    with pytest.raises(RuntimeError, match="checkpoint"):
        model.on_validation_epoch_start()

    assert evaluator.state == "idle"
    assert evaluator.context is None
    model.on_validation_epoch_start()
    ordinary_context = evaluator.context
    assert ordinary_context is not None
    assert (
        ordinary_context.split,
        ordinary_context.pass_kind,
        ordinary_context.checkpoint_id,
    ) == ("val", "fit_epoch", None)
    model.abort_evaluation()


def test_selected_test_missing_digest_resets_before_ordinary_test() -> None:
    """A rejected selected test cannot contaminate the next ordinary test."""
    evaluator = _evaluator()
    model = _model(evaluator)
    model.set_next_test_pass_kind("selected_checkpoint")

    with pytest.raises(RuntimeError, match="checkpoint"):
        model.on_test_epoch_start()

    assert evaluator.state == "idle"
    assert evaluator.context is None
    model.on_test_epoch_start()
    ordinary_context = evaluator.context
    assert ordinary_context is not None
    assert (
        ordinary_context.split,
        ordinary_context.pass_kind,
        ordinary_context.checkpoint_id,
    ) == ("test", "fit_epoch", None)
    model.abort_evaluation()


def test_selected_validation_is_one_shot_without_replacing_lifecycle_hooks() -> (
    None
):
    """Selecting one validation pass changes context, never bound hook identity."""
    evaluator = _evaluator()
    model = _model(evaluator)
    validation_start = model.on_validation_epoch_start
    validation_end = model.on_validation_epoch_end

    model.set_selected_checkpoint_id(_CHECKPOINT_DIGEST)
    model.set_next_validation_pass_kind("selected_checkpoint")
    model.on_validation_epoch_start()

    assert evaluator.context is not None
    assert evaluator.context.pass_kind == "selected_checkpoint"
    assert model.on_validation_epoch_start == validation_start
    assert model.on_validation_epoch_end == validation_end
    model.abort_evaluation()

    model.on_validation_epoch_start()
    assert evaluator.context is not None
    assert evaluator.context.pass_kind == "fit_epoch"
    assert model.on_validation_epoch_start == validation_start
    assert model.on_validation_epoch_end == validation_end
    model.abort_evaluation()


def test_selected_log_failure_discards_result_before_clean_rerun() -> None:
    """A failed logger cannot leave a finalized result blocking the next pass."""
    evaluator = _evaluator()
    model = _model(evaluator)
    model.set_selected_checkpoint_id(_CHECKPOINT_DIGEST)
    model.set_next_validation_pass_kind("selected_checkpoint")
    model.on_validation_epoch_start()
    first_context = evaluator.context
    assert first_context is not None
    evaluator.update(
        EvaluationBatch(
            outputs=torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
            targets=torch.tensor([0, 1]),
            num_examples=2,
            context=first_context,
        )
    )
    model.log.side_effect = RuntimeError("logger failed")

    with pytest.raises(RuntimeError, match="logger failed"):
        model.on_validation_epoch_end()

    assert evaluator.state == "idle"
    model.log = MagicMock()
    model.set_selected_checkpoint_id(_CHECKPOINT_DIGEST)
    model.set_next_validation_pass_kind("selected_checkpoint")
    model.on_validation_epoch_start()
    second_context = evaluator.context
    assert second_context is not None
    evaluator.update(
        EvaluationBatch(
            outputs=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            targets=torch.tensor([0, 1]),
            num_examples=2,
            context=second_context,
        )
    )
    model.on_validation_epoch_end()

    result = model.take_selected_checkpoint_result("val")
    assert result.context.checkpoint_id == _CHECKPOINT_DIGEST
    assert result.num_examples == 2


def test_ordinary_test_defaults_to_fit_epoch_without_checkpoint() -> None:
    """An ordinary test loop does not impersonate a selected-checkpoint rerun."""
    evaluator = _evaluator()
    model = _model(evaluator)

    model.on_test_epoch_start()

    context = evaluator.context
    assert context is not None
    assert (
        context.split,
        context.pass_kind,
        context.checkpoint_id,
    ) == ("test", "fit_epoch", None)
    model.abort_evaluation()


def test_selected_test_pass_kind_is_explicit_and_one_shot() -> None:
    """Only an explicitly identified final test uses selected_checkpoint."""
    evaluator = _evaluator()
    model = _model(evaluator)
    model.set_selected_checkpoint_id(_CHECKPOINT_DIGEST)
    model.set_next_test_pass_kind("selected_checkpoint")

    model.on_test_epoch_start()

    selected_context = evaluator.context
    assert selected_context is not None
    assert (
        selected_context.split,
        selected_context.pass_kind,
        selected_context.checkpoint_id,
    ) == ("test", "selected_checkpoint", _CHECKPOINT_DIGEST)
    model.abort_evaluation()

    model.on_test_epoch_start()
    ordinary_context = evaluator.context
    assert ordinary_context is not None
    assert (
        ordinary_context.split,
        ordinary_context.pass_kind,
        ordinary_context.checkpoint_id,
    ) == ("test", "fit_epoch", None)
    model.abort_evaluation()


@pytest.mark.parametrize("failure_phase", ("val", "test"))
def test_rerun_dataloader_failure_does_not_leave_selected_pass_armed(
    failure_phase: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loader construction failure cannot contaminate the next ordinary pass."""
    evaluator = _evaluator()
    model = _model(evaluator)
    model.register_parameter(
        "checkpoint_test_weight",
        torch.nn.Parameter(torch.tensor(1.0)),
    )
    checkpoint_path = tmp_path / "selected.ckpt"
    torch.save({"state_dict": model.state_dict()}, checkpoint_path)
    checkpoint = ModelCheckpoint()
    checkpoint.best_model_path = str(checkpoint_path)

    class FailingDataModule:
        def val_dataloader(self) -> object:
            if failure_phase == "val":
                raise RuntimeError("validation loader failed")
            return object()

        def test_dataloader(self) -> object:
            raise RuntimeError("test loader failed")

    class LifecycleTrainer:
        def validate(self, *, model: TBModel, dataloaders: object) -> None:
            del dataloaders
            model.on_validation_epoch_start()
            context = evaluator.context
            assert context is not None
            evaluator.update(
                EvaluationBatch(
                    outputs=torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
                    targets=torch.tensor([0, 1]),
                    num_examples=2,
                    context=context,
                )
            )
            model.on_validation_epoch_end()

        def test(self, *, model: TBModel, dataloaders: object) -> None:
            del model, dataloaders
            raise AssertionError("trainer.test must not run")

    monkeypatch.setattr(
        run_module.hydra.utils,
        "instantiate",
        lambda *args, **kwargs: LifecycleTrainer(),
    )
    cfg = OmegaConf.create(
        {
            "trainer": {},
            "paths": {"checkpoint_dir": str(tmp_path)},
            "enable_progress_bar": False,
            "delete_checkpoint_after_test": False,
        }
    )

    with pytest.raises(
        RuntimeError,
        match=f"{failure_phase}.*loader failed",
    ):
        run_module.rerun_best_model_checkpoint(
            checkpoint_model=model,
            cfg=cfg,
            datamodule=FailingDataModule(),
            device=torch.device("cpu"),
            callbacks=[checkpoint],
            logger=[],
        )

    assert evaluator.state == "idle"
    assert evaluator.context is None
    if failure_phase == "val":
        model.on_validation_epoch_start()
    else:
        model.on_test_epoch_start()
    context = evaluator.context
    assert context is not None
    assert (
        context.split,
        context.pass_kind,
        context.checkpoint_id,
    ) == (failure_phase, "fit_epoch", None)
    model.abort_evaluation()


@pytest.mark.parametrize("failure_phase", ("val", "test"))
def test_rerun_trainer_failure_does_not_leave_selected_pass_armed(
    failure_phase: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trainer failure before epoch-start cannot contaminate an ordinary pass."""
    evaluator = _evaluator()
    model = _model(evaluator)
    model.register_parameter(
        "checkpoint_test_weight",
        torch.nn.Parameter(torch.tensor(1.0)),
    )
    checkpoint_path = tmp_path / "trainer-failure.ckpt"
    torch.save({"state_dict": model.state_dict()}, checkpoint_path)
    checkpoint = ModelCheckpoint()
    checkpoint.best_model_path = str(checkpoint_path)

    class DataModule:
        def val_dataloader(self) -> object:
            return object()

        def test_dataloader(self) -> object:
            return object()

    class FailingTrainer:
        def validate(self, *, model: TBModel, dataloaders: object) -> None:
            del dataloaders
            if failure_phase == "val":
                raise RuntimeError("validation trainer failed")
            model.on_validation_epoch_start()
            context = evaluator.context
            assert context is not None
            evaluator.update(
                EvaluationBatch(
                    outputs=torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
                    targets=torch.tensor([0, 1]),
                    num_examples=2,
                    context=context,
                )
            )
            model.on_validation_epoch_end()

        def test(self, *, model: TBModel, dataloaders: object) -> None:
            del model, dataloaders
            raise RuntimeError("test trainer failed")

    monkeypatch.setattr(
        run_module.hydra.utils,
        "instantiate",
        lambda *args, **kwargs: FailingTrainer(),
    )
    cfg = OmegaConf.create(
        {
            "trainer": {},
            "paths": {"checkpoint_dir": str(tmp_path)},
            "enable_progress_bar": False,
            "delete_checkpoint_after_test": False,
        }
    )

    with pytest.raises(
        RuntimeError,
        match=f"{failure_phase}.*trainer failed",
    ):
        run_module.rerun_best_model_checkpoint(
            checkpoint_model=model,
            cfg=cfg,
            datamodule=DataModule(),
            device=torch.device("cpu"),
            callbacks=[checkpoint],
            logger=[],
        )

    assert evaluator.state == "idle"
    assert evaluator.context is None
    if failure_phase == "val":
        model.on_validation_epoch_start()
    else:
        model.on_test_epoch_start()
    context = evaluator.context
    assert context is not None
    assert (
        context.split,
        context.pass_kind,
        context.checkpoint_id,
    ) == (failure_phase, "fit_epoch", None)
    model.abort_evaluation()


def test_model_logger_keeps_stable_phase_keys_and_exact_integer_totals() -> (
    None
):
    """Lightning phase keys remain stable and never average per-batch counts."""
    evaluator = _evaluator()
    cases = (
        ("train", "fit_epoch"),
        ("val", "fit_epoch"),
        ("test", "selected_checkpoint"),
    )

    for split, pass_kind in cases:
        result = _finalize(
            evaluator,
            split=split,
            pass_kind=pass_kind,
            batch_sizes=(2, 3),
        )
        model = _model(evaluator)
        model._log_evaluation_result(result)

        logged = _logged_scalars(model)
        prefix = f"{split}/"
        assert set(logged) == {
            *(f"{prefix}{name}" for name in result.metrics),
            f"{prefix}num_examples",
        }
        for name, value in result.metrics.items():
            assert float(logged[f"{prefix}{name}"]) == float(value)
        assert logged[f"{prefix}num_examples"] == result.num_examples == 5
        assert type(logged[f"{prefix}num_examples"]) is int
        assert not any(
            name.startswith("evaluations/best_checkpoint/") for name in logged
        )


def test_audit_policy_metadata_stays_separate_from_canonical_metric_names() -> (
    None
):
    """Exact names stay canonical while status and thresholds remain structured."""
    result = _finalize(
        _evaluator(),
        split="test",
        pass_kind="selected_checkpoint",
        policy="audit",
    )

    assert {"auroc", "auprc", "somers_d"} <= result.metrics.keys()
    assert {
        "auroc_online",
        "auroc_online_abs_error",
        "auprc_online",
        "auprc_online_abs_error",
        "somers_d_online",
        "somers_d_online_abs_error",
    } <= result.metrics.keys()
    assert result.status["auroc"] == "exact"
    assert result.status["auprc"] == "exact"
    assert result.status["somers_d"] == "exact"
    assert result.status["auroc_online"] == "approximate"
    assert isinstance(result.provenance, Mapping)
    assert "thresholds" in result.provenance
    assert set(result.metrics).isdisjoint(result.provenance)
