"""End-to-end tests for native heterogeneous node-classification pipelines."""

from __future__ import annotations

import hashlib
import logging
import math
import pickle
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import hydra
import pytest
import torch
from hydra.core.global_hydra import GlobalHydra
from lightning import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data, HeteroData

import topobench.evaluator as evaluator_module
import topobench.run as run_module
from topobench.dataloader.heterogeneous import HeterogeneousNodeDataModule
from topobench.evaluator import EvaluationResult
from topobench.model import TBModel
from topobench.nn.capabilities import validate_capability_composition
from topobench.run import rerun_best_model_checkpoint, run
from topobench.utils import instantiate_callbacks
from topobench.utils.checkpoint_io import (
    TrustedCheckpointIO,
    checkpoint_manifest_path,
    checkpoint_state_path,
)
from topobench.utils.config_resolvers import register_all_resolvers
from topobench.utils.model_instantiation import instantiate_model

EXPERIMENTS = (
    pytest.param(
        "heterogeneous_synthetic_hgt_full",
        "hgt",
        id="hgt-full-batch",
    ),
    pytest.param(
        "heterogeneous_synthetic_heterosage_full",
        "heterosage",
        id="heterosage-full-batch",
    ),
)

NEIGHBOR_EXPERIMENTS = (
    pytest.param(
        "heterogeneous_synthetic_hgt_neighbor",
        "hgt",
        id="hgt-neighbor",
    ),
    pytest.param(
        "heterogeneous_synthetic_heterosage_neighbor",
        "heterosage",
        id="heterosage-neighbor",
    ),
)


def _execute_checkpoint_reducer_canary(marker_path: str) -> torch.Tensor:
    """Record unsafe deserialization without performing a harmful action."""
    Path(marker_path).write_text("executed", encoding="utf-8")
    return torch.tensor(0)


class _CheckpointReducerCanary:
    """Pickle payload whose reducer is inert until unsafe deserialization."""

    def __init__(self, marker_path: Path) -> None:
        self._marker_path = marker_path

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (
            _execute_checkpoint_reducer_canary,
            (str(self._marker_path),),
        )


def _compose(
    experiment: str,
    *,
    tmp_path: Path | None = None,
    max_epochs: int = 1,
    logger_override: str = "[]",
    extra_overrides: tuple[str, ...] = (),
) -> DictConfig:
    """Compose one production experiment with bounded test-only overrides."""
    GlobalHydra.instance().clear()
    register_all_resolvers()
    overrides = [f"experiment={experiment}"]
    if tmp_path is not None:
        output_dir = tmp_path / experiment
        overrides.extend(
            [
                # Hydra 1.3 represents a null config-group choice as an empty
                # override list; the resulting root has no logger config.
                f"logger={logger_override}",
                "paths=test",
                f"paths.output_dir={output_dir}",
                f"paths.work_dir={output_dir}",
                f"paths.data_dir={tmp_path / 'datasets'}",
                f"++paths.cache_dir={tmp_path / 'cache'}",
                f"trainer.default_root_dir={output_dir}",
                (
                    "callbacks.model_checkpoint.dirpath="
                    f"{output_dir / 'checkpoints'}"
                ),
                f"trainer.max_epochs={max_epochs}",
                "++trainer.limit_train_batches=2",
                "++trainer.limit_val_batches=2",
                "++trainer.limit_test_batches=2",
            ]
        )
    overrides.extend(extra_overrides)
    with hydra.initialize(version_base="1.3", config_path="../../configs"):
        return hydra.compose(
            config_name="run.yaml",
            overrides=overrides,
        )


def _checkpoint_callback(callbacks: list[object]) -> ModelCheckpoint:
    """Return the one real checkpoint callback used by the run."""
    matches = [
        callback
        for callback in callbacks
        if isinstance(callback, ModelCheckpoint)
    ]
    assert len(matches) == 1
    return matches[0]


def _assert_finite_metrics(
    metrics: Mapping[str, Any],
    *,
    prefix: str,
) -> None:
    """Require at least one finite metric below a phase prefix."""
    matching = {
        name: value
        for name, value in metrics.items()
        if name.startswith(prefix)
    }
    assert matching, f"No metrics with prefix {prefix!r}: {sorted(metrics)}"
    assert all(math.isfinite(float(value)) for value in matching.values())


def _logged_rerun_metrics(
    sink: MagicMock,
    *,
    phase: str,
) -> dict[str, float]:
    """Return one phase's metrics from a mocked external W&B sink."""
    prefix = f"evaluations/best_checkpoint/{phase}/"
    matching = [
        call.args[0]
        for call in sink.log_metrics.call_args_list
        if call.args
        and isinstance(call.args[0], Mapping)
        and call.args[0]
        and all(str(name).startswith(prefix) for name in call.args[0])
    ]
    assert len(matching) == 1
    return {
        str(name).removeprefix(prefix): float(value)
        for name, value in matching[0].items()
    }


def _assert_repeated_rerun_metrics(
    first: Mapping[str, float],
    second: Mapping[str, float],
) -> None:
    """Compare repeated sampled reruns without hiding metric drift."""
    assert first.keys() == second.keys()
    for name in first:
        assert math.isfinite(first[name])
        assert math.isfinite(second[name])
        assert second[name] == first[name]


@pytest.mark.parametrize(("experiment", "model_name"), EXPERIMENTS)
def test_full_batch_experiment_contract(
    experiment: str,
    model_name: str,
) -> None:
    """Both production configs share the heterogeneous experiment identity."""
    cfg = _compose(experiment)

    assert (
        cfg.data_pipeline._target_
        == "topobench.data.pipelines.HeterogeneousNodeDataPipeline"
    )
    assert cfg.dataset.loader.parameters.data_name == "SyntheticHeterogeneous"
    assert cfg.dataset.dataloader_params.mode == "full_batch"
    assert cfg.model.model_domain == "heterogeneous"
    assert cfg.model.model_name == model_name
    assert cfg.seed == 0
    assert cfg.train is True
    assert cfg.test is True
    assert cfg.optimizer.parameters.lr == pytest.approx(0.01)
    assert cfg.optimizer.parameters.weight_decay == pytest.approx(0.0)
    assert cfg.trainer.accelerator == "cpu"
    assert cfg.trainer.devices == 1
    assert cfg.trainer.min_epochs == 1
    assert cfg.trainer.max_epochs == 50
    assert cfg.trainer.check_val_every_n_epoch == 1

    wandb = cfg.logger.wandb
    assert wandb.project == "topobench-heterogeneous"
    assert wandb.offline is False
    assert wandb.name == (
        f"SyntheticHeterogeneous-{model_name}-full_batch-seed0"
    )
    assert wandb.group == "SyntheticHeterogeneous"
    assert wandb.job_type == "full_batch"
    assert list(wandb.tags) == [
        "heterogeneous",
        "synthetic",
        "full_batch",
        model_name,
    ]


def test_heterogeneous_pipeline_emits_observed_capability(
    tmp_path: Path,
) -> None:
    """Capability metadata comes from validated runtime node stores."""
    cfg = _compose("heterogeneous_synthetic_hgt_full", tmp_path=tmp_path)

    output = hydra.utils.instantiate(cfg.data_pipeline).build(cfg)
    capability = output.capability_spec

    assert capability is not None
    assert capability.selector == "heterogeneous/SyntheticHeterogeneous"
    assert capability.data_domain == "heterogeneous"
    assert capability.output_kind == "heterogeneous"
    assert capability.feature_widths == output.data_spec.input_channels
    assert capability.feature_widths == (
        ("author", 8),
        ("paper", 5),
        ("venue", 1),
    )
    assert capability.num_classes == output.data_spec.num_classes == 2
    assert capability.target_node_type == "author"


@pytest.mark.parametrize(
    ("experiment", "model_name"),
    NEIGHBOR_EXPERIMENTS,
)
def test_neighbor_experiment_contract(
    experiment: str,
    model_name: str,
) -> None:
    """Sampled production configs expose one explicit, reproducible contract."""
    cfg = _compose(experiment)

    assert (
        cfg.data_pipeline._target_
        == "topobench.data.pipelines.HeterogeneousNodeDataPipeline"
    )
    assert cfg.dataset.loader.parameters.data_name == "SyntheticHeterogeneous"
    dataloader = cfg.dataset.dataloader_params
    assert dataloader.mode == "neighbor"
    assert dataloader.batch_size == 4
    assert list(dataloader.num_neighbors) == [3, 2]
    assert dataloader.num_workers == 0
    assert dataloader.persistent_workers is False
    assert dataloader.train_shuffle is True
    assert dataloader.replace is False
    assert dataloader.subgraph_type == "directional"
    assert dataloader.filter_per_worker is False
    assert dataloader.evaluation_protocol == "sampled_neighbor_fixed"
    assert dataloader.evaluation_seed == 0
    assert cfg.model.model_domain == "heterogeneous"
    assert cfg.model.model_name == model_name
    assert cfg.model.backbone.num_layers == len(dataloader.num_neighbors)

    wandb = cfg.logger.wandb
    assert wandb.project == "topobench-heterogeneous"
    assert wandb.name == (
        f"SyntheticHeterogeneous-{model_name}-neighbor-seed0"
    )
    assert wandb.group == "SyntheticHeterogeneous"
    assert wandb.job_type == "neighbor"
    assert list(wandb.tags) == [
        "heterogeneous",
        "synthetic",
        "neighbor",
        model_name,
    ]


@pytest.mark.parametrize(("experiment", "_model_name"), EXPERIMENTS)
def test_full_batch_training_validation_checkpoint_and_best_rerun(
    experiment: str,
    _model_name: str,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Run one real training epoch, including both best-checkpoint reruns."""
    cfg = _compose(experiment, tmp_path=tmp_path)
    caplog.set_level(logging.INFO, logger="topobench.run")

    fit_metrics, objects = run(cfg)
    selected = objects["selected_checkpoint_results"]
    assert tuple(selected) == ("val", "test")
    assert all(
        isinstance(selected[split], EvaluationResult)
        for split in ("val", "test")
    )
    assert selected["val"] is not selected["test"]
    assert (
        selected["val"].context.checkpoint_id
        == selected["test"].context.checkpoint_id
    )
    assert selected["val"].context.checkpoint_id is not None

    _assert_finite_metrics(fit_metrics, prefix="train/")
    _assert_finite_metrics(fit_metrics, prefix="val/")
    fit_trainer = objects["trainer"]
    assert fit_trainer.current_epoch == 1

    checkpoint = _checkpoint_callback(objects["callbacks"])
    best_path = Path(checkpoint.best_model_path)
    assert checkpoint.best_model_path
    assert best_path.is_file()
    assert best_path.stat().st_size > 0

    assert "evaluations/best_checkpoint/val/num_examples" in fit_metrics
    assert "evaluations/best_checkpoint/test/num_examples" in fit_metrics

    # The production rerun attaches the model to a dedicated trainer whose
    # final phase is test.
    rerun_model = objects["model"]
    rerun_trainer = rerun_model.trainer
    assert rerun_trainer is not fit_trainer
    assert all(
        math.isfinite(float(value))
        for value in selected["test"].metrics.values()
    )
    assert rerun_model.state_str == "Test"
    assert isinstance(
        objects["datamodule"],
        HeterogeneousNodeDataModule,
    )
    assert objects["datamodule"].mode == "full_batch"
    target_type = objects["datamodule"].spec.target_node_type
    target_store = objects["datamodule"].data[target_type]
    assert int(fit_metrics["train/num_examples"]) == int(
        target_store.train_mask.sum()
    )
    assert int(fit_metrics["val/num_examples"]) == int(
        target_store.val_mask.sum()
    )
    assert selected["val"].num_examples == int(target_store.val_mask.sum())
    assert selected["test"].num_examples == int(target_store.test_mask.sum())
    for split in ("val", "test"):
        count = fit_metrics[
            f"evaluations/best_checkpoint/{split}/num_examples"
        ]
        assert count == selected[split].num_examples
        assert type(count) is int


@pytest.mark.parametrize(
    ("experiment", "_model_name"),
    NEIGHBOR_EXPERIMENTS,
)
def test_neighbor_training_validation_checkpoint_and_best_rerun(
    experiment: str,
    _model_name: str,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Run a real sampled lifecycle through both best-checkpoint reruns."""
    cfg = _compose(
        experiment,
        tmp_path=tmp_path,
        max_epochs=1,
        extra_overrides=(
            "trainer.limit_val_batches=1.0",
            "trainer.limit_test_batches=1.0",
        ),
    )
    assert cfg.trainer.limit_val_batches == 1.0
    assert cfg.trainer.limit_test_batches == 1.0
    caplog.set_level(logging.INFO, logger="topobench.run")

    fit_metrics, objects = run(cfg)
    selected = objects["selected_checkpoint_results"]
    assert tuple(selected) == ("val", "test")
    assert all(
        isinstance(selected[split], EvaluationResult)
        for split in ("val", "test")
    )
    assert selected["val"] is not selected["test"]

    _assert_finite_metrics(fit_metrics, prefix="train/")
    _assert_finite_metrics(fit_metrics, prefix="val/")
    fit_trainer = objects["trainer"]
    assert fit_trainer.current_epoch == 1
    assert fit_trainer.global_step > 1

    checkpoint = _checkpoint_callback(objects["callbacks"])
    best_path = Path(checkpoint.best_model_path)
    assert checkpoint.best_model_path
    assert best_path.is_file()
    assert best_path.stat().st_size > 0

    assert "evaluations/best_checkpoint/val/num_examples" in fit_metrics
    assert "evaluations/best_checkpoint/test/num_examples" in fit_metrics

    rerun_model = objects["model"]
    rerun_trainer = rerun_model.trainer
    assert rerun_trainer is not fit_trainer
    assert all(
        math.isfinite(float(value))
        for value in selected["test"].metrics.values()
    )
    assert rerun_model.state_str == "Test"

    datamodule = objects["datamodule"]
    assert isinstance(datamodule, HeterogeneousNodeDataModule)
    assert datamodule.mode == "neighbor"
    assert datamodule.evaluation_protocol == "sampled_neighbor_fixed"
    assert datamodule.val_dataloader() is datamodule.val_dataloader()
    assert datamodule.test_dataloader() is datamodule.test_dataloader()
    assert not hasattr(datamodule.val_dataloader(), "_cached_batches")
    assert not hasattr(datamodule.test_dataloader(), "_cached_batches")

    target_type = datamodule.spec.target_node_type
    test_batch_sizes = [
        int(batch[target_type].batch_size)
        for batch in datamodule.test_dataloader()
    ]
    expected_test_count = int(datamodule.data[target_type].test_mask.sum())
    assert test_batch_sizes == [4, 4, 2]
    assert sum(test_batch_sizes) == expected_test_count
    target_store = datamodule.data[target_type]
    assert int(fit_metrics["train/num_examples"]) == (
        int(cfg.trainer.limit_train_batches)
        * int(cfg.dataset.dataloader_params.batch_size)
    )
    assert int(fit_metrics["val/num_examples"]) == int(
        target_store.val_mask.sum()
    )
    assert selected["val"].num_examples == int(target_store.val_mask.sum())
    assert selected["test"].num_examples == expected_test_count
    selected_test_count = fit_metrics[
        "evaluations/best_checkpoint/test/num_examples"
    ]
    assert selected_test_count == expected_test_count
    assert type(selected_test_count) is int

    rerun_sinks = [
        MagicMock(spec=WandbLogger),
        MagicMock(spec=WandbLogger),
    ]
    repeated_results: list[dict[str, EvaluationResult]] = []
    for sink in rerun_sinks:
        rng_before = torch.random.get_rng_state().clone()
        repeated_results.append(
            rerun_best_model_checkpoint(
                checkpoint_model=rerun_model,
                cfg=cfg,
                datamodule=datamodule,
                device=torch.device("cpu"),
                callbacks=objects["callbacks"],
                logger=[sink],
                prediction_row_adapter=objects[
                    "pipeline_output"
                ].prediction_row_adapter,
                supervision_counts=objects[
                    "pipeline_output"
                ].supervision_counts,
                provenance_input=objects["pipeline_output"].provenance_input,
                source_graph_id=objects["pipeline_output"].source_graph_id,
            )
        )
        assert torch.equal(torch.random.get_rng_state(), rng_before)
        assert sink.log_metrics.call_count == 2

    for phase in ("val", "test"):
        first_metrics = _logged_rerun_metrics(
            rerun_sinks[0],
            phase=phase,
        )
        second_metrics = _logged_rerun_metrics(
            rerun_sinks[1],
            phase=phase,
        )
        _assert_repeated_rerun_metrics(first_metrics, second_metrics)
        assert repeated_results[0][phase] is not repeated_results[1][phase]
        assert repeated_results[0][phase].context.split == phase
        assert (
            repeated_results[0][phase].context.pass_kind
            == "selected_checkpoint"
        )
        assert repeated_results[0][phase].context.policy == "exact"
        assert repeated_results[0][phase].num_examples == int(
            first_metrics["num_examples"]
        )

        _, direct_metrics = _fixed_evaluation_signature(
            rerun_model,
            datamodule,
            phase,
        )
        direct_metrics.pop("loss")
        assert first_metrics.keys() == direct_metrics.keys()
        assert first_metrics == direct_metrics


@pytest.mark.parametrize(
    ("experiment", "_model_name"),
    NEIGHBOR_EXPERIMENTS,
)
def test_neighbor_process_outputs_accounts_for_every_seed_once(
    experiment: str,
    _model_name: str,
    tmp_path: Path,
) -> None:
    """Every real sampled loss supervises target seeds and no context nodes."""
    seed_everything(0, workers=True)
    cfg = _compose(
        experiment,
        tmp_path=tmp_path,
        extra_overrides=(
            "dataset.dataloader_params.train_shuffle=false",
            "model.backbone.dropout=0.0",
        ),
    )
    datamodule, model, pipeline_output = _build_heterogeneous_runtime(cfg)
    target_type = datamodule.spec.target_node_type
    prediction_adapter = pipeline_output.prediction_row_adapter
    observed_context = False

    for phase in ("train", "val", "test"):
        expected_ids = (
            datamodule.data[target_type][f"{phase}_mask"]
            .nonzero(as_tuple=False)
            .view(-1)
        )
        observed_ids: list[torch.Tensor] = []
        supervised_total = 0
        model.state_str = _phase_state(phase)

        for batch in _phase_loader(datamodule, phase):
            target_store = batch[target_type]
            seed_count = int(target_store.batch_size)
            target_count = int(target_store.num_nodes)
            assert seed_count > 1
            observed_context |= target_count > seed_count
            seed_ids = target_store.n_id[:seed_count]
            assert bool(torch.isin(seed_ids, expected_ids).all())
            observed_ids.append(seed_ids.detach().cpu())

            model.zero_grad(set_to_none=True)
            unfiltered = model.forward(batch)
            supervised = model.supervision_adapter.select(
                unfiltered,
                batch,
                model.state_str,
            )
            payload = prediction_adapter.adapt(
                batch,
                supervised,
                phase=phase,
            )
            assert isinstance(payload, evaluator_module.PredictionPayload)
            assert payload.identity.key == (
                "source_graph_id",
                "target_node_type",
                "n_id",
            )
            exported_ids = payload.identity.columns["n_id"]
            assert exported_ids.tolist() == seed_ids.tolist()
            assert len(exported_ids) == seed_count
            assert payload.prediction.shape[0] == seed_count
            assert (
                payload.identity.columns["target_node_type"].tolist()
                == [target_type] * seed_count
            )
            assert (
                payload.identity.columns["source_graph_id"].tolist()
                == [pipeline_output.source_graph_id] * seed_count
            )
            raw_logits = cast(torch.Tensor, unfiltered["logits"])
            raw_labels = cast(torch.Tensor, unfiltered["labels"])
            raw_logits.retain_grad()
            assert raw_logits.size(0) == target_count

            processed = model.process_outputs(unfiltered, batch)
            assert processed["num_supervised_examples"] == seed_count
            assert torch.equal(processed["logits"], raw_logits[:seed_count])
            assert torch.equal(processed["labels"], raw_labels[:seed_count])
            supervised_total += int(processed["num_supervised_examples"])

            output = model.loss(model_out=processed, batch=batch)
            loss = cast(torch.Tensor, output["loss"])
            assert torch.isfinite(loss)
            loss.backward()
            assert raw_logits.grad is not None
            assert torch.count_nonzero(raw_logits.grad[:seed_count]) > 0
            assert torch.count_nonzero(raw_logits.grad[seed_count:]) == 0
            _assert_expected_module_gradients(model)

        all_ids = torch.cat(observed_ids)
        assert supervised_total == int(expected_ids.numel())
        assert all_ids.numel() == all_ids.unique().numel()
        assert torch.equal(all_ids, expected_ids)

    assert observed_context


@pytest.mark.parametrize(
    ("experiment", "_model_name"),
    NEIGHBOR_EXPERIMENTS,
)
def test_neighbor_fixed_evaluation_repeats_batches_and_metrics(
    experiment: str,
    _model_name: str,
    tmp_path: Path,
) -> None:
    """Fixed validation/test sampling reproduces seed batches and real metrics."""
    seed_everything(0, workers=True)
    cfg = _compose(
        experiment,
        tmp_path=tmp_path,
        extra_overrides=("model.backbone.dropout=0.0",),
    )
    datamodule, model, _ = _build_heterogeneous_runtime(cfg)
    assert datamodule.evaluation_protocol == "sampled_neighbor_fixed"

    for phase in ("val", "test"):
        first_batches, first_metrics = _fixed_evaluation_signature(
            model,
            datamodule,
            phase,
        )
        second_batches, second_metrics = _fixed_evaluation_signature(
            model,
            datamodule,
            phase,
        )

        assert len(first_batches) > 1
        assert second_batches == first_batches
        assert second_metrics.keys() == first_metrics.keys()
        assert all(math.isfinite(value) for value in first_metrics.values())
        assert second_metrics == pytest.approx(first_metrics, abs=0.0, rel=0.0)


def test_full_batch_best_rerun_logs_authoritative_result_namespaces(
    tmp_path: Path,
) -> None:
    """Every external logger receives scalars from the returned result objects."""
    cfg = _compose(
        "heterogeneous_synthetic_hgt_full",
        tmp_path=tmp_path,
        extra_overrides=(
            "evaluator.policy.val=audit",
            "evaluator.policy.test=audit",
        ),
    )
    _, objects = run(cfg)
    checkpoint = _checkpoint_callback(objects["callbacks"])
    best_path = Path(checkpoint.best_model_path)
    assert best_path.is_file()

    sinks = [
        MagicMock(spec=WandbLogger),
        MagicMock(spec=CSVLogger),
    ]
    sinks[1].log_dir = str(tmp_path / "csv-logger")
    results = rerun_best_model_checkpoint(
        checkpoint_model=objects["model"],
        cfg=cfg,
        datamodule=objects["datamodule"],
        device=torch.device("cpu"),
        callbacks=objects["callbacks"],
        logger=sinks,
        prediction_row_adapter=objects[
            "pipeline_output"
        ].prediction_row_adapter,
        supervision_counts=objects["pipeline_output"].supervision_counts,
        provenance_input=objects["pipeline_output"].provenance_input,
        source_graph_id=objects["pipeline_output"].source_graph_id,
    )

    assert tuple(results) == ("val", "test")
    assert results["val"] is not results["test"]
    assert all(sink.log_metrics.call_count == 2 for sink in sinks)
    assert all(
        len(call.args) == 1 and set(call.kwargs) == {"step"}
        for sink in sinks
        for call in sink.log_metrics.call_args_list
    )
    logged_calls = [
        [call.args[0] for call in sink.log_metrics.call_args_list]
        for sink in sinks
    ]
    for index, split in enumerate(("val", "test")):
        result = results[split]
        assert result.context.policy == "audit"
        namespace = f"evaluations/best_checkpoint/{split}/"
        logged = logged_calls[0][index]
        assert logged_calls[0][index] == logged_calls[1][index]
        assert set(logged) == {
            *(f"{namespace}{name}" for name in result.metrics),
            f"{namespace}num_examples",
        }
        assert {
            "accuracy",
            "auroc",
            "auprc",
            "somers_d",
        } <= result.metrics.keys()
        assert result.status["auroc"] == "exact"
        assert result.status["auprc"] == "exact"
        assert {
            "auroc_online",
            "auroc_online_abs_error",
            "auprc_online",
            "auprc_online_abs_error",
            "somers_d_online",
            "somers_d_online_abs_error",
        } <= result.metrics.keys()
        assert result.status["somers_d"] == "exact"
        assert set(result.metrics).isdisjoint(result.provenance)
        for name, value in result.metrics.items():
            assert float(logged[f"{namespace}{name}"]) == float(value)
        count_key = f"{namespace}num_examples"
        assert logged[count_key] == result.num_examples
        assert type(logged[count_key]) is int

    checkpoint_state = torch.load(
        best_path,
        map_location="cpu",
        weights_only=False,
    )["state_dict"]
    current_state = objects["model"].state_dict()
    assert checkpoint_state.keys() == current_state.keys()
    assert all(
        torch.equal(checkpoint_state[name], current_state[name].cpu())
        for name in checkpoint_state
    )


def test_selected_checkpoint_rejects_reducer_without_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selected rerun rejects executable pickle before trainer construction."""
    cfg = _compose(
        "heterogeneous_synthetic_hgt_full",
        tmp_path=tmp_path,
        extra_overrides=(
            "execution_profile=experimental",
            "evaluation_artifacts.enabled=false",
        ),
    )
    datamodule, model, _ = _build_heterogeneous_runtime(cfg)
    callbacks = instantiate_callbacks(cfg.callbacks)
    selected_checkpoint = _checkpoint_callback(callbacks)
    checkpoint_path = (
        Path(cfg.paths.output_dir) / "checkpoints" / "poisoned.ckpt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path = tmp_path / "unsafe-reducer-executed"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": 0,
            "global_step": 0,
            "reducer_canary": _CheckpointReducerCanary(marker_path),
        },
        checkpoint_path,
    )
    assert not marker_path.exists()
    selected_checkpoint.best_model_path = str(checkpoint_path)
    construction_calls: list[object] = []

    def construction_must_not_start(*args: object, **kwargs: object) -> object:
        construction_calls.append((args, kwargs))
        raise AssertionError("unsafe checkpoint must fail before construction")

    monkeypatch.setattr(
        run_module.hydra.utils,
        "instantiate",
        construction_must_not_start,
    )

    with pytest.raises(pickle.UnpicklingError):
        rerun_best_model_checkpoint(
            checkpoint_model=model,
            cfg=cfg,
            datamodule=datamodule,
            device=torch.device("cpu"),
            callbacks=callbacks,
            logger=[],
        )

    assert not marker_path.exists()
    assert construction_calls == []


def test_selected_checkpoint_digest_matches_the_bytes_loaded_during_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A path replacement cannot split checkpoint identity from loaded state."""
    cfg = _compose(
        "heterogeneous_synthetic_hgt_full",
        tmp_path=tmp_path,
        extra_overrides=(
            "execution_profile=experimental",
            "evaluation_artifacts.enabled=false",
        ),
    )
    datamodule, model, _ = _build_heterogeneous_runtime(cfg)
    callbacks = instantiate_callbacks(cfg.callbacks)
    selected_checkpoint = _checkpoint_callback(callbacks)
    checkpoint_dir = Path(cfg.paths.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "selected.ckpt"
    replacement_path = checkpoint_dir / "replacement.ckpt"

    original_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    torch.save(
        {
            "state_dict": original_state,
            "epoch": 0,
            "global_step": 0,
        },
        checkpoint_path,
    )
    original_digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    replacement_state = {
        name: value.detach().clone() for name, value in original_state.items()
    }
    changed_name = next(
        name
        for name, value in replacement_state.items()
        if torch.is_floating_point(value)
    )
    replacement_state[changed_name].add_(1.0)
    torch.save(
        {
            "state_dict": replacement_state,
            "epoch": 1,
            "global_step": 1,
        },
        replacement_path,
    )
    replacement_digest = hashlib.sha256(
        replacement_path.read_bytes()
    ).hexdigest()
    assert replacement_digest != original_digest
    selected_checkpoint.best_model_path = str(checkpoint_path)

    real_torch_load = run_module.torch.load
    load_calls: list[tuple[bool, tuple[Any, ...], dict[str, Any]]] = []
    real_load_state_dict = model.load_state_dict
    strict_load_calls: list[bool] = []

    def record_strict_load(
        state_dict: Mapping[str, torch.Tensor],
        *,
        strict: bool = True,
        assign: bool = False,
    ) -> object:
        strict_load_calls.append(strict)
        return real_load_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
        )

    monkeypatch.setattr(model, "load_state_dict", record_strict_load)

    def replace_before_deserialization(
        source: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        load_calls.append(
            (
                callable(getattr(source, "read", None))
                and not isinstance(source, (str, Path)),
                args,
                dict(kwargs),
            )
        )
        replacement_path.replace(checkpoint_path)
        return real_torch_load(source, *args, **kwargs)

    monkeypatch.setattr(
        run_module.torch,
        "load",
        replace_before_deserialization,
    )

    results = rerun_best_model_checkpoint(
        checkpoint_model=model,
        cfg=cfg,
        datamodule=datamodule,
        device=torch.device("cpu"),
        callbacks=callbacks,
        logger=[],
    )
    assert len(load_calls) == 1
    source_is_stream, _, load_kwargs = load_calls[0]
    assert source_is_stream
    assert load_kwargs["map_location"] == "cpu"
    assert load_kwargs["weights_only"] is True
    assert strict_load_calls == [True]

    loaded_state = model.state_dict()

    def matches(expected: Mapping[str, torch.Tensor]) -> bool:
        return loaded_state.keys() == expected.keys() and all(
            torch.equal(loaded_state[name].cpu(), expected[name].cpu())
            for name in expected
        )

    loaded_original = matches(original_state)
    loaded_replacement = matches(replacement_state)
    assert loaded_original != loaded_replacement
    loaded_digest = original_digest if loaded_original else replacement_digest
    assert all(
        result.context.checkpoint_id == loaded_digest
        for result in results.values()
    )


def test_selected_checkpoint_cleanup_preserves_post_load_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup removes only the trusted objects loaded before a path swap."""
    cfg = _compose(
        "heterogeneous_synthetic_hgt_full",
        tmp_path=tmp_path,
        extra_overrides=(
            "execution_profile=experimental",
            "evaluation_artifacts.enabled=false",
            "delete_checkpoint_after_test=true",
        ),
    )
    datamodule, model, _ = _build_heterogeneous_runtime(cfg)
    callbacks = instantiate_callbacks(cfg.callbacks)
    selected_checkpoint = _checkpoint_callback(callbacks)
    checkpoint_dir = Path(cfg.paths.checkpoint_dir)
    checkpoint_path = checkpoint_dir / "selected.ckpt"
    replacement_path = checkpoint_dir / "replacement.ckpt"
    original_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    checkpoint_io = TrustedCheckpointIO(
        output_root=cfg.paths.output_dir,
        checkpoint_root=checkpoint_dir,
    )
    checkpoint_io.save_checkpoint(
        {
            "state_dict": original_state,
            "epoch": 0,
            "global_step": 0,
        },
        checkpoint_path,
    )
    original_digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    replacement_state = {
        name: value.detach().clone() for name, value in original_state.items()
    }
    changed_name = next(
        name
        for name, value in replacement_state.items()
        if torch.is_floating_point(value)
    )
    replacement_state[changed_name].add_(1.0)
    torch.save(
        {
            "state_dict": replacement_state,
            "epoch": 1,
            "global_step": 1,
        },
        replacement_path,
    )
    replacement_bytes = replacement_path.read_bytes()
    replacement_digest = hashlib.sha256(replacement_bytes).hexdigest()
    assert replacement_digest != original_digest
    selected_checkpoint.best_model_path = str(checkpoint_path)

    real_load_selected_checkpoint = run_module.load_selected_checkpoint

    def replace_after_trusted_load(*args: Any, **kwargs: Any) -> object:
        loaded = real_load_selected_checkpoint(*args, **kwargs)
        replacement_path.replace(checkpoint_path)
        return loaded

    monkeypatch.setattr(
        run_module,
        "load_selected_checkpoint",
        replace_after_trusted_load,
    )

    results = rerun_best_model_checkpoint(
        checkpoint_model=model,
        cfg=cfg,
        datamodule=datamodule,
        device=torch.device("cpu"),
        callbacks=callbacks,
        logger=[],
    )

    loaded_state = model.state_dict()
    assert loaded_state.keys() == original_state.keys()
    assert all(
        torch.equal(loaded_state[name].cpu(), original_state[name].cpu())
        for name in original_state
    )
    assert all(
        result.context.checkpoint_id == original_digest
        for result in results.values()
    )
    assert checkpoint_path.read_bytes() == replacement_bytes
    assert not checkpoint_manifest_path(checkpoint_path).exists()
    assert not checkpoint_state_path(checkpoint_path).exists()


def test_active_selected_evaluation_failure_preserves_root_and_cleans_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup leaves capture reusable without masking an active-pass failure."""
    cfg = _compose(
        "heterogeneous_synthetic_hgt_full",
        tmp_path=tmp_path,
        extra_overrides=("evaluation_artifacts.enabled=true",),
    )
    datamodule, model, pipeline_output = _build_heterogeneous_runtime(cfg)
    callbacks = instantiate_callbacks(cfg.callbacks)
    selected_checkpoint = _checkpoint_callback(callbacks)
    checkpoint_path = (
        Path(cfg.paths.output_dir) / "checkpoints" / "selected.ckpt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": 0,
            "global_step": 0,
        },
        checkpoint_path,
    )
    selected_checkpoint.best_model_path = str(checkpoint_path)

    class SelectedEvaluationFailure(RuntimeError):
        pass

    sentinel = SelectedEvaluationFailure("selected validation failed")
    evaluation_started = False

    class FailingTrainer:
        world_size = 1
        global_rank = 0

        def validate(
            self,
            *,
            model: TBModel,
            dataloaders: object,
        ) -> None:
            nonlocal evaluation_started
            del dataloaders
            model.on_validation_epoch_start()
            assert model.evaluator.state == "active"
            evaluation_started = True
            raise sentinel

        def test(
            self,
            *,
            model: TBModel,
            dataloaders: object,
        ) -> None:
            del model, dataloaders
            raise AssertionError("test must not run after validation failure")

    monkeypatch.setattr(
        run_module.hydra.utils,
        "instantiate",
        lambda *args, **kwargs: FailingTrainer(),
    )

    with pytest.raises(SelectedEvaluationFailure) as error:
        rerun_best_model_checkpoint(
            checkpoint_model=model,
            cfg=cfg,
            datamodule=datamodule,
            device=torch.device("cpu"),
            callbacks=callbacks,
            logger=[],
            prediction_row_adapter=pipeline_output.prediction_row_adapter,
            supervision_counts=pipeline_output.supervision_counts,
            provenance_input=pipeline_output.provenance_input,
            source_graph_id=pipeline_output.source_graph_id,
        )

    assert error.value is sentinel
    assert evaluation_started
    assert model.evaluator.state == "idle"
    assert model.evaluator.context is None

    selected_counts = {
        split: int(pipeline_output.supervision_counts[split])
        for split in ("val", "test")
    }
    model.configure_prediction_artifact_capture(
        pipeline_output.prediction_row_adapter,
        MagicMock(),
        selected_counts,
    )
    model.clear_prediction_artifact_capture()


def _direct_model_loss(
    model: TBModel,
    batch: Data | HeteroData,
) -> torch.Tensor:
    """Evaluate one fresh training-supervision batch without metric carryover."""
    model.on_train_epoch_start()
    try:
        output = model.model_step(batch)
    finally:
        model.abort_evaluation()
    return output["loss"]


def _assert_expected_module_gradients(model: TBModel) -> None:
    """Require fresh, finite, nonzero gradients through every model stage."""
    for stage_name in ("feature_encoder", "backbone", "readout"):
        stage = getattr(model, stage_name)
        gradients = [
            parameter.grad
            for parameter in stage.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        assert gradients, f"{stage_name} produced no parameter gradients"
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def _build_heterogeneous_runtime(
    cfg: DictConfig,
) -> tuple[HeterogeneousNodeDataModule, TBModel, Any]:
    """Build the real heterogeneous pipeline, data module, and model."""
    pipeline = hydra.utils.instantiate(cfg.data_pipeline)
    pipeline_output = pipeline.build(cfg)
    capability_validation = validate_capability_composition(
        cfg,
        observed=pipeline_output.capability_spec,
    )
    lightning_model = instantiate_model(
        cfg,
        data_spec=pipeline_output.data_spec,
        capability_validation=capability_validation,
    )
    assert isinstance(pipeline_output.datamodule, HeterogeneousNodeDataModule)
    assert isinstance(lightning_model, TBModel)
    return pipeline_output.datamodule, lightning_model, pipeline_output


def _phase_loader(
    datamodule: HeterogeneousNodeDataModule,
    phase: str,
) -> Iterable[HeteroData]:
    """Return the real loader corresponding to one canonical phase."""
    return {
        "train": datamodule.train_dataloader,
        "val": datamodule.val_dataloader,
        "test": datamodule.test_dataloader,
    }[phase]()


def _phase_state(phase: str) -> str:
    """Translate canonical loader phases into the model state contract."""
    return {
        "train": "Training",
        "val": "Validation",
        "test": "Test",
    }[phase]


def _fixed_evaluation_signature(
    model: TBModel,
    datamodule: HeterogeneousNodeDataModule,
    phase: str,
) -> tuple[tuple[tuple[int, ...], ...], dict[str, float]]:
    """Evaluate one complete sampled phase and retain its seed/metric identity."""
    model.eval()
    {
        "train": model.on_train_epoch_start,
        "val": model.on_validation_epoch_start,
        "test": model.on_test_epoch_start,
    }[phase]()
    seed_batches: list[tuple[int, ...]] = []
    weighted_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in _phase_loader(datamodule, phase):
            target_store = batch[datamodule.spec.target_node_type]
            seed_count = int(target_store.batch_size)
            seed_batches.append(
                tuple(
                    int(node_id)
                    for node_id in target_store.n_id[:seed_count].tolist()
                )
            )
            output = model.model_step(batch)
            assert output["num_supervised_examples"] == seed_count
            weighted_loss += float(output["loss"]) * seed_count
            total_examples += seed_count
    assert total_examples > 0
    result = model.evaluator.snapshot()
    metrics = {name: float(value) for name, value in result.metrics.items()}
    metrics["num_examples"] = float(result.num_examples)
    assert result.num_examples == total_examples
    model.abort_evaluation()
    metrics["loss"] = weighted_loss / total_examples
    return tuple(seed_batches), metrics


@pytest.mark.parametrize(("experiment", "_model_name"), EXPERIMENTS)
def test_full_batch_model_has_an_overfit_signal(
    experiment: str,
    _model_name: str,
    tmp_path: Path,
) -> None:
    """Both real backbones reduce supervised loss on fresh graph batches."""
    seed_everything(0, workers=True)
    cfg = _compose(
        experiment,
        tmp_path=tmp_path,
        extra_overrides=("model.backbone.dropout=0.0",),
    )
    pipeline = hydra.utils.instantiate(cfg.data_pipeline)
    pipeline_output = pipeline.build(cfg)
    datamodule = pipeline_output.datamodule
    capability_validation = validate_capability_composition(
        cfg,
        observed=pipeline_output.capability_spec,
    )
    lightning_model = instantiate_model(
        cfg,
        data_spec=pipeline_output.data_spec,
        capability_validation=capability_validation,
    )
    assert isinstance(lightning_model, TBModel)
    model = lightning_model
    optimizer_config = model.configure_optimizers()
    optimizer = cast(torch.optim.Optimizer, optimizer_config["optimizer"])

    model.eval()
    with torch.no_grad():
        initial_loss = float(
            _direct_model_loss(
                model,
                next(iter(datamodule.train_dataloader())),
            )
        )

    model.train()
    previous_batch: Data | HeteroData | None = None
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        batch = next(iter(datamodule.train_dataloader()))
        assert batch is not previous_batch
        loss = _direct_model_loss(
            model,
            batch,
        )
        loss.backward()
        _assert_expected_module_gradients(model)
        optimizer.step()
        previous_batch = batch

    model.eval()
    with torch.no_grad():
        final_loss = float(
            _direct_model_loss(
                model,
                next(iter(datamodule.train_dataloader())),
            )
        )

    assert math.isfinite(initial_loss)
    assert math.isfinite(final_loss)
    assert final_loss < initial_loss


@pytest.mark.parametrize(
    ("logger_override", "expect_learning_rate_monitor"),
    [
        pytest.param("[]", False, id="no-logger"),
        pytest.param("csv", True, id="csv-logger"),
    ],
)
def test_run_filters_learning_rate_monitor_only_without_a_logger(
    logger_override: str,
    expect_learning_rate_monitor: bool,
    tmp_path: Path,
) -> None:
    """Logger-free runs remove only the callback requiring a logger."""
    cfg = _compose(
        "heterogeneous_synthetic_hgt_full",
        tmp_path=tmp_path,
        logger_override=logger_override,
        extra_overrides=("train=false", "test=false"),
    )
    before = OmegaConf.to_container(cfg, resolve=False)
    configured_callbacks = instantiate_callbacks(cfg.callbacks)

    _, objects = run(cfg)

    assert OmegaConf.to_container(cfg, resolve=False) == before
    actual_callbacks = objects["callbacks"]
    actual_types = Counter(type(callback) for callback in actual_callbacks)
    expected_types = Counter(
        type(callback)
        for callback in configured_callbacks
        if expect_learning_rate_monitor
        or not isinstance(callback, LearningRateMonitor)
    )
    assert actual_types == expected_types
    assert (
        any(
            isinstance(callback, LearningRateMonitor)
            for callback in actual_callbacks
        )
        is expect_learning_rate_monitor
    )

    if expect_learning_rate_monitor:
        assert len(objects["logger"]) == 1
        assert isinstance(objects["logger"][0], CSVLogger)
    else:
        assert objects["logger"] == []
