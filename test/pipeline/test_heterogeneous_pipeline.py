"""End-to-end tests for native heterogeneous node-classification pipelines."""

from __future__ import annotations

import logging
import math
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

from topobench.dataloader.heterogeneous import HeterogeneousNodeDataModule
from topobench.model import TBModel
from topobench.run import rerun_best_model_checkpoint, run
from topobench.utils import instantiate_callbacks
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
_DIRECT_LOSS_REDUCTION_ABS_TOLERANCE = 1e-8


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
    prefix = f"{phase}_best_rerun/"
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

    _assert_finite_metrics(fit_metrics, prefix="train/")
    _assert_finite_metrics(fit_metrics, prefix="val/")
    fit_trainer = objects["trainer"]
    assert fit_trainer.current_epoch == 1

    checkpoint = _checkpoint_callback(objects["callbacks"])
    best_path = Path(checkpoint.best_model_path)
    assert checkpoint.best_model_path
    assert best_path.is_file()
    assert best_path.stat().st_size > 0

    rerun_messages = [record.getMessage() for record in caplog.records]
    assert any("val_best_rerun/" in message for message in rerun_messages)
    assert any("test_best_rerun/" in message for message in rerun_messages)

    # The production rerun attaches the model to a dedicated trainer whose
    # final phase is test.
    rerun_model = objects["model"]
    rerun_trainer = rerun_model.trainer
    assert rerun_trainer is not fit_trainer
    _assert_finite_metrics(
        rerun_trainer.callback_metrics,
        prefix="test/",
    )
    assert rerun_model.state_str == "Test"
    assert isinstance(
        objects["datamodule"],
        HeterogeneousNodeDataModule,
    )
    assert objects["datamodule"].mode == "full_batch"


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

    rerun_messages = [record.getMessage() for record in caplog.records]
    assert any("val_best_rerun/" in message for message in rerun_messages)
    assert any("test_best_rerun/" in message for message in rerun_messages)

    rerun_model = objects["model"]
    rerun_trainer = rerun_model.trainer
    assert rerun_trainer is not fit_trainer
    _assert_finite_metrics(rerun_trainer.callback_metrics, prefix="test/")
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

    rerun_sinks = [
        MagicMock(spec=WandbLogger),
        MagicMock(spec=WandbLogger),
    ]
    for sink in rerun_sinks:
        rng_before = torch.random.get_rng_state().clone()
        rerun_best_model_checkpoint(
            checkpoint_model=rerun_model,
            cfg=cfg,
            datamodule=datamodule,
            device=torch.device("cpu"),
            callbacks=objects["callbacks"],
            logger=[sink],
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

        _, direct_metrics = _fixed_evaluation_signature(
            rerun_model,
            datamodule,
            phase,
        )
        assert first_metrics.keys() == direct_metrics.keys()
        for name in first_metrics:
            if name == "loss":
                assert first_metrics[name] == pytest.approx(
                    direct_metrics[name],
                    abs=_DIRECT_LOSS_REDUCTION_ABS_TOLERANCE,
                    rel=0,
                )
            else:
                assert first_metrics[name] == direct_metrics[name]


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
    datamodule, model = _build_heterogeneous_runtime(cfg)
    target_type = datamodule.spec.target_node_type
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
    datamodule, model = _build_heterogeneous_runtime(cfg)
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


def test_full_batch_best_rerun_logs_prefixed_finite_metrics(
    tmp_path: Path,
) -> None:
    """Log real validation and test reruns to only a mocked external sink."""
    cfg = _compose(
        "heterogeneous_synthetic_hgt_full",
        tmp_path=tmp_path,
    )
    _, objects = run(cfg)
    checkpoint = _checkpoint_callback(objects["callbacks"])
    best_path = Path(checkpoint.best_model_path)
    assert best_path.is_file()

    sink = MagicMock(spec=WandbLogger)
    rerun_best_model_checkpoint(
        checkpoint_model=objects["model"],
        cfg=cfg,
        datamodule=objects["datamodule"],
        device=torch.device("cpu"),
        callbacks=objects["callbacks"],
        logger=[sink],
    )

    expected_suffixes = {
        "loss",
        "accuracy",
        "auroc",
        "f1",
        "precision",
        "recall",
    }
    assert sink.log_metrics.call_count == 2
    assert all(
        len(call.args) == 1 and not call.kwargs
        for call in sink.log_metrics.call_args_list
    )
    logged_calls = [call.args[0] for call in sink.log_metrics.call_args_list]
    assert set(logged_calls[0]) == {
        f"val_best_rerun/{suffix}" for suffix in expected_suffixes
    }
    assert set(logged_calls[1]) == {
        f"test_best_rerun/{suffix}" for suffix in expected_suffixes
    }
    assert all(
        math.isfinite(float(value))
        for metrics in logged_calls
        for value in metrics.values()
    )

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


def _direct_model_loss(
    model: TBModel,
    batch: Data | HeteroData,
) -> torch.Tensor:
    """Evaluate one fresh training-supervision batch without metric carryover."""
    model.state_str = "Training"
    output = model.model_step(batch)
    model.evaluator.reset()
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
) -> tuple[HeterogeneousNodeDataModule, TBModel]:
    """Build the real heterogeneous datamodule and model from one config."""
    pipeline = hydra.utils.instantiate(cfg.data_pipeline)
    pipeline_output = pipeline.build(cfg)
    lightning_model = instantiate_model(
        cfg,
        data_spec=pipeline_output.data_spec,
    )
    assert isinstance(pipeline_output.datamodule, HeterogeneousNodeDataModule)
    assert isinstance(lightning_model, TBModel)
    return pipeline_output.datamodule, lightning_model


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
    model.state_str = _phase_state(phase)
    model.evaluator.reset()
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
    metrics = {
        name: float(value) for name, value in model.evaluator.compute().items()
    }
    model.evaluator.reset()
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
    lightning_model = instantiate_model(
        cfg,
        data_spec=pipeline_output.data_spec,
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
