"""End-to-end tests for native heterogeneous node-classification pipelines."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import hydra
import pytest
import torch
from hydra.core.global_hydra import GlobalHydra
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig
from torch_geometric.data import Data, HeteroData

from topobench.dataloader.heterogeneous import HeterogeneousNodeDataModule
from topobench.model import TBModel
from topobench.run import rerun_best_model_checkpoint, run
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


def _compose(
    experiment: str,
    *,
    tmp_path: Path | None = None,
    max_epochs: int = 2,
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
                "logger=[]",
                "paths=test",
                f"paths.output_dir={output_dir}",
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
            return_hydra_config=True,
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


@pytest.mark.parametrize(("experiment", "_model_name"), EXPERIMENTS)
def test_full_batch_training_validation_checkpoint_and_best_rerun(
    experiment: str,
    _model_name: str,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Run the real two-epoch lifecycle, including both best-checkpoint reruns."""
    cfg = _compose(experiment, tmp_path=tmp_path)
    caplog.set_level(logging.INFO, logger="topobench.run")

    fit_metrics, objects = run(cfg)

    _assert_finite_metrics(fit_metrics, prefix="train/")
    _assert_finite_metrics(fit_metrics, prefix="val/")
    fit_trainer = objects["trainer"]
    assert fit_trainer.current_epoch == 2

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

    logged_calls = [call.args[0] for call in sink.log_metrics.call_args_list]
    logged = {
        name: value
        for metrics in logged_calls
        for name, value in metrics.items()
    }
    _assert_finite_metrics(logged, prefix="val_best_rerun/")
    _assert_finite_metrics(logged, prefix="test_best_rerun/")

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
    model = cast(TBModel, lightning_model)
    optimizer_config = cast(
        dict[str, Any],
        model.configure_optimizers(),
    )
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
