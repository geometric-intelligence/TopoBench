"""Unit and real lifecycle tests for best-epoch checkpoint metrics."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import hydra
import pytest
import torch
from hydra.core.global_hydra import GlobalHydra
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig

import topobench.callbacks as callback_module
import topobench.run as run_module
from topobench.callbacks import BestEpochMetricsCallback
from topobench.evaluator import EvaluationResult
from topobench.utils import get_metric_value
from topobench.utils.config_resolvers import register_all_resolvers


class TestBestEpochMetricsCallback:
    """Test the BestEpochMetricsCallback class."""

    def test_init(self):
        """Test callback initialization."""
        callback = BestEpochMetricsCallback(monitor="val/loss", mode="min")
        assert callback.monitor_metric == "val/loss"
        assert callback.mode == "min"
        assert callback.best_monitored_value is None
        assert callback.best_epoch_number is None
        assert callback.best_epoch_metrics == {}

    def test_init_with_max_mode(self):
        """Test callback initialization with max mode."""
        callback = BestEpochMetricsCallback(monitor="val/accuracy", mode="max")
        assert callback.monitor_metric == "val/accuracy"
        assert callback.mode == "max"

    def test_on_train_start_finds_checkpoint_callback(self):
        """Test that on_train_start finds ModelCheckpoint callback."""
        callback = BestEpochMetricsCallback(monitor="val/loss", mode="min")

        # Create mock trainer with ModelCheckpoint callback
        trainer = Mock()
        checkpoint_callback = ModelCheckpoint()
        trainer.callbacks = [checkpoint_callback, Mock()]
        pl_module = Mock()

        callback.on_train_start(trainer, pl_module)

        assert callback.checkpoint_callback is checkpoint_callback

    def test_on_train_start_without_checkpoint_callback(self):
        """Test that on_train_start works without ModelCheckpoint (checkpoint_callback stays None)."""
        callback = BestEpochMetricsCallback(monitor="val/loss", mode="min")

        # Create mock trainer without ModelCheckpoint callback
        trainer = Mock()
        trainer.callbacks = [Mock(), Mock()]
        pl_module = Mock()

        callback.on_train_start(trainer, pl_module)

        assert callback.checkpoint_callback is None

    def test_on_train_epoch_end_captures_metrics(self):
        """Test that training metrics are captured at end of training epoch."""
        callback = BestEpochMetricsCallback(monitor="val/loss", mode="min")

        trainer = Mock()
        trainer.callback_metrics = {
            "train/loss": torch.tensor(0.5),
            "train/accuracy": torch.tensor(0.85),
            "val/loss": torch.tensor(0.6),  # Should not be captured
        }
        pl_module = Mock()

        callback.on_train_epoch_end(trainer, pl_module)

        assert "train/loss" in callback.current_epoch_train_metrics
        assert "train/accuracy" in callback.current_epoch_train_metrics
        assert "val/loss" not in callback.current_epoch_train_metrics
        assert callback.current_epoch_train_metrics["train/loss"] == 0.5

    def test_on_validation_end_first_epoch(self):
        """Test that the first epoch is always considered best."""
        callback = BestEpochMetricsCallback(monitor="val/loss", mode="min")

        # Setup training metrics
        callback.current_epoch_train_metrics = {
            "train/loss": 0.5,
            "train/accuracy": 0.8,
        }

        trainer = Mock()
        trainer.current_epoch = 0
        trainer.callback_metrics = {
            "val/loss": torch.tensor(0.6),
            "val/accuracy": torch.tensor(0.75),
        }
        pl_module = Mock()

        callback.on_validation_end(trainer, pl_module)

        assert callback.best_monitored_value == pytest.approx(0.6)
        assert callback.best_epoch_number == 0
        assert "train/loss" in callback.best_epoch_metrics
        assert "val/loss" in callback.best_epoch_metrics

    def test_on_validation_end_min_mode_improvement(self):
        """Test detection of improvement in min mode."""
        callback = BestEpochMetricsCallback(monitor="val/loss", mode="min")
        callback.best_monitored_value = 0.6
        callback.best_epoch_number = 0

        # Setup training metrics
        callback.current_epoch_train_metrics = {
            "train/loss": 0.3,
        }

        trainer = Mock()
        trainer.current_epoch = 1
        trainer.callback_metrics = {
            "val/loss": torch.tensor(0.4),  # Better (lower)
        }
        pl_module = Mock()

        callback.on_validation_end(trainer, pl_module)

        assert callback.best_monitored_value == pytest.approx(0.4)
        assert callback.best_epoch_number == 1

    def test_on_validation_end_min_mode_no_improvement(self):
        """Test that worse values don't update in min mode."""
        callback = BestEpochMetricsCallback(monitor="val/loss", mode="min")
        callback.best_monitored_value = 0.4
        callback.best_epoch_number = 1
        old_metrics = {"val/loss": 0.4}
        callback.best_epoch_metrics = old_metrics.copy()

        # Setup training metrics
        callback.current_epoch_train_metrics = {
            "train/loss": 0.5,
        }

        trainer = Mock()
        trainer.current_epoch = 2
        trainer.callback_metrics = {
            "val/loss": torch.tensor(0.6),  # Worse (higher)
        }
        pl_module = Mock()

        callback.on_validation_end(trainer, pl_module)

        # Should not update
        assert callback.best_monitored_value == 0.4
        assert callback.best_epoch_number == 1

    def test_on_validation_end_max_mode_improvement(self):
        """Test detection of improvement in max mode."""
        callback = BestEpochMetricsCallback(monitor="val/accuracy", mode="max")
        callback.best_monitored_value = 0.7
        callback.best_epoch_number = 0

        # Setup training metrics
        callback.current_epoch_train_metrics = {
            "train/accuracy": 0.85,
        }

        trainer = Mock()
        trainer.current_epoch = 1
        trainer.callback_metrics = {
            "val/accuracy": torch.tensor(0.8),  # Better (higher)
        }
        pl_module = Mock()

        callback.on_validation_end(trainer, pl_module)

        assert callback.best_monitored_value == pytest.approx(0.8)
        assert callback.best_epoch_number == 1

    def test_on_validation_end_max_mode_no_improvement(self):
        """Test that worse values don't update in max mode."""
        callback = BestEpochMetricsCallback(monitor="val/accuracy", mode="max")
        callback.best_monitored_value = 0.8
        callback.best_epoch_number = 1

        # Setup training metrics
        callback.current_epoch_train_metrics = {
            "train/accuracy": 0.75,
        }

        trainer = Mock()
        trainer.current_epoch = 2
        trainer.callback_metrics = {
            "val/accuracy": torch.tensor(0.7),  # Worse (lower)
        }
        pl_module = Mock()

        callback.on_validation_end(trainer, pl_module)

        # Should not update
        assert callback.best_monitored_value == 0.8
        assert callback.best_epoch_number == 1

    def test_best_epoch_logging(self):
        """Test that best epoch number and metrics are logged."""
        callback = BestEpochMetricsCallback(monitor="val/loss", mode="min")

        # Setup training metrics
        callback.current_epoch_train_metrics = {
            "train/loss": 0.5,
            "train/accuracy": 0.8,
        }

        trainer = Mock()
        trainer.current_epoch = 5
        trainer.global_step = 7
        trainer.callback_metrics = {
            "val/loss": torch.tensor(0.3),
            "val/accuracy": torch.tensor(0.85),
        }
        logger = Mock()
        trainer.loggers = [logger]
        pl_module = Mock()

        callback.on_validation_end(trainer, pl_module)

        logger.log_metrics.assert_called_once_with(
            {
                "best_epoch": 5,
                "best_epoch/train/loss": 0.5,
                "best_epoch/train/accuracy": 0.8,
                "best_epoch/val/loss": pytest.approx(0.3),
                "best_epoch/val/accuracy": pytest.approx(0.85),
            },
            step=7,
        )

    def test_on_train_end_with_checkpoint(self):
        """Test on_train_end logs checkpoint path when available."""
        callback = BestEpochMetricsCallback(monitor="val/loss", mode="min")

        # Setup checkpoint callback
        checkpoint_callback = Mock()
        checkpoint_callback.best_model_path = "/path/to/checkpoint.ckpt"
        callback.checkpoint_callback = checkpoint_callback

        trainer = Mock()
        pl_module = Mock()
        pl_module.logger = Mock()
        pl_module.logger.experiment.summary = {}

        callback.on_train_end(trainer, pl_module)

        # Check that checkpoint path was logged
        assert (
            pl_module.logger.experiment.summary["best_epoch/checkpoint"]
            == "/path/to/checkpoint.ckpt"
        )
        assert (
            pl_module.logger.experiment.summary["monitored_metric"]
            == "val/loss (min)"
        )

    def test_on_train_end_without_checkpoint(self):
        """Test on_train_end handles missing checkpoint gracefully."""
        callback = BestEpochMetricsCallback(monitor="val/loss", mode="min")
        callback.checkpoint_callback = None

        trainer = Mock()
        pl_module = Mock()

        # Should not raise error
        callback.on_train_end(trainer, pl_module)

    def test_on_train_end_without_logger(self):
        """Test on_train_end handles missing logger gracefully."""
        callback = BestEpochMetricsCallback(monitor="val/loss", mode="min")

        checkpoint_callback = Mock()
        checkpoint_callback.best_model_path = "/path/to/checkpoint.ckpt"
        callback.checkpoint_callback = checkpoint_callback

        trainer = Mock()
        pl_module = Mock()
        pl_module.logger = None

        # Should not raise error
        callback.on_train_end(trainer, pl_module)

    def test_handles_tensor_values(self):
        """Test that callback properly converts tensor values to floats."""
        callback = BestEpochMetricsCallback(monitor="val/loss", mode="min")

        trainer = Mock()
        trainer.current_epoch = 0
        trainer.callback_metrics = {
            "val/loss": torch.tensor(0.5),
            "val/accuracy": torch.tensor(0.85),
        }
        callback.current_epoch_train_metrics = {}
        pl_module = Mock()

        callback.on_validation_end(trainer, pl_module)

        # Values should be converted to floats
        assert isinstance(callback.best_monitored_value, float)
        assert callback.best_monitored_value == 0.5


_LIFECYCLE_CASES = (
    pytest.param(
        "graph_classification",
        (
            "dataset=graph/SyntheticGraph",
            "model=graph/gcn",
        ),
        id="graph-classification",
    ),
    pytest.param(
        "graph_regression",
        (
            "dataset=graph/SyntheticGraphRegression",
            "model=graph/gcn",
        ),
        id="graph-scalar-regression",
    ),
    pytest.param(
        "heterogeneous_neighbor",
        ("experiment=heterogeneous_synthetic_hgt_neighbor",),
        id="heterogeneous-node-classification",
    ),
    pytest.param(
        "hypergraph_node",
        (
            "dataset=hypergraph/SyntheticHypergraph",
            "model=hypergraph/edgnn",
            "data_pipeline=hypergraph_node",
        ),
        id="hypergraph-node-classification",
    ),
)


def _compose_lifecycle(
    family: str,
    selector_overrides: tuple[str, ...],
    tmp_path: Path,
) -> DictConfig:
    """Compose one bounded, logger-free production lifecycle."""
    GlobalHydra.instance().clear()
    register_all_resolvers()
    output_dir = tmp_path / family
    overrides = [
        *selector_overrides,
        "callbacks=default",
        "evaluation_artifacts=default",
        "optimized_metric=best_epoch/val/loss",
        "logger=[]",
        "paths=test",
        f"paths.output_dir={output_dir}",
        f"paths.work_dir={output_dir}",
        f"paths.data_dir={tmp_path / 'datasets'}",
        f"++paths.cache_dir={tmp_path / 'cache'}",
        f"trainer.default_root_dir={output_dir}",
        "trainer.max_epochs=1",
        "trainer.min_epochs=1",
        "trainer.check_val_every_n_epoch=1",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        "++trainer.limit_train_batches=1",
        "++trainer.limit_val_batches=1.0",
        "++trainer.limit_test_batches=1.0",
        "++trainer.enable_progress_bar=false",
        "++enable_progress_bar=false",
    ]
    with hydra.initialize(
        version_base="1.3",
        config_path="../../configs",
        job_name=f"{family}_best_checkpoint_lifecycle",
    ):
        return hydra.compose(config_name="run.yaml", overrides=overrides)


def _one_callback(
    callbacks: list[object],
    callback_type: type[Any],
) -> Any:
    """Return the only callback of one required concrete family."""
    matches = [
        callback
        for callback in callbacks
        if isinstance(callback, callback_type)
    ]
    assert len(matches) == 1
    return matches[0]


def _assert_finite_prefix(
    metrics: Mapping[str, Any],
    prefix: str,
) -> None:
    """Require nonempty finite metrics from one real trainer phase."""
    matching = {
        name: value
        for name, value in metrics.items()
        if name.startswith(prefix)
    }
    assert matching, f"No metrics with prefix {prefix!r}: {sorted(metrics)}"
    assert all(math.isfinite(float(value)) for value in matching.values())


def _captured_rerun_metrics(
    sink: MagicMock,
    phase: str,
) -> dict[str, object]:
    """Extract one complete selected-checkpoint logger payload."""
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
    captured = {str(name): value for name, value in matching[0].items()}
    assert all(math.isfinite(float(value)) for value in captured.values())
    return captured


def _file_sha256(path: Path) -> str:
    """Hash one retained artifact without depending on manifest claims."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _artifact_tree(root: Path) -> dict[Path, str]:
    """Snapshot every authoritative artifact file for no-publication checks."""
    return {
        path.relative_to(root): _file_sha256(path)
        for path in root.rglob("*")
        if path.is_file()
    }


def _assert_batch_contract(
    family: str,
    objects: Mapping[str, Any],
) -> None:
    """Check the family-specific shape or seed invariant on a real batch."""
    datamodule = objects["datamodule"]
    model = objects["model"]
    batch = next(iter(datamodule.train_dataloader()))
    model.eval()
    model.state_str = "Training"

    with torch.no_grad():
        raw = model.forward(batch)
        if family == "hypergraph_node":
            assert raw["logits"].size(0) == raw["labels"].size(0)
            assert raw["logits"].size(0) == batch.y.size(0)
            return

        processed = model.process_outputs(raw, batch)

    if family == "graph_regression":
        expected_shape = (int(batch.num_graphs), 1)
        assert tuple(processed["logits"].shape) == expected_shape
        assert tuple(processed["labels"].shape) == expected_shape
        return

    if family == "graph_classification":
        expected_examples = int(batch.num_graphs)
        assert expected_examples > 1
        assert processed["num_supervised_examples"] == expected_examples
        assert processed["logits"].size(0) == expected_examples
        assert processed["labels"].size(0) == expected_examples
        return

    target_type = datamodule.spec.target_node_type
    seed_count = int(batch[target_type].batch_size)
    assert seed_count > 1
    assert processed["num_supervised_examples"] == seed_count
    assert processed["logits"].size(0) == seed_count
    assert processed["labels"].size(0) == seed_count


@pytest.mark.parametrize(
    ("family", "selector_overrides"),
    _LIFECYCLE_CASES,
)
def test_real_one_epoch_best_checkpoint_rerun_contract(
    family: str,
    selector_overrides: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Train once and publish both canonical selected-checkpoint results."""
    cfg = _compose_lifecycle(family, selector_overrides, tmp_path)
    sink = MagicMock(spec=WandbLogger)
    real_rerun = run_module.rerun_best_model_checkpoint
    captured: dict[str, object] = {}

    def capture_real_rerun(
        **kwargs: Any,
    ) -> dict[str, EvaluationResult]:
        assert kwargs["logger"] == []
        kwargs["logger"] = [sink]
        checkpoint = _one_callback(kwargs["callbacks"], ModelCheckpoint)
        selected_path = checkpoint.best_model_path
        selected_score = checkpoint.best_model_score
        if isinstance(selected_score, torch.Tensor):
            selected_score = selected_score.detach().clone()
        best_epoch = _one_callback(
            kwargs["callbacks"],
            BestEpochMetricsCallback,
        )
        artifact_callback = _one_callback(
            kwargs["callbacks"],
            callback_module.SelectedCheckpointArtifactCallback,
        )
        artifact_root = (
            Path(kwargs["cfg"].paths.output_dir)
            / "evaluations"
            / "best_checkpoint"
        )
        assert dict(artifact_callback.publications) == {}
        assert not artifact_root.exists()
        captured["artifact_callback"] = artifact_callback
        validation_selection = (
            best_epoch.best_epoch_number,
            best_epoch.best_monitored_value,
            dict(best_epoch.best_epoch_metrics),
        )
        checkpoint_model = kwargs["checkpoint_model"]
        validation_start = checkpoint_model.on_validation_epoch_start.__func__
        validation_end = checkpoint_model.on_validation_epoch_end.__func__

        results = real_rerun(**kwargs)

        assert checkpoint.best_model_path == selected_path
        if isinstance(selected_score, torch.Tensor):
            torch.testing.assert_close(
                checkpoint.best_model_score,
                selected_score,
            )
        else:
            assert checkpoint.best_model_score == selected_score
        assert (
            best_epoch.best_epoch_number,
            best_epoch.best_monitored_value,
            best_epoch.best_epoch_metrics,
        ) == validation_selection
        assert (
            checkpoint_model.on_validation_epoch_start.__func__
            is validation_start
        )
        assert (
            checkpoint_model.on_validation_epoch_end.__func__ is validation_end
        )
        captured["results"] = results
        return results

    monkeypatch.setattr(
        run_module,
        "rerun_best_model_checkpoint",
        capture_real_rerun,
    )

    fit_metrics, objects = run_module.run(cfg)

    assert objects["logger"] == []
    assert objects["trainer"].current_epoch == 1
    _assert_finite_prefix(fit_metrics, "train/")
    _assert_finite_prefix(fit_metrics, "val/")

    checkpoint = _one_callback(objects["callbacks"], ModelCheckpoint)
    checkpoint_path = Path(checkpoint.best_model_path)
    assert checkpoint_path.is_file()
    assert checkpoint_path.stat().st_size > 0

    best_epoch = _one_callback(
        objects["callbacks"],
        BestEpochMetricsCallback,
    )
    assert best_epoch.checkpoint_callback is checkpoint
    assert best_epoch.best_epoch_number == 0
    assert math.isfinite(float(best_epoch.best_monitored_value))
    _assert_finite_prefix(best_epoch.best_epoch_metrics, "val/")
    assert all(
        not name.startswith("evaluations/best_checkpoint/")
        for name in best_epoch.best_epoch_metrics
    )
    assert fit_metrics["best_epoch"] == best_epoch.best_epoch_number
    for name, value in best_epoch.best_epoch_metrics.items():
        returned_name = f"best_epoch/{name}"
        assert returned_name in fit_metrics
        assert float(fit_metrics[returned_name]) == float(value)
    optimized_value = get_metric_value(
        metric_dict=fit_metrics,
        metric_name=cfg.optimized_metric,
    )
    assert optimized_value == pytest.approx(
        float(best_epoch.best_epoch_metrics["val/loss"])
    )

    selected = objects["selected_checkpoint_results"]
    assert selected is captured["results"]
    assert tuple(selected) == ("val", "test")
    assert selected["val"] is not selected["test"]
    checkpoint_ids = {
        selected[split].context.checkpoint_id for split in ("val", "test")
    }
    assert len(checkpoint_ids) == 1
    assert None not in checkpoint_ids
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    assert checkpoint_ids == {checkpoint_sha256}
    artifact_callback = captured["artifact_callback"]
    assert artifact_callback is _one_callback(
        objects["callbacks"],
        callback_module.SelectedCheckpointArtifactCallback,
    )
    publications = artifact_callback.publications
    assert tuple(publications) == ("val", "test")
    assert publications["val"] is not publications["test"]
    publication_paths: dict[str, set[Path]] = {}
    publication_logger_names: dict[str, set[str]] = {}

    assert sink.log_metrics.call_count == 2
    for split in ("val", "test"):
        result = selected[split]
        assert result.context.split == split
        assert result.context.pass_kind == "selected_checkpoint"
        assert result.context.policy == "exact"
        namespace = f"evaluations/best_checkpoint/{split}/"
        publication = publications[split]
        split_root = (
            Path(cfg.paths.output_dir)
            / "evaluations"
            / "best_checkpoint"
            / split
        )
        metrics_path = Path(publication.metrics_file.path)
        manifest_path = Path(publication.manifest_file.path)
        shard_paths = tuple(
            Path(shard.path) for shard in publication.shard_files
        )
        assert publication.split == split
        assert publication.checkpoint_sha256 == checkpoint_sha256
        assert publication.num_examples == result.num_examples
        assert metrics_path == split_root / "metrics.json"
        assert manifest_path == split_root / "predictions" / "manifest.json"
        assert shard_paths
        assert all(
            path.parent == manifest_path.parent
            and path.name.startswith("part-")
            and path.suffix == ".npz"
            for path in shard_paths
        )
        artifact_files = (
            publication.metrics_file,
            publication.manifest_file,
            *publication.shard_files,
        )
        assert all(Path(file.path).is_file() for file in artifact_files)
        assert all(
            _file_sha256(Path(file.path)) == file.sha256
            for file in artifact_files
        )
        expected_logger_names = (
            f"best-checkpoint-{split}-metrics",
            f"best-checkpoint-{split}-predictions-manifest",
            *(
                f"best-checkpoint-{split}-predictions-{path.stem}"
                for path in shard_paths
            ),
        )
        assert (
            tuple(file.registration_name for file in artifact_files)
            == expected_logger_names
        )
        publication_paths[split] = {Path(file.path) for file in artifact_files}
        publication_logger_names[split] = set(expected_logger_names)

        metrics_document = json.loads(metrics_path.read_text())
        manifest_document = json.loads(manifest_path.read_text())
        assert metrics_document["split"] == split
        assert type(metrics_document["num_examples"]) is int
        assert metrics_document["num_examples"] == result.num_examples
        assert metrics_document["metrics"].keys() == result.metrics.keys()
        for name, value in result.metrics.items():
            assert float(metrics_document["metrics"][name]) == float(value)
        assert manifest_document["split"] == split
        assert manifest_document["checkpoint"]["sha256"] == checkpoint_sha256
        assert manifest_document["observed_rows"] == result.num_examples
        assert len(manifest_document["shards"]) == len(shard_paths)
        assert {shard["sha256"] for shard in manifest_document["shards"]} == {
            shard.sha256 for shard in publication.shard_files
        }

        manifest_path_key = f"{namespace}predictions/manifest_path"
        manifest_digest_key = f"{namespace}predictions/manifest_sha256"
        assert Path(fit_metrics[manifest_path_key]) == manifest_path
        assert (
            fit_metrics[manifest_digest_key]
            == publication.manifest_file.sha256
        )
        logged = _captured_rerun_metrics(sink, split)
        assert set(logged) == {
            *(f"{namespace}{name}" for name in result.metrics),
            f"{namespace}num_examples",
        }
        selected_output = {
            name for name in fit_metrics if name.startswith(namespace)
        }
        assert selected_output == {
            *logged,
            manifest_path_key,
            manifest_digest_key,
        }
        assert set(result.metrics).isdisjoint(result.provenance)
        for name, value in result.metrics.items():
            assert float(logged[f"{namespace}{name}"]) == float(value)
            assert float(fit_metrics[f"{namespace}{name}"]) == float(value)
        count_key = f"{namespace}num_examples"
        assert logged[count_key] == result.num_examples
        assert fit_metrics[count_key] == result.num_examples
        assert type(logged[count_key]) is int
        assert type(fit_metrics[count_key]) is int

    assert publication_paths["val"].isdisjoint(publication_paths["test"])
    assert publication_logger_names["val"].isdisjoint(
        publication_logger_names["test"]
    )
    assert (
        Path(publications["val"].manifest_file.path).parent
        != Path(publications["test"].manifest_file.path).parent
    )

    checkpoint_state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )["state_dict"]
    current_state = objects["model"].state_dict()
    assert checkpoint_state.keys() == current_state.keys()
    assert all(
        torch.equal(checkpoint_state[name], current_state[name].cpu())
        for name in checkpoint_state
    )

    _assert_batch_contract(family, objects)
    if family == "graph_classification":
        artifact_root = (
            Path(cfg.paths.output_dir) / "evaluations" / "best_checkpoint"
        )
        publications_before_ad_hoc_test = artifact_callback.publications
        artifacts_before_ad_hoc_test = _artifact_tree(artifact_root)
        rerun_trainer = objects["model"].trainer
        assert artifact_callback in rerun_trainer.callbacks
        rerun_trainer.test(
            model=objects["model"],
            dataloaders=objects["datamodule"].test_dataloader(),
            verbose=False,
        )
        assert (
            artifact_callback.publications == publications_before_ad_hoc_test
        )
        assert _artifact_tree(artifact_root) == artifacts_before_ad_hoc_test
