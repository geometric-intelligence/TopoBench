"""Broad native graph pipeline sentinels."""

from pathlib import Path

import hydra
import pytest
import torch
from omegaconf import open_dict

from test._utils.simplified_pipeline import run as run_simplified
from topobench.evaluator import EvaluationResult
from topobench.run import run as run_production
from topobench.utils.config_resolvers import register_all_resolvers


def _compose(dataset: str, *, epochs: int = 2):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    register_all_resolvers()
    with hydra.initialize(
        version_base="1.3",
        config_path="../../configs",
        job_name="native_graph_pipeline",
    ):
        return hydra.compose(
            config_name="run.yaml",
            overrides=[
                "model=graph/gcn",
                f"dataset=graph/{dataset}",
                f"trainer.max_epochs={epochs}",
                "trainer.min_epochs=1",
                "trainer.check_val_every_n_epoch=1",
                "trainer.accelerator=cpu",
                "trainer.devices=1",
                "dataset.dataloader_params.batch_size=4",
                "paths=test",
                "callbacks=model_checkpoint",
            ],
        )


@pytest.mark.parametrize(
    "dataset",
    ("SyntheticGraph", "SyntheticGraphRegression"),
)
def test_native_graph_pipeline_runs_two_epochs_with_real_batches(
    dataset: str,
) -> None:
    result = run_simplified(_compose(dataset))

    assert result["epochs_completed"] >= 2
    assert result["observed_train_batch_size"] > 1
    assert {"train/loss", "val/loss"} <= result["fit_metrics"].keys()
    assert result["test_results"]
    assert result["test_results"][0]["test/num_examples"] > 0


def test_graph_run_returns_authoritative_selected_checkpoint_results(
    tmp_path: Path,
) -> None:
    """The production graph run exposes both exact final results and scalars."""
    output_dir = tmp_path / "graph-selected-checkpoint"
    cfg = _compose(
        "SyntheticGraph",
        epochs=1,
    )
    with open_dict(cfg):
        cfg.logger = {}
        cfg.paths.output_dir = str(output_dir)
        cfg.paths.work_dir = str(output_dir)
        cfg.trainer.default_root_dir = str(output_dir)
        cfg.trainer.limit_train_batches = 2
        cfg.trainer.limit_val_batches = 1.0
        cfg.trainer.limit_test_batches = 1.0
        cfg.trainer.enable_progress_bar = False
        cfg.enable_progress_bar = False

    metrics, objects = run_production(cfg)

    selected = objects["selected_checkpoint_results"]
    assert tuple(selected) == ("val", "test")
    assert all(
        isinstance(selected[split], EvaluationResult)
        for split in ("val", "test")
    )
    assert selected["val"] is not selected["test"]
    assert {
        (result.context.split, result.context.pass_kind, result.context.policy)
        for result in selected.values()
    } == {
        ("val", "selected_checkpoint", "exact"),
        ("test", "selected_checkpoint", "exact"),
    }
    assert (
        selected["val"].context.checkpoint_id
        == selected["test"].context.checkpoint_id
    )
    assert selected["val"].context.checkpoint_id is not None

    datamodule = objects["datamodule"]
    expected_counts = {
        "val": sum(
            int(batch.num_graphs) for batch in datamodule.val_dataloader()
        ),
        "test": sum(
            int(batch.num_graphs) for batch in datamodule.test_dataloader()
        ),
    }
    assert {"train/num_examples", "val/num_examples"} <= metrics.keys()
    assert not any(
        name.startswith(("val_best_rerun/", "test_best_rerun/"))
        for name in metrics
    )
    for split, result in selected.items():
        namespace = f"evaluations/best_checkpoint/{split}/"
        assert result.num_examples == expected_counts[split]
        count = metrics[f"{namespace}num_examples"]
        assert count == result.num_examples
        assert type(count) is int
        for name, value in result.metrics.items():
            torch.testing.assert_close(
                torch.as_tensor(metrics[f"{namespace}{name}"]),
                torch.as_tensor(value),
            )


@pytest.mark.download
@pytest.mark.parametrize("dataset", ("MUTAG", "AQSOL"))
def test_real_graph_release_lifecycle(dataset: str) -> None:
    """Keep real classification and scalar-regression download gates."""
    result = run_simplified(_compose(dataset, epochs=1))

    assert result["epochs_completed"] >= 1
    assert result["observed_train_batch_size"] > 1
    assert result["test_results"]


def test_native_graph_run_configuration_enables_automatic_preflight() -> None:
    cfg = _compose("SyntheticGraph", epochs=1)

    assert cfg.execution_profile == "qualified"
    assert cfg.preflight.enabled is True
    assert cfg.preflight.execution_probe is True
