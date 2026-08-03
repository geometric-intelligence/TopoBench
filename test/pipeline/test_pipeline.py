"""Broad native graph pipeline sentinels."""

import hydra
import pytest

from test._utils.simplified_pipeline import run
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
    result = run(_compose(dataset))

    assert result["epochs_completed"] >= 2
    assert result["observed_train_batch_size"] > 1
    assert {"train/loss", "val/loss"} <= result["fit_metrics"].keys()
    assert result["test_results"]
    assert result["test_results"][0]["test/num_examples"] > 0


@pytest.mark.download
@pytest.mark.parametrize("dataset", ("MUTAG", "AQSOL"))
def test_real_graph_release_lifecycle(dataset: str) -> None:
    """Keep real classification and scalar-regression download gates."""
    result = run(_compose(dataset, epochs=1))

    assert result["epochs_completed"] >= 1
    assert result["observed_train_batch_size"] > 1
    assert result["test_results"]


def test_native_graph_run_configuration_enables_automatic_preflight() -> None:
    cfg = _compose("SyntheticGraph", epochs=1)

    assert cfg.execution_profile == "qualified"
    assert cfg.preflight.enabled is True
    assert cfg.preflight.execution_probe is True
