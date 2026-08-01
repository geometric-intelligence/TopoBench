"""Network-free composition gates for native hypergraph models."""

from __future__ import annotations

import hydra
import pytest
import torch
from omegaconf import DictConfig

from test._utils.simplified_pipeline import run
from topobench.utils.config_resolvers import register_all_resolvers
from topobench.utils.model_instantiation import instantiate_model


def _compose(model_selector: str) -> DictConfig:
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    register_all_resolvers()
    with hydra.initialize(
        version_base="1.3",
        config_path="../../configs",
        job_name="hypergraph_model_composition",
    ):
        return hydra.compose(
            config_name="run.yaml",
            overrides=[
                "dataset=hypergraph/SyntheticHypergraph",
                f"model=hypergraph/{model_selector}",
                "data_pipeline=hypergraph_node",
                "trainer.max_epochs=1",
                "trainer.min_epochs=1",
                "trainer.check_val_every_n_epoch=1",
                "trainer.accelerator=cpu",
                "trainer.devices=1",
                "callbacks=model_checkpoint",
                "paths=test",
            ],
        )


@pytest.mark.parametrize("model_selector", ("edgnn", "hypergraph_conv"))
def test_synthetic_hypergraph_composes_to_finite_node_logits(
    model_selector: str,
) -> None:
    """Both selectors compose over native fields with one logit row per label."""
    cfg = _compose(model_selector)
    pipeline_output = hydra.utils.instantiate(cfg.data_pipeline).build(cfg)
    batch = next(iter(pipeline_output.datamodule.train_dataloader()))
    model = instantiate_model(cfg, data_spec=None)
    model.eval()

    model_out = model.forward(batch)

    assert set(model_out) == {"x", "labels", "batch", "logits"}
    assert model_out["labels"] is batch.y
    assert model_out["batch"] is batch.batch
    assert model_out["logits"].shape == (
        batch.y.size(0),
        int(cfg.dataset.parameters.num_classes),
    )
    assert torch.isfinite(model_out["logits"]).all()


@pytest.mark.parametrize("model_selector", ("edgnn", "hypergraph_conv"))
def test_synthetic_hypergraph_runs_full_lifecycle(
    model_selector: str,
) -> None:
    """Both native models train, checkpoint, and test without downloads."""
    result = run(_compose(model_selector))

    assert result["epochs_completed"] >= 1
    assert result["observed_train_batch_size"] == 1
    assert {"train/loss", "val/loss"} <= result["fit_metrics"].keys()
    assert all(
        torch.isfinite(torch.tensor(value))
        for value in result["fit_metrics"].values()
    )
    assert result["test_results"]
    assert torch.isfinite(
        torch.tensor(result["test_results"][0]["test/loss"])
    )
