"""Opt-in real DBLP pipeline and optimizer smoke test."""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

if os.environ.get("TOPOBENCH_ALLOW_DOWNLOADS") != "1":
    pytest.skip(
        "Set TOPOBENCH_ALLOW_DOWNLOADS=1 to run real-data integration tests",
        allow_module_level=True,
    )

import hydra
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from topobench.data import validate_heterogeneous_node_data
from topobench.utils.config_resolvers import register_all_resolvers
from topobench.utils.model_instantiation import instantiate_model


def _compose(experiment: str, tmp_path: Path) -> DictConfig:
    """Compose one isolated real-data experiment."""
    GlobalHydra.instance().clear()
    register_all_resolvers()
    config_path = str(Path(__file__).resolve().parents[2] / "configs")
    try:
        with hydra.initialize_config_dir(
            version_base="1.3",
            config_dir=config_path,
            job_name=f"test_real_{experiment}",
        ):
            return hydra.compose(
                config_name="run.yaml",
                overrides=[
                    f"experiment={experiment}",
                    f"paths.data_dir={tmp_path / 'data'}",
                    f"paths.output_dir={tmp_path / 'output'}",
                    "trainer.max_epochs=1",
                    "logger=[]",
                ],
            )
    finally:
        GlobalHydra.instance().clear()


@pytest.mark.integration
@pytest.mark.download
def test_real_dblp_smoke(tmp_path: Path) -> None:
    """Process DBLP once and take one real optimizer step per backbone."""
    hgt_cfg = _compose("heterogeneous_dblp_hgt", tmp_path)
    pipeline = hydra.utils.instantiate(hgt_cfg.data_pipeline)
    pipeline_output = pipeline.build(hgt_cfg)
    datamodule = pipeline_output.datamodule
    spec = pipeline_output.data_spec
    assert spec is not None
    data = datamodule.data
    assert isinstance(data, HeteroData)
    assert (
        validate_heterogeneous_node_data(
            data,
            target_node_type="author",
            num_classes=4,
        )
        == spec
    )
    assert all("x" in data[node_type] for node_type in data.node_types)
    assert all(
        mask_name in data["author"]
        for mask_name in ("train_mask", "val_mask", "test_mask")
    )

    for experiment in (
        "heterogeneous_dblp_hgt",
        "heterogeneous_dblp_heterosage",
    ):
        cfg = (
            hgt_cfg
            if experiment == "heterogeneous_dblp_hgt"
            else _compose(experiment, tmp_path)
        )
        model = instantiate_model(cfg, data_spec=spec)
        model.train()
        model.on_train_epoch_start()
        try:
            optimizer = model.configure_optimizers()["optimizer"]
            batch = next(iter(datamodule.train_dataloader()))
            optimizer.zero_grad(set_to_none=True)
            loss = model.model_step(batch)["loss"]

            assert math.isfinite(float(loss.detach()))
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            assert gradients
            assert all(
                torch.isfinite(gradient).all() for gradient in gradients
            )
            assert any(
                torch.count_nonzero(gradient) > 0 for gradient in gradients
            )
            optimizer.step()
        finally:
            model.abort_evaluation()
