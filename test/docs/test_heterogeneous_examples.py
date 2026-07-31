"""Executable examples for the native heterogeneous-graph documentation."""

from __future__ import annotations

import math
import shlex
from pathlib import Path

import hydra
import pytest
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from topobench.dataloader.heterogeneous import HeterogeneousNodeDataModule
from topobench.model import TBModel
from topobench.utils.config_resolvers import register_all_resolvers
from topobench.utils.model_instantiation import instantiate_model

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_GUIDE = _PROJECT_ROOT / "docs" / "heterogeneous_graphs.md"

SYNTHETIC_EXAMPLES = (
    pytest.param(
        "uv run python -m topobench "
        "experiment=heterogeneous_synthetic_hgt_full "
        "train=false test=false logger=[]",
        "hgt",
        "full_batch",
        id="hgt-full-batch",
    ),
    pytest.param(
        "uv run python -m topobench "
        "experiment=heterogeneous_synthetic_heterosage_full "
        "train=false test=false logger=[]",
        "heterosage",
        "full_batch",
        id="heterosage-full-batch",
    ),
    pytest.param(
        "uv run python -m topobench "
        "experiment=heterogeneous_synthetic_hgt_neighbor "
        "train=false test=false logger=[]",
        "hgt",
        "neighbor",
        id="hgt-neighbor",
    ),
    pytest.param(
        "uv run python -m topobench "
        "experiment=heterogeneous_synthetic_heterosage_neighbor "
        "train=false test=false logger=[]",
        "heterosage",
        "neighbor",
        id="heterosage-neighbor",
    ),
)


def _compose_documented_command(command: str, tmp_path: Path) -> DictConfig:
    """Compose the exact documented overrides plus isolated test paths."""
    words = shlex.split(command)
    assert words[:5] == ["uv", "run", "python", "-m", "topobench"]
    overrides = [
        *words[5:],
        f"paths.data_dir={tmp_path / 'data'}",
        f"paths.output_dir={tmp_path / 'output'}",
        f"paths.work_dir={tmp_path}",
    ]

    GlobalHydra.instance().clear()
    register_all_resolvers()
    try:
        with hydra.initialize_config_dir(
            version_base="1.3",
            config_dir=str(_PROJECT_ROOT / "configs"),
            job_name="test_heterogeneous_documentation",
        ):
            return hydra.compose(config_name="run.yaml", overrides=overrides)
    finally:
        GlobalHydra.instance().clear()


@pytest.mark.parametrize(
    ("command", "model_name", "mode"),
    SYNTHETIC_EXAMPLES,
)
def test_documented_synthetic_example_builds_and_forwards(
    command: str,
    model_name: str,
    mode: str,
    tmp_path: Path,
) -> None:
    """Every offline example builds its real pipeline and runs one model step."""
    guide = _GUIDE.read_text(encoding="utf-8")
    assert command in guide

    cfg = _compose_documented_command(command, tmp_path)
    assert cfg.train is False
    assert cfg.test is False
    assert "logger" not in cfg
    assert cfg.model.model_name == model_name
    assert cfg.dataset.dataloader_params.mode == mode

    pipeline = hydra.utils.instantiate(cfg.data_pipeline)
    pipeline_output = pipeline.build(cfg)
    datamodule = pipeline_output.datamodule
    assert isinstance(datamodule, HeterogeneousNodeDataModule)
    assert datamodule.mode == mode
    assert pipeline_output.data_spec is not None

    model = instantiate_model(cfg, data_spec=pipeline_output.data_spec)
    assert isinstance(model, TBModel)
    model.eval()
    model.state_str = "Training"
    batch = next(iter(datamodule.train_dataloader()))
    assert isinstance(batch, HeteroData)
    with torch.no_grad():
        output = model.model_step(batch)

    assert math.isfinite(float(output["loss"]))
    expected_examples = (
        int(batch[datamodule.spec.target_node_type].batch_size)
        if mode == "neighbor"
        else int(
            batch[datamodule.spec.target_node_type].train_mask.sum().item()
        )
    )
    assert output["num_supervised_examples"] == expected_examples
    assert output["logits"].size(0) == expected_examples
    assert output["labels"].size(0) == expected_examples
