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
        "experiment=heterogeneous_synthetic_hgt_full",
        "hgt",
        "full_batch",
        id="hgt-full-batch",
    ),
    pytest.param(
        "uv run python -m topobench "
        "experiment=heterogeneous_synthetic_heterosage_full",
        "heterosage",
        "full_batch",
        id="heterosage-full-batch",
    ),
    pytest.param(
        "uv run python -m topobench "
        "experiment=heterogeneous_synthetic_hgt_neighbor",
        "hgt",
        "neighbor",
        id="hgt-neighbor",
    ),
    pytest.param(
        "uv run python -m topobench "
        "experiment=heterogeneous_synthetic_heterosage_neighbor",
        "heterosage",
        "neighbor",
        id="heterosage-neighbor",
    ),
)

DBLP_EXAMPLES = (
    pytest.param(
        "TOPOBENCH_ALLOW_DOWNLOADS=1 uv run python -m topobench "
        "experiment=heterogeneous_dblp_hgt",
        "hgt",
        id="hgt",
    ),
    pytest.param(
        "TOPOBENCH_ALLOW_DOWNLOADS=1 uv run python -m topobench "
        "experiment=heterogeneous_dblp_heterosage",
        "heterosage",
        id="heterosage",
    ),
)


OGB_MAG_BOUNDED_EXAMPLES = (
    pytest.param(
        "uv run python -m topobench "
        "experiment=heterogeneous_ogb_mag_hgt "
        "seed=0 "
        "trainer.min_epochs=1 "
        "trainer.max_epochs=1 "
        "trainer.limit_train_batches=10 "
        "trainer.limit_val_batches=5 "
        "trainer.limit_test_batches=5 "
        "logger=heterogeneous_wandb "
        "logger.wandb.name=ogb-mag-hgt-neighbor-bounded-seed0",
        "hgt",
        "ogb-mag-hgt-neighbor-bounded-seed0",
        id="hgt",
    ),
    pytest.param(
        "uv run python -m topobench "
        "experiment=heterogeneous_ogb_mag_heterosage "
        "seed=0 "
        "trainer.min_epochs=1 "
        "trainer.max_epochs=1 "
        "trainer.limit_train_batches=10 "
        "trainer.limit_val_batches=5 "
        "trainer.limit_test_batches=5 "
        "logger=heterogeneous_wandb "
        "logger.wandb.name=ogb-mag-heterosage-neighbor-bounded-seed0",
        "heterosage",
        "ogb-mag-heterosage-neighbor-bounded-seed0",
        id="heterosage",
    ),
)


def _shell_words(command: str) -> list[str]:
    """Tokenize a documented shell command after joining continuations."""
    return shlex.split(command.replace("\\\n", " "))


def _compose_documented_command(command: str, tmp_path: Path) -> DictConfig:
    """Compose the exact documented overrides plus isolated test paths."""
    words = _shell_words(command)
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


def _compose_without_data(command: str) -> DictConfig:
    """Compose a documented command without constructing its dataset."""
    words = _shell_words(command)
    assert words[:5] == ["uv", "run", "python", "-m", "topobench"]

    GlobalHydra.instance().clear()
    register_all_resolvers()
    try:
        with hydra.initialize_config_dir(
            version_base="1.3",
            config_dir=str(_PROJECT_ROOT / "configs"),
            job_name="test_heterogeneous_bounded_documentation",
        ):
            return hydra.compose(config_name="run.yaml", overrides=words[5:])
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
    assert cfg.model.model_name == model_name
    assert cfg.model.model_domain == "heterogeneous"
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
    model.on_train_epoch_start()
    batch = next(iter(datamodule.train_dataloader()))
    assert isinstance(batch, HeteroData)
    try:
        with torch.no_grad():
            output = model.model_step(batch)
    finally:
        model.abort_evaluation()

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


@pytest.mark.parametrize(
    ("command", "model_name"),
    DBLP_EXAMPLES,
)
def test_documented_dblp_commands_compose_without_loading_data(
    command: str,
    model_name: str,
) -> None:
    """Both DBLP commands retain full-batch target-node semantics."""
    guide = _GUIDE.read_text(encoding="utf-8")
    assert command in guide

    command_without_environment = command.split(maxsplit=1)[1]
    cfg = _compose_without_data(command_without_environment)
    assert cfg.dataset.loader.parameters.data_name == "DBLP"
    assert cfg.dataset.parameters.target_node_type == "author"
    assert cfg.dataset.dataloader_params.mode == "full_batch"
    assert cfg.model.model_name == model_name
    assert cfg.model.model_domain == "heterogeneous"


@pytest.mark.parametrize(
    ("command", "model_name", "run_name"),
    OGB_MAG_BOUNDED_EXAMPLES,
)
def test_documented_bounded_ogb_command_composes_without_loading_data(
    command: str,
    model_name: str,
    run_name: str,
) -> None:
    """Both bounded OGB-MAG commands remain strict-Hydra compatible."""
    guide = _GUIDE.read_text(encoding="utf-8")
    assert command in guide

    cfg = _compose_without_data(command)
    assert cfg.dataset.loader.parameters.data_name == "OGB_MAG"
    assert cfg.dataset.dataloader_params.mode == "neighbor"
    assert cfg.model.model_name == model_name
    assert cfg.model.model_domain == "heterogeneous"
    assert cfg.trainer.min_epochs == 1
    assert cfg.trainer.max_epochs == 1
    assert cfg.trainer.limit_train_batches == 10
    assert cfg.trainer.limit_val_batches == 5
    assert cfg.trainer.limit_test_batches == 5
    assert cfg.logger.wandb.project == "topobench-heterogeneous"
    assert cfg.logger.wandb.name == run_name
