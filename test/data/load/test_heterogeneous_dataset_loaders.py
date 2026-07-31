"""Tests for native heterogeneous dataset loaders."""

from pathlib import Path

import hydra
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import HeteroData

from topobench.data.datasets import make_synthetic_heterogeneous_data
from topobench.data.loaders import SyntheticHeterogeneousDatasetLoader
from topobench.data.loaders.heterogeneous.synthetic import (
    SyntheticHeterogeneousDatasetLoader as ModuleSyntheticLoader,
)
from topobench.utils.config_resolvers import register_all_resolvers


def _parameters(tmp_path: Path, **overrides: int) -> DictConfig:
    """Build the minimal loader configuration used by focused tests."""
    return OmegaConf.create(
        {
            "data_dir": str(tmp_path),
            "data_name": "SyntheticHeterogeneous",
            "seed": 11,
            **overrides,
        }
    )


def test_synthetic_loader_returns_one_native_heterogeneous_graph(
    tmp_path: Path,
) -> None:
    """The loader exposes the fixture as a one-graph PyG dataset."""
    dataset, data_dir = SyntheticHeterogeneousDatasetLoader(
        _parameters(tmp_path)
    ).load()

    assert len(dataset) == 1
    assert isinstance(dataset[0], HeteroData)
    assert Path(data_dir) == tmp_path / "SyntheticHeterogeneous"


def test_synthetic_loader_forwards_seed_to_canonical_factory(
    tmp_path: Path,
) -> None:
    """The configured seed controls fixture generation without replacement."""
    configured_seed = 17
    dataset, _ = SyntheticHeterogeneousDatasetLoader(
        _parameters(tmp_path, seed=configured_seed)
    ).load()
    actual = dataset[0]
    expected = make_synthetic_heterogeneous_data(seed=configured_seed)
    default_seed = make_synthetic_heterogeneous_data(seed=0)

    assert not torch.equal(
        expected["author"].x,
        default_seed["author"].x,
    )
    assert torch.equal(actual["author"].x, expected["author"].x)
    assert torch.equal(
        actual["author"].train_mask,
        expected["author"].train_mask,
    )


def test_synthetic_loader_forwards_fixture_sizes(tmp_path: Path) -> None:
    """Optional node counts are forwarded without loader-side policy."""
    dataset, _ = SyntheticHeterogeneousDatasetLoader(
        _parameters(
            tmp_path,
            num_authors=20,
            num_papers=12,
            num_venues=3,
        )
    ).load()
    data = dataset[0]

    assert data["author"].num_nodes == 20
    assert data["paper"].num_nodes == 12
    assert data["venue"].num_nodes == 3


def test_public_loader_export_is_the_canonical_module_class() -> None:
    """Public and module imports must resolve to one stable class object."""
    assert SyntheticHeterogeneousDatasetLoader is ModuleSyntheticLoader
    assert (
        SyntheticHeterogeneousDatasetLoader.__module__
        == "topobench.data.loaders.heterogeneous.synthetic"
    )


def test_synthetic_heterogeneous_config_composes_exact_contract() -> None:
    """The Hydra config declares only the supported staged data contract."""
    GlobalHydra.instance().clear()
    register_all_resolvers()
    config_path = str(Path(__file__).resolve().parents[3] / "configs")
    try:
        with hydra.initialize_config_dir(
            version_base="1.3",
            config_dir=config_path,
            job_name="test_synthetic_heterogeneous_config",
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "dataset=heterogeneous/SyntheticHeterogeneous",
                    "model=cell/hgt",
                    "transforms=no_transform",
                    "train=false",
                    "test=false",
                ],
            )
    finally:
        GlobalHydra.instance().clear()

    assert (
        cfg.dataset.loader._target_
        == "topobench.data.loaders.SyntheticHeterogeneousDatasetLoader"
    )
    assert cfg.dataset.loader.parameters.data_domain == "heterogeneous"
    assert cfg.dataset.loader.parameters.data_type == "synthetic"
    assert cfg.dataset.loader.parameters.data_name == "SyntheticHeterogeneous"
    assert cfg.dataset.loader.parameters.seed == cfg.seed
    assert "num_features" not in cfg.dataset.parameters
    assert cfg.dataset.parameters.target_node_type == "author"
    assert cfg.dataset.parameters.task_level == "node"
    assert cfg.dataset.split_params.learning_setting == "transductive"
    assert cfg.dataset.split_params.source == "official_masks"
    assert cfg.dataset.dataloader_params.mode == "full_batch"
    assert cfg.dataset.dataloader_params.evaluation_protocol == "full_graph"
    assert cfg.dataset.dataloader_params.evaluation_seed == cfg.seed
    assert cfg.dataset.dataloader_params.num_workers == 0
    assert cfg.dataset.dataloader_params.pin_memory is False
    assert cfg.dataset.dataloader_params.persistent_workers is False
