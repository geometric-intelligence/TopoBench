"""Tests for native heterogeneous dataset loaders."""

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import ClassVar

import hydra
import pytest
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Dataset, HeteroData

from topobench.data import validate_heterogeneous_node_data
from topobench.data.datasets import make_synthetic_heterogeneous_data
from topobench.data.loaders import (
    DBLPDatasetLoader,
    SyntheticHeterogeneousDatasetLoader,
)
from topobench.data.loaders.heterogeneous.dblp import (
    DBLPDatasetLoader as ModuleDBLPLoader,
)
from topobench.data.loaders.heterogeneous.synthetic import (
    SyntheticHeterogeneousDatasetLoader as ModuleSyntheticLoader,
)
from topobench.data.preprocessor import PreProcessor
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


def _offline_dblp_graph() -> HeteroData:
    """Build a small DBLP-shaped graph with official author supervision."""
    data = HeteroData()
    data.provenance = {
        "dataset": "DBLP",
        "split": "official",
        "version": 1,
    }
    data["author"].x = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    data["author"].y = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long)
    data["author"].train_mask = torch.tensor(
        [True, True, True, True, False, False]
    )
    data["author"].val_mask = torch.tensor(
        [False, False, False, False, True, False]
    )
    data["author"].test_mask = torch.tensor(
        [False, False, False, False, False, True]
    )
    data["paper"].x = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    data["term"].x = torch.arange(4, dtype=torch.float32).reshape(4, 1)
    data["conference"].num_nodes = 2

    author_to_paper = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [0, 1, 1, 2, 3, 4]],
        dtype=torch.long,
    )
    paper_to_term = torch.tensor(
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 0]],
        dtype=torch.long,
    )
    paper_to_conference = torch.tensor(
        [[0, 1, 2, 3, 4], [0, 0, 1, 1, 1]],
        dtype=torch.long,
    )
    data["author", "to", "paper"].edge_index = author_to_paper
    data["paper", "to", "author"].edge_index = author_to_paper.flip(0)
    data["paper", "to", "term"].edge_index = paper_to_term
    data["paper", "to", "conference"].edge_index = paper_to_conference
    data["term", "to", "paper"].edge_index = paper_to_term.flip(0)
    data["conference", "to", "paper"].edge_index = paper_to_conference.flip(0)
    return data


class _OfflineDBLPDataset(Dataset):
    """One-graph PyG dataset that records the requested DBLP root."""

    constructed_roots: ClassVar[list[str]] = []

    def __init__(self, root: str) -> None:
        type(self).constructed_roots.append(root)
        self.graph = _offline_dblp_graph()
        super().__init__(root=None)

    def len(self) -> int:
        """Return the one-graph dataset size."""
        return 1

    def get(self, idx: int) -> HeteroData:
        """Return an isolated copy of the only graph."""
        if idx != 0:
            raise IndexError(idx)
        return copy.deepcopy(self.graph)


@pytest.fixture
def offline_dblp(monkeypatch: pytest.MonkeyPatch) -> type[_OfflineDBLPDataset]:
    """Replace the PyG DBLP constructor without touching the network."""
    _OfflineDBLPDataset.constructed_roots.clear()
    monkeypatch.setattr("torch_geometric.datasets.DBLP", _OfflineDBLPDataset)
    return _OfflineDBLPDataset


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


def test_dblp_loader_uses_canonical_root_and_preserves_native_graph(
    tmp_path: Path,
    offline_dblp: type[_OfflineDBLPDataset],
) -> None:
    """The loader delegates to PyG without changing DBLP data or metadata."""
    parameters = OmegaConf.create(
        {
            "data_dir": str(tmp_path),
            "data_name": "DBLP",
        }
    )
    dataset, data_dir = DBLPDatasetLoader(parameters).load()

    expected_root = tmp_path / "DBLP"
    assert offline_dblp.constructed_roots == [str(expected_root)]
    assert Path(data_dir) == expected_root
    assert isinstance(dataset, Dataset)
    actual = dataset[0]
    expected = _offline_dblp_graph()
    assert actual.metadata() == expected.metadata()
    assert actual.edge_types == [
        ("author", "to", "paper"),
        ("paper", "to", "author"),
        ("paper", "to", "term"),
        ("paper", "to", "conference"),
        ("term", "to", "paper"),
        ("conference", "to", "paper"),
    ]
    assert actual.provenance == expected.provenance
    assert actual["author"].y.dtype == torch.long
    for mask_name in ("train_mask", "val_mask", "test_mask"):
        assert torch.equal(
            actual["author"][mask_name],
            expected["author"][mask_name],
        )


def test_dblp_public_export_is_canonical_and_pickleable() -> None:
    """DBLP has one stable public loader identity for Hydra and checkpoints."""
    assert DBLPDatasetLoader is ModuleDBLPLoader
    assert DBLPDatasetLoader.__module__ == (
        "topobench.data.loaders.heterogeneous.dblp"
    )
    assert pickle.loads(pickle.dumps(DBLPDatasetLoader)) is DBLPDatasetLoader


def test_dblp_loader_preprocessor_contract_preserves_official_supervision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    offline_dblp: type[_OfflineDBLPDataset],
) -> None:
    """Only conference features are added before validating DBLP metadata."""

    def _unexpected_random_split(*args: object, **kwargs: object) -> None:
        raise AssertionError("DBLP must not invoke TopoBench split utilities")

    monkeypatch.setattr(
        "topobench.data.preprocessor.preprocessor.load_transductive_splits",
        _unexpected_random_split,
    )
    parameters = OmegaConf.create(
        {
            "data_dir": str(tmp_path),
            "data_name": "DBLP",
        }
    )
    dataset, data_dir = DBLPDatasetLoader(parameters).load()
    original = dataset[0]
    transforms = OmegaConf.create(
        {
            "conference_features": {
                "transform_name": "HeterogeneousConstantFeatures",
                "transform_type": "data manipulation",
                "node_types": "conference",
                "value": 1.0,
                "cat": False,
            }
        }
    )

    processed_dataset = PreProcessor(dataset, data_dir, transforms)
    processed = processed_dataset[0]
    spec = validate_heterogeneous_node_data(
        processed,
        target_node_type="author",
        num_classes=4,
    )

    assert spec.target_node_type == "author"
    assert spec.num_classes == 4
    assert spec.node_types == ("author", "paper", "term", "conference")
    assert spec.input_channels_dict == {
        "author": 3,
        "paper": 2,
        "term": 1,
        "conference": 1,
    }
    assert processed.edge_types == [
        ("author", "to", "paper"),
        ("paper", "to", "author"),
        ("paper", "to", "term"),
        ("paper", "to", "conference"),
        ("term", "to", "paper"),
        ("conference", "to", "paper"),
    ]
    assert processed.provenance == original.provenance
    assert torch.equal(processed["author"].x, original["author"].x)
    assert torch.equal(processed["paper"].x, original["paper"].x)
    assert torch.equal(processed["term"].x, original["term"].x)
    assert torch.equal(
        processed["conference"].x,
        torch.ones(2, 1),
    )
    for mask_name in ("train_mask", "val_mask", "test_mask"):
        assert torch.equal(
            processed["author"][mask_name],
            original["author"][mask_name],
        )


@pytest.mark.parametrize(
    ("experiment", "model_name"),
    [
        ("heterogeneous_dblp_hgt", "hgt"),
        ("heterogeneous_dblp_heterosage", "heterosage"),
    ],
)
def test_dblp_experiment_composes_controlled_full_batch_contract(
    experiment: str,
    model_name: str,
) -> None:
    """Both DBLP experiments share data policy and differ only by model."""
    GlobalHydra.instance().clear()
    register_all_resolvers()
    config_path = str(Path(__file__).resolve().parents[3] / "configs")
    try:
        with hydra.initialize_config_dir(
            version_base="1.3",
            config_dir=config_path,
            job_name=f"test_{experiment}",
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    f"experiment={experiment}",
                    "train=false",
                    "test=false",
                ],
            )
    finally:
        GlobalHydra.instance().clear()

    assert (
        cfg.data_pipeline._target_
        == "topobench.data.pipelines.HeterogeneousNodeDataPipeline"
    )
    assert cfg.dataset.loader._target_ == (
        "topobench.data.loaders.DBLPDatasetLoader"
    )
    assert cfg.dataset.loader.parameters.data_domain == "heterogeneous"
    assert cfg.dataset.loader.parameters.data_type == "bibliographic"
    assert cfg.dataset.loader.parameters.data_name == "DBLP"
    assert cfg.dataset.parameters.target_node_type == "author"
    assert cfg.dataset.parameters.num_classes == 4
    assert cfg.dataset.split_params.learning_setting == "transductive"
    assert cfg.dataset.split_params.source == "official_masks"
    assert cfg.dataset.dataloader_params.mode == "full_batch"
    assert cfg.dataset.dataloader_params.num_workers == 0
    assert cfg.dataset.dataloader_params.pin_memory is False
    assert cfg.dataset.dataloader_params.persistent_workers is False
    assert tuple(cfg.transforms) == ("conference_features",)
    assert cfg.transforms.conference_features.transform_name == (
        "HeterogeneousConstantFeatures"
    )
    assert cfg.transforms.conference_features.node_types == "conference"
    assert cfg.transforms.conference_features.value == pytest.approx(1.0)
    assert cfg.transforms.conference_features.cat is False
    assert cfg.model.model_name == model_name
    assert cfg.logger.wandb.project == "topobench-heterogeneous"
    assert cfg.logger.wandb.name == f"DBLP-{model_name}-full_batch-seed0"
    assert cfg.logger.wandb.group == "DBLP"
    assert cfg.optimizer.parameters.lr == pytest.approx(0.005)
    assert cfg.optimizer.parameters.weight_decay == pytest.approx(0.001)
    assert cfg.trainer.accelerator == "auto"
    assert cfg.trainer.devices == 1
    assert cfg.trainer.min_epochs == 1
    assert cfg.trainer.max_epochs == 200
    assert cfg.trainer.check_val_every_n_epoch == 1
    assert tuple(cfg.tags) == (
        "heterogeneous",
        "DBLP",
        "full_batch",
        model_name,
    )
