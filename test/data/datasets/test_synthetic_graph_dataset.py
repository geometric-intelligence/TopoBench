"""Tests for packaged deterministic native homogeneous graph fixtures."""

from __future__ import annotations

import hydra
import pytest
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, Dataset

from topobench.data.datasets import SyntheticGraphDataset
from topobench.data.loaders import SyntheticGraphDatasetLoader
from topobench.data.splits import validate_transductive_masks


@pytest.mark.parametrize(
    ("task", "expected_length"),
    [
        ("graph_classification", 12),
        ("graph_regression", 12),
        ("node_classification", 1),
    ],
)
def test_synthetic_graph_dataset_is_deterministic(
    task: str,
    expected_length: int,
) -> None:
    """The fixture uses a local seed and reproduces every native tensor."""
    first = SyntheticGraphDataset(task=task, seed=7)
    second = SyntheticGraphDataset(task=task, seed=7)

    assert len(first) == expected_length
    assert len(second) == expected_length
    for left, right in zip(first, second, strict=True):
        assert isinstance(left, Data)
        assert left.to_dict().keys() == right.to_dict().keys()
        for key in left.to_dict():
            assert torch.equal(left[key], right[key])


def test_synthetic_graph_classification_has_fixed_native_graph_splits() -> None:
    """Classification fixtures exercise the production fixed-split path."""
    dataset = SyntheticGraphDataset(task="graph_classification")

    assert isinstance(dataset, Dataset)
    assert dataset.split_idx.keys() == {"train", "valid", "test"}
    assert [len(dataset.split_idx[key]) for key in ("train", "valid", "test")] == [
        8,
        2,
        2,
    ]
    assert all(isinstance(graph, Data) for graph in dataset)
    assert all(graph.y.shape == torch.Size([1]) for graph in dataset)
    assert all(graph.y.dtype == torch.long for graph in dataset)
    assert {int(graph.y.item()) for graph in dataset} == {0, 1}


def test_synthetic_graph_regression_labels_are_scalar_floats() -> None:
    """Each graph target batches to [B, 1], including a smaller final batch."""
    dataset = SyntheticGraphDataset(task="graph_regression")

    assert dataset.split_idx.keys() == {"train", "valid", "test"}
    assert all(graph.y.shape == torch.Size([1]) for graph in dataset)
    assert all(graph.y.is_floating_point() for graph in dataset)


def test_synthetic_node_graph_has_canonical_masks() -> None:
    """The node fixture stores one complete boolean split on its source graph."""
    dataset = SyntheticGraphDataset(task="node_classification")
    data = dataset[0]

    validate_transductive_masks(data)
    assert data.y.shape == torch.Size([data.num_nodes])
    assert all(
        getattr(data, name).shape == torch.Size([data.num_nodes])
        for name in ("train_mask", "val_mask", "test_mask")
    )


@pytest.mark.parametrize(
    ("data_name", "task"),
    [
        ("SyntheticGraph", "graph_classification"),
        ("SyntheticGraphRegression", "graph_regression"),
        ("SyntheticNodeGraph", "node_classification"),
    ],
)
def test_synthetic_loader_returns_native_pyg_dataset(
    tmp_path,
    data_name: str,
    task: str,
) -> None:
    """All config names resolve through one explicit public loader."""
    loader = SyntheticGraphDatasetLoader(
        DictConfig(
            {
                "data_dir": str(tmp_path),
                "data_name": data_name,
                "seed": 11,
            }
        )
    )

    dataset = loader.load_dataset()

    assert isinstance(dataset, SyntheticGraphDataset)
    assert dataset.task == task
    assert all(isinstance(data, Data) for data in dataset)


@pytest.mark.parametrize(
    "config_name",
    ["SyntheticGraph", "SyntheticGraphRegression", "SyntheticNodeGraph"],
)
def test_synthetic_yaml_targets_resolve(config_name: str) -> None:
    """Packaged YAML targets are public through the explicit registries."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize(
        version_base="1.3",
        config_path="../../../configs",
        job_name="synthetic_graph_contract",
    ):
        cfg = hydra.compose(
            config_name="run.yaml",
            overrides=[f"dataset=graph/{config_name}"],
        )

    assert hydra.utils.get_class(cfg.dataset.loader._target_) is (
        SyntheticGraphDatasetLoader
    )
    if config_name == "SyntheticNodeGraph":
        assert cfg.dataset.split_params.learning_setting == "transductive"
        assert cfg.dataset.dataloader_params.batch_size == 1
    else:
        assert cfg.dataset.split_params.split_type == "fixed"
        assert cfg.dataset.dataloader_params.batch_size == 4
