"""Tests for the deterministic native synthetic hypergraph dataset."""

from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig
from torch_geometric.data import Dataset

from topobench.data import (
    HypergraphData,
    validate_hypergraph_node_data,
)
from topobench.data.datasets.synthetic_hypergraph_dataset import (
    SyntheticHypergraphDataset,
    make_synthetic_hypergraph_data,
)
from topobench.data.loaders.hypergraph.synthetic import (
    SyntheticHypergraphDatasetLoader,
)


def _assert_same_data(left: HypergraphData, right: HypergraphData) -> None:
    """Compare every stored native field exactly."""
    assert left.to_dict().keys() == right.to_dict().keys()
    for key in left.to_dict():
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, torch.Tensor):
            assert torch.equal(left_value, right_value)
        else:
            assert left_value == right_value


def test_production_factory_is_deterministic_and_generator_local() -> None:
    """Equal seeds reproduce data without consuming global RNG state."""
    torch.manual_seed(123)
    expected_next = torch.rand(4)
    torch.manual_seed(123)

    first = make_synthetic_hypergraph_data(seed=11)
    actual_next = torch.rand(4)
    second = make_synthetic_hypergraph_data(seed=11)

    _assert_same_data(first, second)
    assert torch.equal(actual_next, expected_next)
    assert validate_hypergraph_node_data(first, num_classes=2) is first


def test_fixture_clones_the_production_factory(
    synthetic_hypergraph: HypergraphData,
) -> None:
    """The shared fixture owns no duplicate topology and returns fresh storage."""
    production = make_synthetic_hypergraph_data()

    _assert_same_data(synthetic_hypergraph, production)
    assert synthetic_hypergraph is not production
    for key, value in production.to_dict().items():
        if isinstance(value, torch.Tensor):
            assert synthetic_hypergraph[key].data_ptr() != value.data_ptr()


def test_synthetic_dataset_is_deterministic_and_native() -> None:
    """Dataset construction packages one validated native PyG example."""
    first = SyntheticHypergraphDataset(seed=5, num_nodes=12, num_hyperedges=5)
    second = SyntheticHypergraphDataset(seed=5, num_nodes=12, num_hyperedges=5)

    assert isinstance(first, Dataset)
    assert len(first) == len(second) == 1
    left = first[0]
    right = second[0]
    assert isinstance(left, HypergraphData)
    _assert_same_data(left, right)
    assert validate_hypergraph_node_data(left, num_classes=2) is left


def test_synthetic_loader_is_deterministic_and_network_free(tmp_path) -> None:
    """The production loader constructs local data without downloads or caches."""
    parameters = DictConfig(
        {
            "data_dir": str(tmp_path),
            "data_name": "SyntheticHypergraph",
            "seed": 17,
            "num_nodes": 12,
            "num_hyperedges": 5,
        }
    )

    first = SyntheticHypergraphDatasetLoader(parameters).load_dataset()
    second = SyntheticHypergraphDatasetLoader(parameters).load_dataset()

    assert isinstance(first, SyntheticHypergraphDataset)
    _assert_same_data(first[0], second[0])
    assert list(tmp_path.iterdir()) == []


def test_synthetic_yaml_targets_the_production_loader() -> None:
    """The packaged selector resolves directly to the native loader."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize(
        version_base="1.3",
        config_path="../../../configs",
        job_name="synthetic_hypergraph_contract",
    ):
        cfg = hydra.compose(
            config_name="run.yaml",
            overrides=[
                "dataset=hypergraph/SyntheticHypergraph",
                "model=hypergraph/edgnn",
            ],
        )

    assert hydra.utils.get_class(cfg.dataset.loader._target_) is (
        SyntheticHypergraphDatasetLoader
    )
    assert cfg.dataset.loader.parameters.data_domain == "hypergraph"
    assert cfg.dataset.loader.parameters.data_type == "synthetic"
    assert cfg.dataset.dataloader_params.batch_size == 1
