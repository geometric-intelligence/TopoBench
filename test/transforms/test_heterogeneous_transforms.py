"""Tests for native heterogeneous data-manipulation transforms."""

from __future__ import annotations

import importlib
import pickle
from collections.abc import Iterator

import hydra
import pytest
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform

from topobench.data.datasets import (
    SyntheticHeterogeneousDataset,
    make_synthetic_heterogeneous_data,
)
from topobench.data.preprocessor import PreProcessor
from topobench.transforms import TRANSFORMS
from topobench.transforms.data_manipulations import DATA_MANIPULATIONS
from topobench.transforms.data_transform import DataTransform
from topobench.utils.config_resolvers import register_all_resolvers


@pytest.fixture
def heterogeneous_transforms_module():
    """Import the heterogeneous wrappers after test collection."""
    return importlib.import_module(
        "topobench.transforms.data_manipulations.heterogeneous"
    )


@pytest.fixture
def synthetic_data() -> HeteroData:
    """Return a fresh canonical heterogeneous graph."""
    return make_synthetic_heterogeneous_data(seed=7)


@pytest.fixture
def synthetic_transform_config() -> Iterator[DictConfig]:
    """Compose the synthetic dataset's default transform configuration."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    register_all_resolvers()
    with hydra.initialize(version_base="1.3", config_path="../../configs"):
        config = hydra.compose(
            config_name="run.yaml",
            overrides=[
                "dataset=heterogeneous/SyntheticHeterogeneous",
                "model=cell/hgt",
                "train=false",
                "test=false",
            ],
        )
        yield config.transforms
    hydra.core.global_hydra.GlobalHydra.instance().clear()


def test_heterogeneous_constant_features_fills_only_selected_store(
    heterogeneous_transforms_module,
    synthetic_data: HeteroData,
) -> None:
    """Constant features fill venue nodes without changing other stores."""
    original = synthetic_data.clone()
    transform = heterogeneous_transforms_module.HeterogeneousConstantFeatures(
        node_types="venue",
        value=2.5,
        cat=False,
    )

    transformed = transform(synthetic_data)

    assert transformed is not synthetic_data
    assert torch.equal(transformed["author"].x, original["author"].x)
    assert torch.equal(transformed["paper"].x, original["paper"].x)
    assert transformed["venue"].x.shape == (
        original["venue"].num_nodes,
        1,
    )
    assert torch.equal(
        transformed["venue"].x,
        torch.full((original["venue"].num_nodes, 1), 2.5),
    )
    for attribute in ("y", "train_mask", "val_mask", "test_mask"):
        assert torch.equal(
            transformed["author"][attribute],
            original["author"][attribute],
        )


def test_to_undirected_adds_reverse_typed_relations_without_merging(
    heterogeneous_transforms_module,
    synthetic_data: HeteroData,
) -> None:
    """The wrapper creates exact PyG reverse relation stores."""
    original = synthetic_data.clone()
    transform = heterogeneous_transforms_module.HeterogeneousToUndirected(
        merge=False,
        reduce="add",
    )

    transformed = transform(synthetic_data)

    assert transformed is not synthetic_data
    expected_reverse_relations = {
        ("paper", "rev_writes", "author"): (
            "author",
            "writes",
            "paper",
        ),
        ("venue", "rev_published_in", "paper"): (
            "paper",
            "published_in",
            "venue",
        ),
    }
    assert transformed.edge_types == [
        *original.edge_types,
        *expected_reverse_relations,
    ]
    for reverse_type, forward_type in expected_reverse_relations.items():
        assert torch.equal(
            transformed[reverse_type].edge_index,
            original[forward_type].edge_index.flip(0),
        )
    for forward_type in original.edge_types:
        assert torch.equal(
            transformed[forward_type].edge_index,
            original[forward_type].edge_index,
        )


def test_heterogeneous_transforms_compose_in_declared_order(
    synthetic_transform_config: DictConfig,
    synthetic_data: HeteroData,
) -> None:
    """Hydra composition preserves the preprocessor's execution order."""
    assert list(synthetic_transform_config) == [
        "venue_features",
        "reverse_relations",
    ]
    assert dict(synthetic_transform_config.venue_features) == {
        "transform_name": "HeterogeneousConstantFeatures",
        "transform_type": "data manipulation",
        "node_types": "venue",
        "value": 1.0,
        "cat": False,
    }
    assert dict(synthetic_transform_config.reverse_relations) == {
        "transform_name": "HeterogeneousToUndirected",
        "transform_type": "data manipulation",
        "merge": False,
        "reduce": "add",
    }

    preprocessor = object.__new__(PreProcessor)
    composed = preprocessor.instantiate_pre_transform(
        "unused",
        synthetic_transform_config,
    )
    original = synthetic_data.clone()
    transformed = composed(synthetic_data)

    assert torch.equal(transformed["author"].x, original["author"].x)
    assert torch.equal(transformed["paper"].x, original["paper"].x)
    assert torch.equal(
        transformed["venue"].x,
        torch.ones(original["venue"].num_nodes, 1),
    )
    for attribute in ("y", "train_mask", "val_mask", "test_mask"):
        assert torch.equal(
            transformed["author"][attribute],
            original["author"][attribute],
        )
    assert transformed.edge_types == [
        *original.edge_types,
        ("paper", "rev_writes", "author"),
        ("venue", "rev_published_in", "paper"),
    ]


def test_data_transform_rejects_unmarked_transform_for_heterodata(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_data: HeteroData,
) -> None:
    """An existing transform must explicitly opt in to HeteroData."""

    class UnmarkedTransform(BaseTransform):
        def __init__(self, **_):
            super().__init__()

        def forward(self, data):
            return data

    monkeypatch.setitem(TRANSFORMS, "UnmarkedTransform", UnmarkedTransform)
    transform = DataTransform(transform_name="UnmarkedTransform")

    with pytest.raises(TypeError) as error:
        transform(synthetic_data)

    message = str(error.value)
    assert "UnmarkedTransform" in message
    for node_type in synthetic_data.node_types:
        assert node_type in message
    for edge_type in synthetic_data.edge_types:
        assert repr(edge_type) in message


@pytest.mark.parametrize("data", [Data(x=torch.ones(2, 1)), HeteroData()])
def test_data_transform_without_name_is_identity(
    data: Data | HeteroData,
) -> None:
    """A disabled transform is an identity for either PyG representation."""
    transform = DataTransform(transform_name=None)

    assert transform.forward(data) is data
    transformed = transform(data)
    assert transformed is not data
    assert transformed.to_dict() == data.to_dict()


def test_marked_transform_rejects_unsupported_return_type(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_data: HeteroData,
) -> None:
    """DataTransform reports both producer and unsupported result types."""

    class InvalidResultTransform(BaseTransform):
        supports_heterodata = True

        def __init__(self, **_):
            super().__init__()

        def forward(self, data):
            return {"data": data}

    monkeypatch.setitem(
        TRANSFORMS,
        "InvalidResultTransform",
        InvalidResultTransform,
    )
    transform = DataTransform(transform_name="InvalidResultTransform")

    with pytest.raises(
        TypeError,
        match="InvalidResultTransform returned unsupported type dict",
    ):
        transform(synthetic_data)


def test_unmarked_transform_still_supports_homogeneous_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compatibility gate does not change homogeneous behavior."""

    class HomogeneousTransform(BaseTransform):
        def __init__(self, **_):
            super().__init__()

        def forward(self, data):
            data.answer = 42
            return data

    monkeypatch.setitem(
        TRANSFORMS,
        "HomogeneousTransform",
        HomogeneousTransform,
    )
    data = Data(x=torch.ones(2, 1))

    transformed = DataTransform(transform_name="HomogeneousTransform")(data)

    assert transformed is not data
    assert transformed.answer == 42
    assert "answer" not in data


def test_wrapper_exports_and_registries_use_canonical_classes(
    heterogeneous_transforms_module,
) -> None:
    """All public lookups retain one importable, pickle-stable class."""
    manipulations_module = importlib.import_module(
        "topobench.transforms.data_manipulations"
    )
    for class_name in (
        "HeterogeneousConstantFeatures",
        "HeterogeneousToUndirected",
    ):
        canonical_class = getattr(heterogeneous_transforms_module, class_name)
        assert canonical_class.__module__ == (
            "topobench.transforms.data_manipulations.heterogeneous"
        )
        assert getattr(manipulations_module, class_name) is canonical_class
        assert DATA_MANIPULATIONS[class_name] is canonical_class
        assert TRANSFORMS[class_name] is canonical_class

        kwargs = (
            {"node_types": "venue"}
            if class_name == "HeterogeneousConstantFeatures"
            else {}
        )
        restored = pickle.loads(pickle.dumps(canonical_class(**kwargs)))
        assert type(restored) is canonical_class


def test_preprocessor_persists_composed_heterogeneous_transforms(
    tmp_path,
    synthetic_transform_config: DictConfig,
) -> None:
    """The real preprocessing path saves and reloads transformed HeteroData."""
    dataset = SyntheticHeterogeneousDataset(seed=11)
    original = dataset[0].clone()

    preprocessor = PreProcessor(
        dataset,
        str(tmp_path),
        synthetic_transform_config,
    )
    reloaded_preprocessor = PreProcessor(
        dataset,
        str(tmp_path),
        synthetic_transform_config,
    )
    transformed = preprocessor[0]
    reloaded = reloaded_preprocessor[0]

    assert isinstance(transformed, HeteroData)
    assert isinstance(reloaded, HeteroData)
    assert torch.equal(transformed["author"].x, original["author"].x)
    assert torch.equal(transformed["paper"].x, original["paper"].x)
    assert torch.equal(
        transformed["venue"].x,
        torch.ones(original["venue"].num_nodes, 1),
    )
    assert transformed.edge_types == [
        *original.edge_types,
        ("paper", "rev_writes", "author"),
        ("venue", "rev_published_in", "paper"),
    ]
    for attribute in ("y", "train_mask", "val_mask", "test_mask"):
        assert torch.equal(
            transformed["author"][attribute],
            original["author"][attribute],
        )
    assert transformed.metadata() == reloaded.metadata()
    for node_type in transformed.node_types:
        assert torch.equal(transformed[node_type].x, reloaded[node_type].x)
    for edge_type in transformed.edge_types:
        assert torch.equal(
            transformed[edge_type].edge_index,
            reloaded[edge_type].edge_index,
        )
