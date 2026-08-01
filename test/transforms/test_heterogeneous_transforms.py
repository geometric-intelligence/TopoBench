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


def _assert_heterodata_equal(
    expected: HeteroData,
    actual: HeteroData,
) -> None:
    """Assert exact metadata and store attributes for two small graphs."""
    assert actual.metadata() == expected.metadata()
    for store_type in (*expected.node_types, *expected.edge_types):
        expected_store = expected[store_type]
        actual_store = actual[store_type]
        assert set(actual_store.keys()) == set(expected_store.keys())
        for key, expected_value in expected_store.items():
            actual_value = actual_store[key]
            if torch.is_tensor(expected_value):
                assert torch.equal(actual_value, expected_value)
            else:
                assert actual_value == expected_value


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
                "model=heterogeneous/hgt",
                "transforms=dataset_defaults/SyntheticHeterogeneous",
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


def test_to_undirected_handles_same_node_type_relation(
    heterogeneous_transforms_module,
    synthetic_data: HeteroData,
) -> None:
    """Merge-disabled PyG semantics create a typed reverse self relation."""
    relation = ("author", "collaborates", "author")
    reverse_relation = ("author", "rev_collaborates", "author")
    synthetic_data[relation].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 0]],
    )
    original = synthetic_data.clone()

    transformed = heterogeneous_transforms_module.HeterogeneousToUndirected(
        merge=False,
    )(synthetic_data)

    assert torch.equal(
        transformed[reverse_relation].edge_index,
        original[relation].edge_index.flip(0),
    )


def test_to_undirected_rejects_collision_before_any_mutation(
    heterogeneous_transforms_module,
    synthetic_data: HeteroData,
) -> None:
    """A pre-existing reverse store is never overwritten or partly updated."""
    source_type = ("author", "writes", "paper")
    collision_type = ("paper", "rev_writes", "author")
    synthetic_data[collision_type].edge_index = torch.tensor([[0], [0]])
    synthetic_data[collision_type].stale_edge_attr = torch.tensor([17.0])
    original = synthetic_data.clone()
    transform = heterogeneous_transforms_module.HeterogeneousToUndirected(
        merge=False,
    )

    with pytest.raises(ValueError) as error:
        transform.forward(synthetic_data)

    message = str(error.value)
    assert repr(source_type) in message
    assert repr(collision_type) in message
    _assert_heterodata_equal(original, synthetic_data)


def test_to_undirected_rejects_repeated_application_without_mutation(
    heterogeneous_transforms_module,
    synthetic_data: HeteroData,
) -> None:
    """A second application errors instead of creating nested reverse stores."""
    transform = heterogeneous_transforms_module.HeterogeneousToUndirected(
        merge=False,
    )
    transformed = transform(synthetic_data)
    original = transformed.clone()

    with pytest.raises(ValueError, match="already exists"):
        transform.forward(transformed)

    assert not any(
        relation.startswith("rev_rev_")
        for _, relation, _ in transformed.edge_types
    )
    _assert_heterodata_equal(original, transformed)


@pytest.mark.parametrize(
    ("wrapper_name", "kwargs"),
    [
        ("HeterogeneousConstantFeatures", {"node_types": None}),
        ("HeterogeneousToUndirected", {}),
    ],
)
def test_heterogeneous_wrappers_reject_homogeneous_data(
    heterogeneous_transforms_module,
    wrapper_name: str,
    kwargs: dict[str, object],
) -> None:
    """Direct wrapper calls enforce their native HeteroData contract."""
    wrapper = getattr(heterogeneous_transforms_module, wrapper_name)(**kwargs)

    with pytest.raises(
        TypeError,
        match=rf"{wrapper_name}.*HeteroData.*Data",
    ):
        wrapper.forward(Data(x=torch.ones(2, 1)))


def test_constant_features_none_selects_all_node_stores(
    heterogeneous_transforms_module,
    synthetic_data: HeteroData,
) -> None:
    """The reusable null selection retains PyG's all-node-store semantics."""
    transform = heterogeneous_transforms_module.HeterogeneousConstantFeatures(
        node_types=None,
        value=4.0,
        cat=False,
    )

    transformed = transform(synthetic_data)

    assert transform.node_types is None
    for node_type in transformed.node_types:
        assert torch.equal(
            transformed[node_type].x,
            torch.full((transformed[node_type].num_nodes, 1), 4.0),
        )


@pytest.mark.parametrize(
    ("node_types", "error_type", "match"),
    [
        ("", ValueError, "non-empty"),
        ([], ValueError, "at least one"),
        (["venue", ""], ValueError, "non-empty"),
        (["venue", 1], TypeError, "strings"),
    ],
)
def test_constant_features_rejects_invalid_node_type_selection(
    heterogeneous_transforms_module,
    node_types,
    error_type: type[Exception],
    match: str,
) -> None:
    """Selected node types are normalized only after structural validation."""
    with pytest.raises(error_type, match=match):
        heterogeneous_transforms_module.HeterogeneousConstantFeatures(
            node_types=node_types,
        )


def test_constant_features_rejects_unknown_node_type_before_mutation(
    heterogeneous_transforms_module,
    synthetic_data: HeteroData,
) -> None:
    """An unknown selected store raises with available types and no mutation."""
    original = synthetic_data.clone()
    transform = heterogeneous_transforms_module.HeterogeneousConstantFeatures(
        node_types=["venue", "institution"],
    )
    assert transform.node_types == ("venue", "institution")

    with pytest.raises(ValueError) as error:
        transform.forward(synthetic_data)

    message = str(error.value)
    assert "institution" in message
    for node_type in synthetic_data.node_types:
        assert node_type in message
    _assert_heterodata_equal(original, synthetic_data)


def test_wrapper_constructor_rejects_unknown_keyword() -> None:
    """Registry construction does not silently discard misspelled options."""
    with pytest.raises(TypeError, match="vaule"):
        DataTransform(
            transform_name="HeterogeneousConstantFeatures",
            transform_type="data manipulation",
            node_types="venue",
            vaule=2.0,
        )
    with pytest.raises(TypeError, match="merg"):
        DataTransform(
            transform_name="HeterogeneousToUndirected",
            transform_type="data manipulation",
            merg=False,
        )


@pytest.mark.parametrize("cat", [0, 1, "false", None])
def test_constant_features_requires_actual_bool_cat(
    heterogeneous_transforms_module,
    cat,
) -> None:
    """The cat option rejects bool-like integers, strings, and null."""
    with pytest.raises(TypeError, match="cat must be bool"):
        heterogeneous_transforms_module.HeterogeneousConstantFeatures(
            node_types="venue",
            cat=cat,
        )


@pytest.mark.parametrize("merge", [0, 1, "false", None])
def test_to_undirected_requires_actual_bool_merge(
    heterogeneous_transforms_module,
    merge,
) -> None:
    """The merge option rejects bool-like integers, strings, and null."""
    with pytest.raises(TypeError, match="merge must be bool"):
        heterogeneous_transforms_module.HeterogeneousToUndirected(
            merge=merge,
        )


@pytest.mark.parametrize(
    ("reduce", "expected_weight"),
    [
        pytest.param("add", 6.0, id="add"),
        pytest.param("sum", 6.0, id="sum"),
        pytest.param("mean", 2.0, id="mean"),
        pytest.param("min", 2.0, id="min"),
        pytest.param("max", 2.0, id="max"),
        pytest.param("amin", 2.0, id="amin"),
        pytest.param("amax", 2.0, id="amax"),
        pytest.param("mul", 8.0, id="mul"),
        pytest.param("any", 2.0, id="any"),
    ],
)
def test_to_undirected_executes_all_pyg_coalesce_reductions(
    heterogeneous_transforms_module,
    reduce: str,
    expected_weight: float,
) -> None:
    """Every accepted reduction executes PyG's duplicate-edge coalescing."""
    edge_type = ("node", "links", "node")
    data = HeteroData()
    data["node"].x = torch.ones(2, 1)
    data[edge_type].edge_index = torch.tensor(
        [[0, 0, 1], [1, 1, 0]],
    )
    data[edge_type].edge_weight = torch.full((3,), 2.0)
    original = data.clone()
    transform = heterogeneous_transforms_module.HeterogeneousToUndirected(
        reduce=reduce,
        merge=True,
    )

    transformed = transform(data)

    # Undirecting contributes the three weights to each orientation. Thus
    # sum/add=6, mul=8, and every idempotent/mean reduction remains 2.
    assert transformed is not data
    assert transformed.edge_types == [edge_type]
    assert torch.equal(
        transformed[edge_type].edge_index,
        torch.tensor([[0, 1], [1, 0]]),
    )
    torch.testing.assert_close(
        transformed[edge_type].edge_weight,
        torch.full((2,), expected_weight),
    )
    _assert_heterodata_equal(original, data)


def test_to_undirected_rejects_unknown_reduction(
    heterogeneous_transforms_module,
) -> None:
    """Invalid reductions fail at construction with the supported values."""
    with pytest.raises(ValueError, match=r"median.*add.*sum.*mean"):
        heterogeneous_transforms_module.HeterogeneousToUndirected(
            reduce="median",
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
def test_data_transform_none_is_identity(
    data: Data | HeteroData,
) -> None:
    """An omitted transform is an identity for either representation."""
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
