"""Tests for validated native heterogeneous node-classification metadata."""

from __future__ import annotations

import copy
import pickle
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch
from torch_geometric.data import Data, HeteroData

from topobench.data import (
    HeterogeneousDataSpec,
    validate_heterogeneous_node_data,
)
from topobench.data.datasets import make_synthetic_heterogeneous_data
from topobench.transforms.data_manipulations.heterogeneous import (
    HeterogeneousConstantFeatures,
    HeterogeneousToUndirected,
)

TARGET = "author"
NUM_CLASSES = 2
WRITES = ("author", "writes", "paper")


@pytest.fixture
def transformed_data() -> HeteroData:
    """Return the canonical graph after its declared preprocessing."""
    data = make_synthetic_heterogeneous_data(seed=7)
    data = HeterogeneousConstantFeatures(node_types="venue")(data)
    return HeterogeneousToUndirected(merge=False)(data)


def _validate(
    data: object,
    *,
    num_classes: object = NUM_CLASSES,
) -> HeterogeneousDataSpec:
    """Validate with the canonical node-classification settings."""
    return validate_heterogeneous_node_data(
        data,
        target_node_type=TARGET,
        num_classes=num_classes,
    )


def _assert_nested_state_equal(
    expected: object,
    actual: object,
    *,
    path: str,
) -> None:
    """Compare a complete nested PyG store snapshot with diagnostics."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), path
        assert actual.dtype == expected.dtype, path
        assert actual.shape == expected.shape, path
        assert actual.device == expected.device, path
        assert actual.layout == expected.layout, path
        assert actual.requires_grad == expected.requires_grad, path
        assert torch.equal(actual, expected), path
        return

    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert tuple(actual) == tuple(expected), path
        for key, expected_value in expected.items():
            _assert_nested_state_equal(
                expected_value,
                actual[key],
                path=f"{path}[{key!r}]",
            )
        return

    if isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected), path
        assert len(actual) == len(expected), path
        for index, (expected_value, actual_value) in enumerate(
            zip(expected, actual, strict=True)
        ):
            _assert_nested_state_equal(
                expected_value,
                actual_value,
                path=f"{path}[{index}]",
            )
        return

    assert type(actual) is type(expected), path
    assert actual == expected, path


def _assert_complete_heterodata_state_equal(
    expected: HeteroData,
    actual: HeteroData,
) -> None:
    """Compare ordered stores and all global, node, and edge attributes."""
    assert actual.node_types == expected.node_types
    assert actual.edge_types == expected.edge_types
    _assert_nested_state_equal(
        expected.to_dict(),
        actual.to_dict(),
        path="heterodata",
    )


def test_valid_data_produces_exact_ordered_spec(
    transformed_data: HeteroData,
) -> None:
    """Validation preserves PyG metadata order and per-type widths."""
    spec = _validate(transformed_data)

    assert spec == HeterogeneousDataSpec(
        node_types=("author", "paper", "venue"),
        edge_types=(
            ("author", "writes", "paper"),
            ("paper", "published_in", "venue"),
            ("paper", "rev_writes", "author"),
            ("venue", "rev_published_in", "paper"),
        ),
        target_node_type="author",
        num_classes=2,
        input_channels=(("author", 8), ("paper", 5), ("venue", 1)),
    )


def test_validation_does_not_mutate_any_heterogeneous_data_state(
    transformed_data: HeteroData,
) -> None:
    """Validation preserves every ordered store and nested attribute."""
    transformed_data.validation_marker = {
        "fold": 0,
        "tags": ("synthetic", "fully-transformed"),
        "tensor": torch.tensor([1.0, 2.0]),
    }
    snapshot = copy.deepcopy(transformed_data)

    _validate(transformed_data)

    _assert_complete_heterodata_state_equal(snapshot, transformed_data)


def test_spec_is_frozen_hashable_pickleable_and_has_fresh_views(
    transformed_data: HeteroData,
) -> None:
    """Metadata is immutable while compatibility views are disposable."""
    spec = _validate(transformed_data)

    assert hash(spec) == hash(pickle.loads(pickle.dumps(spec)))
    assert pickle.loads(pickle.dumps(spec)) == spec
    assert isinstance(spec.node_types, tuple)
    assert isinstance(spec.edge_types, tuple)
    assert isinstance(spec.input_channels, tuple)
    with pytest.raises(FrozenInstanceError):
        spec.num_classes = 3

    channels = spec.input_channels_dict
    metadata = spec.pyg_metadata()
    channels["author"] = 99
    metadata[0].append("mutation")
    metadata[1].clear()

    assert spec.input_channels_dict == {
        "author": 8,
        "paper": 5,
        "venue": 1,
    }
    assert spec.pyg_metadata() == (
        ["author", "paper", "venue"],
        [
            ("author", "writes", "paper"),
            ("paper", "published_in", "venue"),
            ("paper", "rev_writes", "author"),
            ("venue", "rev_published_in", "paper"),
        ],
    )
    assert spec.input_channels_dict is not spec.input_channels_dict
    assert spec.pyg_metadata()[0] is not spec.pyg_metadata()[0]
    assert spec.pyg_metadata()[1] is not spec.pyg_metadata()[1]


def test_direct_spec_construction_normalizes_nested_sequences_deeply() -> None:
    """Direct construction copies mutable inputs into canonical tuple state."""
    node_types = ["author", "paper"]
    relation = ["author", "writes", "paper"]
    edge_types = [relation]
    author_channels = ["author", np.int64(8)]
    input_channels = [author_channels, ["paper", 5]]

    spec = HeterogeneousDataSpec(
        node_types=node_types,
        edge_types=edge_types,
        target_node_type="author",
        num_classes=np.int64(2),
        input_channels=input_channels,
    )
    node_types[0] = "mutated"
    relation[0] = "mutated"
    edge_types.clear()
    author_channels[1] = 999
    input_channels.clear()

    assert spec.node_types == ("author", "paper")
    assert spec.edge_types == (("author", "writes", "paper"),)
    assert spec.input_channels == (("author", 8), ("paper", 5))
    assert type(spec.num_classes) is int
    assert type(spec.input_channels[0][1]) is int
    assert pickle.loads(pickle.dumps(spec)) == spec
    assert isinstance(hash(spec), int)
    with pytest.raises(TypeError):
        spec.edge_types[0][0] = "mutated"
    with pytest.raises(TypeError):
        spec.input_channels[0][1] = 999


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    [
        ({"node_types": "author"}, TypeError, "node_types.*sequence"),
        ({"node_types": ["author", 1]}, TypeError, "node_types.*strings"),
        (
            {"node_types": ["author", "author"]},
            ValueError,
            "node_types.*duplicate",
        ),
        ({"edge_types": ["writes"]}, TypeError, "edge_types.*triple"),
        (
            {"edge_types": [["author", "writes"]]},
            ValueError,
            "edge_types.*three",
        ),
        (
            {"edge_types": [["author", 1, "paper"]]},
            TypeError,
            "edge_types.*strings",
        ),
        (
            {"edge_types": [["unknown", "writes", "paper"]]},
            ValueError,
            "unknown node type",
        ),
        ({"target_node_type": 1}, TypeError, "target_node_type.*str"),
        (
            {"target_node_type": "unknown"},
            ValueError,
            "target_node_type.*node_types",
        ),
        ({"num_classes": True}, TypeError, "num_classes.*integer"),
        ({"num_classes": 2.0}, TypeError, "num_classes.*integer"),
        ({"num_classes": 1}, ValueError, "num_classes.*at least 2"),
        (
            {"input_channels": "author"},
            TypeError,
            "input_channels.*sequence",
        ),
        (
            {"input_channels": [["author", 8, 9], ["paper", 5]]},
            ValueError,
            "input_channels.*two",
        ),
        (
            {"input_channels": [[1, 8], ["paper", 5]]},
            TypeError,
            "input_channels.*node type.*str",
        ),
        (
            {"input_channels": [["author", True], ["paper", 5]]},
            TypeError,
            "input_channels.*width.*integer",
        ),
        (
            {"input_channels": [["author", 0], ["paper", 5]]},
            ValueError,
            "input_channels.*positive",
        ),
        (
            {"input_channels": [["paper", 5], ["author", 8]]},
            ValueError,
            "input_channels.*node_types.*order",
        ),
    ],
)
def test_direct_spec_construction_rejects_invalid_state(
    overrides: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    """The public frozen value object rejects malformed direct state."""
    kwargs = {
        "node_types": ["author", "paper"],
        "edge_types": [["author", "writes", "paper"]],
        "target_node_type": "author",
        "num_classes": 2,
        "input_channels": [["author", 8], ["paper", 5]],
    }
    kwargs.update(overrides)

    with pytest.raises(error_type, match=message):
        HeterogeneousDataSpec(**kwargs)


def test_rejects_non_heterogeneous_top_level_data() -> None:
    """The public contract accepts native HeteroData only."""
    with pytest.raises(
        TypeError,
        match="Expected native torch_geometric.data.HeteroData",
    ):
        _validate(Data(x=torch.ones(2, 1)))


@pytest.mark.parametrize(
    "num_classes",
    [True, 2.0, 2.5, "2", torch.tensor(2)],
)
def test_rejects_non_integral_num_classes(
    transformed_data: HeteroData,
    num_classes: object,
) -> None:
    """Invalid scalar types fail at the public boundary without comparison."""
    with pytest.raises(TypeError, match="num_classes must be an integer"):
        _validate(transformed_data, num_classes=num_classes)


@pytest.mark.parametrize("num_classes", [2, np.int64(2)])
def test_normalizes_integral_num_classes_to_builtin_int(
    transformed_data: HeteroData,
    num_classes: object,
) -> None:
    """Integral subclasses are accepted but never retained in the spec."""
    spec = _validate(transformed_data, num_classes=num_classes)

    assert spec.num_classes == 2
    assert type(spec.num_classes) is int


@pytest.mark.parametrize("target_node_type", [None, 1, ("author",)])
def test_rejects_non_string_target_node_type_early(
    transformed_data: HeteroData,
    target_node_type: object,
) -> None:
    """The public validator rejects misleading target selector types."""
    with pytest.raises(TypeError, match="target_node_type must be str"):
        validate_heterogeneous_node_data(
            transformed_data,
            target_node_type=target_node_type,
            num_classes=NUM_CLASSES,
        )


@pytest.mark.parametrize("num_classes", [-1, 0, 1])
def test_rejects_num_classes_below_two(
    transformed_data: HeteroData,
    num_classes: int,
) -> None:
    """Node classification requires at least two classes."""
    with pytest.raises(ValueError, match="num_classes must be at least 2"):
        _validate(transformed_data, num_classes=num_classes)


def test_unknown_target_names_available_node_types(
    transformed_data: HeteroData,
) -> None:
    """An unknown target error includes the ordered available choices."""
    with pytest.raises(ValueError) as error:
        validate_heterogeneous_node_data(
            transformed_data,
            target_node_type="institution",
            num_classes=NUM_CLASSES,
        )

    assert "institution" in str(error.value)
    assert repr(tuple(transformed_data.node_types)) in str(error.value)


def test_rejects_target_without_labels(
    transformed_data: HeteroData,
) -> None:
    """The target node store owns the label vector."""
    del transformed_data[TARGET].y

    with pytest.raises(ValueError, match=r"'author'.*\by\b"):
        _validate(transformed_data)


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (torch.zeros(36, dtype=torch.int32), "torch.long"),
        (torch.zeros(36, 1, dtype=torch.long), "one-dimensional"),
        (torch.zeros(35, dtype=torch.long), "36 nodes"),
    ],
)
def test_rejects_invalid_target_labels(
    transformed_data: HeteroData,
    labels: torch.Tensor,
    message: str,
) -> None:
    """Labels must be a full-length one-dimensional long tensor."""
    transformed_data[TARGET].y = labels

    with pytest.raises((TypeError, ValueError), match=message):
        _validate(transformed_data)


@pytest.mark.parametrize("mask_name", ["train_mask", "val_mask", "test_mask"])
def test_rejects_missing_split_mask(
    transformed_data: HeteroData,
    mask_name: str,
) -> None:
    """Every explicit split mask is required on the target store."""
    del transformed_data[TARGET][mask_name]

    with pytest.raises(ValueError, match=rf"'author'.*{mask_name}"):
        _validate(transformed_data)


@pytest.mark.parametrize("mask_name", ["train_mask", "val_mask", "test_mask"])
@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (torch.zeros(36, dtype=torch.int64), "boolean"),
        (torch.zeros(36, 1, dtype=torch.bool), "shape"),
        (torch.zeros(35, dtype=torch.bool), "shape"),
    ],
)
def test_rejects_invalid_split_mask(
    transformed_data: HeteroData,
    mask_name: str,
    mask: torch.Tensor,
    message: str,
) -> None:
    """Split masks must be boolean vectors aligned with target labels."""
    transformed_data[TARGET][mask_name] = mask

    with pytest.raises(
        (TypeError, ValueError), match=rf"{mask_name}.*{message}"
    ):
        _validate(transformed_data)


@pytest.mark.parametrize("mask_name", ["train_mask", "val_mask", "test_mask"])
def test_rejects_empty_split_mask(
    transformed_data: HeteroData,
    mask_name: str,
) -> None:
    """Every split must supervise at least one target node."""
    transformed_data[TARGET][mask_name] = torch.zeros(
        transformed_data[TARGET].num_nodes,
        dtype=torch.bool,
    )

    with pytest.raises(ValueError, match=rf"{mask_name}.*non-empty"):
        _validate(transformed_data)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("train_mask", "val_mask"),
        ("train_mask", "test_mask"),
        ("val_mask", "test_mask"),
    ],
)
def test_rejects_pairwise_split_overlap(
    transformed_data: HeteroData,
    first: str,
    second: str,
) -> None:
    """Train, validation, and test supervision must be disjoint."""
    overlap_index = transformed_data[TARGET][first].nonzero()[0].item()
    transformed_data[TARGET][second][overlap_index] = True

    with pytest.raises(ValueError) as error:
        _validate(transformed_data)

    assert first in str(error.value)
    assert second in str(error.value)
    assert "overlap" in str(error.value)


def test_rejects_out_of_range_supervised_label(
    transformed_data: HeteroData,
) -> None:
    """Only class IDs in the configured range may be supervised."""
    supervised = (
        transformed_data[TARGET].train_mask
        | transformed_data[TARGET].val_mask
        | transformed_data[TARGET].test_mask
    )
    transformed_data[TARGET].y[supervised.nonzero()[0]] = NUM_CLASSES

    with pytest.raises(ValueError, match=r"'author'.*\[0, 2\)"):
        _validate(transformed_data)


def test_rejects_negative_label_on_unsupervised_node(
    transformed_data: HeteroData,
) -> None:
    """The qualified full source rejects sentinel class IDs everywhere."""
    target = transformed_data[TARGET]
    index = target.train_mask.nonzero()[0]
    target.train_mask[index] = False
    target.y[index] = -1

    with pytest.raises(ValueError) as error:
        _validate(transformed_data)

    message = str(error.value)
    assert "<heterogeneous>" in message
    assert TARGET in message
    assert "y" in message
    assert "range" in message


def test_masks_may_be_nonexhaustive(
    transformed_data: HeteroData,
) -> None:
    """Disjoint nonempty masks need not cover every target node."""
    target = transformed_data[TARGET]
    target.train_mask = torch.zeros(target.num_nodes, dtype=torch.bool)
    target.val_mask = torch.zeros(target.num_nodes, dtype=torch.bool)
    target.test_mask = torch.zeros(target.num_nodes, dtype=torch.bool)
    target.train_mask[0] = True
    target.val_mask[1] = True
    target.test_mask[2] = True

    assert _validate(transformed_data).target_node_type == TARGET


def test_rejects_node_without_num_nodes(
    transformed_data: HeteroData,
) -> None:
    """Every node type must expose an unambiguous node count."""
    transformed_data["orphan"]

    with (
        pytest.warns(UserWarning),
        pytest.raises(ValueError, match=r"'orphan'.*num_nodes"),
    ):
        _validate(transformed_data)


def test_rejects_missing_features_after_preprocessing(
    transformed_data: HeteroData,
) -> None:
    """Every post-transform node store must have features."""
    del transformed_data["venue"].x

    with pytest.raises(ValueError, match=r"'venue'.*no x after preprocessing"):
        _validate(transformed_data)


@pytest.mark.parametrize(
    ("features", "message"),
    [
        (torch.zeros(24), "invalid x shape"),
        (torch.zeros(23, 5), "invalid x shape"),
        (torch.zeros(24, 0), "zero feature width"),
        (torch.zeros(24, 5, dtype=torch.long), "floating point"),
    ],
)
def test_rejects_invalid_node_features(
    transformed_data: HeteroData,
    features: torch.Tensor,
    message: str,
) -> None:
    """Features must be nonempty floating matrices aligned to node count."""
    transformed_data["paper"].num_nodes = 24
    transformed_data["paper"].x = features

    with pytest.raises((TypeError, ValueError), match=rf"'paper'.*{message}"):
        _validate(transformed_data)


def test_rejects_zero_node_heterogeneous_store_contextually(
    transformed_data: HeteroData,
) -> None:
    transformed_data["orphan"].num_nodes = 0
    transformed_data["orphan"].x = torch.empty((0, 1))

    with pytest.raises(ValueError) as error:
        _validate(transformed_data)

    message = str(error.value)
    assert "<heterogeneous>" in message
    assert "orphan" in message
    assert "field x" in message
    assert "shape=(0, 1)" in message
    assert "dtype=torch.float32" in message
    assert "at least one node" in message


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_rejects_nonfinite_heterogeneous_features_contextually(
    transformed_data: HeteroData,
    invalid: float,
) -> None:
    transformed_data["paper"].x[0, 0] = invalid

    with pytest.raises(ValueError) as error:
        _validate(transformed_data)

    message = str(error.value)
    assert "<heterogeneous>" in message
    assert "paper" in message
    assert "x" in message
    assert "finite" in message


def test_rejects_runtime_vocabulary_missing_configured_class(
    transformed_data: HeteroData,
) -> None:
    with pytest.raises(ValueError) as error:
        _validate(transformed_data, num_classes=3)

    message = str(error.value)
    assert "<heterogeneous>" in message
    assert TARGET in message
    assert "y" in message
    assert "missing" in message
    assert "2" in message


def test_full_heterogeneous_source_allows_one_phase_to_omit_class(
    transformed_data: HeteroData,
) -> None:
    target = transformed_data[TARGET]
    zero = (target.y == 0).nonzero()[0]
    one = (target.y == 1).nonzero()[0]
    target.train_mask.zero_()
    target.val_mask.zero_()
    target.test_mask.zero_()
    target.train_mask[zero] = True
    target.val_mask[one] = True
    target.test_mask[(target.y == 0).nonzero()[1]] = True

    assert _validate(transformed_data).num_classes == NUM_CLASSES


@pytest.mark.parametrize(
    "edge_index",
    [
        torch.zeros(3, 2, dtype=torch.long),
        torch.zeros(2, 2, 1, dtype=torch.long),
    ],
)
def test_rejects_malformed_typed_edge_index(
    transformed_data: HeteroData,
    edge_index: torch.Tensor,
) -> None:
    """A malformed relation error names the full typed relation."""
    transformed_data[WRITES].edge_index = edge_index

    with pytest.raises(ValueError) as error:
        _validate(transformed_data)

    assert repr(WRITES) in str(error.value)
    assert "shape" in str(error.value)


@pytest.mark.parametrize(
    ("edge_index", "message"),
    [
        (torch.tensor([[-1], [0]]), "negative"),
        (torch.tensor([[36], [0]]), "source"),
        (torch.tensor([[0], [24]]), "destination"),
    ],
)
def test_rejects_out_of_bounds_typed_edges(
    transformed_data: HeteroData,
    edge_index: torch.Tensor,
    message: str,
) -> None:
    """Typed-edge bounds errors retain relation and endpoint context."""
    transformed_data[WRITES].edge_index = edge_index

    with pytest.raises(ValueError) as error:
        _validate(transformed_data)

    assert repr(WRITES) in str(error.value)
    assert message in str(error.value)


@pytest.mark.parametrize(
    "edge_index",
    [
        [[0], [0]],
        torch.tensor([[0.0], [0.0]]),
        torch.tensor([[False], [True]]),
        torch.tensor([[0], [0]], dtype=torch.int32),
    ],
)
def test_rejects_non_long_typed_edge_indices_without_mutation(
    transformed_data: HeteroData,
    edge_index: object,
) -> None:
    """Every present typed edge index is a real long tensor."""
    transformed_data[WRITES].edge_index = edge_index
    snapshot = copy.deepcopy(transformed_data)

    with pytest.raises(TypeError) as error:
        _validate(transformed_data)

    assert repr(WRITES) in str(error.value)
    if isinstance(edge_index, torch.Tensor):
        assert "torch.long" in str(error.value)
    else:
        assert "torch.Tensor" in str(error.value)
    _assert_complete_heterodata_state_equal(snapshot, transformed_data)


def test_accepts_valid_empty_long_typed_relation(
    transformed_data: HeteroData,
) -> None:
    """An empty relation remains valid when its shape and dtype are explicit."""
    empty_relation = ("paper", "cites", "paper")
    transformed_data[empty_relation].edge_index = torch.empty(
        (2, 0),
        dtype=torch.long,
    )

    spec = _validate(transformed_data)

    assert spec.edge_types[-1] == empty_relation
