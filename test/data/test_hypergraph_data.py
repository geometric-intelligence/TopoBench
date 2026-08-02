"""Behavioral tests for the native hypergraph data contract."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch_geometric.data import Batch

from topobench.data import (
    HYPERGRAPH_CACHE_FILENAME,
    HYPERGRAPH_REPRESENTATION_VERSION,
    HypergraphData,
    validate_hypergraph_node_data,
    validate_hypergraph_structure,
)
import topobench.data.datasets.synthetic_hypergraph_dataset as synthetic_module
from topobench.data.datasets.synthetic_hypergraph_dataset import (
    make_synthetic_hypergraph_data,
)


def _valid_data() -> HypergraphData:
    """Return a fresh, valid native hypergraph."""
    return make_synthetic_hypergraph_data(
        seed=7, num_nodes=9, num_hyperedges=4
    )


def _assert_data_unchanged(
    data: HypergraphData,
    before: dict[str, object],
) -> None:
    """Assert that validation did not mutate any stored attribute."""
    assert data.to_dict().keys() == before.keys()
    for key, expected in before.items():
        actual = data[key]
        if isinstance(expected, torch.Tensor):
            assert isinstance(actual, torch.Tensor)
            torch.testing.assert_close(
                actual,
                expected,
                rtol=0,
                atol=0,
                equal_nan=True,
            )
        else:
            assert actual == expected


def _snapshot(data: HypergraphData) -> dict[str, object]:
    """Clone stored values so mutation is observable after an exception."""
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in data.to_dict().items()
    }


def test_synthetic_factory_supplies_authoritative_class_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    validator = synthetic_module.validate_hypergraph_node_data

    def record_qualification(
        data: HypergraphData,
        *,
        selector: str,
        num_classes: object,
    ) -> HypergraphData:
        observed["selector"] = selector
        observed["num_classes"] = num_classes
        return validator(
            data,
            selector=selector,
            num_classes=num_classes,
        )

    monkeypatch.setattr(
        synthetic_module,
        "validate_hypergraph_node_data",
        record_qualification,
    )

    data = synthetic_module.make_synthetic_hypergraph_data(seed=7)

    assert isinstance(data, HypergraphData)
    assert observed == {
        "selector": "SyntheticHypergraph",
        "num_classes": 2,
    }


def test_hypergraph_batch_offsets_nodes_and_hyperedges_independently() -> None:
    """Each incidence row uses the count from its own entity family."""
    first = make_synthetic_hypergraph_data(
        seed=1,
        num_nodes=6,
        num_hyperedges=2,
    )
    second = make_synthetic_hypergraph_data(
        seed=2,
        num_nodes=8,
        num_hyperedges=3,
    )

    batch = Batch.from_data_list([first, second])
    second_incidence = batch.hyperedge_index[
        :, first.hyperedge_index.size(1) :
    ]

    assert torch.equal(
        second_incidence[0],
        second.hyperedge_index[0] + first.num_nodes,
    )
    assert torch.equal(
        second_incidence[1],
        second.hyperedge_index[1] + first.num_hyperedges,
    )


def test_batch_preserves_per_example_hyperedge_counts() -> None:
    """A collated count tensor is metadata per graph, not a scalar total."""
    first = make_synthetic_hypergraph_data(
        num_nodes=6,
        num_hyperedges=2,
    )
    second = make_synthetic_hypergraph_data(
        num_nodes=8,
        num_hyperedges=3,
    )

    batch = Batch.from_data_list([first, second])

    assert torch.equal(batch.num_hyperedges, torch.tensor([2, 3]))
    assert batch.num_hyperedges.ndim == 1
    assert batch.num_hyperedges.numel() == 2
    assert int(batch.num_hyperedges.sum()) == 5


def test_validation_is_transactional_and_returns_identical_object() -> None:
    """Successful validation observes the input without normalizing it."""
    data = _valid_data()
    before = _snapshot(data)

    validated = validate_hypergraph_node_data(data, num_classes=2)

    assert validated is data
    _assert_data_unchanged(data, before)


def test_structure_validation_precedes_split_generation_without_mutation() -> (
    None
):
    """The pipeline can validate native structure before attaching masks."""
    data = _valid_data()
    del data.y
    del data.train_mask
    del data.val_mask
    del data.test_mask
    before = _snapshot(data)

    validated = validate_hypergraph_structure(data)

    assert validated is data
    _assert_data_unchanged(data, before)


def test_structure_validation_accepts_stored_builtin_int_version() -> None:
    """The current schema marker is accepted when stored as a plain int."""
    data = _valid_data()
    data.representation_version = HYPERGRAPH_REPRESENTATION_VERSION

    validated = validate_hypergraph_structure(data)

    assert validated is data
    assert type(data["representation_version"]) is int


def test_structure_validation_requires_stored_version_marker() -> None:
    """A class-level default cannot stand in for serialized schema metadata."""
    data = _valid_data()
    del data.representation_version

    with pytest.raises(ValueError, match="requires representation_version"):
        validate_hypergraph_structure(data)


@pytest.mark.parametrize("marker", [True, "2", 2.0])
def test_structure_validation_rejects_wrong_typed_version_markers(
    marker: object,
) -> None:
    """Cache schema metadata is never normalized through integer coercion."""
    data = _valid_data()
    data.representation_version = marker

    with pytest.raises(
        TypeError, match="representation_version.*built-in int"
    ):
        validate_hypergraph_structure(data)


def test_structure_validation_rejects_wrong_version_value() -> None:
    """A well-typed marker for another schema does not cross the boundary."""
    data = _valid_data()
    data.representation_version = HYPERGRAPH_REPRESENTATION_VERSION + 1

    with pytest.raises(ValueError, match="representation_version.*2"):
        validate_hypergraph_structure(data)


@pytest.mark.parametrize(
    "labels",
    [
        lambda labels: labels.float(),
        lambda labels: labels.int(),
        lambda labels: labels.bool(),
    ],
    ids=["float", "int32", "bool"],
)
def test_node_data_validation_requires_long_labels(
    labels: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """Malformed class-label dtypes fail at the complete data boundary."""
    data = _valid_data()
    data.y = labels(data.y)

    with pytest.raises(TypeError, match="y.*torch.long"):
        validate_hypergraph_node_data(data, num_classes=2)


def test_invalid_incidence_is_not_renumbered_or_otherwise_mutated() -> None:
    """A late structural failure leaves every original value untouched."""
    data = _valid_data()
    data.hyperedge_index[1].masked_fill_(data.hyperedge_index[1] == 1, 2)
    before = _snapshot(data)

    with pytest.raises(ValueError, match="contiguous|empty"):
        validate_hypergraph_node_data(data, num_classes=2)

    _assert_data_unchanged(data, before)


def test_hypergraph_rejects_zero_feature_width_before_splitting() -> None:
    data = _valid_data()
    data.x = torch.empty((data.num_nodes, 0))

    with pytest.raises(ValueError) as error:
        validate_hypergraph_node_data(
            data,
            num_classes=2,
            selector="SyntheticHypergraph",
        )

    message = str(error.value)
    assert "SyntheticHypergraph" in message
    assert "node field x" in message
    assert f"shape=({data.num_nodes}, 0)" in message
    assert "dtype=torch.float32" in message
    assert "positive feature width" in message




@pytest.mark.parametrize(
    ("mutate", "error_type", "message"),
    [
        (
            lambda data: setattr(data, "x", data.x.unsqueeze(0)),
            ValueError,
            "x.*rank-2",
        ),
        (
            lambda data: setattr(data, "x", data.x.long()),
            TypeError,
            "x.*floating",
        ),
        (
            lambda data: data.x.__setitem__((0, 0), float("nan")),
            ValueError,
            "x.*finite",
        ),
        (
            lambda data: data.x.__setitem__((0, 0), float("inf")),
            ValueError,
            "x.*finite",
        ),
        (
            lambda data: setattr(
                data, "hyperedge_index", data.hyperedge_index[0]
            ),
            ValueError,
            r"hyperedge_index.*\[2, M\]",
        ),
        (
            lambda data: setattr(
                data, "hyperedge_index", data.hyperedge_index.int()
            ),
            TypeError,
            "hyperedge_index.*torch.long",
        ),
        (
            lambda data: data.hyperedge_index.__setitem__((0, 0), -1),
            ValueError,
            "node indices.*nonnegative",
        ),
        (
            lambda data: data.hyperedge_index.__setitem__((1, 0), -1),
            ValueError,
            "hyperedge IDs.*nonnegative",
        ),
        (
            lambda data: data.hyperedge_index.__setitem__(
                (0, 0), data.num_nodes
            ),
            ValueError,
            "node indices.*num_nodes",
        ),
        (
            lambda data: data.hyperedge_index.__setitem__(
                (1, 0), data.num_hyperedges
            ),
            ValueError,
            "hyperedge IDs.*num_hyperedges",
        ),
        (
            lambda data: data.__delattr__("num_hyperedges"),
            ValueError,
            "num_hyperedges",
        ),
        (
            lambda data: setattr(data, "num_hyperedges", True),
            TypeError,
            "integer",
        ),
        (
            lambda data: setattr(data, "num_hyperedges", 0),
            ValueError,
            "positive",
        ),
        (
            lambda data: data.hyperedge_index[1].add_(1),
            ValueError,
            "hyperedge IDs.*num_hyperedges|contiguous",
        ),
        (
            lambda data: setattr(
                data, "num_hyperedges", data.num_hyperedges + 1
            ),
            ValueError,
            "empty|contiguous",
        ),
        (
            lambda data: setattr(data, "y", data.y[:-1]),
            ValueError,
            "y.*num_nodes",
        ),
        (
            lambda data: setattr(data, "y", data.y.unsqueeze(1)),
            ValueError,
            "y.*rank-1",
        ),
        (
            lambda data: setattr(data, "train_mask", data.train_mask.long()),
            TypeError,
            "train_mask.*boolean",
        ),
        (
            lambda data: setattr(data, "val_mask", data.val_mask.unsqueeze(1)),
            ValueError,
            "val_mask.*rank-1",
        ),
        (
            lambda data: setattr(data, "test_mask", data.test_mask[:-1]),
            ValueError,
            "test_mask.*num_nodes",
        ),
    ],
)
def test_validation_rejects_malformed_native_fields_without_mutation(
    mutate: Callable[[HypergraphData], None],
    error_type: type[Exception],
    message: str,
) -> None:
    """Every native field fails locally with its field name in the error."""
    data = _valid_data()
    mutate(data)
    before = _snapshot(data)

    with pytest.raises(error_type, match=message):
        validate_hypergraph_node_data(data, num_classes=2)

    _assert_data_unchanged(data, before)


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (lambda data: data.y.float(), "dtype"),
        (lambda data: data.y.unsqueeze(1), "rank-1"),
        (lambda data: data.y.masked_fill(data.y == 0, -1), "range"),
        (lambda data: data.y.masked_fill(data.y == 1, 2), "range"),
    ],
    ids=["dtype", "rank", "negative", "above-range"],
)
def test_hypergraph_rejects_malformed_labels_contextually(
    labels: Callable[[HypergraphData], torch.Tensor],
    expected: str,
) -> None:
    data = _valid_data()
    data.y = labels(data)

    with pytest.raises((TypeError, ValueError)) as error:
        validate_hypergraph_node_data(
            data,
            selector="synthetic_hypergraph",
            num_classes=2,
        )

    message = str(error.value)
    assert "synthetic_hypergraph" in message
    assert "node" in message
    assert "y" in message
    assert expected in message


def test_hypergraph_rejects_missing_runtime_class() -> None:
    data = _valid_data()
    data.y.zero_()

    with pytest.raises(ValueError, match=r"synthetic_hypergraph.*y.*missing.*1"):
        validate_hypergraph_node_data(
            data,
            selector="synthetic_hypergraph",
            num_classes=2,
        )


def test_full_hypergraph_source_allows_one_phase_to_omit_class() -> None:
    data = _valid_data()
    zero = (data.y == 0).nonzero(as_tuple=False).flatten()
    one = (data.y == 1).nonzero(as_tuple=False).flatten()
    data.train_mask.zero_()
    data.val_mask.zero_()
    data.test_mask.fill_(True)
    data.train_mask[zero[0]] = True
    data.test_mask[zero[0]] = False
    data.val_mask[one] = True
    data.test_mask[one] = False

    assert (
        validate_hypergraph_node_data(
            data,
            selector="synthetic_hypergraph",
            num_classes=2,
        )
        is data
    )


def test_validation_rejects_overlapping_masks_without_mutation() -> None:
    """A node cannot belong to two supervised phases."""
    data = _valid_data()
    data.val_mask[0] = True
    before = _snapshot(data)

    with pytest.raises(ValueError, match="train_mask.*val_mask.*overlap"):
        validate_hypergraph_node_data(data, num_classes=2)

    _assert_data_unchanged(data, before)


def test_validation_rejects_masks_that_do_not_cover_every_labeled_node() -> (
    None
):
    """Exactly one mask must select every node carrying a label."""
    data = _valid_data()
    data.train_mask[0] = False
    before = _snapshot(data)

    with pytest.raises(ValueError, match="cover.*labeled nodes"):
        validate_hypergraph_node_data(data, num_classes=2)

    _assert_data_unchanged(data, before)


@pytest.mark.parametrize("mask_name", ["train_mask", "val_mask", "test_mask"])
def test_validation_rejects_empty_supervised_splits(mask_name: str) -> None:
    """Every phase must select at least one labeled node."""
    data = _valid_data()
    getattr(data, mask_name).zero_()
    before = _snapshot(data)

    with pytest.raises(
        ValueError, match=rf"{mask_name}.*at least one labeled node"
    ):
        validate_hypergraph_node_data(data, num_classes=2)

    _assert_data_unchanged(data, before)


def test_representation_and_cache_versions_are_shared_public_constants() -> (
    None
):
    """Loaders and data objects use one explicit native cache schema."""
    assert HYPERGRAPH_REPRESENTATION_VERSION == 2
    assert (
        HypergraphData.representation_version
        == HYPERGRAPH_REPRESENTATION_VERSION
    )
    assert HYPERGRAPH_CACHE_FILENAME == "hypergraph_data_v2.pt"
