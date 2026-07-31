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
from topobench.data.datasets.synthetic_hypergraph_dataset import (
    make_synthetic_hypergraph_data,
)


def _valid_data() -> HypergraphData:
    """Return a fresh, valid native hypergraph."""
    return make_synthetic_hypergraph_data(seed=7, num_nodes=9, num_hyperedges=4)


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

    validated = validate_hypergraph_node_data(data)

    assert validated is data
    _assert_data_unchanged(data, before)


def test_structure_validation_precedes_split_generation_without_mutation() -> None:
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


def test_invalid_incidence_is_not_renumbered_or_otherwise_mutated() -> None:
    """A late structural failure leaves every original value untouched."""
    data = _valid_data()
    data.hyperedge_index[1].masked_fill_(data.hyperedge_index[1] == 1, 2)
    before = _snapshot(data)

    with pytest.raises(ValueError, match="contiguous|empty"):
        validate_hypergraph_node_data(data)

    _assert_data_unchanged(data, before)


@pytest.mark.parametrize(
    ("mutate", "error_type", "message"),
    [
        (lambda data: setattr(data, "x", data.x.unsqueeze(0)), ValueError, "x.*rank-2"),
        (lambda data: setattr(data, "x", data.x.long()), TypeError, "x.*floating"),
        (
            lambda data: data.x.__setitem__((0, 0), float("nan")),
            ValueError,
            "x.*finite",
        ),
        (
            lambda data: setattr(data, "hyperedge_index", data.hyperedge_index[0]),
            ValueError,
            r"hyperedge_index.*\[2, M\]",
        ),
        (
            lambda data: setattr(data, "hyperedge_index", data.hyperedge_index.int()),
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
            lambda data: data.hyperedge_index.__setitem__((0, 0), data.num_nodes),
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
        (lambda data: data.__delattr__("num_hyperedges"), ValueError, "num_hyperedges"),
        (lambda data: setattr(data, "num_hyperedges", True), TypeError, "integer"),
        (lambda data: setattr(data, "num_hyperedges", 0), ValueError, "positive"),
        (
            lambda data: data.hyperedge_index[1].add_(1),
            ValueError,
            "hyperedge IDs.*num_hyperedges|contiguous",
        ),
        (
            lambda data: setattr(data, "num_hyperedges", data.num_hyperedges + 1),
            ValueError,
            "empty|contiguous",
        ),
        (lambda data: setattr(data, "y", data.y[:-1]), ValueError, "y.*num_nodes"),
        (lambda data: setattr(data, "y", data.y.unsqueeze(1)), ValueError, "y.*rank-1"),
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
        validate_hypergraph_node_data(data)

    _assert_data_unchanged(data, before)


def test_validation_rejects_overlapping_masks_without_mutation() -> None:
    """A node cannot belong to two supervised phases."""
    data = _valid_data()
    data.val_mask[0] = True
    before = _snapshot(data)

    with pytest.raises(ValueError, match="train_mask.*val_mask.*overlap"):
        validate_hypergraph_node_data(data)

    _assert_data_unchanged(data, before)


def test_validation_rejects_masks_that_do_not_cover_every_labeled_node() -> None:
    """Exactly one mask must select every node carrying a label."""
    data = _valid_data()
    data.train_mask[0] = False
    before = _snapshot(data)

    with pytest.raises(ValueError, match="cover.*labeled nodes"):
        validate_hypergraph_node_data(data)

    _assert_data_unchanged(data, before)


@pytest.mark.parametrize("mask_name", ["train_mask", "val_mask", "test_mask"])
def test_validation_rejects_empty_supervised_splits(mask_name: str) -> None:
    """Every phase must select at least one labeled node."""
    data = _valid_data()
    getattr(data, mask_name).zero_()
    before = _snapshot(data)

    with pytest.raises(ValueError, match=rf"{mask_name}.*at least one labeled node"):
        validate_hypergraph_node_data(data)

    _assert_data_unchanged(data, before)


def test_representation_and_cache_versions_are_shared_public_constants() -> None:
    """Loaders and data objects use one explicit native cache schema."""
    assert HYPERGRAPH_REPRESENTATION_VERSION == 2
    assert HypergraphData.representation_version == HYPERGRAPH_REPRESENTATION_VERSION
    assert HYPERGRAPH_CACHE_FILENAME == "hypergraph_data_v2.pt"
