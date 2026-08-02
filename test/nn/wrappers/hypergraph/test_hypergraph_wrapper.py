"""Native hypergraph wrapper contract tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import Tensor, nn
from torch_geometric.data import Batch

from topobench.data import HypergraphData
from topobench.dataloader.graph import mark_hypergraph_validated
from topobench.nn.wrappers.hypergraph import (
    WRAPPER_CLASSES,
    HypergraphWrapper,
)


class RecordingBackbone(nn.Module):
    """Record exact native arguments and return a configurable output."""

    def __init__(
        self,
        output: Callable[[Tensor], object] | None = None,
    ) -> None:
        super().__init__()
        self.calls: list[tuple[Tensor, Tensor]] = []
        self.output = output or (lambda x: x[:, :2] + 1)

    def forward(self, x: Tensor, hyperedge_index: Tensor) -> object:
        self.calls.append((x, hyperedge_index))
        return self.output(x)


def _native_batch() -> Batch:
    first = HypergraphData(
        x=torch.arange(9, dtype=torch.float32).reshape(3, 3),
        hyperedge_index=torch.tensor(
            [[0, 1, 1, 2], [0, 0, 1, 1]], dtype=torch.long
        ),
        num_hyperedges=2,
        y=torch.tensor([0, 1, 0], dtype=torch.long),
    )
    second = HypergraphData(
        x=torch.arange(6, dtype=torch.float32).reshape(2, 3),
        hyperedge_index=torch.tensor(
            [[0, 1], [0, 0]], dtype=torch.long
        ),
        num_hyperedges=1,
        y=torch.tensor([1, 0], dtype=torch.long),
    )
    batch = Batch.from_data_list([first, second])
    split_ids = torch.arange(batch.num_nodes) % 3
    batch.train_mask = split_ids == 0
    batch.val_mask = split_ids == 1
    batch.test_mask = split_ids == 2
    return mark_hypergraph_validated(batch)


def test_hypergraph_wrapper_export_is_explicit_and_narrow() -> None:
    """The hypergraph wrapper package exposes only its native adapter."""
    assert {"HypergraphWrapper": HypergraphWrapper} == WRAPPER_CLASSES
    assert HypergraphWrapper.__bases__ == (nn.Module,)


def test_wrapper_uses_exact_native_fields_and_returns_exact_keys() -> None:
    """Collated hyperedge counts are validated, never passed to a backbone."""
    batch = _native_batch()
    backbone = RecordingBackbone()

    result = HypergraphWrapper(backbone)(batch)

    assert set(result) == {"x", "labels", "batch"}
    assert result["labels"] is batch.y
    assert result["batch"] is batch.batch
    assert torch.equal(result["x"], batch.x[:, :2] + 1)
    assert backbone.calls == [(batch.x, batch.hyperedge_index)]
    assert batch.num_hyperedges.shape == (2,)


def _without(batch: Batch, field: str) -> Batch:
    clone = mark_hypergraph_validated(batch.clone())
    del clone[field]
    return clone


def _invalid_cases() -> tuple[tuple[str, Batch], ...]:
    base = _native_batch()
    cases = [
        (field, _without(base, field))
        for field in ("x", "hyperedge_index", "num_hyperedges", "y", "batch")
    ]
    invalid_values: tuple[tuple[str, object], ...] = (
        ("x", torch.ones(5, 3, dtype=torch.long)),
        ("x", torch.randn(5)),
        ("x", torch.full((5, 3), float("nan"))),
        ("hyperedge_index", torch.tensor([0, 1], dtype=torch.long)),
        ("hyperedge_index", torch.tensor([[0.0], [0.0]])),
        (
            "hyperedge_index",
            torch.tensor([[0, 5, 1, 2, 3, 4], [0, 0, 1, 1, 2, 2]]),
        ),
        (
            "hyperedge_index",
            torch.tensor([[0, 1, 1, 2, 3, 4], [0, 0, 1, 1, 2, 3]]),
        ),
        (
            "hyperedge_index",
            torch.tensor([[0, 1, 1, 2, 3, 4], [0, 0, 2, 2, 2, 2]]),
        ),
        (
            "hyperedge_index",
            torch.tensor([[0, 1, 1, 2, 0, 4], [0, 0, 1, 1, 2, 2]]),
        ),
        ("num_hyperedges", torch.tensor([2.0, 1.0])),
        ("num_hyperedges", torch.tensor([3], dtype=torch.long)),
        ("num_hyperedges", torch.tensor([2, 0], dtype=torch.long)),
        ("y", torch.tensor([[0, 1, 0, 1, 0]], dtype=torch.long)),
        ("y", torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])),
        ("y", torch.tensor([0, 1, 0, 1], dtype=torch.long)),
        ("batch", torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.long)),
        ("batch", torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])),
        ("batch", torch.tensor([0, 0, 0, 1], dtype=torch.long)),
        ("batch", torch.tensor([0, 0, 0, 2, 2], dtype=torch.long)),
    )
    for field, value in invalid_values:
        batch = mark_hypergraph_validated(base.clone())
        batch[field] = value
        cases.append((field, batch))
    return tuple(cases)


@pytest.mark.parametrize(("field", "batch"), _invalid_cases())
def test_invalid_native_fields_fail_before_backbone(
    field: str, batch: Batch
) -> None:
    """All malformed inputs fail transactionally before backbone side effects."""
    backbone = RecordingBackbone()

    with pytest.raises((TypeError, ValueError), match=field):
        HypergraphWrapper(backbone)(batch)

    assert backbone.calls == []


@pytest.mark.parametrize(
    "mutate",
    (
        lambda batch: setattr(batch, "x", batch.x.clone()),
        lambda batch: setattr(batch, "y", batch.y.clone()),
        lambda batch: batch.hyperedge_index.add_(0),
        lambda batch: batch.train_mask.logical_not_(),
    ),
)
def test_wrapper_rejects_mutation_after_boundary_validation(
    mutate: Callable[[Batch], object],
) -> None:
    """Tensor replacement and in-place writes invalidate the cheap marker."""
    batch = _native_batch()
    backbone = RecordingBackbone()
    mutate(batch)

    with pytest.raises(ValueError, match="changed after boundary validation"):
        HypergraphWrapper(backbone)(batch)

    assert backbone.calls == []


def test_wrapper_reuses_boundary_validation_without_full_tensor_scans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward performs only constant-size field and marker checks."""
    batch = _native_batch()
    backbone = RecordingBackbone()

    def forbidden_scan(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("full tensor scan reached wrapper forward")

    for scan_name in ("isfinite", "bincount", "bucketize", "equal"):
        monkeypatch.setattr(torch, scan_name, forbidden_scan)

    result = HypergraphWrapper(backbone)(batch)

    assert result["x"].shape == (batch.num_nodes, 2)
    assert len(backbone.calls) == 1


def test_wrapper_accepts_valid_data_device_transfer() -> None:
    """Boundary metadata survives framework-managed Data.to transfers."""
    batch = _native_batch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = batch.to(device)

    result = HypergraphWrapper(RecordingBackbone())(batch)

    assert result["x"].device == device
    assert result["labels"].device == device
    assert result["batch"].device == device


@pytest.mark.parametrize(
    ("output", "error", "message"),
    [
        (lambda x: (x, None), TypeError, "tensor"),
        (lambda x: x[:, 0], ValueError, "rank-2"),
        (lambda x: x[:-1], ValueError, "node"),
    ],
)
def test_invalid_backbone_output_is_rejected(
    output: Callable[[Tensor], object],
    error: type[Exception],
    message: str,
) -> None:
    """The wrapper enforces one rank-2 embedding row per native node."""
    backbone = RecordingBackbone(output)

    with pytest.raises(error, match=message):
        HypergraphWrapper(backbone)(_native_batch())

    assert len(backbone.calls) == 1
