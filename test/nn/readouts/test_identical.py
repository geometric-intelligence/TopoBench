"""Tests for the native homogeneous ``NoReadOut`` contract."""

import pytest
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from topobench.nn.readouts.base import AbstractReadout
from topobench.nn.readouts.identical import NoReadOut


@pytest.mark.parametrize("pooling", ["sum", "mean", "max"])
def test_graph_readout_pools_model_output_batch(pooling: str) -> None:
    """Graph logits pool native node embeddings with PyG scatter."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    batch_index = torch.tensor([0, 0, 1, 1])
    model_out = {"x": x, "batch": batch_index, "labels": torch.tensor([0, 1])}
    readout = NoReadOut(
        hidden_dim=2,
        out_channels=2,
        task_level="graph",
        pooling_type=pooling,
        logits_linear_layer=False,
    )

    result = readout(model_out, Data())

    assert result is model_out
    assert torch.equal(
        result["logits"], scatter(x, batch_index, dim=0, reduce=pooling)
    )
    assert isinstance(readout.linear, nn.Identity)


def test_node_classification_does_not_pool() -> None:
    """Node logits retain one output per native node even with batch indices."""
    x = torch.randn(5, 4)
    model_out = {
        "x": x,
        "batch": torch.tensor([0, 0, 1, 1, 1]),
        "labels": torch.tensor([0, 1, 0, 1, 0]),
    }
    readout = NoReadOut(
        hidden_dim=4,
        out_channels=3,
        task_level="node",
    )

    result = readout(model_out, Data())

    assert result["logits"].shape == (5, 3)
    assert result["x"] is x


def test_readout_uses_model_output_batch_over_data_batch() -> None:
    """The model output owns graph assignment when input data disagrees."""
    model_out = {
        "x": torch.randn(4, 2),
        "batch": torch.tensor([0, 0, 1, 1]),
    }
    data = Data(batch=torch.zeros(4, dtype=torch.long))
    readout = NoReadOut(
        hidden_dim=2,
        out_channels=2,
        task_level="graph",
        logits_linear_layer=False,
    )

    result = readout(model_out, data)

    assert result["logits"].shape == (2, 2)


def test_existing_logits_are_preserved() -> None:
    """A specialized readout may provide logits before the base hook."""
    logits = torch.randn(3, 2)
    model_out = {
        "x": torch.randn(3, 4),
        "batch": None,
        "logits": logits,
    }
    readout = NoReadOut(
        hidden_dim=4,
        out_channels=2,
        task_level="node",
    )

    assert readout(model_out, Data())["logits"] is logits


def test_public_name_and_native_base_class_are_stable() -> None:
    """NoReadOut uses the native graph readout base."""
    readout = NoReadOut(hidden_dim=4, out_channels=2, task_level="node")

    assert isinstance(readout, AbstractReadout)
    assert repr(readout) == "NoReadOut()"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"task_level": "invalid"}, "task_level"),
        ({"pooling_type": "median"}, "pooling_type"),
    ],
)
def test_invalid_readout_modes_fail_at_construction(
    kwargs: dict[str, str], message: str
) -> None:
    """Readout task and pooling behavior cannot remain implicit."""
    config = {"hidden_dim": 4, "out_channels": 2, "task_level": "graph"}
    config.update(kwargs)

    with pytest.raises(ValueError, match=message):
        NoReadOut(**config)


def test_graph_readout_requires_native_batch_tensor() -> None:
    """Graph pooling rejects a missing batch instead of consulting Data."""
    readout = NoReadOut(
        hidden_dim=4,
        out_channels=2,
        task_level="graph",
    )

    with pytest.raises(TypeError, match="model_out.*batch"):
        readout({"x": torch.randn(3, 4), "batch": None}, Data())
