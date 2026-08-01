"""Tests for the self-contained native ``MLPReadout``."""

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from topobench.nn.readouts.mlp_readout import MLPReadout


def _readout(task_level: str, pooling_type: str = "sum") -> MLPReadout:
    return MLPReadout(
        in_channels=4,
        hidden_layers=[6, 5],
        out_channels=3,
        task_level=task_level,
        pooling_type=pooling_type,
        dropout=0.0,
        act="relu",
    )


def test_mlp_is_a_self_contained_module_with_owned_sequential() -> None:
    """The readout does not inherit the removable non-relational backbone."""
    readout = _readout("node")

    assert type(readout).__bases__ == (nn.Module,)
    assert isinstance(readout.mlp_layers, nn.Sequential)
    assert not any(
        module.__class__.__module__.startswith(
            "topobench.nn.backbones.non_relational"
        )
        for module in readout.modules()
    )


def test_node_readout_applies_mlp_without_pooling() -> None:
    """Node classification preserves the native node leading dimension."""
    x = torch.randn(7, 4)
    model_out = {
        "x": x,
        "batch": torch.tensor([0, 0, 0, 1, 1, 1, 1]),
        "labels": torch.randint(0, 3, (7,)),
        "sentinel": object(),
    }

    result = _readout("node")(model_out, Data())

    assert result is model_out
    assert result["x"].shape == (7, 3)
    assert result["logits"] is result["x"]
    assert "sentinel" in result


@pytest.mark.parametrize("pooling_type", ["sum", "mean", "max"])
def test_graph_readout_pools_using_model_output_batch(
    pooling_type: str,
) -> None:
    """Graph readout emits one row per graph for every supported reduction."""
    model_out = {
        "x": torch.randn(7, 4),
        "batch": torch.tensor([0, 0, 0, 1, 1, 2, 2]),
        "labels": torch.tensor([0, 1, 2]),
    }

    result = _readout("graph", pooling_type)(model_out, Data())

    assert result["x"].shape == (3, 3)
    assert result["logits"] is result["x"]


def test_graph_readout_uses_model_output_batch_over_data_batch() -> None:
    """Only model_out['batch'] drives graph pooling."""
    model_out = {
        "x": torch.randn(6, 4),
        "batch": torch.tensor([0, 0, 1, 1, 2, 2]),
    }
    data = Data(batch=torch.zeros(6, dtype=torch.long))

    result = _readout("graph")(model_out, data)

    assert result["logits"].shape == (3, 3)


def test_empty_hidden_layers_remain_supported() -> None:
    """A direct linear head is still an owned Sequential readout."""
    readout = MLPReadout(
        in_channels=4,
        hidden_layers=[],
        out_channels=2,
        task_level="graph",
        pooling_type="mean",
        dropout=0.0,
    )
    model_out = {
        "x": torch.randn(4, 4),
        "batch": torch.tensor([0, 0, 1, 1]),
    }

    result = readout(model_out, Data())

    assert isinstance(readout.mlp_layers, nn.Sequential)
    assert result["logits"].shape == (2, 2)


@pytest.mark.parametrize("task_level", ["node", "node_inductive"])
def test_all_node_task_levels_skip_pooling(task_level: str) -> None:
    """Inductive and transductive node heads share no-pooling semantics."""
    model_out = {"x": torch.randn(4, 4), "batch": None}

    result = _readout(task_level)(model_out, Data())

    assert result["logits"].shape == (4, 3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"task_level": "edge"}, "task_level"),
        ({"pooling_type": "median"}, "pooling_type"),
    ],
)
def test_invalid_modes_fail_at_construction(
    kwargs: dict[str, str], message: str
) -> None:
    """Task and pooling behavior are validated before forward."""
    config: dict[str, object] = {
        "in_channels": 4,
        "hidden_layers": 5,
        "out_channels": 2,
        "task_level": "graph",
    }
    config.update(kwargs)

    with pytest.raises(ValueError, match=message):
        MLPReadout(**config)


def test_graph_readout_requires_native_batch() -> None:
    """Missing graph membership fails instead of silently pooling all nodes."""
    with pytest.raises(TypeError, match="model_out.*batch"):
        _readout("graph")({"x": torch.randn(3, 4), "batch": None}, Data())
