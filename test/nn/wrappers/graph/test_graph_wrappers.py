"""Native homogeneous graph wrapper contract tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data

from topobench.nn.readouts import NoReadOut
from topobench.nn.wrappers.graph import (
    WRAPPER_CLASSES,
    GNNWrapper,
    GraphMLPWrapper,
)


def test_graph_wrapper_exports_are_explicit_and_narrow() -> None:
    """Only native graph adapters are registered from this package."""
    assert {
        "GNNWrapper": GNNWrapper,
        "GraphMLPWrapper": GraphMLPWrapper,
    } == WRAPPER_CLASSES
    assert GNNWrapper.__bases__ == (nn.Module,)
    assert GraphMLPWrapper.__bases__ == (nn.Module,)


class RecordingGNN(nn.Module):
    """Record the translated GNN call without depending on a real model."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[Tensor, Tensor, dict[str, object]]] = []

    def forward(
        self, x: Tensor, edge_index: Tensor, **kwargs: object
    ) -> Tensor:
        self.calls.append((x, edge_index, kwargs))
        return x + 1


class RecordingMLP(nn.Module):
    """Record the translated feature-only GraphMLP call."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[Tensor, dict[str, object]]] = []

    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        self.calls.append((x, kwargs))
        return x + 1


WrapperFactory = Callable[[nn.Module, str, str], nn.Module]


def _gnn(backbone: nn.Module, edge_attr: str, edge_weight: str) -> nn.Module:
    return GNNWrapper(
        backbone,
        edge_attr_mode=edge_attr,
        edge_weight_mode=edge_weight,
    )


def _mlp(backbone: nn.Module, edge_attr: str, edge_weight: str) -> nn.Module:
    return GraphMLPWrapper(
        backbone,
        edge_attr_mode=edge_attr,
        edge_weight_mode=edge_weight,
    )


WRAPPERS: tuple[tuple[str, WrapperFactory, type[nn.Module]], ...] = (
    ("gnn", _gnn, RecordingGNN),
    ("graph_mlp", _mlp, RecordingMLP),
)


def _graph_batch(*, regression: bool = False) -> Data:
    x = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    labels = (
        torch.tensor([[1.5], [-0.5]]) if regression else torch.tensor([0, 2])
    )
    return Data(
        x=x,
        edge_index=torch.tensor(
            [[0, 1, 2, 3, 4, 2], [1, 0, 3, 4, 3, 2]],
            dtype=torch.long,
        ),
        y=labels,
        batch=torch.tensor([0, 0, 0, 1, 1]),
    )


def _rank_one_regression_batch() -> Data:
    data = _graph_batch(regression=True)
    data.y = data.y.squeeze(1)
    return data


def _node_graph() -> Data:
    return Data(
        x=torch.randn(4, 3),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        y=torch.tensor([0, 1, 0, 2]),
    )


@pytest.mark.parametrize(("name", "factory", "backbone_type"), WRAPPERS)
@pytest.mark.parametrize(
    "data", [_graph_batch(), _graph_batch(regression=True), _node_graph()]
)
def test_native_wrappers_return_exact_contract(
    name: str,
    factory: WrapperFactory,
    backbone_type: type[nn.Module],
    data: Data,
) -> None:
    """Classification, regression, and node batches expose only native keys."""
    del name
    backbone = backbone_type()
    wrapper = factory(backbone, "ignore", "ignore")

    result = wrapper(data)

    assert set(result) == {"x", "labels", "batch"}
    assert torch.equal(result["x"], data.x + 1)
    assert result["labels"] is data.y
    assert result["batch"] is data.get("batch")


@pytest.mark.parametrize(("name", "factory", "backbone_type"), WRAPPERS)
@pytest.mark.parametrize(
    ("data", "task_level", "out_channels", "expected_shape"),
    [
        (_graph_batch(), "graph", 3, (2, 3)),
        (_graph_batch(regression=True), "graph", 1, (2, 1)),
        (_node_graph(), "node", 3, (4, 3)),
    ],
)
def test_wrapper_output_flows_directly_into_native_readout(
    name: str,
    factory: WrapperFactory,
    backbone_type: type[nn.Module],
    data: Data,
    task_level: str,
    out_channels: int,
    expected_shape: tuple[int, int],
) -> None:
    """Native adapters and readouts compose without rank aliases."""
    del name
    model_out = factory(
        backbone_type(),
        "ignore",
        "ignore",
    )(data)
    readout = NoReadOut(
        hidden_dim=3,
        out_channels=out_channels,
        task_level=task_level,
    )

    result = readout(model_out, data)

    assert result["logits"].shape == expected_shape


@pytest.mark.parametrize(
    "data",
    [
        _graph_batch(),
        _graph_batch(regression=True),
        _rank_one_regression_batch(),
        _node_graph(),
    ],
    ids=[
        "graph_classification",
        "graph_regression_column",
        "graph_regression_vector",
        "node_classification",
    ],
)
def test_gnn_forwards_supported_native_targets(data: Data) -> None:
    """Every supported target contract reaches the backbone unchanged."""
    backbone = RecordingGNN()
    result = GNNWrapper(
        backbone,
        edge_attr_mode="ignore",
        edge_weight_mode="ignore",
    )(data)

    assert len(backbone.calls) == 1
    assert result["labels"] is data.y


@pytest.mark.parametrize(
    ("data", "labels"),
    [
        (_graph_batch(), torch.ones(2, 2)),
        (_graph_batch(), torch.ones(2, 1, 1)),
        (_graph_batch(), torch.ones(2, 1, dtype=torch.long)),
        (_graph_batch(), torch.tensor([0, 1], dtype=torch.int32)),
        (_graph_batch(), torch.tensor([False, True])),
        (_graph_batch(), torch.ones(2, 1, dtype=torch.complex64)),
        (_graph_batch(), torch.tensor([0, 1, 2])),
        (_graph_batch(), torch.ones(3, 1)),
        (_node_graph(), torch.arange(4, dtype=torch.float32)),
        (_node_graph(), torch.ones(4, 1)),
        (_node_graph(), torch.ones(4, 1, dtype=torch.long)),
        (_node_graph(), torch.arange(4, dtype=torch.int32)),
        (_node_graph(), torch.tensor([0, 1, 2])),
    ],
    ids=[
        "graph_regression_width_two",
        "graph_regression_rank_three",
        "graph_integer_rank_two",
        "graph_integer_non_long",
        "graph_bool",
        "graph_complex",
        "classification_count_mismatch",
        "regression_count_mismatch",
        "node_float_rank_one",
        "node_float_rank_two",
        "node_integer_rank_two",
        "node_integer_non_long",
        "node_count_mismatch",
    ],
)
def test_gnn_rejects_malformed_targets_before_backbone(
    data: Data,
    labels: Tensor,
) -> None:
    """Unsupported target shape, dtype, task level, or count fails locally."""
    data.y = labels
    backbone = RecordingGNN()
    wrapper = GNNWrapper(
        backbone,
        edge_attr_mode="ignore",
        edge_weight_mode="ignore",
    )

    with pytest.raises((TypeError, ValueError), match=r"batch\.y"):
        wrapper(data)

    assert backbone.calls == []


@pytest.mark.parametrize(("name", "factory", "backbone_type"), WRAPPERS)
@pytest.mark.parametrize("field", ["edge_attr", "edge_weight"])
@pytest.mark.parametrize("mode", ["consume", "ignore", "reject"])
def test_optional_edge_field_modes_are_explicit(
    name: str,
    factory: WrapperFactory,
    backbone_type: type[nn.Module],
    field: str,
    mode: str,
) -> None:
    """Consume forwards, ignore omits, and reject fails before the call."""
    del name
    data = _graph_batch()
    value = (
        torch.randn(data.edge_index.size(1), 2)
        if field == "edge_attr"
        else torch.rand(data.edge_index.size(1))
    )
    data[field] = value
    modes = {"edge_attr": "ignore", "edge_weight": "ignore"}
    modes[field] = mode
    backbone = backbone_type()
    wrapper = factory(backbone, modes["edge_attr"], modes["edge_weight"])

    if mode == "reject":
        with pytest.raises(ValueError, match=rf"{field} is unsupported"):
            wrapper(data)
        assert backbone.calls == []
        return

    wrapper(data)
    kwargs = backbone.calls[0][-1]
    if mode == "consume":
        assert kwargs[field] is value
    else:
        assert field not in kwargs


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("edge_weight", torch.tensor(1.0)),
        ("edge_weight", torch.ones(6, 1)),
        ("edge_weight", torch.ones(5)),
        ("edge_weight", [1.0] * 6),
        ("edge_attr", torch.tensor(1.0)),
        ("edge_attr", torch.ones(5, 2)),
        ("edge_attr", [[1.0]] * 6),
    ],
    ids=[
        "edge_weight_scalar",
        "edge_weight_rank_two",
        "edge_weight_count",
        "edge_weight_non_tensor",
        "edge_attr_scalar",
        "edge_attr_count",
        "edge_attr_non_tensor",
    ],
)
def test_gnn_rejects_malformed_consumed_edges_before_backbone(
    field: str,
    value: object,
) -> None:
    """Only structurally valid consumed edge fields reach the backbone."""
    data = _graph_batch()
    data[field] = value
    modes = {"edge_attr": "ignore", "edge_weight": "ignore"}
    modes[field] = "consume"
    backbone = RecordingGNN()
    wrapper = GNNWrapper(
        backbone,
        edge_attr_mode=modes["edge_attr"],
        edge_weight_mode=modes["edge_weight"],
    )

    with pytest.raises((TypeError, ValueError), match=field):
        wrapper(data)

    assert backbone.calls == []


@pytest.mark.parametrize("field", ["edge_attr", "edge_weight"])
def test_gnn_ignores_malformed_ignored_edge_fields(field: str) -> None:
    """Ignore mode neither validates nor forwards optional edge fields."""
    data = _graph_batch()
    data[field] = torch.tensor(1.0)
    backbone = RecordingGNN()

    GNNWrapper(
        backbone,
        edge_attr_mode="ignore",
        edge_weight_mode="ignore",
    )(data)

    assert field not in backbone.calls[0][-1]


def test_gnn_consumes_rank_one_edge_attributes() -> None:
    """A scalar feature per edge satisfies the edge-attribute contract."""
    data = _graph_batch()
    data.edge_attr = torch.arange(data.edge_index.size(1))
    backbone = RecordingGNN()

    GNNWrapper(
        backbone,
        edge_attr_mode="consume",
        edge_weight_mode="ignore",
    )(data)

    assert backbone.calls[0][-1]["edge_attr"] is data.edge_attr


@pytest.mark.parametrize(("name", "factory", "backbone_type"), WRAPPERS)
@pytest.mark.parametrize("field", ["edge_attr", "edge_weight"])
def test_reject_mode_allows_absent_optional_field(
    name: str,
    factory: WrapperFactory,
    backbone_type: type[nn.Module],
    field: str,
) -> None:
    """Reject means reject a present field, not reject every batch."""
    del name
    modes = {"edge_attr": "ignore", "edge_weight": "ignore"}
    modes[field] = "reject"
    wrapper = factory(
        backbone_type(), modes["edge_attr"], modes["edge_weight"]
    )

    assert set(wrapper(_graph_batch())) == {"x", "labels", "batch"}


@pytest.mark.parametrize(("name", "factory", "backbone_type"), WRAPPERS)
@pytest.mark.parametrize("mode_name", ["edge_attr", "edge_weight"])
@pytest.mark.parametrize("invalid", [None, "unknown", "", 1])
def test_invalid_edge_modes_fail_during_construction(
    name: str,
    factory: WrapperFactory,
    backbone_type: type[nn.Module],
    mode_name: str,
    invalid: object,
) -> None:
    """No implicit optional-field policy reaches runtime."""
    del name
    modes: dict[str, object] = {"edge_attr": "ignore", "edge_weight": "ignore"}
    modes[mode_name] = invalid

    with pytest.raises((TypeError, ValueError), match=mode_name):
        factory(backbone_type(), modes["edge_attr"], modes["edge_weight"])  # type: ignore[arg-type]


def _without(data: Data, field: str) -> Data:
    clone = data.clone()
    del clone[field]
    return clone


def _invalid_cases() -> tuple[tuple[str, Data], ...]:
    base = _graph_batch()
    cases: list[tuple[str, Data]] = [
        ("x", _without(base, "x")),
        ("edge_index", _without(base, "edge_index")),
        ("y", _without(base, "y")),
        ("batch", _without(base, "batch")),
    ]
    invalid_values: tuple[tuple[str, object], ...] = (
        ("x", [1.0, 2.0]),
        ("x", torch.randn(5)),
        ("x", torch.ones(5, 3, dtype=torch.long)),
        ("edge_index", [[0, 1], [1, 0]]),
        ("edge_index", torch.tensor([0, 1])),
        ("edge_index", torch.tensor([[0.0], [1.0]])),
        ("edge_index", torch.tensor([[0], [5]])),
        ("y", [0, 1]),
        ("y", torch.tensor(1)),
        ("batch", [0, 0, 1, 1, 1]),
        ("batch", torch.tensor([[0, 0, 0, 1, 1]])),
        ("batch", torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])),
        ("batch", torch.tensor([0, 0, 1, 1])),
        ("batch", torch.tensor([0, 0, 0, -1, -1])),
        ("batch", torch.tensor([0, 0, 0, 2, 2])),
    )
    for field, value in invalid_values:
        data = base.clone()
        data[field] = value
        cases.append((field, data))
    return tuple(cases)


@pytest.mark.parametrize(("name", "factory", "backbone_type"), WRAPPERS)
@pytest.mark.parametrize(("field", "data"), _invalid_cases())
def test_invalid_native_fields_fail_before_backbone(
    name: str,
    factory: WrapperFactory,
    backbone_type: type[nn.Module],
    field: str,
    data: Data,
) -> None:
    """Malformed or incomplete native batches never enter the backbone."""
    del name
    backbone = backbone_type()
    wrapper = factory(backbone, "ignore", "ignore")

    with pytest.raises((TypeError, ValueError), match=field):
        wrapper(data)

    assert backbone.calls == []


@pytest.mark.parametrize(("name", "factory", "backbone_type"), WRAPPERS)
def test_missing_batch_rejects_ambiguous_batched_one_node_graphs(
    name: str,
    factory: WrapperFactory,
    backbone_type: type[nn.Module],
) -> None:
    """Native Batch boundaries prevent graph data from posing as node data."""
    del name
    data = Batch.from_data_list(
        [
            Data(
                x=torch.tensor([[float(label)]]),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                y=torch.tensor([label]),
            )
            for label in (0, 1)
        ]
    )
    del data.batch
    backbone = backbone_type()
    wrapper = factory(backbone, "ignore", "ignore")

    with pytest.raises(
        ValueError, match="batch.batch is required for graph-level targets"
    ):
        wrapper(data)

    assert backbone.calls == []


def test_gnn_translates_native_structural_arguments() -> None:
    """GNN receives x/edge_index positionally and native batch explicitly."""
    data = _graph_batch()
    backbone = RecordingGNN()

    GNNWrapper(
        backbone,
        edge_attr_mode="ignore",
        edge_weight_mode="ignore",
    )(data)

    x, edge_index, kwargs = backbone.calls[0]
    assert x is data.x
    assert edge_index is data.edge_index
    assert kwargs == {"batch": data.batch}


def test_graph_mlp_requires_exact_tensor_backbone_output() -> None:
    """GraphMLP returns only the native wrapper contract."""
    data = _node_graph()
    backbone = RecordingMLP()

    result = GraphMLPWrapper(
        backbone,
        edge_attr_mode="ignore",
        edge_weight_mode="ignore",
    )(data)

    assert set(result) == {"x", "labels", "batch"}
    called_x, kwargs = backbone.calls[0]
    assert called_x is data.x
    assert kwargs == {}


def test_graph_mlp_rejects_legacy_auxiliary_tuple() -> None:
    """The removed global distance side channel has no compatibility path."""

    class TupleBackbone(nn.Module):
        def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
            return x, x @ x.T

    with pytest.raises(TypeError, match="must return a tensor"):
        GraphMLPWrapper(
            TupleBackbone(),
            edge_attr_mode="ignore",
            edge_weight_mode="ignore",
        )(_node_graph())
