"""Contract tests for the reusable heterogeneous GraphSAGE backbone."""

from __future__ import annotations

import io
import pickle
import warnings
from collections.abc import Mapping

import pytest
import torch
from torch.nn.parameter import UninitializedParameter
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, SAGEConv

import topobench.nn.backbones as backbone_registry
from topobench.nn.backbones.heterogeneous import HeteroSAGEBackbone

HIDDEN_CHANNELS = 8
NODE_TYPES = ["author", "paper", "venue"]
EDGE_TYPES = [
    ("author", "writes", "paper"),
    ("paper", "rev_writes", "author"),
    ("paper", "published_in", "venue"),
    ("venue", "rev_published_in", "paper"),
]


def make_data() -> HeteroData:
    """Build a small fully typed graph with active relations."""
    generator = torch.Generator().manual_seed(19)
    data = HeteroData()
    data["author"].x = torch.randn(5, HIDDEN_CHANNELS, generator=generator)
    data["paper"].x = torch.randn(4, HIDDEN_CHANNELS, generator=generator)
    data["venue"].x = torch.randn(2, HIDDEN_CHANNELS, generator=generator)
    data["author", "writes", "paper"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 0]]
    )
    data["paper", "rev_writes", "author"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 0], [0, 1, 2, 3, 4]]
    )
    data["paper", "published_in", "venue"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 0, 1]]
    )
    data["venue", "rev_published_in", "paper"].edge_index = torch.tensor(
        [[0, 1, 0, 1], [0, 1, 2, 3]]
    )
    return data


def make_model(
    *,
    metadata: tuple[list[str], list[tuple[str, str, str]]] | None = None,
    hidden_channels: int = HIDDEN_CHANNELS,
    num_layers: int = 2,
) -> HeteroSAGEBackbone:
    """Construct the smallest useful heterogeneous GraphSAGE."""
    return HeteroSAGEBackbone(
        metadata=metadata or (NODE_TYPES, EDGE_TYPES),
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=0.0,
        activation="relu",
    )


def test_registry_exports_canonical_pickle_stable_class() -> None:
    """Automatic discovery exposes only the canonical trainable class."""
    assert backbone_registry.HeteroSAGEBackbone is HeteroSAGEBackbone
    assert (
        backbone_registry.MODEL_CLASSES["HeteroSAGEBackbone"]
        is HeteroSAGEBackbone
    )
    assert HeteroSAGEBackbone.__module__ == (
        "topobench.nn.backbones.heterogeneous.heterosage"
    )

    restored = pickle.loads(pickle.dumps(make_model()))

    assert restored.__class__ is HeteroSAGEBackbone


def test_constructor_eagerly_builds_every_relation_for_every_layer() -> None:
    """Optimizer and checkpoint state are complete before first forward."""
    model = make_model(num_layers=3)
    initial_parameter_ids = {id(parameter) for parameter in model.parameters()}
    optimizer = torch.optim.Adam(model.parameters())

    assert len(model.convs) == 3
    assert initial_parameter_ids
    assert {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    } == initial_parameter_ids
    assert not any(
        isinstance(parameter, UninitializedParameter)
        for parameter in model.parameters()
    )
    for conv in model.convs:
        assert isinstance(conv, HeteroConv)
        assert conv.aggr == "sum"
        assert list(conv.convs.keys()) == model.internal_metadata[1]
        assert len(conv.convs) == len(EDGE_TYPES)
        assert all(
            isinstance(relation_conv, SAGEConv)
            for relation_conv in conv.convs.values()
        )
        assert all(
            relation_conv.in_channels == (HIDDEN_CHANNELS, HIDDEN_CHANNELS)
            and relation_conv.out_channels == HIDDEN_CHANNELS
            for relation_conv in conv.convs.values()
        )

    before_keys = tuple(model.state_dict())
    data = make_data()
    model(data.x_dict, data.edge_index_dict)

    assert tuple(model.state_dict()) == before_keys
    assert {id(parameter) for parameter in model.parameters()} == (
        initial_parameter_ids
    )


def test_forward_returns_complete_external_dictionary_without_mutation() -> (
    None
):
    """Full-batch input and output follow the same API as HGTBackbone."""
    data = make_data()
    original = {key: value.clone() for key, value in data.x_dict.items()}

    output = make_model()(data.x_dict, data.edge_index_dict)

    assert list(output) == NODE_TYPES
    assert {
        node_type: tuple(features.shape)
        for node_type, features in output.items()
    } == {"author": (5, 8), "paper": (4, 8), "venue": (2, 8)}
    for node_type in NODE_TYPES:
        torch.testing.assert_close(data[node_type].x, original[node_type])


def test_changing_one_relation_changes_only_its_layer_destination() -> None:
    """Relation-wise convolutions route messages to the declared target."""
    torch.manual_seed(7)
    metadata = (
        ["source", "middle", "destination"],
        [
            ("source", "to_middle", "middle"),
            ("middle", "to_destination", "destination"),
        ],
    )
    x_dict = {
        "source": torch.randn(4, HIDDEN_CHANNELS),
        "middle": torch.randn(3, HIDDEN_CHANNELS),
        "destination": torch.randn(2, HIDDEN_CHANNELS),
    }
    edge_index_dict = {
        metadata[1][0]: torch.tensor([[0, 1, 2], [0, 1, 2]]),
        metadata[1][1]: torch.tensor([[0, 1, 2], [0, 1, 0]]),
    }
    perturbed = dict(edge_index_dict)
    perturbed[metadata[1][0]] = torch.tensor([[3, 3, 3], [0, 1, 2]])
    model = make_model(metadata=metadata, num_layers=1).eval()

    reference = model(x_dict, edge_index_dict)
    changed = model(x_dict, perturbed)

    torch.testing.assert_close(changed["source"], reference["source"])
    torch.testing.assert_close(
        changed["destination"], reference["destination"]
    )
    assert not torch.equal(changed["middle"], reference["middle"])


def test_all_relation_specific_convolutions_receive_gradients() -> None:
    """Every eager relation module in every layer participates in backward."""
    data = make_data()
    model = make_model(num_layers=3)

    output = model(data.x_dict, data.edge_index_dict)
    sum(value.square().mean() for value in output.values()).backward()

    relation_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("convs.")
    ]
    assert relation_parameters
    assert all(parameter.grad is not None for parameter in relation_parameters)
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in relation_parameters
        if parameter.grad is not None
    )


def test_source_only_and_isolated_nodes_are_carried_through_all_layers() -> (
    None
):
    """Types without an update retain their previous representation."""
    metadata = (
        ["source", "destination", "isolated"],
        [("source", "to", "destination")],
    )
    x_dict = {
        "source": torch.randn(3, HIDDEN_CHANNELS),
        "destination": torch.randn(2, HIDDEN_CHANNELS),
        "isolated": torch.randn(4, HIDDEN_CHANNELS),
    }
    edges = {metadata[1][0]: torch.tensor([[0, 1, 2], [0, 1, 1]])}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = make_model(metadata=metadata, num_layers=3)(x_dict, edges)

    torch.testing.assert_close(output["source"], x_dict["source"])
    torch.testing.assert_close(output["isolated"], x_dict["isolated"])
    assert not any(
        "representations do not get updated" in str(item.message)
        for item in caught
    )


def test_omitted_and_present_empty_relations_have_distinct_semantics() -> None:
    """Omission carries a type while an explicit empty relation updates it."""
    metadata = (["source", "target"], [("source", "to", "target")])
    x_dict = {
        "source": torch.randn(3, HIDDEN_CHANNELS),
        "target": torch.randn(2, HIDDEN_CHANNELS),
    }
    edge_type = metadata[1][0]
    model = make_model(metadata=metadata, num_layers=1).eval()

    omitted = model(x_dict, {})
    present = model(
        x_dict,
        {edge_type: torch.empty((2, 0), dtype=torch.long)},
    )

    torch.testing.assert_close(omitted["target"], x_dict["target"])
    assert not torch.equal(present["target"], omitted["target"])


def test_neighbor_loader_sample_forwards_with_relation_subset() -> None:
    """A real sampled mini-batch satisfies the dictionary contract."""
    data = make_data()
    loader = NeighborLoader(
        data,
        input_nodes=("author", torch.tensor([0, 1])),
        num_neighbors=[-1],
        batch_size=2,
        shuffle=False,
    )
    try:
        sample = next(iter(loader))
    except ImportError as error:
        pytest.fail(f"PyG neighbor sampling backend is unavailable: {error}")

    output = make_model()(sample.x_dict, sample.edge_index_dict)

    assert list(output) == NODE_TYPES
    for node_type, features in output.items():
        assert features.shape == (
            sample[node_type].num_nodes,
            HIDDEN_CHANNELS,
        )


def test_adversarial_metadata_is_lossless_and_checkpoint_order_independent() -> (
    None
):
    """Internal PyG keys never expose its lossy tuple-key conversion."""
    surrogate_node = "\ud800.node"
    node_types = [
        "paper.type",
        "items",
        "研究者",
        surrogate_node,
        "a___b",
        "a",
        "d",
    ]
    edge_types = [
        ("paper.type", "forward", "items"),
        ("items", "rel.ation", "研究者"),
        ("研究者", "state_dict", "paper.type"),
        (surrogate_node, "returns", "items"),
        ("a___b", "c", "d"),
        ("a", "b___c", "d"),
    ]
    assert "___".join(edge_types[-2]) == "___".join(edge_types[-1])
    metadata = (node_types, edge_types)
    generator = torch.Generator().manual_seed(31)
    x_dict = {
        node_type: torch.randn(
            index + 1,
            HIDDEN_CHANNELS,
            generator=generator,
        )
        for index, node_type in enumerate(node_types)
    }
    edge_index_dict = {
        edge_type: torch.tensor([[0], [0]]) for edge_type in edge_types
    }
    reference = make_model(metadata=metadata).eval()
    reversed_model = make_model(
        metadata=(node_types, list(reversed(edge_types)))
    ).eval()

    expected = reference(x_dict, edge_index_dict)
    reversed_model.load_state_dict(reference.state_dict(), strict=True)
    actual = reversed_model(
        x_dict, dict(reversed(list(edge_index_dict.items())))
    )

    assert list(expected) == node_types
    assert reversed_model.internal_metadata == reference.internal_metadata
    assert len(reference.convs[0].convs) == len(edge_types)
    internal_subset = reference.metadata_adapter.to_internal_edge_index_dict(
        {
            edge_types[4]: edge_index_dict[edge_types[4]],
            edge_types[0]: edge_index_dict[edge_types[0]],
            edge_types[2]: edge_index_dict[edge_types[2]],
        }
    )
    assert list(internal_subset) == [
        edge_type
        for edge_type in reference.internal_metadata[1]
        if edge_type in internal_subset
    ]
    for node_type in node_types:
        torch.testing.assert_close(actual[node_type], expected[node_type])

    checkpoint = io.BytesIO()
    torch.save(reference.state_dict(), checkpoint)
    checkpoint.seek(0)
    roundtrip = pickle.loads(
        pickle.dumps(make_model(metadata=metadata))
    ).eval()
    roundtrip.load_state_dict(torch.load(checkpoint, weights_only=True))
    restored = roundtrip(x_dict, edge_index_dict)
    for node_type in node_types:
        torch.testing.assert_close(restored[node_type], expected[node_type])


def test_hash_names_are_encoded_before_pyg_tuple_key_conversion() -> None:
    """PyG's lossy ``#``-to-dot decoding never changes semantic keys."""
    metadata = (
        ["hash#source", "target"],
        [("hash#source", "hash#relation", "target")],
    )
    x_dict = {
        "hash#source": torch.randn(3, HIDDEN_CHANNELS),
        "target": torch.randn(2, HIDDEN_CHANNELS),
    }
    edge_index_dict = {metadata[1][0]: torch.tensor([[0, 1, 2], [0, 1, 1]])}
    model = make_model(metadata=metadata, num_layers=1)

    output = model(x_dict, edge_index_dict)
    output["target"].square().mean().backward()

    internal_nodes, internal_edges = model.internal_metadata
    assert all("#" not in node_type for node_type in internal_nodes)
    assert all(
        "#" not in component
        for edge_type in internal_edges
        for component in edge_type
    )
    assert all(
        parameter.grad is not None for parameter in model.convs[0].parameters()
    )


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (([], []), "node types.*not be empty"),
        ((["author", "author"], []), "node types.*unique"),
        ((["author"], []), "edge types.*not be empty"),
        (
            (["author"], [("author", "writes", "paper")]),
            "endpoint.*paper",
        ),
    ],
)
def test_constructor_reuses_shared_metadata_validation(
    metadata: tuple[list[str], list[tuple[str, str, str]]],
    message: str,
) -> None:
    """SAGE rejects the same invalid metadata contract as HGT."""
    with pytest.raises((TypeError, ValueError), match=message):
        make_model(metadata=metadata)


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    [
        ({"hidden_channels": True}, TypeError, "hidden_channels.*integer"),
        ({"hidden_channels": 0}, ValueError, "hidden_channels.*positive"),
        ({"num_layers": 0}, ValueError, "num_layers.*at least 1"),
        ({"dropout": float("nan")}, ValueError, "dropout.*finite"),
        ({"dropout": 1.0}, ValueError, r"dropout.*\[0, 1\)"),
    ],
)
def test_constructor_reuses_shared_scalar_validation(
    overrides: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    """Heads-free validation keeps every other backbone invariant."""
    arguments: dict[str, object] = {
        "metadata": (NODE_TYPES, EDGE_TYPES),
        "hidden_channels": HIDDEN_CHANNELS,
        "num_layers": 2,
        "dropout": 0.0,
        "activation": "relu",
    }
    arguments.update(overrides)

    with pytest.raises(error_type, match=message):
        HeteroSAGEBackbone(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("x_dict", "edge_index_dict", "error_type", "message"),
    [
        (
            {"author": torch.randn(2, 8), "paper": torch.randn(2, 8)},
            {},
            ValueError,
            "node types.*missing.*venue",
        ),
        (
            {
                "author": torch.randn(2, 8),
                "paper": torch.randn(2, 8),
                "venue": torch.randn(2, 8),
            },
            {
                ("paper", "unknown", "venue"): torch.empty(
                    (2, 0), dtype=torch.long
                )
            },
            ValueError,
            "Unknown edge type.*unknown",
        ),
        (
            {
                "author": torch.randn(2, 8),
                "paper": torch.randn(2, 8),
                "venue": torch.randn(2, 8),
            },
            {
                ("author", "writes", "paper"): torch.zeros(
                    3, 2, dtype=torch.long
                )
            },
            ValueError,
            r"shape.*\[2, E\]",
        ),
        (
            {
                "author": torch.randn(2, 8),
                "paper": torch.randn(2, 8),
                "venue": torch.randn(2, 8),
            },
            {("author", "writes", "paper"): torch.tensor([[2], [0]])},
            ValueError,
            "source.*range",
        ),
    ],
)
def test_forward_reuses_shared_dictionary_validation(
    x_dict: Mapping[str, torch.Tensor],
    edge_index_dict: Mapping[tuple[str, str, str], torch.Tensor],
    error_type: type[Exception],
    message: str,
) -> None:
    """Forward diagnostics retain external metadata names."""
    with pytest.raises(error_type, match=message):
        make_model()(x_dict, edge_index_dict)


def test_double_precision_and_cpu_autocast_forward() -> None:
    """The eager backbone respects standard module dtype/device mechanics."""
    data = make_data()
    double_model = make_model().double().eval()
    double_output = double_model(
        {key: value.double() for key, value in data.x_dict.items()},
        data.edge_index_dict,
    )

    assert all(
        value.dtype == torch.float64 for value in double_output.values()
    )

    autocast_model = make_model().eval()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast_output = autocast_model(
            data.x_dict,
            data.edge_index_dict,
        )

    assert set(autocast_output) == set(NODE_TYPES)
    assert all(
        torch.isfinite(value).all() for value in autocast_output.values()
    )
