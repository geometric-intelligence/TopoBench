"""Contract tests for the reusable native heterogeneous HGT backbone."""

from __future__ import annotations

import io
import pickle
from collections.abc import Mapping

import pytest
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

import topobench.nn.backbones as backbone_registry
from topobench.nn.backbones.heterogeneous import HGTBackbone

HIDDEN_CHANNELS = 8
NODE_TYPES = ["author", "paper", "venue"]
EDGE_TYPES = [
    ("author", "writes", "paper"),
    ("paper", "rev_writes", "author"),
    ("paper", "published_in", "venue"),
    ("venue", "rev_published_in", "paper"),
]
EXPECTED_REGISTERED_BACKBONES = {
    "EDGNN",
    "GPSEncoder",
    "GraphMLP",
    "HGTBackbone",
    "HeteroSAGEBackbone",
    "NSDEncoder",
}


def make_data() -> HeteroData:
    """Build a small fully connected-by-type heterogeneous graph."""
    data = HeteroData()
    data["author"].x = torch.randn(5, HIDDEN_CHANNELS)
    data["paper"].x = torch.randn(4, HIDDEN_CHANNELS)
    data["venue"].x = torch.randn(2, HIDDEN_CHANNELS)
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
    num_layers: int = 2,
) -> HGTBackbone:
    """Construct the smallest useful reusable HGT."""
    return HGTBackbone(
        metadata=metadata or (NODE_TYPES, EDGE_TYPES),
        hidden_channels=HIDDEN_CHANNELS,
        num_layers=num_layers,
        heads=2,
        dropout=0.0,
        activation="relu",
    )


def test_top_level_registry_does_not_publish_metadata_infrastructure() -> None:
    """Registry exports canonical, pickle-stable trainable classes only."""
    assert (
        set(backbone_registry.MODEL_CLASSES) == EXPECTED_REGISTERED_BACKBONES
    )
    assert backbone_registry.HGTBackbone is HGTBackbone
    assert backbone_registry.MODEL_CLASSES["HGTBackbone"] is HGTBackbone
    assert HGTBackbone.__module__ == (
        "topobench.nn.backbones.heterogeneous.hgt"
    )
    assert not hasattr(backbone_registry, "HeterogeneousMetadataAdapter")
    model = make_model()
    restored = pickle.loads(pickle.dumps(model))
    assert restored.__class__ is HGTBackbone


def test_forward_returns_exact_external_node_keys_and_hidden_shapes() -> None:
    """The dictionary API preserves semantic keys and node counts."""
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


def test_every_hgt_layer_executes_and_receives_finite_gradients() -> None:
    """All configured layers participate in forward and backward."""
    data = make_data()
    model = make_model(num_layers=3)
    call_counts = [0, 0, 0]
    handles = []
    for index, conv in enumerate(model.convs):
        handles.append(
            conv.register_forward_hook(
                lambda _module, _args, _result, layer=index: (
                    call_counts.__setitem__(layer, call_counts[layer] + 1)
                )
            )
        )

    output = model(data.x_dict, data.edge_index_dict)
    sum(value.square().mean() for value in output.values()).backward()
    for handle in handles:
        handle.remove()

    assert call_counts == [1, 1, 1]
    gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("convs.")
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_checkpoint_is_relation_order_independent() -> None:
    """Positional HGT relation weights use canonical metadata ordering."""
    data = make_data()
    reference = make_model().eval()
    reversed_metadata = (NODE_TYPES, list(reversed(EDGE_TYPES)))
    restored = make_model(metadata=reversed_metadata).eval()

    restored.load_state_dict(reference.state_dict(), strict=True)
    expected = reference(data.x_dict, data.edge_index_dict)
    reversed_edges = dict(reversed(list(data.edge_index_dict.items())))
    actual = restored(data.x_dict, reversed_edges)

    assert restored.internal_metadata == reference.internal_metadata
    for node_type in NODE_TYPES:
        torch.testing.assert_close(actual[node_type], expected[node_type])


def test_forward_is_independent_of_edge_dictionary_iteration_order() -> None:
    """HGT receives present relations in canonical metadata order."""
    data = make_data()
    model = make_model().eval()

    expected = model(data.x_dict, data.edge_index_dict)
    reversed_edges = dict(reversed(list(data.edge_index_dict.items())))
    actual = model(data.x_dict, reversed_edges)

    for node_type in NODE_TYPES:
        assert torch.equal(actual[node_type], expected[node_type])


def test_omitted_relation_subset_uses_canonical_internal_order() -> None:
    """Filtering sampled relations never reintroduces mapping-order effects."""
    data = make_data()
    model = make_model().eval()
    subset_types = [EDGE_TYPES[3], EDGE_TYPES[0], EDGE_TYPES[2]]
    subset = {
        edge_type: data.edge_index_dict[edge_type]
        for edge_type in subset_types
    }

    internal = model.metadata_adapter.to_internal_edge_index_dict(subset)
    expected_order = [
        edge_type
        for edge_type in model.internal_metadata[1]
        if edge_type in internal
    ]

    assert list(internal) == expected_order
    assert set(model(data.x_dict, subset)) == set(NODE_TYPES)


def test_known_relations_may_be_omitted_from_a_sample() -> None:
    """A sampled subgraph may omit configured relations locally."""
    data = make_data()
    relations = dict(data.edge_index_dict)
    relations.pop(("paper", "published_in", "venue"))

    output = make_model()(data.x_dict, relations)

    assert set(output) == set(NODE_TYPES)


def test_sample_with_no_local_relations_carries_every_node_forward() -> None:
    """An edgeless sampled mini-batch remains a valid relation subset."""
    data = make_data()

    output = make_model(num_layers=3)(data.x_dict, {})

    for node_type in NODE_TYPES:
        torch.testing.assert_close(output[node_type], data[node_type].x)


def test_present_empty_relation_runs_hgt_instead_of_carry() -> None:
    """An explicit empty relation retains HGTConv's learned update semantics."""
    metadata = (["source", "target"], [("source", "to", "target")])
    x_dict = {
        "source": torch.randn(3, HIDDEN_CHANNELS),
        "target": torch.randn(2, HIDDEN_CHANNELS),
    }
    edge_type = metadata[1][0]
    empty_relation = torch.empty((2, 0), dtype=torch.long)
    model = make_model(metadata=metadata, num_layers=1).eval()

    omitted = model(x_dict, {})
    present = model(x_dict, {edge_type: empty_relation})

    torch.testing.assert_close(omitted["target"], x_dict["target"])
    assert not torch.equal(present["target"], omitted["target"])


def test_node_without_incoming_relation_is_carried_forward() -> None:
    """Missing HGT updates preserve the previous node representation."""
    metadata = (
        ["source", "destination", "isolated"],
        [("source", "to", "destination")],
    )
    x_dict = {
        "source": torch.randn(3, 8),
        "destination": torch.randn(2, 8),
        "isolated": torch.randn(4, 8),
    }
    edges = {("source", "to", "destination"): torch.tensor([[0, 1], [0, 1]])}

    output = make_model(metadata=metadata, num_layers=3)(x_dict, edges)

    torch.testing.assert_close(output["source"], x_dict["source"])
    torch.testing.assert_close(output["isolated"], x_dict["isolated"])


def test_neighbor_loader_sample_forwards_with_omitted_relations() -> None:
    """A real PyG neighbor sample satisfies the reusable dictionary API."""
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

    assert set(output) == set(NODE_TYPES)
    for node_type, features in output.items():
        assert features.shape == (
            sample[node_type].num_nodes,
            HIDDEN_CHANNELS,
        )


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (([], []), "node types.*not be empty"),
        ((["author", "author"], []), "node types.*unique"),
        ((["author"], []), "edge types.*not be empty"),
        (
            (
                ["author", "paper"],
                [
                    ("author", "writes", "paper"),
                    ("author", "writes", "paper"),
                ],
            ),
            "edge types.*unique",
        ),
        (
            (["author"], [("author", "writes", "paper")]),
            "endpoint.*paper",
        ),
    ],
)
def test_constructor_rejects_invalid_metadata(
    metadata: tuple[list[str], list[tuple[str, str, str]]],
    message: str,
) -> None:
    """Metadata is non-empty, canonical, unique, and closed over nodes."""
    with pytest.raises((TypeError, ValueError), match=message):
        make_model(metadata=metadata)


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    [
        ({"hidden_channels": True}, TypeError, "hidden_channels.*integer"),
        ({"hidden_channels": 0}, ValueError, "hidden_channels.*positive"),
        ({"num_layers": 0}, ValueError, "num_layers.*at least 1"),
        ({"heads": 0}, ValueError, "heads.*positive"),
        ({"heads": None}, TypeError, "heads.*integer"),
        (
            {"hidden_channels": 10, "heads": 4},
            ValueError,
            "divisible",
        ),
        ({"dropout": float("nan")}, ValueError, "dropout.*finite"),
        ({"dropout": -0.1}, ValueError, r"dropout.*\[0, 1\)"),
        ({"dropout": 1.0}, ValueError, r"dropout.*\[0, 1\)"),
    ],
)
def test_constructor_rejects_invalid_arguments(
    overrides: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    """Backbone hyperparameters fail before PyG module construction."""
    arguments: dict[str, object] = {
        "metadata": (NODE_TYPES, EDGE_TYPES),
        "hidden_channels": HIDDEN_CHANNELS,
        "num_layers": 2,
        "heads": 2,
        "dropout": 0.0,
        "activation": "relu",
    }
    arguments.update(overrides)
    with pytest.raises(error_type, match=message):
        HGTBackbone(**arguments)  # type: ignore[arg-type]


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
                "extra": torch.randn(2, 8),
            },
            {},
            ValueError,
            "node types.*unexpected.*extra",
        ),
        (
            {
                "author": torch.randn(2, 8),
                "paper": torch.randn(2, 8),
                "venue": torch.randn(2, 8),
                "extra": torch.randn(2, 8),
                7: torch.randn(2, 8),
            },
            {},
            ValueError,
            "node types.*unexpected.*7",
        ),
        (
            {
                "author": torch.randn(2, 7),
                "paper": torch.randn(2, 8),
                "venue": torch.randn(2, 8),
            },
            {},
            ValueError,
            "author.*width.*8.*7",
        ),
        (
            {
                "author": torch.randn(2, 8),
                "paper": torch.randn(2, 8),
                "venue": torch.randn(2, 8),
            },
            {
                ("paper", "unknown", "author"): torch.empty(
                    (2, 0), dtype=torch.long
                )
            },
            ValueError,
            "Unknown edge type.*unknown",
        ),
    ],
)
def test_forward_rejects_dictionary_contract_violations(
    x_dict: Mapping[str, torch.Tensor],
    edge_index_dict: Mapping[tuple[str, str, str], torch.Tensor],
    error_type: type[Exception],
    message: str,
) -> None:
    """Forward requires exact nodes and a known relation subset."""
    with pytest.raises(error_type, match=message):
        make_model()(x_dict, edge_index_dict)


@pytest.mark.parametrize(
    ("edge_index", "error_type", "message"),
    [
        (torch.zeros(3, 2, dtype=torch.long), ValueError, r"shape.*\[2, E\]"),
        (torch.zeros(2, 2), TypeError, "integer"),
        (
            torch.zeros(2, 2, dtype=torch.int32),
            TypeError,
            "torch.long",
        ),
        (torch.tensor([[-1], [0]]), ValueError, "source.*range"),
        (torch.tensor([[5], [0]]), ValueError, "source.*range"),
        (torch.tensor([[0], [4]]), ValueError, "destination.*range"),
    ],
)
def test_forward_rejects_invalid_edge_indices(
    edge_index: torch.Tensor,
    error_type: type[Exception],
    message: str,
) -> None:
    """Edge indices must be valid integer COO coordinates."""
    data = make_data()
    relation = ("author", "writes", "paper")
    with pytest.raises(error_type, match=message):
        make_model()(data.x_dict, {relation: edge_index})


def test_adversarial_external_names_are_losslessly_adapted() -> None:
    """PyG names unsafe for PyTorch and HGT joins remain fully supported."""
    node_types = ["paper.type", "items", "研究者", "a__b", "a", "d"]
    edge_types = [
        ("paper.type", "forward", "items"),
        ("items", "rel.ation", "研究者"),
        ("研究者", "state_dict", "paper.type"),
        ("a__b", "c", "d"),
        ("a", "b__c", "d"),
    ]
    assert "__".join(edge_types[-2]) == "__".join(edge_types[-1])
    x_dict = {
        node_type: torch.randn(index + 1, HIDDEN_CHANNELS)
        for index, node_type in enumerate(node_types)
    }
    edge_index_dict = {
        edge_type: torch.tensor([[0], [0]]) for edge_type in edge_types
    }

    model = make_model(metadata=(node_types, edge_types))
    output = model(x_dict, edge_index_dict)

    assert list(output) == node_types
    assert model.metadata == (node_types, edge_types)
    internal_nodes, internal_edges = model.internal_metadata
    assert len(set(internal_nodes)) == len(node_types)
    assert len({"__".join(edge_type) for edge_type in internal_edges}) == len(
        edge_types
    )
    checkpoint_keys = tuple(model.state_dict())
    assert all("paper.type" not in key for key in checkpoint_keys)
    assert all(".items." not in key for key in checkpoint_keys)
    for node_type, features in output.items():
        assert features.shape == x_dict[node_type].shape


def test_unknown_relation_diagnostic_uses_external_names() -> None:
    """Validation errors never leak opaque internal aliases."""
    data = make_data()
    unknown = ("paper", "mystery.relation", "venue")

    with pytest.raises(ValueError) as error:
        make_model()(data.x_dict, {unknown: torch.tensor([[0], [0]])})

    assert repr(unknown) in str(error.value)


def test_external_relation_may_equal_the_default_encoded_alias() -> None:
    """The internal alias namespace never excludes a valid external name."""
    unsafe_edge = ("a", "rel.ation", "b")
    payload = b"".join(
        len(encoded).to_bytes(8, "big") + encoded
        for part in unsafe_edge
        for encoded in (part.encode("utf-8"),)
    )
    alias_shaped_relation = f"tbE{payload.hex()}"
    metadata = (
        ["a", "b"],
        [unsafe_edge, ("a", alias_shaped_relation, "b")],
    )
    x_dict = {
        "a": torch.randn(2, HIDDEN_CHANNELS),
        "b": torch.randn(2, HIDDEN_CHANNELS),
    }
    edge_index_dict = {
        edge_type: torch.tensor([[0], [0]]) for edge_type in metadata[1]
    }

    output = make_model(metadata=metadata)(x_dict, edge_index_dict)

    assert set(output) == {"a", "b"}


def test_lone_surrogate_metadata_forward_and_checkpoint_roundtrip() -> None:
    """Alias encoding is total for arbitrary valid Python string metadata."""
    surrogate_node = "\ud800.node"
    surrogate_relation = "\udfff.rel"
    metadata = (
        [surrogate_node, "target"],
        [
            (surrogate_node, surrogate_relation, "target"),
            ("target", "returns", surrogate_node),
        ],
    )
    x_dict = {
        surrogate_node: torch.randn(2, HIDDEN_CHANNELS),
        "target": torch.randn(3, HIDDEN_CHANNELS),
    }
    edge_index_dict = {
        metadata[1][0]: torch.tensor([[0, 1], [0, 1]]),
        metadata[1][1]: torch.tensor([[0, 1], [0, 1]]),
    }
    model = make_model(metadata=metadata).eval()

    expected = model(x_dict, edge_index_dict)
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = make_model(metadata=metadata).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    actual = restored(x_dict, edge_index_dict)

    assert set(actual) == set(metadata[0])
    assert all(
        surrogate_node not in key and surrogate_relation not in key
        for key in model.state_dict()
    )
    for node_type in metadata[0]:
        torch.testing.assert_close(actual[node_type], expected[node_type])
