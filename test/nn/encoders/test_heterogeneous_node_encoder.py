"""Tests for deterministic per-node-type heterogeneous feature encoding."""

from __future__ import annotations

import copy
import math
import pickle
import subprocess
import sys
from collections import OrderedDict

import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.data import Data, HeteroData

import topobench.nn.encoders as encoder_registry
from topobench.data.datasets import make_synthetic_heterogeneous_data
from topobench.nn import make_activation
from topobench.nn.encoders import HeterogeneousNodeFeatureEncoder
from topobench.nn.encoders.heterogeneous_node_encoder import (
    HeterogeneousNodeFeatureEncoder as CanonicalEncoder,
)
from topobench.transforms.data_manipulations.heterogeneous import (
    HeterogeneousConstantFeatures,
    HeterogeneousToUndirected,
)

INPUT_CHANNELS = OrderedDict((("author", 8), ("paper", 5), ("venue", 1)))
NODE_COUNTS = {"author": 36, "paper": 24, "venue": 6}
PYTORCH_RESERVED_NODE_TYPES = OrderedDict(
    (
        ("paper.type", 2),
        ("training", 3),
        ("items", 4),
        ("forward", 5),
        ("state_dict", 6),
    )
)


def make_data() -> HeteroData:
    """Return a fresh canonical, fully transformed synthetic graph."""
    data = make_synthetic_heterogeneous_data(seed=7)
    data = HeterogeneousConstantFeatures(node_types="venue")(data)
    return HeterogeneousToUndirected(merge=False)(data)


def make_encoder(
    *,
    hidden_channels: int = 16,
    activation: str = "relu",
    dropout: float = 0.0,
) -> HeterogeneousNodeFeatureEncoder:
    """Return an encoder matching the canonical synthetic graph."""
    return HeterogeneousNodeFeatureEncoder(
        input_channels=INPUT_CHANNELS,
        hidden_channels=hidden_channels,
        activation=activation,
        dropout=dropout,
    )


def make_reserved_name_data() -> HeteroData:
    """Return feature stores whose valid PyG names are unsafe module keys."""
    data = HeteroData()
    for index, (node_type, width) in enumerate(
        PYTORCH_RESERVED_NODE_TYPES.items(),
        start=1,
    ):
        data[node_type].x = torch.randn(index + 1, width)
    return data


@pytest.mark.parametrize(
    ("name", "module_type"),
    [
        ("relu", nn.ReLU),
        ("elu", nn.ELU),
        ("tanh", nn.Tanh),
        ("gelu", nn.GELU),
        ("id", nn.Identity),
    ],
)
def test_make_activation_supports_exact_legacy_names(
    name: str,
    module_type: type[nn.Module],
) -> None:
    """Each supported name creates a fresh module of the expected type."""
    first = make_activation(name)
    second = make_activation(name)

    assert type(first) is module_type
    assert type(second) is module_type
    assert first is not second


@pytest.mark.parametrize("name", [None, 1, True, object()])
def test_make_activation_rejects_non_strings(name: object) -> None:
    """Activation selection has a clear string-only contract."""
    with pytest.raises(TypeError, match="activation name.*string"):
        make_activation(name)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["ReLU", "identity", "", "swish"])
def test_make_activation_rejects_unknown_names(name: str) -> None:
    """Unsupported names retain CellHGT's exact legacy error text."""
    with pytest.raises(ValueError) as error:
        make_activation(name)
    assert str(error.value) == f"Unsupported activation: {name}"


def test_identity_activation_preserves_values() -> None:
    """The identity helper is numerically an identity operation."""
    values = torch.tensor([[-2.0, 0.0, 3.5]])
    assert torch.equal(make_activation("id")(values), values)


@pytest.mark.parametrize(
    ("input_channels", "error_type", "message"),
    [
        ({}, ValueError, "input_channels.*not be empty"),
        ([], TypeError, "input_channels.*mapping"),
        ("author", TypeError, "input_channels.*mapping"),
        ({1: 8}, TypeError, "node type.*string"),
        ({"": 8}, ValueError, "node type.*non-empty"),
        ({"author": True}, TypeError, "width.*integer"),
        ({"author": 1.5}, TypeError, "width.*integer"),
        ({"author": 0}, ValueError, "width.*positive"),
        ({"author": -1}, ValueError, "width.*positive"),
    ],
)
def test_constructor_validates_input_channel_mapping(
    input_channels: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Input metadata is complete, ordered, and strictly typed."""
    with pytest.raises(error_type, match=message):
        HeterogeneousNodeFeatureEncoder(
            input_channels=input_channels,  # type: ignore[arg-type]
            hidden_channels=8,
        )


@pytest.mark.parametrize(
    ("hidden_channels", "error_type", "message"),
    [
        (True, TypeError, "hidden_channels.*integer"),
        (1.5, TypeError, "hidden_channels.*integer"),
        (0, ValueError, "hidden_channels.*positive"),
        (-2, ValueError, "hidden_channels.*positive"),
    ],
)
def test_constructor_validates_hidden_channels(
    hidden_channels: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """The common output width is a non-boolean positive integer."""
    with pytest.raises(error_type, match=message):
        HeterogeneousNodeFeatureEncoder(
            input_channels={"author": 8},
            hidden_channels=hidden_channels,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("dropout", "error_type", "message"),
    [
        (True, TypeError, "dropout.*real"),
        ("0.1", TypeError, "dropout.*real"),
        (float("nan"), ValueError, "dropout.*finite"),
        (float("inf"), ValueError, "dropout.*finite"),
        (-0.1, ValueError, r"dropout.*\[0, 1\)"),
        (1.0, ValueError, r"dropout.*\[0, 1\)"),
    ],
)
def test_constructor_validates_dropout(
    dropout: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Dropout is finite and lies in the probability interval."""
    with pytest.raises(error_type, match=message):
        HeterogeneousNodeFeatureEncoder(
            input_channels={"author": 8},
            hidden_channels=4,
            dropout=dropout,  # type: ignore[arg-type]
        )


def test_constructor_copies_and_normalizes_ordered_metadata() -> None:
    """Caller mutation cannot alter deterministic projection construction."""
    source = OrderedDict((("paper", np.int64(5)), ("author", np.int32(8))))
    encoder = HeterogeneousNodeFeatureEncoder(
        input_channels=source,
        hidden_channels=np.int64(12),
        dropout=np.float64(0.25),
    )
    source["paper"] = np.int64(999)
    source["venue"] = np.int64(1)

    assert encoder.input_channels == {"paper": 5, "author": 8}
    assert tuple(encoder.input_channels) == ("paper", "author")
    assert tuple(encoder.projections) == ("paper", "author")
    assert type(encoder.hidden_channels) is int
    assert encoder.hidden_channels == 12
    assert math.isclose(encoder.dropout.p, 0.25)
    assert {
        node_type: (projection.in_features, projection.out_features)
        for node_type, projection in encoder.projections.items()
    } == {"paper": (5, 12), "author": (8, 12)}


def test_input_channel_property_is_a_defensive_metadata_copy() -> None:
    """Public mapping mutation cannot alter immutable internal metadata."""
    encoder = make_encoder()
    first = encoder.input_channels
    second = encoder.input_channels
    first["author"] = 999
    first["unexpected"] = 4

    assert first is not second
    assert second == dict(INPUT_CHANNELS)
    assert encoder.input_channels == dict(INPUT_CHANNELS)
    assert encoder.projection_for("author").in_features == 8


def test_valid_pyg_node_names_unsafe_for_moduledict_are_supported() -> None:
    """External node names never become raw PyTorch child-module names."""
    data = make_reserved_name_data()
    original_features = {}
    for node_type in data.node_types:
        data[node_type].x.requires_grad_(True)
        original_features[node_type] = data[node_type].x
    encoder = HeterogeneousNodeFeatureEncoder(
        input_channels=PYTORCH_RESERVED_NODE_TYPES,
        hidden_channels=7,
        activation="elu",
    )
    names_before = tuple(encoder.named_parameters())
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.1)

    result = encoder(data)
    sum(
        features.square().sum() for features in result.x_dict.values()
    ).backward()

    assert result is data
    assert tuple(result.x_dict) == tuple(PYTORCH_RESERVED_NODE_TYPES)
    assert {
        node_type: tuple(features.shape)
        for node_type, features in result.x_dict.items()
    } == {
        node_type: (index + 1, 7)
        for index, node_type in enumerate(
            PYTORCH_RESERVED_NODE_TYPES,
            start=1,
        )
    }
    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimizer_parameters == {
        id(parameter) for _, parameter in names_before
    }
    assert tuple(encoder.named_parameters()) == names_before
    for node_type in PYTORCH_RESERVED_NODE_TYPES:
        projection = encoder.projection_for(node_type)
        assert projection.weight.grad is not None
        assert projection.bias.grad is not None
        assert original_features[node_type].grad is not None


def test_state_dict_uses_collision_free_name_derived_module_keys() -> None:
    """Checkpoint keys safely and deterministically encode semantic names."""
    encoder = HeterogeneousNodeFeatureEncoder(
        input_channels=PYTORCH_RESERVED_NODE_TYPES,
        hidden_channels=7,
    )

    assert set(encoder.state_dict()) == {
        f"_projections.node_type_{node_type.encode('utf-8').hex()}.{suffix}"
        for node_type in PYTORCH_RESERVED_NODE_TYPES
        for suffix in ("weight", "bias")
    }


def test_state_dict_strictly_loads_across_equivalent_metadata_order() -> None:
    """Checkpoint parameter names are independent of metadata insertion order."""
    first = HeterogeneousNodeFeatureEncoder(
        input_channels=PYTORCH_RESERVED_NODE_TYPES,
        hidden_channels=7,
    )
    reversed_channels = OrderedDict(
        reversed(tuple(PYTORCH_RESERVED_NODE_TYPES.items()))
    )
    second = HeterogeneousNodeFeatureEncoder(
        input_channels=reversed_channels,
        hidden_channels=7,
    )

    load_result = second.load_state_dict(first.state_dict(), strict=True)

    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
    for node_type in PYTORCH_RESERVED_NODE_TYPES:
        first_projection = first.projection_for(node_type)
        second_projection = second.projection_for(node_type)
        assert torch.equal(
            first_projection.weight,
            second_projection.weight,
        )
        assert torch.equal(first_projection.bias, second_projection.bias)


def test_projection_lookup_reports_unknown_external_node_type() -> None:
    """Semantic projection lookup retains external names in diagnostics."""
    with pytest.raises(KeyError, match="Unknown node type.*institution"):
        make_encoder().projection_for("institution")


def test_forward_encodes_all_node_types_to_common_width_in_place() -> None:
    """Different feature widths map to one hidden width on the same graph."""
    data = make_data()
    encoder = make_encoder()
    original_node_types = tuple(data.node_types)
    original_edge_types = tuple(data.edge_types)
    original_x_keys = tuple(data.x_dict)

    result = encoder(data)

    assert result is data
    assert tuple(result.node_types) == original_node_types
    assert tuple(result.edge_types) == original_edge_types
    assert tuple(result.x_dict) == original_x_keys
    assert {
        key: tuple(value.shape) for key, value in result.x_dict.items()
    } == {
        "author": (36, 16),
        "paper": (24, 16),
        "venue": (6, 16),
    }


def test_forward_changes_only_node_features() -> None:
    """Labels, masks, relations, globals, and store order remain unchanged."""
    data = make_data()
    data.audit_marker = {"name": "synthetic", "fold": 0}
    snapshot = copy.deepcopy(data)

    make_encoder()(data)

    assert data.node_types == snapshot.node_types
    assert data.edge_types == snapshot.edge_types
    assert data.audit_marker == snapshot.audit_marker
    for node_type in data.node_types:
        assert tuple(data[node_type]) == tuple(snapshot[node_type])
        for key, expected in snapshot[node_type].items():
            if key != "x":
                actual = data[node_type][key]
                if isinstance(expected, torch.Tensor):
                    assert torch.equal(actual, expected)
                else:
                    assert actual == expected
    for edge_type in data.edge_types:
        assert tuple(data[edge_type]) == tuple(snapshot[edge_type])
        for key, expected in snapshot[edge_type].items():
            assert torch.equal(data[edge_type][key], expected)


def test_every_metadata_type_has_exactly_one_prebuilt_projection() -> None:
    """Projection topology is complete before the first batch."""
    encoder = make_encoder()

    assert tuple(encoder.projections) == tuple(INPUT_CHANNELS)
    assert len(encoder.projections) == len(INPUT_CHANNELS)
    for node_type, input_width in INPUT_CHANNELS.items():
        projection = encoder.projections[node_type]
        assert projection.in_features == input_width
        assert projection.out_features == 16


def test_forward_rejects_missing_feature_type_by_name() -> None:
    """A store without features is reported as missing metadata."""
    data = make_data()
    del data["venue"].x

    with pytest.raises(ValueError, match=r"missing=.*venue"):
        make_encoder()(data)


def test_forward_rejects_unexpected_feature_type_by_name() -> None:
    """Feature-bearing stores absent from metadata are rejected."""
    data = make_data()
    data["institution"].x = torch.ones(2, 3)

    with pytest.raises(ValueError, match=r"unexpected=.*institution"):
        make_encoder()(data)


def test_forward_rejects_wrong_input_family() -> None:
    """Homogeneous PyG data cannot silently enter this encoder."""
    with pytest.raises(TypeError, match="requires.*HeteroData"):
        make_encoder()(Data(x=torch.ones(2, 8)))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("node_type", "features", "error_type", "message"),
    [
        ("paper", [1.0, 2.0], TypeError, "paper.*torch.Tensor"),
        ("paper", torch.ones(24, 5, 1), ValueError, "paper.*rank 2"),
        ("paper", torch.ones(23, 5), ValueError, "paper.*24.*23"),
        ("paper", torch.ones(24, 4), ValueError, "paper.*width 5.*4"),
        (
            "paper",
            torch.ones(24, 5, dtype=torch.long),
            TypeError,
            "paper.*floating",
        ),
    ],
)
def test_forward_validates_each_feature_tensor(
    node_type: str,
    features: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Feature tensors receive node-specific structural diagnostics."""
    data = make_data()
    data[node_type].num_nodes = NODE_COUNTS[node_type]
    data[node_type].x = features

    with pytest.raises(error_type, match=message):
        make_encoder()(data)


def test_gradients_reach_every_projection_and_original_feature_tensor() -> (
    None
):
    """Encoding preserves autograd connectivity without detach or copies."""
    data = make_data()
    original_features = {}
    for node_type in data.node_types:
        data[node_type].x.requires_grad_(True)
        original_features[node_type] = data[node_type].x
    encoder = make_encoder(activation="elu")

    encoded = encoder(data)
    sum(value.square().sum() for value in encoded.x_dict.values()).backward()

    for node_type, projection in encoder.projections.items():
        assert projection.weight.grad is not None, node_type
        assert projection.bias.grad is not None, node_type
        assert bool(projection.weight.grad.abs().sum()), node_type
        assert original_features[node_type].grad is not None, node_type
        assert bool(original_features[node_type].grad.abs().sum()), node_type


def test_all_parameters_exist_before_forward_and_optimizer_sees_them() -> None:
    """Forward never creates or replaces trainable state."""
    encoder = make_encoder()
    names_before = tuple(encoder.named_parameters())
    identities_before = tuple(id(parameter) for _, parameter in names_before)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    encoder(make_data())
    encoder(make_data())

    names_after = tuple(encoder.named_parameters())
    assert tuple(name for name, _ in names_after) == tuple(
        name for name, _ in names_before
    )
    assert tuple(id(parameter) for _, parameter in names_after) == (
        identities_before
    )
    assert optimizer_ids == set(identities_before)


def test_dropout_obeys_train_and_eval_semantics() -> None:
    """Dropout is stochastic only while the encoder is training."""
    data = make_data()
    encoder = make_encoder(hidden_channels=32, activation="id", dropout=0.75)
    encoder.train()
    torch.manual_seed(1)
    first_train = {
        key: value.clone() for key, value in encoder(data).x_dict.items()
    }
    torch.manual_seed(2)
    second_train = {
        key: value.clone()
        for key, value in encoder(make_data()).x_dict.items()
    }
    assert any(
        not torch.equal(first_train[key], second_train[key])
        for key in first_train
    )

    encoder.eval()
    first_eval = {
        key: value.clone()
        for key, value in encoder(make_data()).x_dict.items()
    }
    second_eval = {
        key: value.clone()
        for key, value in encoder(make_data()).x_dict.items()
    }
    assert all(
        torch.equal(first_eval[key], second_eval[key]) for key in first_eval
    )


def test_configured_activation_is_applied() -> None:
    """A ReLU encoder cannot emit negative encoded features."""
    encoder = make_encoder(activation="relu")
    encoder.eval()

    assert all(
        bool((features >= 0).all())
        for features in encoder(make_data()).x_dict.values()
    )


def test_preflight_validation_is_transactional() -> None:
    """A later invalid store leaves every earlier feature object untouched."""
    data = make_data()
    original = {key: value for key, value in data.x_dict.items()}
    data["venue"].x = torch.ones(6, 2)

    with pytest.raises(ValueError, match="venue.*width"):
        make_encoder()(data)

    assert all(
        data[key].x is original[key] for key in original if key != "venue"
    )
    assert data["venue"].x.shape == (6, 2)


def test_projection_failure_is_transactional() -> None:
    """A failed later projection cannot partially commit earlier outputs."""
    data = make_data()
    original = {key: value for key, value in data.x_dict.items()}
    encoder = make_encoder()

    def fail_projection(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        raise RuntimeError("injected projection failure")

    handle = encoder.projections["paper"].register_forward_hook(
        fail_projection
    )
    try:
        with pytest.raises(RuntimeError, match="injected projection failure"):
            encoder(data)
    finally:
        handle.remove()

    assert all(data[key].x is original[key] for key in original)


def test_public_registry_uses_one_canonical_pickleable_class() -> None:
    """Public import, registry lookup, and module import share one identity."""
    assert HeterogeneousNodeFeatureEncoder is CanonicalEncoder
    assert (
        encoder_registry.FEATURE_ENCODERS["HeterogeneousNodeFeatureEncoder"]
        is CanonicalEncoder
    )
    assert CanonicalEncoder.__module__ == (
        "topobench.nn.encoders.heterogeneous_node_encoder"
    )
    restored = pickle.loads(pickle.dumps(make_encoder()))
    assert restored.__class__ is CanonicalEncoder


def test_clean_process_registry_and_pickle_identity_are_stable() -> None:
    """Canonical identity does not depend on prior package import order."""
    script = """
import pickle
import topobench.nn.encoders as public
from topobench.nn.encoders.heterogeneous_node_encoder import (
    HeterogeneousNodeFeatureEncoder as canonical,
)
encoder = canonical({"author": 8}, 4)
assert public.HeterogeneousNodeFeatureEncoder is canonical
assert public.FEATURE_ENCODERS["HeterogeneousNodeFeatureEncoder"] is canonical
assert pickle.loads(pickle.dumps(encoder)).__class__ is canonical
assert canonical.__module__ == (
    "topobench.nn.encoders.heterogeneous_node_encoder"
)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
