"""Contract tests for the native heterogeneous backbone wrapper."""

from __future__ import annotations

import importlib
import pickle
import subprocess
import sys
from collections.abc import Mapping

import pytest
import torch
from torch import Tensor, nn
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.typing import EdgeType

import topobench.nn.wrappers as wrapper_registry
from topobench.data.datasets import make_synthetic_heterogeneous_data
from topobench.nn.backbones.heterogeneous import (
    HeteroSAGEBackbone,
    HGTBackbone,
)
from topobench.nn.encoders import HeterogeneousNodeFeatureEncoder
from topobench.nn.readouts import HeterogeneousNodeReadout
from topobench.nn.wrappers.heterogeneous import HeterogeneousWrapper
from topobench.nn.wrappers.heterogeneous.heterogeneous_wrapper import (
    HeterogeneousWrapper as CanonicalWrapper,
)
from topobench.transforms.data_manipulations.heterogeneous import (
    HeterogeneousConstantFeatures,
    HeterogeneousToUndirected,
)

HIDDEN_CHANNELS = 8
INPUT_CHANNELS = {"author": 8, "paper": 5, "venue": 1}
EXPECTED_REGISTERED_WRAPPERS = {
    "GNNWrapper",
    "GraphMLPWrapper",
    "HeterogeneousWrapper",
    "HypergraphWrapper",
}


def make_data() -> HeteroData:
    """Return the real synthetic fixture with all model inputs present."""
    data = make_synthetic_heterogeneous_data(seed=7)
    data = HeterogeneousConstantFeatures(node_types="venue")(data)
    return HeterogeneousToUndirected(merge=False)(data)


def encode(data: HeteroData) -> HeteroData:
    """Project every type to the shared backbone width."""
    return HeterogeneousNodeFeatureEncoder(
        input_channels=INPUT_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        activation="id",
        dropout=0.0,
    )(data)


def make_backbone(
    backbone_type: type[HGTBackbone] | type[HeteroSAGEBackbone],
    data: HeteroData,
) -> nn.Module:
    """Construct one compatible backbone against exact graph metadata."""
    common: dict[str, object] = {
        "metadata": data.metadata(),
        "hidden_channels": HIDDEN_CHANNELS,
        "num_layers": 1,
        "dropout": 0.0,
        "activation": "relu",
    }
    if backbone_type is HGTBackbone:
        common["heads"] = 2
    return backbone_type(**common)


class SpyBackbone(nn.Module):
    """Record the exact wrapper call while returning valid typed features."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[
            tuple[Mapping[str, Tensor], Mapping[EdgeType, Tensor]]
        ] = []

    def forward(
        self,
        x_dict: Mapping[str, Tensor],
        edge_index_dict: Mapping[EdgeType, Tensor],
    ) -> dict[str, Tensor]:
        """Record one positional dictionary call."""
        self.calls.append((x_dict, edge_index_dict))
        return {
            node_type: features + 1 for node_type, features in x_dict.items()
        }


@pytest.mark.parametrize("target_node_type", [None, 1, "", "   "])
def test_constructor_rejects_invalid_target_node_type(
    target_node_type: object,
) -> None:
    """The target type is a required meaningful string."""
    with pytest.raises(
        (TypeError, ValueError),
        match="target_node_type must be a non-empty string",
    ):
        HeterogeneousWrapper(SpyBackbone(), target_node_type)  # type: ignore[arg-type]


def test_constructor_requires_module_backbone() -> None:
    """Only registered modules provide safe optimizer/checkpoint behavior."""
    with pytest.raises(TypeError, match="backbone must be a torch.nn.Module"):
        HeterogeneousWrapper(object(), "author")  # type: ignore[arg-type]


def test_wrapper_calls_compatible_backbone_once_and_preserves_batch_metadata() -> (
    None
):
    """The wrapper owns translation only, never supervision selection."""
    data = encode(make_data())
    data["author"].n_id = torch.arange(data["author"].num_nodes)
    data["author"].batch_size = 5
    before = {
        "n_id": data["author"].n_id,
        "batch_size": data["author"].batch_size,
        "train_mask": data["author"].train_mask,
        "val_mask": data["author"].val_mask,
        "test_mask": data["author"].test_mask,
    }
    input_x = dict(data.x_dict)
    input_edges = dict(data.edge_index_dict)
    backbone = SpyBackbone()

    result = HeterogeneousWrapper(backbone, "author")(data)

    assert len(backbone.calls) == 1
    called_x, called_edges = backbone.calls[0]
    assert list(called_x) == list(input_x)
    assert list(called_edges) == list(input_edges)
    assert all(called_x[key] is input_x[key] for key in input_x)
    assert all(called_edges[key] is input_edges[key] for key in input_edges)
    assert list(result) == ["x_dict", "labels"]
    assert list(result["x_dict"]) == list(data.node_types)
    assert result["labels"] is data["author"].y
    assert data["author"].n_id is before["n_id"]
    assert data["author"].batch_size == before["batch_size"]
    for mask_name in ("train_mask", "val_mask", "test_mask"):
        assert data["author"][mask_name] is before[mask_name]


@pytest.mark.parametrize("backbone_type", [HGTBackbone, HeteroSAGEBackbone])
def test_same_wrapper_interface_runs_both_real_backbones(
    backbone_type: type[HGTBackbone] | type[HeteroSAGEBackbone],
) -> None:
    """Backbone selection does not leak into the wrapper contract."""
    data = encode(make_data())
    original_features = {
        node_type: features.clone()
        for node_type, features in data.x_dict.items()
    }
    original_edges = {
        edge_type: edge_index.clone()
        for edge_type, edge_index in data.edge_index_dict.items()
    }
    backbone = make_backbone(backbone_type, data)

    result = HeterogeneousWrapper(backbone, "author")(data)

    assert list(result) == ["x_dict", "labels"]
    assert list(result["x_dict"]) == list(data.node_types)
    assert result["labels"] is data["author"].y
    for node_type in data.node_types:
        assert result["x_dict"][node_type].shape == (
            data[node_type].num_nodes,
            HIDDEN_CHANNELS,
        )
        torch.testing.assert_close(
            data[node_type].x, original_features[node_type]
        )
    for edge_type, edge_index in original_edges.items():
        torch.testing.assert_close(data[edge_type].edge_index, edge_index)


@pytest.mark.parametrize("backbone_type", [HGTBackbone, HeteroSAGEBackbone])
def test_real_neighbor_sample_preserves_seed_metadata_and_forwards(
    backbone_type: type[HGTBackbone] | type[HeteroSAGEBackbone],
) -> None:
    """A sampled batch follows the same model-agnostic wrapper API."""
    full_data = make_data()
    loader = NeighborLoader(
        full_data,
        input_nodes=("author", torch.tensor([0, 1, 2])),
        num_neighbors=[-1],
        batch_size=2,
        shuffle=False,
    )
    try:
        sample = next(iter(loader))
    except ImportError as error:
        pytest.fail(f"PyG neighbor sampling backend is unavailable: {error}")
    sample = encode(sample)
    n_id = sample["author"].n_id
    batch_size = sample["author"].batch_size
    backbone = make_backbone(backbone_type, full_data)

    result = HeterogeneousWrapper(backbone, "author")(sample)

    assert result["labels"] is sample["author"].y
    assert result["labels"].shape[0] == sample["author"].num_nodes
    assert sample["author"].n_id is n_id
    assert sample["author"].batch_size == batch_size == 2
    assert list(result["x_dict"]) == list(sample.node_types)


def test_wrapper_preserves_autograd_from_every_typed_output() -> None:
    """The returned feature dictionary remains connected to input features."""
    data = encode(make_data())
    for features in data.x_dict.values():
        features.retain_grad()
    result = HeterogeneousWrapper(SpyBackbone(), "author")(data)

    sum(features.sum() for features in result["x_dict"].values()).backward()

    assert all(features.grad is not None for features in data.x_dict.values())


@pytest.mark.parametrize("backbone_type", [HGTBackbone, HeteroSAGEBackbone])
def test_surrogate_metadata_runs_complete_trainable_model_path(
    backbone_type: type[HGTBackbone] | type[HeteroSAGEBackbone],
) -> None:
    """External names survive encoder, backbone, wrapper, and readout."""
    surrogate_target = "target\ud800.type"
    context_type = "context\udfff.type"
    data = HeteroData()
    target_features = torch.randn(4, 3, requires_grad=True)
    context_features = torch.randn(3, 2, requires_grad=True)
    data[surrogate_target].x = target_features
    data[surrogate_target].y = torch.tensor([0, 1, 0, 1])
    data[context_type].x = context_features
    data[
        surrogate_target, "links\ud800.to", context_type
    ].edge_index = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 0]])
    data[
        context_type, "returns\udfff.to", surrogate_target
    ].edge_index = torch.tensor([[0, 1, 2, 0], [0, 1, 2, 3]])
    metadata = data.metadata()
    encoder = HeterogeneousNodeFeatureEncoder(
        input_channels={surrogate_target: 3, context_type: 2},
        hidden_channels=HIDDEN_CHANNELS,
        activation="relu",
        dropout=0.0,
    )
    backbone = make_backbone(backbone_type, data)
    wrapper = HeterogeneousWrapper(backbone, surrogate_target)
    readout = HeterogeneousNodeReadout(
        surrogate_target,
        HIDDEN_CHANNELS,
        2,
    )

    encoded = encoder(data)
    model_out = wrapper(encoded)
    result = readout(model_out=model_out, batch=data)
    result["logits"].square().mean().backward()

    assert data.metadata() == metadata
    assert tuple(data.node_types) == (surrogate_target, context_type)
    assert tuple(model_out["x_dict"]) == (surrogate_target, context_type)
    assert model_out["labels"] is data[surrogate_target].y
    assert result["logits"].shape == (4, 2)
    assert target_features.grad is not None
    assert context_features.grad is not None
    for component in (encoder, backbone, readout):
        gradients = [
            parameter.grad
            for parameter in component.parameters()
            if parameter.requires_grad
        ]
        assert gradients
        assert any(gradient is not None for gradient in gradients)
        assert all(
            gradient is None or torch.isfinite(gradient).all()
            for gradient in gradients
        )


@pytest.mark.parametrize(
    ("batch", "error_type", "message"),
    [
        (Data(), TypeError, "requires native HeteroData"),
        (HeteroData(), ValueError, "missing target node store 'author'"),
    ],
)
def test_wrapper_rejects_invalid_batch_before_backbone_call(
    batch: Data | HeteroData,
    error_type: type[Exception],
    message: str,
) -> None:
    """Invalid boundary inputs fail before any trainable work executes."""
    backbone = SpyBackbone()
    with pytest.raises(error_type, match=message):
        HeterogeneousWrapper(backbone, "author")(batch)  # type: ignore[arg-type]
    assert backbone.calls == []


@pytest.mark.parametrize(
    ("label_value", "error_type", "message"),
    [
        (None, TypeError, "target store 'author'.*tensor y"),
        ([0, 1], TypeError, "target store 'author'.*tensor y"),
        (torch.empty(0, dtype=torch.long), ValueError, "tensor y.*non-empty"),
        (torch.zeros(2, 1, dtype=torch.long), ValueError, "tensor y.*rank-1"),
        (
            torch.zeros(2, dtype=torch.long),
            ValueError,
            "tensor y count must match target nodes",
        ),
    ],
)
def test_wrapper_validates_target_labels_before_backbone_call(
    label_value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Labels are copied unfiltered only after their typed contract is valid."""
    data = encode(make_data())
    if label_value is None:
        del data["author"].y
    else:
        data["author"].y = label_value
    backbone = SpyBackbone()

    with pytest.raises(error_type, match=message):
        HeterogeneousWrapper(backbone, "author")(data)

    assert backbone.calls == []


@pytest.mark.parametrize(
    ("output_factory", "error_type", "message"),
    [
        (lambda data: [], TypeError, "backbone output must be a mapping"),
        (
            lambda data: {"author": data["author"].x},
            ValueError,
            "backbone output node types.*missing",
        ),
        (
            lambda data: {
                **data.x_dict,
                "unexpected": torch.ones(1, HIDDEN_CHANNELS),
            },
            ValueError,
            "backbone output node types.*unexpected",
        ),
        (
            lambda data: {
                **data.x_dict,
                1: torch.ones(1, HIDDEN_CHANNELS),
            },
            TypeError,
            "backbone output node-type keys must be strings",
        ),
        (
            lambda data: {**data.x_dict, "author": object()},
            TypeError,
            "backbone output.*'author'.*tensor",
        ),
        (
            lambda data: {
                **data.x_dict,
                "author": torch.ones(data["author"].num_nodes),
            },
            ValueError,
            "backbone output.*'author'.*rank-2",
        ),
    ],
)
def test_wrapper_rejects_incomplete_or_invalid_backbone_outputs_transactionally(
    output_factory,
    error_type: type[Exception],
    message: str,
) -> None:
    """A bad backbone result cannot alter labels or target-store metadata."""
    data = encode(make_data())
    original = {
        node_type: features for node_type, features in data.x_dict.items()
    }
    labels = data["author"].y

    class InvalidBackbone(nn.Module):
        def forward(self, x_dict, edge_index_dict):
            del x_dict, edge_index_dict
            return output_factory(data)

    with pytest.raises(error_type, match=message):
        HeterogeneousWrapper(InvalidBackbone(), "author")(data)

    assert data["author"].y is labels
    assert all(
        data[node_type].x is features
        for node_type, features in original.items()
    )


def test_registry_exports_canonical_pickle_stable_classes() -> None:
    """Reduced registry exports canonical, pickle-stable wrapper classes."""
    expected = EXPECTED_REGISTERED_WRAPPERS
    assert set(wrapper_registry.WRAPPER_CLASSES) == expected
    assert list(wrapper_registry.WRAPPER_CLASSES) == sorted(expected)
    assert HeterogeneousWrapper is CanonicalWrapper
    assert wrapper_registry.HeterogeneousWrapper is CanonicalWrapper
    assert (
        wrapper_registry.WRAPPER_CLASSES["HeterogeneousWrapper"]
        is CanonicalWrapper
    )
    for name, wrapper_class in wrapper_registry.WRAPPER_CLASSES.items():
        module = importlib.import_module(wrapper_class.__module__)
        assert getattr(module, name) is wrapper_class
        assert getattr(wrapper_registry, name) is wrapper_class
        assert pickle.loads(pickle.dumps(wrapper_class)) is wrapper_class


def test_clean_process_wrapper_registry_is_canonical_and_pickle_stable() -> (
    None
):
    """Registry identity is independent of import order in a fresh process."""
    script = """
import importlib
import pickle
import topobench.nn.wrappers as public
from topobench.nn.wrappers.heterogeneous import HeterogeneousWrapper
from topobench.nn.wrappers.heterogeneous.heterogeneous_wrapper import (
    HeterogeneousWrapper as canonical,
)
assert HeterogeneousWrapper is canonical
assert public.HeterogeneousWrapper is canonical
assert list(public.WRAPPER_CLASSES) == sorted(public.WRAPPER_CLASSES)
for name, cls in public.WRAPPER_CLASSES.items():
    module = importlib.import_module(cls.__module__)
    assert getattr(module, name) is cls
    assert getattr(public, name) is cls
    assert pickle.loads(pickle.dumps(cls)) is cls
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
