"""Contract tests for the heterogeneous target-node classifier."""

from __future__ import annotations

import importlib
import pickle
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader

import topobench.nn.readouts as readout_registry
from topobench.data.datasets import make_synthetic_heterogeneous_data
from topobench.nn.readouts import HeterogeneousNodeReadout
from topobench.nn.readouts.heterogeneous_node import (
    HeterogeneousNodeReadout as CanonicalReadout,
)
from topobench.transforms.data_manipulations.heterogeneous import (
    HeterogeneousToUndirected,
)

HIDDEN_CHANNELS = 8
OUT_CHANNELS = 2
NUM_AUTHORS = 36
EXISTING_READOUT_NAMES = {
    "MLPReadout",
    "NoReadOut",
}


def make_batch() -> HeteroData:
    """Return the canonical synthetic batch."""
    return make_synthetic_heterogeneous_data(seed=7)


def make_model_out(
    batch: HeteroData,
    *,
    width: int = HIDDEN_CHANNELS,
    dtype: torch.dtype = torch.float32,
) -> dict[str, object]:
    """Return typed embeddings and unrelated fields to preserve."""
    generator = torch.Generator().manual_seed(13)
    return {
        "x_dict": {
            node_type: torch.randn(
                batch[node_type].num_nodes,
                width,
                dtype=dtype,
                generator=generator,
            )
            for node_type in batch.node_types
        },
        "labels": batch["author"].y,
        "diagnostic": object(),
    }


@pytest.mark.parametrize("target_node_type", [None, 1, "", "   "])
def test_constructor_rejects_invalid_target_node_type(
    target_node_type: object,
) -> None:
    """The target node type must be a meaningful string."""
    with pytest.raises(
        (TypeError, ValueError),
        match="target_node_type must be a non-empty string",
    ):
        HeterogeneousNodeReadout(
            target_node_type,
            HIDDEN_CHANNELS,
            OUT_CHANNELS,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("hidden_channels", None),
        ("hidden_channels", True),
        ("hidden_channels", 1.5),
        ("hidden_channels", 0),
        ("hidden_channels", -1),
        ("out_channels", None),
        ("out_channels", False),
        ("out_channels", 2.5),
        ("out_channels", 1),
        ("out_channels", 0),
    ],
)
def test_constructor_validates_nonboolean_integer_dimensions(
    argument: str,
    value: object,
) -> None:
    """Classifier dimensions are eager, positive integer contracts."""
    parameters = {
        "target_node_type": "author",
        "hidden_channels": HIDDEN_CHANNELS,
        "out_channels": OUT_CHANNELS,
    }
    parameters[argument] = value
    with pytest.raises(
        (TypeError, ValueError),
        match=rf"{argument}.*{'integer' if value is None or isinstance(value, (bool, float)) else ''}|{argument}.*at least",
    ):
        HeterogeneousNodeReadout(**parameters)


def test_constructor_normalizes_integral_dimensions_to_builtin_ints() -> None:
    """NumPy integral configuration values remain safe and serializable."""
    readout = HeterogeneousNodeReadout(
        "author",
        np.int64(HIDDEN_CHANNELS),
        np.int64(OUT_CHANNELS),
    )

    assert type(readout.hidden_channels) is int
    assert type(readout.out_channels) is int
    assert readout.hidden_channels == HIDDEN_CHANNELS
    assert readout.out_channels == OUT_CHANNELS


def test_parameters_and_checkpoint_state_are_complete_before_forward() -> None:
    """Optimizer construction never depends on a warm-up batch."""
    readout = HeterogeneousNodeReadout("author", HIDDEN_CHANNELS, OUT_CHANNELS)
    initial_ids = {id(parameter) for parameter in readout.parameters()}
    initial_keys = tuple(readout.state_dict())
    optimizer = torch.optim.Adam(readout.parameters())

    readout(make_model_out(make_batch()), make_batch())

    assert initial_ids
    assert initial_ids == {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert (
        tuple(readout.state_dict())
        == initial_keys
        == (
            "linear.weight",
            "linear.bias",
        )
    )
    assert readout.task_level == "node"


def test_readout_selects_exact_target_embedding_and_writes_only_logits() -> (
    None
):
    """The readout classifies all target nodes without supervision filtering."""
    batch = make_batch()
    model_out = make_model_out(batch)
    original = dict(model_out)
    original_x_dict = model_out["x_dict"]
    readout = HeterogeneousNodeReadout("author", HIDDEN_CHANNELS, OUT_CHANNELS)

    result = readout(model_out=model_out, batch=batch)

    assert result is model_out
    assert set(result) == {*original, "logits"}
    assert result["x_dict"] is original_x_dict
    assert result["labels"] is original["labels"]
    assert result["diagnostic"] is original["diagnostic"]
    assert result["logits"].shape == (
        batch["author"].num_nodes,
        OUT_CHANNELS,
    )
    assert result["labels"].shape[0] == batch["author"].num_nodes


def test_readout_overwrites_only_existing_logits() -> None:
    """A repeated pipeline call replaces logits without touching other fields."""
    batch = make_batch()
    model_out = make_model_out(batch)
    previous_logits = torch.randn(1, 1)
    model_out["logits"] = previous_logits
    labels = model_out["labels"]
    x_dict = model_out["x_dict"]

    result = HeterogeneousNodeReadout("author", HIDDEN_CHANNELS, OUT_CHANNELS)(
        model_out=model_out, batch=batch
    )

    assert result["logits"] is not previous_logits
    assert result["labels"] is labels
    assert result["x_dict"] is x_dict


def test_readout_preserves_autograd_to_embedding_and_classifier() -> None:
    """Classification remains end-to-end differentiable."""
    batch = make_batch()
    model_out = make_model_out(batch)
    target_embedding = model_out["x_dict"]["author"]
    target_embedding.requires_grad_()
    readout = HeterogeneousNodeReadout("author", HIDDEN_CHANNELS, OUT_CHANNELS)

    result = readout(model_out=model_out, batch=batch)
    result["logits"].square().mean().backward()

    assert target_embedding.grad is not None
    assert torch.isfinite(target_embedding.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in readout.parameters()
    )


def test_readout_supports_explicit_double_precision() -> None:
    """Normal module dtype conversion follows PyTorch conventions."""
    batch = make_batch()
    model_out = make_model_out(batch, dtype=torch.float64)
    readout = HeterogeneousNodeReadout(
        "author", HIDDEN_CHANNELS, OUT_CHANNELS
    ).double()

    result = readout(model_out=model_out, batch=batch)

    assert result["logits"].dtype is torch.float64


def test_readout_supports_cpu_autocast() -> None:
    """The eager classifier participates in mixed precision."""
    batch = make_batch()
    model_out = make_model_out(batch)
    readout = HeterogeneousNodeReadout("author", HIDDEN_CHANNELS, OUT_CHANNELS)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = readout(model_out=model_out, batch=batch)

    assert result["logits"].dtype is torch.bfloat16


def test_real_neighbor_sample_classifies_context_without_selecting_seeds() -> (
    None
):
    """Neighbor supervision remains a later adapter responsibility."""
    full_data = HeterogeneousToUndirected(merge=False)(make_batch())
    loader = NeighborLoader(
        full_data,
        input_nodes=("author", torch.tensor([0, 1, 2])),
        num_neighbors=[-1, -1],
        batch_size=2,
        shuffle=False,
    )
    try:
        sample = next(iter(loader))
    except ImportError as error:
        pytest.fail(f"PyG neighbor sampling backend is unavailable: {error}")
    seed_count = sample["author"].batch_size
    n_id = sample["author"].n_id
    model_out = make_model_out(sample)

    result = HeterogeneousNodeReadout("author", HIDDEN_CHANNELS, OUT_CHANNELS)(
        model_out=model_out, batch=sample
    )

    assert sample["author"].num_nodes > seed_count
    assert result["logits"].shape[0] == sample["author"].num_nodes
    assert result["labels"].shape[0] == sample["author"].num_nodes
    assert sample["author"].batch_size == seed_count
    assert sample["author"].n_id is n_id


@pytest.mark.parametrize(
    ("model_out", "batch", "error_type", "message"),
    [
        ([], make_batch(), TypeError, "model_out must be a dictionary"),
        ({}, make_batch(), TypeError, "model_out.*x_dict.*mapping"),
        (
            {"x_dict": []},
            make_batch(),
            TypeError,
            "model_out.*x_dict.*mapping",
        ),
        (
            {"x_dict": {"paper": torch.ones(2, HIDDEN_CHANNELS)}},
            make_batch(),
            ValueError,
            "no embeddings for target node type 'author'",
        ),
        (
            {"x_dict": {"author": object()}},
            make_batch(),
            TypeError,
            "target embeddings 'author'.*tensor",
        ),
        (
            {"x_dict": {"author": torch.ones(HIDDEN_CHANNELS)}},
            make_batch(),
            ValueError,
            "target embeddings 'author'.*rank-2",
        ),
        (
            {
                "x_dict": {
                    "author": torch.ones(
                        NUM_AUTHORS,
                        HIDDEN_CHANNELS + 1,
                    )
                }
            },
            make_batch(),
            ValueError,
            "target embeddings 'author'.*width.*8.*received 9",
        ),
        (
            {
                "x_dict": {
                    "author": torch.ones(
                        NUM_AUTHORS - 1,
                        HIDDEN_CHANNELS,
                    )
                }
            },
            make_batch(),
            ValueError,
            "target embeddings 'author'.*node count",
        ),
        (
            {
                "x_dict": {
                    "author": torch.ones(
                        NUM_AUTHORS,
                        HIDDEN_CHANNELS,
                    )
                }
            },
            Data(),
            TypeError,
            "requires native HeteroData batch",
        ),
        (
            {
                "x_dict": {
                    "author": torch.ones(
                        NUM_AUTHORS,
                        HIDDEN_CHANNELS,
                    )
                }
            },
            HeteroData(),
            ValueError,
            "batch is missing target node store 'author'",
        ),
    ],
)
def test_readout_reports_clear_boundary_errors_transactionally(
    model_out: object,
    batch: Data | HeteroData,
    error_type: type[Exception],
    message: str,
) -> None:
    """Validation errors cannot partially commit a new logits field."""
    existing_logits = torch.tensor([[7.0]])
    if isinstance(model_out, dict):
        model_out["logits"] = existing_logits
        snapshot = dict(model_out)
    readout = HeterogeneousNodeReadout("author", HIDDEN_CHANNELS, OUT_CHANNELS)

    with pytest.raises(error_type, match=message):
        readout(model_out=model_out, batch=batch)  # type: ignore[arg-type]

    if isinstance(model_out, dict):
        assert model_out == snapshot
        assert model_out["logits"] is existing_logits


def test_registry_exports_canonical_pickle_stable_classes() -> None:
    """The readout registry exposes only native concrete classes."""
    expected = EXISTING_READOUT_NAMES | {"HeterogeneousNodeReadout"}
    assert set(readout_registry.READOUT_CLASSES) == expected
    assert list(readout_registry.READOUT_CLASSES) == sorted(expected)
    assert HeterogeneousNodeReadout is CanonicalReadout
    assert readout_registry.HeterogeneousNodeReadout is CanonicalReadout
    assert (
        readout_registry.READOUT_CLASSES["HeterogeneousNodeReadout"]
        is CanonicalReadout
    )
    for name, readout_class in readout_registry.READOUT_CLASSES.items():
        module = importlib.import_module(readout_class.__module__)
        assert getattr(module, name) is readout_class
        assert getattr(readout_registry, name) is readout_class
        assert pickle.loads(pickle.dumps(readout_class)) is readout_class


def test_clean_process_readout_registry_is_canonical_and_pickle_stable() -> (
    None
):
    """All registered readout classes remain importable in a fresh process."""
    script = """
import importlib
import pickle
import topobench.nn.readouts as public
from topobench.nn.readouts.heterogeneous_node import (
    HeterogeneousNodeReadout as canonical,
)
assert public.HeterogeneousNodeReadout is canonical
assert list(public.READOUT_CLASSES) == sorted(public.READOUT_CLASSES)
for name, cls in public.READOUT_CLASSES.items():
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
