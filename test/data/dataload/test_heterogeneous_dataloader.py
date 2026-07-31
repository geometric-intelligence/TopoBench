"""Contracts for the native heterogeneous node data module."""

from __future__ import annotations

from enum import IntEnum
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader, NeighborLoader

from topobench.data.datasets import make_synthetic_heterogeneous_data
from topobench.data.heterogeneous import (
    HeterogeneousDataSpec,
    validate_heterogeneous_node_data,
)
from topobench.dataloader.heterogeneous import (
    HeterogeneousNodeDataModule,
    _normalize_fanout,
)
from topobench.transforms.data_manipulations.heterogeneous import (
    HeterogeneousConstantFeatures,
    HeterogeneousToUndirected,
)


class IntegralValue(IntEnum):
    """Non-built-in integral values used to test normalization."""

    TWO = 2
    THREE = 3
    FOUR = 4


@pytest.fixture
def heterogeneous_data() -> HeteroData:
    """Return the canonical fully transformed synthetic graph."""
    data = make_synthetic_heterogeneous_data(seed=7)
    data = HeterogeneousConstantFeatures(node_types="venue")(data)
    return HeterogeneousToUndirected(merge=False)(data)


@pytest.fixture
def heterogeneous_spec(
    heterogeneous_data: HeteroData,
) -> HeterogeneousDataSpec:
    """Return the validated immutable contract for the synthetic graph."""
    return validate_heterogeneous_node_data(
        heterogeneous_data,
        target_node_type="author",
        num_classes=2,
    )


def _assert_store_equal(expected: object, actual: object) -> None:
    """Assert original attributes survive PyG's added batch bookkeeping."""
    assert set(expected.keys()) <= set(actual.keys())  # type: ignore[union-attr]
    for key, expected_value in expected.items():  # type: ignore[union-attr]
        actual_value = actual[key]  # type: ignore[index]
        if torch.is_tensor(expected_value):
            assert torch.equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


@pytest.mark.parametrize(
    "loader_method",
    [
        "train_dataloader",
        "val_dataloader",
        "test_dataloader",
    ],
)
def test_full_batch_loaders_preserve_one_native_graph(
    loader_method: str,
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Each phase batches exactly one complete native heterogeneous graph."""
    datamodule = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="full_batch",
        batch_size=19,
        train_shuffle=True,
    )

    loader = getattr(datamodule, loader_method)()
    batch = next(iter(loader))

    assert type(loader) is DataLoader
    assert isinstance(batch, HeteroData)
    assert batch.num_graphs == 1
    assert batch.metadata() == heterogeneous_data.metadata()
    for store_type in (
        *heterogeneous_data.node_types,
        *heterogeneous_data.edge_types,
    ):
        _assert_store_equal(
            heterogeneous_data[store_type],
            batch[store_type],
        )


def test_full_batch_constructs_fresh_fixed_loaders(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Full-graph loading ignores graph batch size and never shuffles."""
    datamodule = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="full_batch",
        batch_size=17,
        train_shuffle=True,
    )

    loaders = [
        datamodule.train_dataloader(),
        datamodule.val_dataloader(),
        datamodule.test_dataloader(),
        datamodule.train_dataloader(),
    ]

    assert len({id(loader) for loader in loaders}) == len(loaders)
    for loader in loaders:
        assert type(loader) is DataLoader
        assert loader.batch_size == 1
        assert type(loader.sampler).__name__ == "SequentialSampler"


def test_neighbor_loaders_execute_real_seed_isolated_batches(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Real PyG sampling uses only phase seeds and preserves seed ordering."""
    datamodule = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="neighbor",
        num_neighbors=[3, 2],
        batch_size=4,
        train_shuffle=True,
    )

    for phase, loader in (
        ("train", datamodule.train_dataloader()),
        ("val", datamodule.val_dataloader()),
        ("test", datamodule.test_dataloader()),
    ):
        assert type(loader) is NeighborLoader
        try:
            batches = list(loader)
        except ImportError as error:
            pytest.fail(
                "NeighborLoader execution requires the PyG sampling backend "
                f"provided by pyg-lib or torch-sparse: {error}"
            )
        assert batches
        expected_ids = (
            heterogeneous_data["author"][f"{phase}_mask"]
            .nonzero(as_tuple=False)
            .view(-1)
        )
        observed_seed_ids: list[torch.Tensor] = []
        for batch in batches:
            target = batch["author"]
            assert 0 < target.batch_size <= 4
            assert "n_id" in target
            seed_ids = target.n_id[: target.batch_size]
            assert bool(torch.isin(seed_ids, expected_ids).all())
            observed_seed_ids.append(seed_ids)
        assert torch.equal(
            torch.cat(observed_seed_ids).sort().values,
            expected_ids.sort().values,
        )

    train_loader = datamodule.train_dataloader()
    assert train_loader.node_sampler.num_neighbors.values == [3, 2]
    assert set(train_loader.node_sampler.edge_types) == set(
        heterogeneous_spec.edge_types
    )


def test_neighbor_loader_forwards_exact_phase_and_sampling_contract(
    monkeypatch: pytest.MonkeyPatch,
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Constructor spying fixes masks, shuffle policy, and sampler kwargs."""
    import topobench.dataloader.heterogeneous as module

    sentinels = [object(), object(), object()]
    constructor = MagicMock(side_effect=sentinels)
    monkeypatch.setattr(module, "NeighborLoader", constructor)
    datamodule = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="neighbor",
        batch_size=4,
        num_neighbors=[3, 2],
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        train_shuffle=True,
        replace=True,
        subgraph_type="directional",
        filter_per_worker=True,
    )

    actual = [
        datamodule.train_dataloader(),
        datamodule.val_dataloader(),
        datamodule.test_dataloader(),
    ]

    assert actual == sentinels
    assert constructor.call_count == 3
    for phase, call_args in zip(
        ("train", "val", "test"),
        constructor.call_args_list,
        strict=True,
    ):
        args, kwargs = call_args
        assert args == (heterogeneous_data,)
        assert kwargs == {
            "input_nodes": (
                "author",
                heterogeneous_data["author"][f"{phase}_mask"],
            ),
            "num_neighbors": [3, 2],
            "batch_size": 4,
            "shuffle": phase == "train",
            "replace": True,
            "subgraph_type": "directional",
            "filter_per_worker": True,
            "num_workers": 2,
            "pin_memory": True,
            "persistent_workers": True,
        }


def test_datamodule_is_model_agnostic_and_avoids_topological_path() -> None:
    """The separate heterogeneous module has no model or topo-batch coupling."""
    import inspect

    import topobench.dataloader.heterogeneous as module

    signature = inspect.signature(HeterogeneousNodeDataModule)
    assert "model" not in signature.parameters
    assert "backbone" not in signature.parameters
    source = inspect.getsource(module)
    assert "HGT" not in source
    assert "HeteroSAGE" not in source
    assert not hasattr(module, "TBDataloader")
    assert not hasattr(module, "DataloadDataset")
    assert "collate_fn" not in source


def test_generic_fanout_is_copied_and_normalized() -> None:
    """Ordered generic fanouts become fresh positive built-in integer lists."""
    edge_types = (
        ("author", "writes", "paper"),
        ("paper", "rev_writes", "author"),
    )
    values = OmegaConf.create([IntegralValue.THREE, IntegralValue.TWO])

    normalized = _normalize_fanout(values, edge_types)

    assert normalized == [3, 2]
    assert normalized is not values
    assert all(type(value) is int for value in normalized)
    values[0] = 9
    assert normalized == [3, 2]


def test_relation_fanout_is_reordered_and_deeply_copied() -> None:
    """Typed fanouts follow metadata order and share no mutable lists."""
    first = ("author", "writes", "paper")
    second = ("paper", "rev_writes", "author")
    second_values = [IntegralValue.FOUR, IntegralValue.TWO]
    first_values = (IntegralValue.THREE, IntegralValue.TWO)
    fanout = {second: second_values, first: first_values}

    normalized = _normalize_fanout(fanout, (first, second))

    assert list(normalized) == [first, second]
    assert normalized == {first: [3, 2], second: [4, 2]}
    assert normalized[first] is not first_values
    assert normalized[second] is not second_values
    assert all(
        type(value) is int
        for values in normalized.values()
        for value in values
    )
    second_values[0] = 99
    assert normalized[second] == [4, 2]


@pytest.mark.parametrize(
    ("fanout", "error_type", "message"),
    [
        ("3,2", TypeError, "ordered sequence or relation mapping"),
        (b"3,2", TypeError, "ordered sequence or relation mapping"),
        ([], ValueError, "must not be empty"),
        ([True], TypeError, "positive integers"),
        ([1.5], TypeError, "positive integers"),
        ([0], ValueError, "positive"),
        ([-1], ValueError, "positive"),
        ({("author", "writes", "paper"): []}, ValueError, "must not be empty"),
    ],
)
def test_fanout_rejects_invalid_values(
    fanout: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Invalid generic and relation fanouts fail before loader construction."""
    edge_types = (("author", "writes", "paper"),)

    with pytest.raises(error_type, match=message):
        _normalize_fanout(fanout, edge_types)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "fanout",
    [
        {},
        {("author", "writes", "paper"): [2]},
        {
            ("author", "writes", "paper"): [2],
            ("paper", "rev_writes", "author"): [2],
            ("author", "extra", "author"): [2],
        },
    ],
)
def test_relation_fanout_requires_exact_keys(fanout: object) -> None:
    """Missing and extra typed relations are reported with full key context."""
    edge_types = (
        ("author", "writes", "paper"),
        ("paper", "rev_writes", "author"),
    )

    with pytest.raises(
        ValueError,
        match=r"exactly match.*missing=.*extra=",
    ):
        _normalize_fanout(fanout, edge_types)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"mode": "invalid"}, ValueError, "Unsupported.*mode"),
        ({"mode": []}, TypeError, "mode must be a string"),
        ({"mode": "neighbor", "batch_size": True}, TypeError, "batch_size"),
        ({"mode": "neighbor", "batch_size": 1.5}, TypeError, "batch_size"),
        ({"mode": "neighbor", "batch_size": 0}, ValueError, "batch_size"),
        ({"mode": "neighbor", "num_workers": True}, TypeError, "num_workers"),
        ({"mode": "neighbor", "num_workers": 1.5}, TypeError, "num_workers"),
        ({"mode": "neighbor", "num_workers": -1}, ValueError, "num_workers"),
        (
            {"mode": "neighbor", "persistent_workers": True},
            ValueError,
            "persistent_workers requires",
        ),
        (
            {"mode": "neighbor", "subgraph_type": "bidirectional"},
            ValueError,
            "directional",
        ),
        ({"mode": "neighbor", "pin_memory": 1}, TypeError, "pin_memory"),
        (
            {"mode": "neighbor", "persistent_workers": 1},
            TypeError,
            "persistent_workers",
        ),
        (
            {"mode": "neighbor", "train_shuffle": 1},
            TypeError,
            "train_shuffle",
        ),
        ({"mode": "neighbor", "replace": 1}, TypeError, "replace"),
        (
            {"mode": "neighbor", "filter_per_worker": 1},
            TypeError,
            "filter_per_worker",
        ),
    ],
)
def test_constructor_rejects_invalid_options(
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Loader options use strict, early, actionable validation."""
    with pytest.raises(error_type, match=message):
        HeterogeneousNodeDataModule(
            heterogeneous_data,
            heterogeneous_spec,
            **kwargs,  # type: ignore[arg-type]
        )


def test_constructor_normalizes_integral_options(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Integral subclasses normalize consistently to built-in integers."""
    datamodule = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="neighbor",
        batch_size=IntegralValue.FOUR,
        num_workers=IntegralValue.TWO,
    )

    assert datamodule.batch_size == 4
    assert type(datamodule.batch_size) is int
    assert datamodule.num_workers == 2
    assert type(datamodule.num_workers) is int


def test_constructor_rejects_invalid_data_and_spec(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Data and runtime contract families are checked explicitly."""
    with pytest.raises(TypeError, match="data must be.*HeteroData"):
        HeterogeneousNodeDataModule(  # type: ignore[arg-type]
            object(),
            heterogeneous_spec,
            mode="full_batch",
        )
    with pytest.raises(TypeError, match="spec must be.*HeterogeneousDataSpec"):
        HeterogeneousNodeDataModule(  # type: ignore[arg-type]
            heterogeneous_data,
            object(),
            mode="full_batch",
        )


def test_constructor_rejects_data_spec_metadata_mismatch(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Metadata alignment failures surface before a phase loader is requested."""
    mismatched = heterogeneous_data.clone()
    del mismatched[("venue", "rev_published_in", "paper")]

    with pytest.raises(ValueError, match="metadata.*spec"):
        HeterogeneousNodeDataModule(
            mismatched,
            heterogeneous_spec,
            mode="full_batch",
        )


def test_datamodule_hparams_never_serialize_graph_or_spec(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Lightning hyperparameters never retain heavyweight runtime objects."""
    datamodule = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="full_batch",
    )

    assert "data" not in datamodule.hparams
    assert "spec" not in datamodule.hparams
