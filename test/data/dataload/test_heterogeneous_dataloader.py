"""Contracts for the native heterogeneous node data module."""

from __future__ import annotations

import json
import subprocess
import sys
from enum import IntEnum
from pathlib import Path
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


def _batch_tensor_signature(batch: HeteroData) -> tuple[object, ...]:
    """Return an ordered byte-level signature of every sampled tensor."""
    signature: list[object] = [batch.metadata()]
    for store_type in (*batch.node_types, *batch.edge_types):
        store = batch[store_type]
        signature.append(store_type)
        for key in sorted(store.keys()):
            value = store[key]
            if torch.is_tensor(value):
                cpu_value = value.detach().cpu().contiguous()
                signature.append(
                    (
                        key,
                        str(cpu_value.dtype),
                        tuple(cpu_value.shape),
                        cpu_value.numpy().tobytes(),
                    )
                )
            else:
                signature.append((key, value))
    return tuple(signature)


def _loader_signature(loader: object) -> tuple[tuple[object, ...], ...]:
    """Materialize one loader traversal into ordered batch signatures."""
    return tuple(_batch_tensor_signature(batch) for batch in loader)


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
        assert isinstance(loader, NeighborLoader)
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

    sentinels = [object()]
    constructor = MagicMock(side_effect=sentinels)
    evaluation_sentinels = [object(), object()]
    evaluation_constructor = MagicMock(side_effect=evaluation_sentinels)
    monkeypatch.setattr(module, "NeighborLoader", constructor)
    monkeypatch.setattr(
        module,
        "_FixedEvaluationNeighborLoader",
        evaluation_constructor,
    )
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

    assert actual == [*sentinels, *evaluation_sentinels]
    constructor.assert_called_once()
    train_args, train_kwargs = constructor.call_args
    assert train_args == (heterogeneous_data,)
    assert train_kwargs == {
        "input_nodes": (
            "author",
            heterogeneous_data["author"].train_mask,
        ),
        "num_neighbors": [3, 2],
        "batch_size": 4,
        "shuffle": True,
        "replace": True,
        "subgraph_type": "directional",
        "filter_per_worker": True,
        "num_workers": 2,
        "pin_memory": True,
        "persistent_workers": True,
    }
    assert evaluation_constructor.call_count == 2
    for phase, call_args in zip(
        ("val", "test"),
        evaluation_constructor.call_args_list,
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
            "evaluation_owner": datamodule,
            "evaluation_phase": phase,
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


@pytest.mark.parametrize(
    ("mode", "protocol"),
    [
        ("full_batch", "full_graph"),
        ("neighbor", "sampled_neighbor_fixed"),
    ],
)
def test_evaluation_protocol_is_derived_or_validated(
    mode: str,
    protocol: str,
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """The protocol is explicit at runtime and canonical for its loader mode."""
    derived = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode=mode,  # type: ignore[arg-type]
    )
    explicit = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode=mode,  # type: ignore[arg-type]
        evaluation_protocol=protocol,
    )

    assert derived.evaluation_protocol == protocol
    assert explicit.evaluation_protocol == protocol
    assert derived.data is heterogeneous_data


@pytest.mark.parametrize(
    ("mode", "protocol"),
    [
        ("full_batch", "sampled_neighbor_fixed"),
        ("neighbor", "full_graph"),
        ("neighbor", "sampled_neighbor"),
    ],
)
def test_evaluation_protocol_rejects_mode_mismatch(
    mode: str,
    protocol: str,
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Mislabelled evaluation semantics fail before any loader is built."""
    with pytest.raises(
        ValueError,
        match=r"evaluation_protocol.*mode",
    ):
        HeterogeneousNodeDataModule(
            heterogeneous_data,
            heterogeneous_spec,
            mode=mode,  # type: ignore[arg-type]
            evaluation_protocol=protocol,
        )


@pytest.mark.parametrize(
    ("seed", "error_type", "message"),
    [
        (True, TypeError, "evaluation_seed must be an integer"),
        (1.5, TypeError, "evaluation_seed must be an integer"),
        (-1, ValueError, "evaluation_seed must be non-negative"),
        (
            2**63,
            ValueError,
            "evaluation_seed must be no greater than",
        ),
    ],
)
def test_evaluation_seed_rejects_invalid_values(
    seed: object,
    error_type: type[Exception],
    message: str,
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Fixed evaluation seeds use an explicit torch-compatible range."""
    with pytest.raises(error_type, match=message):
        HeterogeneousNodeDataModule(
            heterogeneous_data,
            heterogeneous_spec,
            mode="neighbor",
            evaluation_seed=seed,  # type: ignore[arg-type]
        )


def test_evaluation_seed_normalizes_integral_subclasses(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Integral seed subclasses become built-in non-negative integers."""
    datamodule = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="neighbor",
        evaluation_seed=IntegralValue.THREE,
    )

    assert datamodule.evaluation_seed == 3
    assert type(datamodule.evaluation_seed) is int


def test_evaluation_cache_descriptor_is_stable_explicit_json(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Cache identity has no dependency on randomized Python hashing."""
    datamodule = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="neighbor",
        batch_size=4,
        num_neighbors=[1, 1],
        evaluation_protocol="sampled_neighbor_fixed",
        evaluation_seed=17,
        replace=False,
        subgraph_type="directional",
    )

    descriptor = datamodule.evaluation_cache_descriptor("val")

    assert descriptor == json.dumps(
        {
            "batch_size": 4,
            "evaluation_protocol": "sampled_neighbor_fixed",
            "evaluation_seed": 17,
            "fanout": [1, 1],
            "phase": "val",
            "replace": False,
            "subgraph_type": "directional",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    assert datamodule.evaluation_cache_descriptor("val") == descriptor
    assert datamodule.evaluation_cache_descriptor("test") != descriptor


@pytest.mark.parametrize("phase", ["val", "test"])
def test_sampled_evaluation_replays_identical_batches_across_all_traversals(
    phase: str,
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """One materialization is replayed byte-identically by all phase loaders."""
    datamodule = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="neighbor",
        num_neighbors=[1, 1],
        batch_size=4,
        evaluation_seed=23,
    )
    loader_method = getattr(datamodule, f"{phase}_dataloader")
    loader = loader_method()

    first = _loader_signature(loader)
    same_loader_replay = _loader_signature(loader)
    fresh_loader_replay = _loader_signature(loader_method())

    assert first
    assert same_loader_replay == first
    assert fresh_loader_replay == first
    assert len(datamodule._evaluation_batch_cache[phase]) == len(first)


def test_sampled_evaluation_seed_controls_context_without_cross_run_drift(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Equal seeds match exactly while a distinct seed changes sampled hops."""

    def make_datamodule(seed: int) -> HeterogeneousNodeDataModule:
        return HeterogeneousNodeDataModule(
            heterogeneous_data,
            heterogeneous_spec,
            mode="neighbor",
            num_neighbors=[1, 1],
            batch_size=4,
            evaluation_seed=seed,
        )

    first = _loader_signature(make_datamodule(101).val_dataloader())
    same = _loader_signature(make_datamodule(101).val_dataloader())
    different = _loader_signature(make_datamodule(202).val_dataloader())

    assert first == same
    assert first != different


def test_cached_evaluation_yields_fresh_cpu_clones(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Consumer mutation or transfer cannot corrupt cached evaluation data."""
    datamodule = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="neighbor",
        num_neighbors=[1, 1],
        batch_size=4,
        evaluation_seed=31,
    )
    loader = datamodule.val_dataloader()
    first_batches = list(loader)
    expected = tuple(_batch_tensor_signature(batch) for batch in first_batches)

    first_batches[0]["author"].x.zero_()
    first_batches[-1].to("meta")
    replay = list(loader)

    assert all(
        tensor.device.type == "cpu"
        for batch in replay
        for store_type in (*batch.node_types, *batch.edge_types)
        for tensor in batch[store_type].values()
        if torch.is_tensor(tensor)
    )
    assert (
        tuple(_batch_tensor_signature(batch) for batch in replay) == expected
    )
    assert all(
        replayed is not cached
        for replayed, cached in zip(
            replay,
            datamodule._evaluation_batch_cache["val"],
            strict=True,
        )
    )


def test_sampled_evaluation_does_not_perturb_global_torch_rng(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """First sampling and later cache replay preserve the caller RNG state."""
    datamodule = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="neighbor",
        num_neighbors=[1, 1],
        batch_size=4,
        evaluation_seed=37,
    )
    loader = datamodule.val_dataloader()

    state_before_materialization = torch.random.get_rng_state().clone()
    list(loader)
    state_after_materialization = torch.random.get_rng_state().clone()
    list(loader)
    state_after_replay = torch.random.get_rng_state().clone()

    assert torch.equal(
        state_after_materialization,
        state_before_materialization,
    )
    assert torch.equal(state_after_replay, state_before_materialization)


def test_train_neighbor_loading_stays_uncached_and_stochastic_capable(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Training keeps the ordinary fresh shuffled NeighborLoader path."""
    datamodule = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="neighbor",
        num_neighbors=[1, 1],
        batch_size=4,
        train_shuffle=True,
        evaluation_seed=41,
    )

    first_loader = datamodule.train_dataloader()
    second_loader = datamodule.train_dataloader()
    list(first_loader)
    list(second_loader)

    assert type(first_loader) is NeighborLoader
    assert type(second_loader) is NeighborLoader
    assert first_loader is not second_loader
    assert datamodule._evaluation_batch_cache == {}


def test_full_graph_evaluation_does_not_use_sampled_cache(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Full-graph protocol remains deterministic without sampling materialization."""
    datamodule = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="full_batch",
        evaluation_protocol="full_graph",
        evaluation_seed=43,
    )

    first = _loader_signature(datamodule.val_dataloader())
    second = _loader_signature(datamodule.val_dataloader())

    assert first == second
    assert datamodule._evaluation_batch_cache == {}


def test_directional_sampling_retains_required_reverse_relations(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Reverse relations produced by preprocessing remain available for hops."""
    datamodule = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="neighbor",
        num_neighbors=[1, 1],
        batch_size=4,
        evaluation_seed=47,
    )

    batches = list(datamodule.val_dataloader())

    reverse_relations = {
        ("paper", "rev_writes", "author"),
        ("venue", "rev_published_in", "paper"),
    }
    assert reverse_relations <= set(heterogeneous_spec.edge_types)
    assert all(reverse_relations <= set(batch.edge_types) for batch in batches)
    assert any(
        batch["paper", "rev_writes", "author"].edge_index.numel() > 0
        for batch in batches
    )


def test_sampling_backend_failure_has_precise_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Backend import/runtime failures name the required PyG capabilities."""
    import topobench.dataloader.heterogeneous as module

    datamodule = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="neighbor",
        num_neighbors=[1, 1],
        evaluation_seed=53,
    )
    loader = datamodule.val_dataloader()
    monkeypatch.setattr(
        module._FixedEvaluationNeighborLoader,
        "_uncached_iterator",
        MagicMock(side_effect=ImportError("missing sampling operator")),
    )

    with pytest.raises(
        RuntimeError,
        match=r"pyg-lib.*torch-sparse.*val",
    ):
        list(loader)


def test_fixed_evaluation_replays_with_real_persistent_workers() -> None:
    """A bounded clean process executes deterministic two-worker replay."""
    verifier = Path(__file__).with_name(
        "verify_fixed_heterogeneous_neighbor_workers.py"
    )

    completed = subprocess.run(
        [sys.executable, str(verifier)],
        cwd=Path(__file__).parents[3],
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )

    assert completed.stdout.strip() == "worker-replay-ok"
