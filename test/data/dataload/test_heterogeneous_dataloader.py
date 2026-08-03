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
from test.preflight.test_data_probe import make_observations, run_probe

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

_CHOICE_RICH_FANOUT = [2, 3, 1]


class IntegralValue(IntEnum):
    """Non-built-in integral values used to test normalization."""

    TWO = 2
    THREE = 3
    FOUR = 4


def _make_every_relation_choice_rich(data: HeteroData) -> HeteroData:
    """Give every relation more candidates than the largest test fanout."""
    candidate_count = max(_CHOICE_RICH_FANOUT) + 1
    author_count = data["author"].num_nodes
    paper_count = data["paper"].num_nodes
    venue_count = data["venue"].num_nodes
    assert author_count is not None
    assert paper_count is not None
    assert venue_count is not None

    author_ids = torch.arange(author_count).repeat_interleave(candidate_count)
    author_slots = torch.arange(candidate_count).repeat(author_count)
    paper_ids = (author_ids + author_slots) % paper_count
    data["author", "writes", "paper"].edge_index = torch.stack(
        [author_ids, paper_ids]
    )

    paper_ids = torch.arange(paper_count).repeat_interleave(candidate_count)
    paper_slots = torch.arange(candidate_count).repeat(paper_count)
    venue_ids = (paper_ids + paper_slots) % venue_count
    data["paper", "published_in", "venue"].edge_index = torch.stack(
        [paper_ids, venue_ids]
    )
    return data


@pytest.fixture
def heterogeneous_data() -> HeteroData:
    """Return the canonical fully transformed synthetic graph."""
    data = _make_every_relation_choice_rich(
        make_synthetic_heterogeneous_data(seed=7)
    )
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


def _assert_every_relation_exceeds_fanout(
    data: HeteroData,
    fanout: list[int],
) -> None:
    """Require a genuine neighbor choice for every relation at every hop."""
    required_degree = max(fanout)
    for edge_type in data.edge_types:
        destination_type = edge_type[2]
        destination_count = data[destination_type].num_nodes
        assert destination_count is not None
        in_degree = torch.bincount(
            data[edge_type].edge_index[1],
            minlength=destination_count,
        )
        assert torch.all(in_degree > required_degree), (
            f"{edge_type!r} has minimum in-degree "
            f"{int(in_degree.min())}, which does not exceed every fanout "
            f"in {fanout!r}"
        )


def test_sampling_fixture_exceeds_every_relation_fanout(
    heterogeneous_data: HeteroData,
) -> None:
    """Every relation offers more candidates than all configured hop fanouts."""
    _assert_every_relation_exceeds_fanout(
        heterogeneous_data,
        _CHOICE_RICH_FANOUT,
    )


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
            "persistent_workers": False,
            "evaluation_phase": phase,
            "evaluation_phase_seed": datamodule._phase_evaluation_seed(phase),
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


def test_evaluation_settings_descriptor_is_stable_and_complete(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Settings identity is stable, explicit, and not a batch-content claim."""
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

    descriptor = datamodule.evaluation_settings_descriptor("val")
    settings = json.loads(descriptor)

    assert settings["phase"] == "val"
    assert settings["mode"] == "neighbor"
    assert settings["evaluation_protocol"] == "sampled_neighbor_fixed"
    assert settings["evaluation_seed"] == 17
    assert settings["phase_seed"] == datamodule._phase_evaluation_seed("val")
    assert settings["target_node_type"] == "author"
    assert settings["node_types"] == ["author", "paper", "venue"]
    assert settings["edge_types"] == [
        list(edge_type) for edge_type in heterogeneous_spec.edge_types
    ]
    assert settings["node_counts"] == {
        "author": 36,
        "paper": 24,
        "venue": 6,
    }
    assert settings["edge_counts"] == [144, 96, 144, 96]
    assert settings["phase_seed_count"] == int(
        heterogeneous_data["author"].val_mask.sum()
    )
    assert len(settings["phase_seed_ids_sha256"]) == 64
    assert settings["batch_size"] == 4
    assert settings["fanout"] == [1, 1]
    assert settings["replace"] is False
    assert settings["subgraph_type"] == "directional"
    assert settings["filter_per_worker"] is False
    assert settings["evaluation_num_workers"] == 0
    assert settings["evaluation_persistent_workers"] is False
    assert "torch_geometric" in settings["versions"]
    assert "pyg-lib" in settings["versions"]
    assert "torch-sparse" in settings["versions"]
    assert datamodule.evaluation_settings_descriptor("val") == descriptor
    assert datamodule.evaluation_settings_descriptor("test") != descriptor
    assert not hasattr(datamodule, "evaluation_cache_descriptor")


def test_full_graph_evaluation_descriptor_reports_actual_loader_contract(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Full-graph identity excludes inactive neighbor-sampling settings."""
    datamodule = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="full_batch",
        batch_size=17,
        num_neighbors=[3, 2],
        num_workers=2,
        persistent_workers=True,
        replace=True,
        filter_per_worker=True,
        evaluation_seed=71,
    )

    descriptor = datamodule.evaluation_settings_descriptor("test")
    settings = json.loads(descriptor)
    loader = datamodule.test_dataloader()

    assert settings["phase"] == "test"
    assert settings["mode"] == "full_batch"
    assert settings["evaluation_protocol"] == "full_graph"
    assert settings["batch_size"] == loader.batch_size == 1
    assert settings["evaluation_num_workers"] == loader.num_workers == 2
    assert settings["evaluation_persistent_workers"] is True
    assert loader.persistent_workers is True
    for sampled_field in (
        "evaluation_seed",
        "phase_seed",
        "fanout",
        "replace",
        "subgraph_type",
        "filter_per_worker",
    ):
        assert settings[sampled_field] is None
    assert settings["versions"]["torch_geometric"]
    assert settings["versions"]["pyg-lib"] is None
    assert settings["versions"]["torch-sparse"] is None
    assert datamodule.evaluation_settings_descriptor("test") == descriptor
    assert datamodule.evaluation_settings_descriptor("val") != descriptor


def test_evaluation_protocol_changes_settings_identity(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Full-graph and sampled-neighbor evaluation have distinct identities."""
    full_graph = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="full_batch",
    ).evaluation_settings_descriptor("val")
    sampled_neighbor = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="neighbor",
    ).evaluation_settings_descriptor("val")

    assert full_graph != sampled_neighbor


def test_evaluation_settings_descriptor_rejects_train(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Training has no fixed-evaluation settings identity."""
    datamodule = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="neighbor",
    )

    with pytest.raises(ValueError, match="val or test"):
        datamodule.evaluation_settings_descriptor("train")  # type: ignore[arg-type]


def test_evaluation_settings_descriptor_changes_with_runtime_identity(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Graph, split, seed, worker, and fanout changes alter settings identity."""

    def descriptor(
        data: HeteroData,
        *,
        seed: int = 17,
        workers: int = 0,
        fanout: list[int] | None = None,
    ) -> str:
        return HeterogeneousNodeDataModule(
            data,
            heterogeneous_spec,
            mode="neighbor",
            evaluation_seed=seed,
            num_workers=workers,
            num_neighbors=[1, 1] if fanout is None else fanout,
        ).evaluation_settings_descriptor("val")

    baseline = descriptor(heterogeneous_data)
    graph_changed = heterogeneous_data.clone()
    relation = ("author", "writes", "paper")
    graph_changed[relation].edge_index = torch.cat(
        [
            graph_changed[relation].edge_index,
            torch.tensor([[0], [0]], dtype=torch.long),
        ],
        dim=1,
    )
    split_changed = heterogeneous_data.clone()
    val_ids = split_changed["author"].val_mask.nonzero().view(-1)
    test_ids = split_changed["author"].test_mask.nonzero().view(-1)
    split_changed["author"].val_mask[val_ids[0]] = False
    split_changed["author"].test_mask[val_ids[0]] = True
    split_changed["author"].test_mask[test_ids[0]] = False
    split_changed["author"].val_mask[test_ids[0]] = True

    assert descriptor(graph_changed) != baseline
    assert descriptor(split_changed) != baseline
    assert descriptor(heterogeneous_data, seed=18) != baseline
    assert descriptor(heterogeneous_data, workers=2) != baseline
    assert descriptor(heterogeneous_data, fanout=[2, 1]) != baseline


@pytest.mark.parametrize("phase", ["val", "test"])
def test_sampled_evaluation_replays_identical_batches_across_all_traversals(
    phase: str,
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Each seeded streaming traversal is byte-identical without batch caching."""
    datamodule = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="neighbor",
        num_neighbors=_CHOICE_RICH_FANOUT,
        batch_size=4,
        evaluation_seed=23,
    )
    loader_method = getattr(datamodule, f"{phase}_dataloader")
    loader = loader_method()

    first = _loader_signature(loader)
    same_loader_replay = _loader_signature(loader)
    retrieved_loader = loader_method()
    retrieved_loader_replay = _loader_signature(retrieved_loader)

    assert first
    assert same_loader_replay == first
    assert retrieved_loader_replay == first
    assert retrieved_loader is loader
    assert not hasattr(datamodule, "_evaluation_batch_cache")
    assert not any(
        isinstance(value, (list, tuple))
        and any(isinstance(item, HeteroData) for item in value)
        for value in vars(datamodule).values()
    )


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
            num_neighbors=_CHOICE_RICH_FANOUT,
            batch_size=4,
            evaluation_seed=seed,
        )

    first = _loader_signature(make_datamodule(101).val_dataloader())
    same = _loader_signature(make_datamodule(101).val_dataloader())
    different = _loader_signature(make_datamodule(202).val_dataloader())

    assert first == same
    assert first != different


def test_resampled_evaluation_isolated_from_consumer_mutation(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Consumer mutation or transfer cannot alter data or later resampling."""
    datamodule = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="neighbor",
        num_neighbors=_CHOICE_RICH_FANOUT,
        batch_size=4,
        evaluation_seed=31,
    )
    loader = datamodule.val_dataloader()
    canonical_before = _batch_tensor_signature(heterogeneous_data)
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
    assert _batch_tensor_signature(heterogeneous_data) == canonical_before


def test_sampled_evaluation_does_not_perturb_global_torch_rng(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Every streaming traversal preserves the caller RNG state."""
    datamodule = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="neighbor",
        num_neighbors=_CHOICE_RICH_FANOUT,
        batch_size=4,
        evaluation_seed=37,
    )
    loader = datamodule.val_dataloader()

    state_before_traversal = torch.random.get_rng_state().clone()
    list(loader)
    state_after_first_traversal = torch.random.get_rng_state().clone()
    list(loader)
    state_after_second_traversal = torch.random.get_rng_state().clone()

    assert torch.equal(
        state_after_first_traversal,
        state_before_traversal,
    )
    assert torch.equal(state_after_second_traversal, state_before_traversal)


def test_sampled_evaluation_restores_rng_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """A failed sampling stream restores caller RNG before propagating."""
    import topobench.dataloader.heterogeneous as module

    datamodule = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="neighbor",
        num_neighbors=[1, 1],
        evaluation_seed=39,
    )
    loader = datamodule.val_dataloader()
    monkeypatch.setattr(
        module._FixedEvaluationNeighborLoader,
        "_base_iterator",
        MagicMock(side_effect=ImportError("missing backend")),
    )
    state_before = torch.random.get_rng_state().clone()

    with pytest.raises(RuntimeError, match=r"pyg-lib.*torch-sparse"):
        list(loader)

    assert torch.equal(torch.random.get_rng_state(), state_before)


def test_sampled_evaluation_restores_rng_when_iterator_closed_early(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Explicit early close releases the stream and restores caller RNG."""
    datamodule = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="neighbor",
        num_neighbors=[1, 1],
        batch_size=2,
        evaluation_seed=40,
    )
    state_before = torch.random.get_rng_state().clone()
    iterator = iter(datamodule.val_dataloader())

    next(iterator)
    iterator.close()  # type: ignore[attr-defined]

    assert torch.equal(torch.random.get_rng_state(), state_before)


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
    assert datamodule._evaluation_loaders == {}


def test_full_graph_evaluation_does_not_create_sampled_loaders(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Full-graph protocol stays deterministic without sampled loaders."""
    datamodule = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="full_batch",
        evaluation_protocol="full_graph",
        evaluation_seed=43,
    )

    first_loader = datamodule.val_dataloader()
    second_loader = datamodule.val_dataloader()
    first = _loader_signature(first_loader)
    second = _loader_signature(second_loader)

    assert first == second
    assert first_loader is not second_loader
    assert datamodule._evaluation_loaders == {}


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
        "_base_iterator",
        MagicMock(side_effect=ImportError("missing sampling operator")),
    )

    with pytest.raises(
        RuntimeError,
        match=r"pyg-lib.*torch-sparse.*val",
    ):
        list(loader)


def test_fixed_evaluation_streams_with_workers_and_releases_them() -> None:
    """A bounded clean process verifies replay and worker release."""
    verifier = Path(__file__).with_name(
        "verify_fixed_heterogeneous_neighbor_workers.py"
    )

    completed = subprocess.run(
        [sys.executable, str(verifier)],
        cwd=Path(__file__).parents[3],
        check=True,
        capture_output=True,
        text=True,
        # macOS spawn imports PyTorch/PyG separately in both workers.
        timeout=90,
    )

    assert completed.stdout.strip() == "worker-replay-ok"


def test_neighbor_evaluation_memoizes_loaders_and_forces_nonpersistent_workers(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """One sampler exists per eval phase and workers restart every traversal."""
    datamodule = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="neighbor",
        num_neighbors=[1, 1],
        batch_size=4,
        num_workers=2,
        persistent_workers=True,
        evaluation_seed=61,
    )

    train = datamodule.train_dataloader()
    first_val = datamodule.val_dataloader()
    second_val = datamodule.val_dataloader()
    first_test = datamodule.test_dataloader()
    second_test = datamodule.test_dataloader()

    assert train.persistent_workers is True
    assert first_val is second_val
    assert first_test is second_test
    assert first_val is not first_test
    assert first_val.persistent_workers is False
    assert first_test.persistent_workers is False
    assert datamodule._evaluation_loaders == {
        "val": first_val,
        "test": first_test,
    }


def test_large_many_batch_evaluation_streams_before_sampler_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First yield is lazy and neither module nor loader retains batch lists."""
    data = make_synthetic_heterogeneous_data(
        seed=7,
        num_authors=360,
        num_papers=240,
        num_venues=12,
    )
    data = HeterogeneousConstantFeatures(node_types="venue")(data)
    data = HeterogeneousToUndirected(merge=False)(data)
    spec = validate_heterogeneous_node_data(
        data,
        target_node_type="author",
        num_classes=2,
    )
    datamodule = HeterogeneousNodeDataModule(
        data,
        spec,
        mode="neighbor",
        num_neighbors=[1, 1],
        batch_size=2,
        evaluation_seed=67,
    )
    loader = datamodule.val_dataloader()
    original_base_iterator = loader._base_iterator
    yielded = 0
    exhausted = False

    def instrumented_iterator():
        nonlocal yielded, exhausted
        for batch in original_base_iterator():
            yielded += 1
            yield batch
        exhausted = True

    monkeypatch.setattr(loader, "_base_iterator", instrumented_iterator)
    iterator = iter(loader)

    first_batch = next(iterator)

    assert isinstance(first_batch, HeteroData)
    assert yielded == 1
    assert not exhausted
    assert len(loader) > 20
    for owner in (datamodule, loader):
        assert not any(
            isinstance(value, (list, tuple))
            and any(isinstance(item, HeteroData) for item in value)
            for value in vars(owner).values()
        )
    iterator.close()


def test_preflight_reads_each_full_batch_phase_without_mutating_graph(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Task8 probes all active phases through native full-graph loaders."""
    datamodule = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="full_batch",
    )
    canonical_before = _batch_tensor_signature(heterogeneous_data)
    observations = make_observations()

    result = run_probe(datamodule, observations)

    assert result.passed
    assert [
        event.removeprefix("forward:")
        for event in observations["events"]
        if event.startswith("forward:")
    ] == ["train", "val", "test"]
    assert _batch_tensor_signature(heterogeneous_data) == canonical_before


def test_preflight_preserves_first_canonical_neighbor_batch_and_rng(
    heterogeneous_data: HeteroData,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Task8 cannot advance the first shuffled production seed descriptor."""
    control = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="neighbor",
        num_neighbors=[3, 2],
        batch_size=4,
        train_shuffle=True,
        evaluation_seed=73,
    )
    probed = HeterogeneousNodeDataModule(
        heterogeneous_data,
        heterogeneous_spec,
        mode="neighbor",
        num_neighbors=[3, 2],
        batch_size=4,
        train_shuffle=True,
        evaluation_seed=73,
    )

    torch.manual_seed(448)
    expected = _batch_tensor_signature(next(iter(control.train_dataloader())))
    torch.manual_seed(448)
    rng_before = torch.random.get_rng_state().clone()
    descriptors_before = {
        phase: probed.evaluation_settings_descriptor(phase)
        for phase in ("val", "test")
    }

    result = run_probe(probed, make_observations())

    assert result.passed
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    actual = _batch_tensor_signature(next(iter(probed.train_dataloader())))
    assert actual == expected
    assert {
        phase: probed.evaluation_settings_descriptor(phase)
        for phase in ("val", "test")
    } == descriptors_before
