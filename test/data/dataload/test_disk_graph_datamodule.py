"""Shared Task8 fixtures and strategy-driven data-module lifecycle tests."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch_geometric.data import Data, HeteroData

from test.data.stores.test_topology_only_pyg_partitioner import (
    asymmetric_typed_source,
    homogeneous_source,
)
from test.data.stores.test_typed_graph_store import (
    QualifiedStoreFixture,
    _build_qualified_store,
)
from topobench.data.loaders.parquet import ParquetTypedGraphSource
from topobench.data.stores.typed_graph_store import TypedGraphStore
from topobench.dataloader.disk_graph import (
    DiskGraphDataModule,
    HeterogeneousNeighborStrategy,
    HomogeneousClusterStrategy,
    SamplingDescriptor,
)


def _tensor(value: np.ndarray) -> torch.Tensor:
    """Copy a read-only fixture mmap into a writable reference tensor."""
    return torch.from_numpy(np.array(value, copy=True))


@pytest.fixture(scope="session")
def task8_stores(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, QualifiedStoreFixture]:
    """Build three-way Task7 stores through the real qualified helper."""
    root = tmp_path_factory.mktemp("task8-qualified-stores")
    heterogeneous = _build_qualified_store(
        asymmetric_typed_source(
            root / "heterogeneous-source",
            num_partitions=3,
            memory_limit_bytes=1,
            external_partition_map="external/manifest.json",
        ),
        root / "heterogeneous-stores",
    )
    homogeneous_value = homogeneous_source(root / "homogeneous-source")
    homogeneous_value = ParquetTypedGraphSource(
        replace(
            homogeneous_value.spec,
            partition=replace(
                homogeneous_value.spec.partition,
                num_partitions=3,
                memory_limit_bytes=1,
                external_partition_map="external/manifest.json",
            ),
        )
    )
    homogeneous = _build_qualified_store(
        homogeneous_value,
        root / "homogeneous-stores",
    )
    return {"heterogeneous": heterogeneous, "homogeneous": homogeneous}


def materialized_heterogeneous_reference(store: TypedGraphStore) -> HeteroData:
    """Construct the admitted in-memory oracle from a tiny real Task7 store."""
    data = HeteroData()
    for node_type in store.node_types:
        node_record = store._node(node_type)
        data[node_type].x = _tensor(store.node_features(node_type))
        if node_record["y"] is not None:
            data[node_type].y = _tensor(store.node_labels(node_type))
        for field_name in node_record["fields"]:
            data[node_type][field_name] = _tensor(
                store.node_array(node_type, field_name)
            )
        data[node_type].n_id = torch.arange(
            node_record["count"], dtype=torch.long
        )
        data[node_type].num_nodes = node_record["count"]

    for relation in store.relation_types:
        row, colptr = store.relation_csc(relation)
        destination = np.repeat(
            np.arange(len(colptr) - 1, dtype=np.int64),
            np.diff(colptr),
        )
        data[relation].edge_index = torch.stack((_tensor(row), _tensor(destination)))
        data[relation].edge_id = _tensor(
            store.relation_field(relation, "edge_id")
        )
        for field_name in store._relation(relation)["fields"]:
            data[relation][field_name] = _tensor(
                store.relation_field(relation, field_name)
            )

    target_type = store._manifest["target_node_type"]
    for tag, split in store._manifest["splits"].items():
        for phase in ("train", "val", "test"):
            mask = torch.zeros(data[target_type].num_nodes, dtype=torch.bool)
            mask[_tensor(store.split_ids(tag, phase))] = True
            data[target_type][f"{tag}_{phase}_mask"] = mask
            if tag == store.active_split_tag:
                data[target_type][f"{phase}_mask"] = mask.clone()

    data.content_sha256 = store.content_sha256
    data.active_split_tag = store.active_split_tag
    data.target_node_type = target_type
    data.partition_book_identity = store.partition_book_identity
    return data


def materialized_homogeneous_reference(store: TypedGraphStore) -> Data:
    """Construct the homogeneous in-memory oracle from a tiny real store."""
    node_type = store.node_types[0]
    relation = store.relation_types[0]
    row, colptr = store.relation_csc(relation)
    destination = np.repeat(
        np.arange(len(colptr) - 1, dtype=np.int64),
        np.diff(colptr),
    )
    data = Data(
        x=_tensor(store.node_features(node_type)),
        y=_tensor(store.node_labels(node_type)),
        edge_index=torch.stack((_tensor(row), _tensor(destination))),
        num_nodes=len(store.partition_assignment(node_type)),
    )
    if store._relation(relation)["edge_id"] is not None:
        data.edge_id = _tensor(store.relation_field(relation, "edge_id"))
    for field_name in store._node(node_type)["fields"]:
        data[field_name] = _tensor(store.node_array(node_type, field_name))
    for field_name in store._relation(relation)["fields"]:
        data[field_name] = _tensor(store.relation_field(relation, field_name))
    for phase in ("train", "val", "test"):
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[_tensor(store.split_ids(store.active_split_tag, phase))] = True
        data[f"{phase}_mask"] = mask
    data.content_sha256 = store.content_sha256
    data.active_split_tag = store.active_split_tag
    return data


def assert_heterogeneous_exact(
    actual: HeteroData,
    expected: HeteroData,
    *,
    ignored_global: set[str] | None = None,
) -> None:
    """Assert ordered native tensor parity for every node and relation field."""
    assert tuple(actual.node_types) == tuple(expected.node_types)
    assert tuple(actual.edge_types) == tuple(expected.edge_types)
    for node_type in expected.node_types:
        for field_name, expected_value in expected[node_type].items():
            if isinstance(expected_value, torch.Tensor):
                assert field_name in actual[node_type]
                assert torch.equal(actual[node_type][field_name], expected_value), (
                    node_type,
                    field_name,
                )
    for relation in expected.edge_types:
        for field_name, expected_value in expected[relation].items():
            if isinstance(expected_value, torch.Tensor):
                assert field_name in actual[relation]
                assert torch.equal(actual[relation][field_name], expected_value), (
                    relation,
                    field_name,
                )
    ignored = set() if ignored_global is None else ignored_global
    for field_name, expected_value in expected._global_store.items():
        if field_name in ignored or not isinstance(expected_value, torch.Tensor):
            continue
        assert field_name in actual._global_store
        assert torch.equal(actual[field_name], expected_value)


def exhaustive_fanout(
    relations: tuple[tuple[str, str, str], ...],
) -> dict[tuple[str, str, str], list[int]]:
    """Return a deterministic one-hop exhaustive fanout for every relation."""
    return {relation: [-1] for relation in relations}


def test_disk_graph_datamodule_runs_multiworker_final_batch_and_closes(
    task8_stores: dict[str, QualifiedStoreFixture],
) -> None:
    """One module owns ordered descriptors and lazily opened worker stores."""
    fixture = task8_stores["heterogeneous"]
    with TypedGraphStore.open(fixture.store_build.path) as store:
        target_type = store._manifest["target_node_type"]
        relations = store.relation_types
        expected_train_ids = tuple(
            int(value)
            for value in store.split_ids(store.active_split_tag, "train")
        )

    strategy = HeterogeneousNeighborStrategy(
        batch_size=1,
        num_neighbors=exhaustive_fanout(relations),
        seed=37,
    )
    module = DiskGraphDataModule(
        fixture.store_build.path,
        strategy,
        num_workers=2,
        persistent_workers=False,
        train_shuffle=False,
    )
    assert module._owner is not None
    assert module._owner._store is None
    module.setup("fit")
    assert module._owner._store is None
    descriptors = module.descriptors("train")
    assert tuple(d.target_seed_ids for d in descriptors) == tuple(
        (seed_id,) for seed_id in expected_train_ids
    )

    batches = list(module.train_dataloader())
    assert [batch[target_type].batch_size for batch in batches] == [1, 1]
    assert [int(batch[target_type].n_id[0]) for batch in batches] == list(
        expected_train_ids
    )
    assert all(batch.sampling_descriptor.phase == "train" for batch in batches)
    assert all(
        tensor.device.type == "cpu"
        for batch in batches
        for store in batch.stores
        for tensor in store.values()
        if isinstance(tensor, torch.Tensor)
    )
    assert module._owner._store is None

    module.teardown("fit")
    assert module.closed

def test_materialized_module_multiworker_uses_admitted_snapshot(
    task8_stores: dict[str, QualifiedStoreFixture],
) -> None:
    """Spawned workers sample the admitted snapshot, not later caller mutations."""
    with TypedGraphStore.open(
        task8_stores["heterogeneous"].store_build.path
    ) as store:
        data = materialized_heterogeneous_reference(store)
        target_type = store._manifest["target_node_type"]
        fanout = exhaustive_fanout(store.relation_types)
    strategy = HeterogeneousNeighborStrategy(
        batch_size=1,
        num_neighbors=fanout,
        seed=39,
    )
    module = DiskGraphDataModule(
        data,
        strategy,
        num_workers=2,
        persistent_workers=False,
        train_shuffle=False,
    )
    descriptor = module.descriptors("train")[0]
    expected = strategy.materialize(data, descriptor)
    data[target_type].x.add_(1000)

    actual = next(iter(module.train_dataloader()))

    assert_heterogeneous_exact(actual, expected)
    assert actual.sampling_descriptor == expected.sampling_descriptor
    module.close()


    with pytest.raises(RuntimeError, match="closed"):
        module.train_dataloader()


def test_stage_setup_is_lazy_and_does_not_touch_unrequested_empty_phases() -> None:
    """Stage setup materializes only its phases and reports requested emptiness."""
    relation = ("paper", "written_by", "author")
    data = HeteroData()
    data["paper"].x = torch.ones((1, 1))
    data["author"].x = torch.ones((2, 1))
    data["author"].y = torch.tensor([0, 1])
    data["author"].train_mask = torch.tensor([True, False])
    data["author"].val_mask = torch.tensor([False, False])
    data["author"].test_mask = torch.tensor([False, True])
    data[relation].edge_index = torch.empty((2, 0), dtype=torch.long)
    data.active_split_tag = "primary"
    data.target_node_type = "author"
    strategy = HeterogeneousNeighborStrategy(
        batch_size=1,
        num_neighbors={relation: [-1]},
    )
    module = DiskGraphDataModule(data, strategy, train_shuffle=False)

    module.setup("test")
    assert tuple(module._descriptors) == ("test",)
    assert module.descriptors("test")[0].target_seed_ids == (1,)
    with pytest.raises(ValueError, match="val phase setup failed.*no target"):
        module.val_dataloader()

    prediction = DiskGraphDataModule(data, strategy, train_shuffle=False)
    prediction.setup("predict")
    assert tuple(prediction._descriptors) == ("test",)
    assert prediction.predict_dataloader() is prediction.test_dataloader()
    module.close()
    prediction.close()




def test_descriptor_and_module_validation_are_contextual_and_exact(
    task8_stores: dict[str, QualifiedStoreFixture],
) -> None:
    """Invalid identities and incompatible strategy/store capabilities fail early."""
    common: dict[str, Any] = {
        "content_sha256": "a" * 64,
        "active_split_tag": "primary",
        "phase": "train",
        "strategy": "homogeneous-cluster",
        "strategy_options_json": "{}",
        "batch_ordinal": 0,
        "partition_ids": (0,),
        "participant_counts": (("node", 1),),
        "generator_seed": 1,
        "generator_state_sha256": "b" * 64,
    }
    with pytest.raises(TypeError, match="batch_ordinal"):
        SamplingDescriptor(**(common | {"batch_ordinal": True}))
    with pytest.raises(ValueError, match="duplicate partition"):
        SamplingDescriptor(**(common | {"partition_ids": (0, 0)}))
    with pytest.raises(ValueError, match="cannot bind both"):
        SamplingDescriptor(
            **(
                common
                | {
                    "target_node_type": "author",
                    "target_seed_ids": (1,),
                }
            )
        )
    canonical = SamplingDescriptor(
        **(
            common
            | {
                "partition_ids": [np.int64(2), np.int32(0)],
                "participant_counts": [["zeta", np.int64(1)], ("alpha", 2)],
            }
        )
    )
    assert canonical.partition_ids == (2, 0)
    assert canonical.participant_counts == (("alpha", 2), ("zeta", 1))
    assert isinstance(hash(canonical), int)
    seed_descriptor = SamplingDescriptor(
        **(
            common
            | {
                "partition_ids": [],
                "target_node_type": "author",
                "target_seed_ids": [np.int64(3), np.int32(1)],
            }
        )
    )
    assert seed_descriptor.target_seed_ids == (3, 1)
    with pytest.raises(TypeError, match="participant_counts"):
        SamplingDescriptor(
            **(common | {"participant_counts": (("node", 1, 2),)})
        )
    with pytest.raises(ValueError, match="duplicate participant"):
        SamplingDescriptor(
            **(
                common
                | {"participant_counts": (("node", 1), ("node", 2))}
            )
        )


    heterogeneous_path = task8_stores["heterogeneous"].store_build.path
    with pytest.raises(ValueError, match="homogeneous-cluster.*heterogeneous"):
        DiskGraphDataModule(
            heterogeneous_path,
            HomogeneousClusterStrategy(clusters_per_batch=1),
        )
    with TypedGraphStore.open(heterogeneous_path) as store:
        fanout = {relation: [-1] for relation in store.relation_types}
    with pytest.raises(ValueError, match="active split tag.*diagnostic"):
        DiskGraphDataModule(
            heterogeneous_path,
            HeterogeneousNeighborStrategy(
                batch_size=1,
                num_neighbors=fanout,
            ),
            active_split_tag="diagnostic",
        )


def test_store_move_and_reload_keep_descriptors_identical(
    task8_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
) -> None:
    """Descriptor identity binds content, never an installation path."""
    original = task8_stores["homogeneous"].store_build.path
    moved = tmp_path / original.name
    shutil.copytree(original, moved)
    strategy = HomogeneousClusterStrategy(clusters_per_batch=1, seed=11)

    first = DiskGraphDataModule(original, strategy, train_shuffle=False)
    second = DiskGraphDataModule(moved, strategy, train_shuffle=False)
    first.setup("fit")
    second.setup("fit")
    assert first.descriptors("train") == second.descriptors("train")
    first.close()
    second.close()
