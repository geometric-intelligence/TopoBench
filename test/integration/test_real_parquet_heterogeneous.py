"""Download-gated real multi-file heterogeneous Parquet qualification."""

from __future__ import annotations

import json
import logging
import math
import os
import stat
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from topobench.data.loaders.parquet import ParquetTypedGraphLoader
from topobench.data.stores.store_bundle import StoreBundle
from topobench.data.stores.typed_graph_ingestion import (
    ParquetTypedGraphIngestor,
)
from topobench.data.stores.typed_graph_store import (
    TypedGraphStore,
    TypedGraphStoreWriter,
)
from topobench.data.stores.typed_partition_book import (
    PartitionQualificationLimits,
)
from topobench.dataloader.disk_graph import (
    DiskGraphDataModule,
    HeterogeneousClusterStrategy,
    HeterogeneousNeighborStrategy,
)
from topobench.nn.backbones.heterogeneous.hgt import HGTBackbone
from topobench.transforms.incremental_pca import IncrementalPCATransform

pytestmark = [pytest.mark.integration, pytest.mark.download]

_LIVE_FLAG = "TOPOBENCH_RUN_LIVE_PARQUET"
_URL = "TOPOBENCH_REAL_PARQUET_HETEROGENEOUS_URL"
_SHA256 = "TOPOBENCH_REAL_PARQUET_HETEROGENEOUS_SHA256"
_MAX_DOWNLOAD_BYTES = int(
    os.environ.get("TOPOBENCH_REAL_PARQUET_MAX_BYTES", str(8 * 1024**3))
)
_MAX_MEMBERS = 20_000
_CONTRACT_NAME = "qualification.json"
_CONTRACT_VERSION = "topobench-real-parquet-qualification-v1"
_NUM_CLASSES = 4


class _HGTStepModel(torch.nn.Module):
    def __init__(self, sample: HeteroData, target: str) -> None:
        super().__init__()
        self.target = target
        self.project = torch.nn.ModuleDict(
            {
                node_type: torch.nn.Linear(int(sample[node_type].x.shape[1]), 32)
                for node_type in sample.node_types
            }
        )
        self.backbone = HGTBackbone(
            sample.metadata(),
            hidden_channels=32,
            num_layers=2,
            heads=2,
            dropout=0.0,
            activation="relu",
        )
        self.head = torch.nn.Linear(32, _NUM_CLASSES)

    def forward(self, batch: HeteroData) -> Tensor:
        encoded = {
            node_type: self.project[node_type](batch[node_type].x.float())
            for node_type in batch.node_types
        }
        hidden = self.backbone(encoded, batch.edge_index_dict)
        return self.head(hidden[self.target])


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        pytest.fail(f"live heterogeneous contract {name} must be a mapping")
    return value


def _require_live_bundle(tmp_path: Path) -> tuple[Path, Mapping[str, Any]]:
    if os.environ.get(_LIVE_FLAG) != "1":
        pytest.skip(
            f"live Parquet download disabled; set {_LIVE_FLAG}=1 with digest-pinned URLs"
        )
    url = os.environ.get(_URL)
    digest = os.environ.get(_SHA256)
    if not url or not digest:
        pytest.fail(f"live release gate requires both {_URL} and {_SHA256}")
    archive = StoreBundle.download(
        url,
        tmp_path / "real-heterogeneous.zip",
        expected_sha256=digest,
        max_bytes=_MAX_DOWNLOAD_BYTES,
        timeout_seconds=120.0,
    )
    extracted = tmp_path / "download"
    extracted.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if not members or len(members) > _MAX_MEMBERS:
            pytest.fail("live heterogeneous bundle member count is outside its bound")
        total = 0
        for member in members:
            relative = PurePosixPath(member.filename)
            if (
                relative.is_absolute()
                or not relative.parts
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                pytest.fail(f"unsafe live heterogeneous member path {member.filename!r}")
            if stat.S_ISLNK(member.external_attr >> 16):
                pytest.fail(
                    f"live heterogeneous bundle contains symlink {member.filename!r}"
                )
            total += member.file_size
            if total > _MAX_DOWNLOAD_BYTES:
                pytest.fail("live heterogeneous bundle exceeds its byte bound")
            destination = extracted.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if member.is_dir():
                destination.mkdir(exist_ok=True)
                continue
            with bundle.open(member) as source, destination.open("xb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
    contract_path = extracted / _CONTRACT_NAME
    if not contract_path.is_file():
        pytest.fail(f"live heterogeneous bundle is missing {_CONTRACT_NAME}")
    contract = _mapping(json.loads(contract_path.read_text(encoding="utf-8")), "root")
    if contract.get("format_version") != _CONTRACT_VERSION:
        pytest.fail("live heterogeneous qualification contract version is unsupported")
    if contract.get("output_kind") != "heterogeneous":
        pytest.fail("live heterogeneous contract must declare heterogeneous output")
    return extracted, contract


def _source_from_contract(root: Path, contract: Mapping[str, Any]) -> Any:
    parameters = dict(_mapping(contract.get("loader_parameters"), "loader_parameters"))
    source_subdirectory = contract.get("source_subdirectory", "source")
    if not isinstance(source_subdirectory, str) or not source_subdirectory:
        pytest.fail("source_subdirectory must be a non-empty relative path")
    relative = PurePosixPath(source_subdirectory)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        pytest.fail("source_subdirectory escapes the downloaded bundle")
    parameters["source_root"] = str(root.joinpath(*relative.parts))
    parameters["data_domain"] = "heterogeneous"
    parameters["data_type"] = "parquet_typed"
    parameters["data_name"] = "ParquetTypedGraph"
    parameters["output_kind"] = "heterogeneous"
    return ParquetTypedGraphLoader(parameters).source


def _assert_source_contract(source: Any, contract: Mapping[str, Any]) -> None:
    expectations = _mapping(contract["expectations"], "expectations")
    spec = source.spec
    assert spec.output_kind == "heterogeneous"
    assert spec.reproducibility.save_reproducibility_bundle is True
    if len(spec.node_types) < 2 or len(spec.relations) < 2:
        pytest.fail("representative heterogeneous source needs multiple types/relations")
    if len(spec.files) < 10:
        pytest.fail("representative heterogeneous source needs multiple Parquet files")
    if len(spec.supervision.split_registry.sets) < 2:
        pytest.fail("representative heterogeneous source needs several split triplets")
    assert tuple(node.name for node in spec.node_types) == tuple(
        expectations["node_types"]
    )
    declared_relations = tuple(relation.relation for relation in spec.relations)
    expected_relations = tuple(tuple(value) for value in expectations["relations"])
    assert declared_relations == expected_relations
    assert len(set(declared_relations)) == len(declared_relations)
    roles = _mapping(expectations["semantic_roles"], "semantic_roles")
    assert roles["target_node_type"] == spec.supervision.target_node_type
    assert roles["label_column"] == spec.supervision.label_column
    assert roles["label_dtype"] == spec.supervision.label_dtype
    assert expectations["num_classes"] == _NUM_CLASSES
    widths = {node.name: node.feature_width for node in spec.node_types}
    assert widths == expectations["feature_widths"]
    assert len(set(widths.values())) > 1
    assert all(len(node.paths) >= 2 for node in spec.node_types)


def _assert_store_semantics(
    store: TypedGraphStore,
    contract: Mapping[str, Any],
) -> tuple[str, ...]:
    expectations = _mapping(contract["expectations"], "expectations")
    node_counts = _mapping(expectations["node_counts"], "node_counts")
    relation_counts = _mapping(expectations["relation_counts"], "relation_counts")
    assert tuple(store.node_types) == tuple(expectations["node_types"])
    assert tuple(store.relation_types) == tuple(
        tuple(value) for value in expectations["relations"]
    )
    for node_type, count in node_counts.items():
        assert store._node(node_type)["count"] == count
    for relation_text, count in relation_counts.items():
        relation = tuple(relation_text.split("|"))
        row, colptr = store.relation_csc(relation)
        assert len(row) == count
        assert len(colptr) == node_counts[relation[2]] + 1
    relation_samples = _mapping(
        expectations["relation_samples"],
        "relation_samples",
    )
    assert set(relation_samples) == set(relation_counts)
    for relation_text, rows_value in relation_samples.items():
        if not isinstance(rows_value, list) or not rows_value:
            pytest.fail(f"relation_samples.{relation_text} must be non-empty")
        relation = tuple(relation_text.split("|"))
        row, colptr = store.relation_csc(relation)
        for sample in rows_value:
            source = int(sample["source_internal_id"])
            destination = int(sample["destination_internal_id"])
            start, stop = int(colptr[destination]), int(colptr[destination + 1])
            assert source in row[start:stop].tolist()

    split_counts = _mapping(expectations["split_counts"], "split_counts")
    assert tuple(store._manifest["splits"]) == tuple(split_counts)
    target = store._manifest["target_node_type"]
    assert target == expectations["semantic_roles"]["target_node_type"]
    for tag, phases_value in split_counts.items():
        phases = _mapping(phases_value, f"split_counts.{tag}")
        for phase in ("train", "val", "test"):
            ids = store.split_ids(tag, phase)
            assert len(ids) == phases[phase]
            assert len(np.unique(ids)) == len(ids)

    samples = _mapping(expectations["external_id_samples"], "external_id_samples")
    raw_sentinels: list[str] = []
    for node_type in store.node_types:
        rows_value = samples[node_type]
        if not isinstance(rows_value, list) or not rows_value:
            pytest.fail(f"external_id_samples.{node_type} must be non-empty")
        internal = np.array([int(row["internal_id"]) for row in rows_value], dtype=np.int64)
        expected = [row["external_id"] for row in rows_value]
        assert store.external_ids(node_type, internal) == expected
        raw_sentinels.extend(str(value) for value in expected)
        selected = store.node_features(node_type, internal)
        assert selected.shape == (len(internal), store._node(node_type)["feature_width"])
        assert np.isfinite(selected).all() and selected.flags.writeable is False
        if node_type == target:
            labels = store.node_labels(node_type, internal)
            assert labels.dtype.kind in "iu"
            assert int(labels.min()) >= 0
            assert int(labels.max()) < _NUM_CLASSES
    if not any(len(value) >= 8 for value in raw_sentinels):
        pytest.fail("live heterogeneous contract needs a distinctive raw-ID sentinel")
    return tuple(raw_sentinels)


def _finite_hgt_step(batch: HeteroData, target: str) -> tuple[float, float]:
    torch.manual_seed(20260802)
    model = _HGTStepModel(batch, target)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch)
    mask = batch[target].supervised_mask
    assert bool(mask.any())
    labels = batch[target].y[mask].long()
    loss = torch.nn.functional.cross_entropy(logits[mask], labels)
    loss.backward()
    optimizer.step()
    accuracy = float((logits[mask].argmax(dim=-1) == labels).float().mean())
    assert math.isfinite(float(loss.detach())) and math.isfinite(accuracy)
    return float(loss.detach()), accuracy


def _pca(
    contract: Mapping[str, Any],
    *,
    target: str,
) -> IncrementalPCATransform:
    pca_contract = _mapping(contract["expectations"]["pca"], "pca")
    return IncrementalPCATransform(
        n_components=int(pca_contract["components"]),
        max_batch_rows=int(pca_contract["max_batch_rows"]),
        max_batch_bytes=int(pca_contract["max_batch_bytes"]),
        target_node_type=target,
        input_dtype=str(pca_contract["input_dtype"]),
        output_dtype="float32",
    )


def _write_aggregate_evidence(
    store: TypedGraphStore,
    source: Any,
    contract: Mapping[str, Any],
    *,
    pca_state_key: str,
    cluster_result: tuple[float, float],
    neighbor_result: tuple[float, float],
    cluster_state: Mapping[str, object],
    neighbor_state: Mapping[str, object],
) -> None:
    evidence_root_value = os.environ.get("TOPOBENCH_QUALIFICATION_EVIDENCE_DIR")
    if not evidence_root_value:
        return
    evidence_root = Path(evidence_root_value)
    evidence_root.mkdir(parents=True, exist_ok=True)
    expectations = _mapping(contract["expectations"], "expectations")
    evidence = {
        "schema_version": "real-parquet-qualification-aggregate-v1",
        "status": "passed",
        "dataset": contract["dataset_name"],
        "output_kind": "heterogeneous",
        "source_fingerprint": store._manifest["source_binding"]["source_fingerprint"],
        "store_fingerprint": store.content_sha256,
        "partition_book_identity": store.partition_book_identity,
        "schema_roles": expectations["semantic_roles"],
        "representation": "typed-csc-mmap",
        "feature_widths": expectations["feature_widths"],
        "node_counts": expectations["node_counts"],
        "relation_counts": expectations["relation_counts"],
        "split_counts": expectations["split_counts"],
        "validated_cache_hit": True,
        "fitted_transform_state_key": pca_state_key,
        "strategies": {
            "cluster": {
                "state": dict(cluster_state),
                "loss": cluster_result[0],
                "accuracy": cluster_result[1],
            },
            "neighbor": {
                "state": dict(neighbor_state),
                "loss": neighbor_result[0],
                "accuracy": neighbor_result[1],
            },
        },
        "source_file_count": len(source.files),
    }
    (evidence_root / "real-parquet-heterogeneous.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_real_multifile_parquet_heterogeneous_lifecycle(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Qualify typed roles, PCA, HGT cluster/neighbor, replay, and redaction."""
    caplog.set_level(logging.INFO)
    root, contract = _require_live_bundle(tmp_path)
    source = _source_from_contract(root, contract)
    _assert_source_contract(source, contract)

    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores", threads=1)
    partition = ingestor.build_partitions(limits=PartitionQualificationLimits())
    fresh = TypedGraphStoreWriter(ingestor, partition).build()
    assert fresh.cache_hit is False
    raw_ids = _assert_store_semantics(fresh.store, contract)
    target = fresh.store._manifest["target_node_type"]
    pca_contract = _mapping(contract["expectations"]["pca"], "pca")
    state_root = tmp_path / "fitted-state"

    cluster_strategy = HeterogeneousClusterStrategy(
        clusters_per_batch=1,
        seed=20260802,
    )
    cluster_pca = _pca(contract, target=target)
    cluster_module = DiskGraphDataModule(
        fresh.path,
        cluster_strategy,
        active_split_tag=source.spec.supervision.split_registry.active_tag,
        train_shuffle=False,
        fitted_transform=cluster_pca,
        fitted_state_root=state_root,
    )
    cluster_module.setup("fit")
    cluster_batch = next(iter(cluster_module.train_dataloader()))
    assert isinstance(cluster_batch, HeteroData)
    assert cluster_batch[target].x.shape[1] == pca_contract["components"]
    assert cluster_pca.state_key is not None
    cluster_result = _finite_hgt_step(cluster_batch, target)
    cluster_module.close()

    with TypedGraphStore.open(fresh.path) as store:
        fanout = {relation: [-1, -1] for relation in store.relation_types}
    neighbor_strategy = HeterogeneousNeighborStrategy(
        batch_size=int(pca_contract["neighbor_batch_size"]),
        num_neighbors=fanout,
        seed=20260802,
    )
    neighbor_pca = _pca(contract, target=target)
    neighbor_module = DiskGraphDataModule(
        fresh.path,
        neighbor_strategy,
        active_split_tag=source.spec.supervision.split_registry.active_tag,
        train_shuffle=False,
        fitted_transform=neighbor_pca,
        fitted_state_root=state_root,
    )
    neighbor_module.setup("fit")
    neighbor_batch = next(iter(neighbor_module.train_dataloader()))
    assert isinstance(neighbor_batch, HeteroData)
    assert neighbor_batch[target].x.shape[1] == pca_contract["components"]
    assert neighbor_pca.state_key == cluster_pca.state_key
    neighbor_result = _finite_hgt_step(neighbor_batch, target)
    neighbor_module.close()

    replay_partition = ingestor.build_partitions(limits=PartitionQualificationLimits())
    replay = TypedGraphStoreWriter(ingestor, replay_partition).build()
    assert replay.cache_hit is True
    assert replay.content_sha256 == fresh.content_sha256
    state = replay.store.state()
    replay.store.close()
    with TypedGraphStore.from_state(state) as reopened:
        _assert_store_semantics(reopened, contract)
        assert reopened.partition_book_identity == partition.book.content_identity
        _write_aggregate_evidence(
            reopened,
            source,
            contract,
            pca_state_key=cluster_pca.state_key,
            cluster_result=cluster_result,
            neighbor_result=neighbor_result,
            cluster_state={
                **cluster_strategy.sampler_state(),
                "clusters_per_batch": 1,
            },
            neighbor_state={
                **neighbor_strategy.sampler_state(),
                "batch_size": int(pca_contract["neighbor_batch_size"]),
                "fanout": [
                    {
                        "relation": list(relation),
                        "values": values,
                    }
                    for relation, values in fanout.items()
                ],
            },
        )
    fresh.store.close()

    captured = capsys.readouterr()
    emitted = captured.out + captured.err + "\n".join(
        record.getMessage() for record in caplog.records
    )
    for raw_id in raw_ids:
        assert raw_id not in emitted
