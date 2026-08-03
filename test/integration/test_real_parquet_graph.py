"""Download-gated real multi-file homogeneous Parquet qualification."""

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
from torch_geometric.data import Data
from torch_geometric.nn.models import GCN

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
    HomogeneousClusterStrategy,
)
from topobench.transforms.incremental_pca import IncrementalPCATransform

pytestmark = [pytest.mark.integration, pytest.mark.download]

_LIVE_FLAG = "TOPOBENCH_RUN_LIVE_PARQUET"
_URL = "TOPOBENCH_REAL_PARQUET_GRAPH_URL"
_SHA256 = "TOPOBENCH_REAL_PARQUET_GRAPH_SHA256"
_MAX_DOWNLOAD_BYTES = int(
    os.environ.get("TOPOBENCH_REAL_PARQUET_MAX_BYTES", str(8 * 1024**3))
)
_MAX_MEMBERS = 10_000
_CONTRACT_NAME = "qualification.json"
_CONTRACT_VERSION = "topobench-real-parquet-qualification-v1"
NUM_CLASSES = 4


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        pytest.fail(f"live Parquet contract {name} must be a mapping")
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
        tmp_path / "real-graph.zip",
        expected_sha256=digest,
        max_bytes=_MAX_DOWNLOAD_BYTES,
        timeout_seconds=120.0,
    )
    extracted = tmp_path / "download"
    extracted.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if not members or len(members) > _MAX_MEMBERS:
            pytest.fail(
                "live graph bundle member count is outside the declared bound"
            )
        total = 0
        for member in members:
            relative = PurePosixPath(member.filename)
            if (
                relative.is_absolute()
                or not relative.parts
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                pytest.fail(
                    f"unsafe live graph member path {member.filename!r}"
                )
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                pytest.fail(
                    f"live graph bundle contains symlink {member.filename!r}"
                )
            total += member.file_size
            if total > _MAX_DOWNLOAD_BYTES:
                pytest.fail(
                    "live graph uncompressed bundle exceeds its byte bound"
                )
            destination = extracted.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if member.is_dir():
                destination.mkdir(exist_ok=True)
                continue
            with (
                bundle.open(member) as source,
                destination.open("xb") as target,
            ):
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
    contract_path = extracted / _CONTRACT_NAME
    if not contract_path.is_file():
        pytest.fail(f"live graph bundle is missing {_CONTRACT_NAME}")
    contract = _mapping(
        json.loads(contract_path.read_text(encoding="utf-8")), "root"
    )
    if contract.get("format_version") != _CONTRACT_VERSION:
        pytest.fail("live graph qualification contract version is unsupported")
    if contract.get("output_kind") != "homogeneous":
        pytest.fail(
            "live graph qualification contract must declare homogeneous output"
        )
    return extracted, contract


def _source_from_contract(root: Path, contract: Mapping[str, Any]) -> Any:
    parameters = dict(
        _mapping(contract.get("loader_parameters"), "loader_parameters")
    )
    source_subdirectory = contract.get("source_subdirectory", "source")
    if not isinstance(source_subdirectory, str) or not source_subdirectory:
        pytest.fail("source_subdirectory must be a non-empty relative path")
    relative = PurePosixPath(source_subdirectory)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        pytest.fail("source_subdirectory escapes the downloaded bundle")
    parameters["source_root"] = str(root.joinpath(*relative.parts))
    parameters["data_domain"] = "graph"
    parameters["data_type"] = "parquet_typed"
    parameters["data_name"] = "ParquetTypedGraph"
    parameters["output_kind"] = "homogeneous"
    return ParquetTypedGraphLoader(parameters).source


def _assert_source_contract(source: Any, contract: Mapping[str, Any]) -> None:
    expectations = _mapping(contract.get("expectations"), "expectations")
    spec = source.spec
    if len(spec.files) < 6:
        pytest.fail(
            "representative live graph source must contain multiple node/edge/split files"
        )
    if len(spec.supervision.split_registry.sets) < 2:
        pytest.fail(
            "representative live graph source must provide several named split triplets"
        )
    assert spec.output_kind == "homogeneous"
    assert spec.reproducibility.save_reproducibility_bundle is True
    assert tuple(node.name for node in spec.node_types) == tuple(
        expectations["node_types"]
    )
    assert tuple(relation.relation for relation in spec.relations) == tuple(
        tuple(value) for value in expectations["relations"]
    )
    expected_roles = _mapping(expectations["semantic_roles"], "semantic_roles")
    assert (
        expected_roles["target_node_type"] == spec.supervision.target_node_type
    )
    assert expected_roles["label_column"] == spec.supervision.label_column
    assert expected_roles["label_dtype"] == spec.supervision.label_dtype
    assert expectations["num_classes"] == NUM_CLASSES
    for node in spec.node_types:
        assert node.feature_width > 1
        assert node.feature_dtype.startswith("float")
        assert len(node.paths) >= 2


def _assert_store_semantics(
    store: TypedGraphStore,
    contract: Mapping[str, Any],
) -> tuple[str, ...]:
    expectations = _mapping(contract["expectations"], "expectations")
    node_counts = _mapping(expectations["node_counts"], "node_counts")
    relation_counts = _mapping(
        expectations["relation_counts"], "relation_counts"
    )
    assert tuple(store.node_types) == tuple(expectations["node_types"])
    assert tuple(store.relation_types) == tuple(
        tuple(value) for value in expectations["relations"]
    )
    for node_type, count in node_counts.items():
        assert store._node(node_type)["count"] == count
    for relation_text, count in relation_counts.items():
        relation = tuple(relation_text.split("|"))
        assert len(store.relation_csc(relation)[0]) == count
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
            start, stop = (
                int(colptr[destination]),
                int(colptr[destination + 1]),
            )
            assert source in row[start:stop].tolist()

    split_counts = _mapping(expectations["split_counts"], "split_counts")
    assert tuple(store._manifest["splits"]) == tuple(split_counts)
    for tag, phases_value in split_counts.items():
        phases = _mapping(phases_value, f"split_counts.{tag}")
        for phase in ("train", "val", "test"):
            ids = store.split_ids(tag, phase)
            assert len(ids) == phases[phase]
            assert len(np.unique(ids)) == len(ids)

    samples = _mapping(
        expectations["external_id_samples"], "external_id_samples"
    )
    raw_sentinels: list[str] = []
    for node_type, rows_value in samples.items():
        if not isinstance(rows_value, list) or not rows_value:
            pytest.fail(f"external_id_samples.{node_type} must be non-empty")
        internal = np.array(
            [int(row["internal_id"]) for row in rows_value], dtype=np.int64
        )
        expected = [row["external_id"] for row in rows_value]
        assert store.external_ids(node_type, internal) == expected
        raw_sentinels.extend(str(value) for value in expected)
        feature_rows = store.node_features(node_type, internal)
        assert feature_rows.shape == (
            len(internal),
            store._node(node_type)["feature_width"],
        )
        assert np.isfinite(feature_rows).all()
        assert feature_rows.flags.writeable is False
        if node_type == store._manifest["target_node_type"]:
            labels = store.node_labels(node_type, internal)
            assert labels.dtype.kind in "iu"
            assert int(labels.min()) >= 0
            assert int(labels.max()) < NUM_CLASSES
    if not any(len(value) >= 8 for value in raw_sentinels):
        pytest.fail(
            "live graph contract needs a distinctive raw-ID redaction sentinel"
        )
    return tuple(raw_sentinels)


def _finite_native_step(batch: Data) -> tuple[float, float]:
    torch.manual_seed(20260802)
    model = GCN(
        in_channels=int(batch.x.shape[1]),
        hidden_channels=32,
        num_layers=2,
        out_channels=NUM_CLASSES,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch.x, batch.edge_index)
    mask = batch.supervised_mask
    assert bool(mask.any())
    loss = torch.nn.functional.cross_entropy(
        logits[mask], batch.y[mask].long()
    )
    loss.backward()
    optimizer.step()
    accuracy = float(
        (logits[mask].argmax(dim=-1) == batch.y[mask]).float().mean()
    )
    assert math.isfinite(float(loss.detach())) and math.isfinite(accuracy)
    return float(loss.detach()), accuracy


def _write_aggregate_evidence(
    store: TypedGraphStore,
    source: Any,
    contract: Mapping[str, Any],
    *,
    cache_hit: bool,
    pca_state_key: str,
    loss: float,
    accuracy: float,
    strategy_state: Mapping[str, Any],
) -> None:
    evidence_root_value = os.environ.get(
        "TOPOBENCH_QUALIFICATION_EVIDENCE_DIR"
    )
    if not evidence_root_value:
        return
    evidence_root = Path(evidence_root_value)
    evidence_root.mkdir(parents=True, exist_ok=True)
    expectations = _mapping(contract["expectations"], "expectations")
    evidence = {
        "schema_version": "real-parquet-qualification-aggregate-v1",
        "status": "passed",
        "dataset": contract["dataset_name"],
        "output_kind": "homogeneous",
        "source_fingerprint": store._manifest["source_binding"][
            "source_fingerprint"
        ],
        "store_fingerprint": store.content_sha256,
        "partition_book_identity": store.partition_book_identity,
        "schema_roles": expectations["semantic_roles"],
        "representation": "typed-csc-mmap",
        "node_counts": expectations["node_counts"],
        "relation_counts": expectations["relation_counts"],
        "split_counts": expectations["split_counts"],
        "validated_cache_hit": cache_hit,
        "fitted_transform_state_key": pca_state_key,
        "native_step": {"loss": loss, "accuracy": accuracy},
        "source_file_count": len(source.files),
        "strategy_state": dict(strategy_state),
    }
    (evidence_root / "real-parquet-graph.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_real_multifile_parquet_graph_lifecycle(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Convert, partition, fit PCA, train, replay, and redact real graph data."""
    caplog.set_level(logging.INFO)
    root, contract = _require_live_bundle(tmp_path)
    source = _source_from_contract(root, contract)
    _assert_source_contract(source, contract)

    ingestor = ParquetTypedGraphIngestor(
        source, tmp_path / "stores", threads=1
    )
    partition = ingestor.build_partitions(
        limits=PartitionQualificationLimits()
    )
    fresh = TypedGraphStoreWriter(ingestor, partition).build()
    assert fresh.cache_hit is False
    raw_ids = _assert_store_semantics(fresh.store, contract)

    expectations = _mapping(contract["expectations"], "expectations")
    pca_contract = _mapping(expectations["pca"], "pca")
    pca = IncrementalPCATransform(
        n_components=int(pca_contract["components"]),
        max_batch_rows=int(pca_contract["max_batch_rows"]),
        max_batch_bytes=int(pca_contract["max_batch_bytes"]),
        target_node_type=None,
        input_dtype=str(pca_contract["input_dtype"]),
        output_dtype="float32",
    )
    strategy = HomogeneousClusterStrategy(clusters_per_batch=1, seed=20260802)
    module = DiskGraphDataModule(
        fresh.path,
        strategy,
        active_split_tag=source.spec.supervision.split_registry.active_tag,
        train_shuffle=False,
        fitted_transform=pca,
        fitted_state_root=tmp_path / "fitted-state",
    )
    module.setup("fit")
    batch = next(iter(module.train_dataloader()))
    assert isinstance(batch, Data)
    assert batch.x.shape[1] == pca_contract["components"]
    assert pca.state_key is not None
    loss, accuracy = _finite_native_step(batch)
    module.close()

    replay_partition = ingestor.build_partitions(
        limits=PartitionQualificationLimits()
    )
    replay = TypedGraphStoreWriter(ingestor, replay_partition).build()
    assert replay.cache_hit is True
    assert replay.content_sha256 == fresh.content_sha256
    state = replay.store.state()
    replay.store.close()
    with TypedGraphStore.from_state(state) as reopened:
        assert reopened.content_sha256 == fresh.content_sha256
        assert (
            reopened.partition_book_identity == partition.book.content_identity
        )
        _assert_store_semantics(reopened, contract)
        _write_aggregate_evidence(
            reopened,
            source,
            contract,
            cache_hit=True,
            pca_state_key=pca.state_key,
            loss=loss,
            accuracy=accuracy,
            strategy_state={
                **strategy.sampler_state(),
                "clusters_per_batch": 1,
            },
        )
    fresh.store.close()

    captured = capsys.readouterr()
    emitted = (
        captured.out
        + captured.err
        + "\n".join(record.getMessage() for record in caplog.records)
    )
    for raw_id in raw_ids:
        assert raw_id not in emitted
