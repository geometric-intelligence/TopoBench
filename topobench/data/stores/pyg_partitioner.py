"""Topology-only PyG partition generation, trusted adaptation, and publication."""
from __future__ import annotations
import hashlib
import json
import multiprocessing as mp
import os
import resource
import shutil
import stat
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.distributed import Partitioner
from torch_geometric.distributed.utils import as_str
from topobench.data.stores.typed_partition_book import CanonicalRelation, PartitionQualificationLimits, PartitionStatistics, QualificationCheck, TypedPartitionBook, topology_fingerprint


def _ingestion() -> Any:
    from topobench.data.stores import typed_graph_ingestion
    return typed_graph_ingestion


def _error(code: str, detail: str) -> Exception:
    return _ingestion().ArtifactValidationError(f"{code}: {detail}")


def _safe_json(path: Path, code: str) -> dict[str, Any]:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise _error(code, f"{path.name} is not regular")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            current = os.fstat(descriptor)
            if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
                raise _error(code, f"{path.name} changed while opening")
            with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as stream:
                value = json.load(stream)
        finally:
            os.close(descriptor)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _error(code, f"cannot safely read {path.name}") from exc
    if not isinstance(value, dict):
        raise _error(code, f"{path.name} is not a JSON object")
    return value


def _safe_torch_load(path: Path) -> Any:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise _error("PARTITION-OUTPUT-001", f"{path.name} is not regular")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            current = os.fstat(descriptor)
            if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
                raise _error("PARTITION-OUTPUT-001", f"{path.name} changed while opening")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                return torch.load(stream, map_location="cpu", weights_only=True)
        finally:
            os.close(descriptor)
    except _ingestion().ArtifactValidationError:
        raise
    except Exception as exc:
        raise _error("PARTITION-OUTPUT-001", f"malformed trusted tensor {path.name}") from exc


def _safe_relative(root: Path, relative: str, code: str) -> Path:
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or not parsed.parts or ".." in parsed.parts:
        raise _error(code, "unsafe relative path")
    candidate = root
    for component in parsed.parts:
        candidate /= component
        if os.path.lexists(candidate) and stat.S_ISLNK(candidate.lstat().st_mode):
            raise _error(code, "path contains symlink")
    resolved = (root / parsed).resolve(strict=False)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except (ValueError, FileNotFoundError) as exc:
        raise _error(code, "path escapes trusted root") from exc
    return resolved


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink())


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _relation_label(relation: CanonicalRelation) -> str:
    return json.dumps(relation, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class CanonicalRelationTopology:
    relation: CanonicalRelation
    internal_key: str
    source_internal_key: str
    destination_internal_key: str
    source_count: int
    destination_count: int
    edge_count: int
    colptr_path: Path
    row_path: Path
    metadata: Mapping[str, Any]

    @property
    def source(self) -> np.ndarray:
        return np.load(self.row_path, mmap_mode="r", allow_pickle=False)

    @property
    def destination(self) -> np.ndarray:
        colptr = np.load(self.colptr_path, mmap_mode="r", allow_pickle=False)
        return np.repeat(np.arange(self.destination_count, dtype=np.int64), np.diff(colptr))


@dataclass(frozen=True, slots=True)
class TopologyContext:
    fingerprint: str
    node_types: tuple[str, ...]
    node_internal_keys: Mapping[str, str]
    internal_node_types: Mapping[str, str]
    node_counts: Mapping[str, int]
    relations: Mapping[CanonicalRelation, CanonicalRelationTopology]
    source_binding: Mapping[str, Any]
    arrays_metadata: Mapping[str, Any]
    stage_root: Path
    num_partitions: int
    recursive: bool


@dataclass(frozen=True, slots=True)
class TypedPartitionBuild:
    inventory: Any
    stage_root: Path
    artifact_root: Path
    book: TypedPartitionBook
    limits: PartitionQualificationLimits
    binding: Mapping[str, Any]
    evidence: Mapping[str, Any]
    resumed: bool


class PyGPartitionArtifactAdapter:
    """Adapt only locally-generated, fingerprint-owned PyG tensor artifacts."""
    def __init__(self, topology: TopologyContext, artifact_root: Path, *, trusted_parent: Path) -> None:
        self.topology = topology
        self.artifact_root = Path(artifact_root)
        self.trusted_parent = Path(trusted_parent)
        try:
            self.artifact_root.resolve(strict=True).relative_to(self.trusted_parent.resolve(strict=True))
        except (ValueError, FileNotFoundError) as exc:
            raise _error("PARTITION-OUTPUT-001", "PyG output is outside trusted parent") from exc
        if topology.fingerprint not in self.artifact_root.resolve().parts:
            raise _error("PARTITION-OUTPUT-001", "PyG output is not fingerprint-owned")
        if self.artifact_root.is_symlink():
            raise _error("PARTITION-OUTPUT-001", "PyG output is symlinked")

    def adapt(self, limits: PartitionQualificationLimits, *, estimated_resources: Mapping[str, Any] | None = None, measured_resources: Mapping[str, Any] | None = None) -> TypedPartitionBook:
        meta = _safe_json(self.artifact_root / "META.json", "PARTITION-OUTPUT-001")
        expected_nodes = list(self.topology.internal_node_types)
        if meta.get("num_parts") != self.topology.num_partitions or meta.get("is_hetero") is not True or meta.get("node_types") != expected_nodes:
            raise _error("PARTITION-OUTPUT-001", "PyG metadata differs from request")
        assignments: dict[str, np.ndarray] = {}
        for internal in expected_nodes:
            value = _safe_torch_load(self.artifact_root / "node_map" / f"{internal}.pt")
            if not isinstance(value, torch.Tensor) or value.dtype != torch.int64 or value.ndim != 1:
                raise _error("PARTITION-ID-001", f"node map {internal} is malformed")
            node_type = self.topology.internal_node_types[internal]
            array = value.detach().cpu().numpy()
            if len(array) != self.topology.node_counts[node_type] or np.any(array < 0) or np.any(array >= self.topology.num_partitions):
                raise _error("PARTITION-ID-001", f"node map {internal} has invalid IDs")
            assignments[node_type] = array
        self._validate_node_ids(assignments)
        expected = _derive_ownership(self.topology, assignments)
        for relation, item in self.topology.relations.items():
            edge_type = (item.source_internal_key, item.internal_key, item.destination_internal_key)
            value = _safe_torch_load(self.artifact_root / "edge_map" / f"{as_str(edge_type)}.pt")
            if not isinstance(value, torch.Tensor) or value.dtype != torch.int64 or value.ndim != 1 or not np.array_equal(value.numpy(), expected[relation]):
                raise _error("PARTITION-OUTPUT-001", f"canonical ownership changed for {relation!r}")
        return _qualified_book(self.topology, assignments, limits, backend="pyg", estimated_resources=estimated_resources or {}, measured_resources=measured_resources or {})

    def _validate_node_ids(
        self,
        assignments: Mapping[str, np.ndarray],
    ) -> None:
        seen = {key: np.zeros(self.topology.node_counts[name], dtype=np.bool_) for key, name in self.topology.internal_node_types.items()}
        for partition in range(self.topology.num_partitions):
            value = _safe_torch_load(self.artifact_root / f"part_{partition}" / "node_feats.pt")
            if not isinstance(value, dict): raise _error("PARTITION-OUTPUT-001", "malformed node feature map")
            for internal in self.topology.internal_node_types:
                record = value.get(internal)
                if not isinstance(record, dict) or not isinstance(record.get("id"), torch.Tensor):
                    raise _error("PARTITION-ID-001", f"typed IDs missing for {internal}")
                ids = record["id"].numpy()
                if ids.dtype != np.int64 or ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(seen[internal])) or np.any(seen[internal][ids]):
                    raise _error("PARTITION-ID-001", f"duplicate or invalid IDs for {internal}")
                node_type = self.topology.internal_node_types[internal]
                if np.any(assignments[node_type][ids] != partition):
                    raise _error(
                        "PARTITION-ID-001",
                        f"typed IDs for {internal} disagree with node_map",
                    )
                seen[internal][ids] = True
        if any(not np.all(value) for value in seen.values()): raise _error("PARTITION-ID-001", "typed IDs are incomplete")


class TopologyOnlyPyGPartitioner:
    """Preflight and generate a topology-only typed partition book."""
    def __init__(self, ingestor: Any, relation_build: Any, *, num_partitions: int | None = None) -> None:
        self.ingestor = ingestor
        self.relation_build = relation_build
        resolved_partitions = (
            ingestor.source.spec.partition.num_partitions
            if num_partitions is None
            else num_partitions
        )
        if isinstance(resolved_partitions, bool):
            raise TypeError("num_partitions must be an integer")
        if (
            not isinstance(resolved_partitions, int)
            or resolved_partitions < 2
        ):
            raise ValueError("num_partitions must be at least 2")
        self.num_partitions = resolved_partitions
        self.materialization_count = 0
        self.last_work_root: Path | None = None
        self._synthetic_edge_types: tuple[tuple[str, str, str], ...] = ()
        self._synthetic_edge_types_computed = False
        self.topology_context = self._context()

    def _context(self) -> TopologyContext:
        if self.relation_build.stage_root != self.ingestor.stage_root(self.relation_build.inventory): raise _error("PARTITION-FINGERPRINT-001", "foreign Task4 build")
        relations_meta = _safe_json(self.relation_build.artifact_root / "relations.json", "PARTITION-FINGERPRINT-001")
        arrays_meta = _safe_json(self.relation_build.stage_root / "arrays" / "arrays.json", "PARTITION-FINGERPRINT-001")
        node_keys = {node.name: f"n{index:04d}" for index, node in enumerate(self.ingestor.source.spec.node_types)}
        inverse = {key: name for name, key in node_keys.items()}
        counts = dict(self.relation_build.inventory.node_rows)
        relations: dict[CanonicalRelation, CanonicalRelationTopology] = {}
        for key, record in sorted(relations_meta["relations"].items()):
            relation = tuple(record["relation"])
            relations[relation] = CanonicalRelationTopology(relation, key, record["source_internal_key"], record["destination_internal_key"], record["source_count"], record["destination_count"], record["edge_count"], self.relation_build.artifact_root / record["colptr"]["relative_path"], self.relation_build.artifact_root / record["row"]["relative_path"], MappingProxyType(record))
        completion = _safe_json(self.relation_build.artifact_root / "relations.complete.json", "PARTITION-FINGERPRINT-001")
        binding = {"task4_content_sha256": self.relation_build.content_sha256, "task4_completion_sha256": _file_sha(self.relation_build.artifact_root / "relations.complete.json"), "array_binding": completion["array_binding"], "source_fingerprint": self.relation_build.inventory.source_fingerprint, "config_fingerprint": self.relation_build.inventory.config_fingerprint, "active_split_tag": arrays_meta["active_split_tag"]}
        return TopologyContext(topology_fingerprint.from_relation_build(self.ingestor, self.relation_build), tuple(node.name for node in self.ingestor.source.spec.node_types), MappingProxyType(node_keys), MappingProxyType(inverse), MappingProxyType(counts), MappingProxyType(relations), MappingProxyType(binding), MappingProxyType(arrays_meta), self.relation_build.stage_root, self.num_partitions, self.ingestor.source.spec.partition.recursive)

    @property
    def topology_fingerprint(self) -> str: return self.topology_context.fingerprint
    @property
    def canonical_relations(self) -> Mapping[CanonicalRelation, CanonicalRelationTopology]: return self.topology_context.relations
    @property
    def canonical_edge_types(self) -> tuple[tuple[str, str, str], ...]: return tuple((item.source_internal_key, item.internal_key, item.destination_internal_key) for item in self.topology_context.relations.values())
    @property
    def synthetic_edge_types(
        self,
    ) -> tuple[tuple[str, str, str], ...]:
        self.preflight()
        if not self._synthetic_edge_types_computed:
            self._scoring_edges()
            self._synthetic_edge_types_computed = True
        return self._synthetic_edge_types

    @staticmethod
    def estimate_topology_resources(*, node_count: int, canonical_edge_count: int, relation_count: int, node_type_count: int) -> dict[str, int]:
        scoring = canonical_edge_count * 2
        external_assignment_bytes = node_count * np.dtype(np.int64).itemsize
        memory = 768 * 1024**2 + scoring * 64 + node_count * 40 + external_assignment_bytes + (relation_count + node_type_count + 2) * 4096
        temporary = 8 * 1024**2 + scoring * 80 + node_count * 32
        return {"node_count": node_count, "canonical_edge_count": canonical_edge_count, "scoring_edge_upper_bound": scoring, "external_assignment_bytes": external_assignment_bytes, "peak_memory_bytes": memory, "temporary_disk_bytes": temporary}

    def estimate_resources(self) -> dict[str, int]:
        return self.estimate_topology_resources(node_count=sum(self.topology_context.node_counts.values()), canonical_edge_count=sum(item.edge_count for item in self.topology_context.relations.values()), relation_count=len(self.topology_context.relations), node_type_count=len(self.topology_context.node_types))

    def preflight(self, *, memory_limit_bytes: int | None = None, temp_available_bytes: int | None = None) -> dict[str, int]:
        estimate = self.estimate_resources()
        memory = memory_limit_bytes if memory_limit_bytes is not None else self.ingestor.source.spec.partition.memory_limit_bytes
        if estimate["peak_memory_bytes"] > memory: raise _error("PARTITION-MEMORY-001", f"estimate {estimate['peak_memory_bytes']} exceeds {memory}")
        if temp_available_bytes is None:
            _, available, _ = _ingestion()._filesystem_capacity(
                self.relation_build.inventory.temporary_filesystem_path
            )
            temp_available_bytes = min(available, self.ingestor.disk_limit_bytes) if self.ingestor.disk_limit_bytes is not None else available
        if estimate["temporary_disk_bytes"] > temp_available_bytes: raise _error("PARTITION-TEMP-DISK-001", f"estimate {estimate['temporary_disk_bytes']} exceeds {temp_available_bytes}")
        return estimate

    def _scoring_edges(
        self,
    ) -> dict[tuple[str, str, str], np.ndarray]:
        items = list(self.topology_context.relations.values())
        scoring = {
            (
                item.source_internal_key,
                item.internal_key,
                item.destination_internal_key,
            ): np.stack((item.source, item.destination))
            for item in items
        }
        synthetic: dict[tuple[str, str, str], np.ndarray] = {}
        for item in items:
            reverse_candidates = [
                (candidate.source, candidate.destination)
                for candidate in items
                if (
                    candidate.source_internal_key
                    == item.destination_internal_key
                    and candidate.destination_internal_key
                    == item.source_internal_key
                )
            ]
            missing_source, missing_destination = _missing_reverse_arcs(
                item.source,
                item.destination,
                reverse_candidates,
            )
            if len(missing_source):
                synthetic[
                    (
                        item.destination_internal_key,
                        f"srev_{item.internal_key}",
                        item.source_internal_key,
                    )
                ] = np.stack((missing_source, missing_destination))
        self._synthetic_edge_types = tuple(synthetic)
        scoring.update(synthetic)
        return scoring

    def materialize_topology(self) -> HeteroData:
        self.preflight(); data = HeteroData()
        for internal, node_type in self.topology_context.internal_node_types.items(): data[internal].num_nodes = self.topology_context.node_counts[node_type]
        for edge_type, edges in self._scoring_edges().items(): data[edge_type].edge_index = torch.from_numpy(edges)
        self.materialization_count += 1; return data

    def artifact_adapter(self, root: Path) -> PyGPartitionArtifactAdapter:
        return PyGPartitionArtifactAdapter(self.topology_context, root, trusted_parent=Path(root).parent)

    def adapt_assignments(self, assignments: Mapping[str, np.ndarray], limits: PartitionQualificationLimits, *, backend: str) -> TypedPartitionBook:
        return _qualified_book(self.topology_context, assignments, limits, backend=backend, estimated_resources=self.estimate_resources(), measured_resources={})

    def generate(self, limits: PartitionQualificationLimits) -> TypedPartitionBook:
        estimate = self.preflight()
        ephemeral_root = self.ingestor._new_ephemeral_root(
            self.relation_build.inventory,
            purpose="partition-pyg",
        )
        parent = ephemeral_root / self.topology_fingerprint
        work = parent / "work"
        output = work / "output"
        request_path = work / "request.json"
        response_path = work / "response.json"
        work.mkdir(parents=True, exist_ok=False)
        self.last_work_root = work
        request = {
            "format_version": "topology-only-pyg-worker-v1",
            "fingerprint": self.topology_fingerprint,
            "num_partitions": self.num_partitions,
            "recursive": self.topology_context.recursive,
            "output_root": str(output),
            "nodes": [
                {
                    "internal_key": internal,
                    "count": self.topology_context.node_counts[node_type],
                }
                for internal, node_type
                in self.topology_context.internal_node_types.items()
            ],
            "relations": [
                {
                    "relation": list(relation),
                    "internal_key": item.internal_key,
                    "source_internal_key": item.source_internal_key,
                    "destination_internal_key": item.destination_internal_key,
                    "source_count": item.source_count,
                    "destination_count": item.destination_count,
                    "edge_count": item.edge_count,
                    "colptr_path": str(item.colptr_path),
                    "row_path": str(item.row_path),
                }
                for relation, item in self.topology_context.relations.items()
            ],
        }
        _atomic_json(request_path, request)
        process: Any | None = None
        try:
            process = mp.get_context("spawn").Process(
                target=_partition_worker,
                args=(str(request_path), str(response_path)),
            )
            process.start()
            process.join(timeout=3600)
            if process.is_alive():
                process.terminate()
                process.join()
                raise _error(
                    "PARTITION-BACKEND-001",
                    "isolated PyG/METIS worker exceeded its deadline",
                )
            if process.exitcode != 0 or not response_path.is_file():
                raise _error(
                    "PARTITION-BACKEND-001",
                    "isolated PyG/METIS worker failed",
                )
            response = _safe_json(
                response_path,
                "PARTITION-BACKEND-001",
            )
            if response.get("status") != "ok":
                detail = response.get("detail")
                raise _error(
                    "PARTITION-BACKEND-001",
                    "isolated PyG/METIS worker rejected the topology"
                    + (f": {detail}" if isinstance(detail, str) else ""),
                )
            measured = {
                "measurement_scope": "isolated-worker",
                "peak_rss_bytes": response["peak_rss_bytes"],
                "temporary_disk_bytes": response["temporary_disk_bytes"],
            }
            if measured["peak_rss_bytes"] > estimate["peak_memory_bytes"]:
                raise _error(
                    "PARTITION-MEMORY-001",
                    "measured isolated peak RSS exceeded its declared bound",
                )
            if (
                measured["temporary_disk_bytes"]
                > estimate["temporary_disk_bytes"]
            ):
                raise _error(
                    "PARTITION-TEMP-DISK-001",
                    "measured temporary bytes exceeded their declared bound",
                )
            return PyGPartitionArtifactAdapter(
                self.topology_context,
                output,
                trusted_parent=parent,
            ).adapt(
                limits,
                estimated_resources=estimate,
                measured_resources=measured,
            )
        except _ingestion().ArtifactValidationError:
            raise
        except Exception as exc:
            raise _error(
                "PARTITION-BACKEND-001",
                "PyG/METIS generation failed",
            ) from exc
        finally:
            if process is not None:
                if process.is_alive():
                    process.terminate()
                    process.join()
                process.close()
            shutil.rmtree(ephemeral_root, ignore_errors=True)

    def _reopen_task4_locked(self) -> None:
        """Semantically revalidate the exact Task4 binding under this lock."""
        from topobench.data.stores.typed_graph_arrays import (
            TypedGraphArrayWriter,
        )
        from topobench.data.stores.typed_graph_csc import (
            TypedGraphRelationWriter,
        )

        ingestion = _ingestion()
        pa, pq, duckdb = ingestion._parquet_dependencies()
        inventory = self.relation_build.inventory
        self.ingestor._validate_inventory_current(
            inventory,
            pa=pa,
            pq=pq,
        )
        indexes = self.ingestor._resume(
            inventory,
            self.relation_build.stage_root,
            pa=pa,
            pq=pq,
            duckdb=duckdb,
        )
        arrays = TypedGraphArrayWriter(
            self.ingestor,
            indexes,
        )._open_validated(resumed=True)
        relations = TypedGraphRelationWriter(
            self.ingestor,
            indexes,
            arrays,
        )._open_validated(
            resumed=True,
            pa=pa,
            pq=pq,
            duckdb=duckdb,
        )
        if (
            relations.content_sha256
            != self.relation_build.content_sha256
            or arrays.active_tag
            != self.topology_context.source_binding["active_split_tag"]
        ):
            raise _error(
                "PARTITION-FINGERPRINT-001",
                "Task4 binding changed before partition publication",
            )


    def _clean_task6_scratch_locked(self) -> None:
        """Remove only confined Task6 leftovers for this source identity."""
        stage_root = self.relation_build.stage_root
        for path in stage_root.glob(".partitions-tmp-*"):
            if path.is_symlink() or not path.is_dir():
                raise _error(
                    "PARTITION-OUTPUT-001",
                    "legacy partition staging path is unsafe",
                )
            shutil.rmtree(path)
        scratch_parent = (
            self.relation_build.inventory.temporary_filesystem_path
            / ".topobench-typed-graph-work"
            / stage_root.name
        )
        if scratch_parent.exists():
            if scratch_parent.is_symlink() or not scratch_parent.is_dir():
                raise _error(
                    "PARTITION-OUTPUT-001",
                    "configured Task6 scratch parent is unsafe",
                )
            for path in scratch_parent.glob("partition-pyg-*"):
                if path.is_symlink() or not path.is_dir():
                    raise _error(
                        "PARTITION-OUTPUT-001",
                        "configured PyG scratch path is unsafe",
                    )
                shutil.rmtree(path)
        publish_parent = (
            self.ingestor.store_root
            / ".topobench-partition-work"
            / stage_root.name
        )
        if publish_parent.exists():
            if publish_parent.is_symlink() or not publish_parent.is_dir():
                raise _error(
                    "PARTITION-OUTPUT-001",
                    "partition publish scratch parent is unsafe",
                )
            for path in publish_parent.iterdir():
                if path.is_symlink() or not path.is_dir():
                    raise _error(
                        "PARTITION-OUTPUT-001",
                        "partition publish scratch path is unsafe",
                    )
                shutil.rmtree(path)


    def build(self, limits: PartitionQualificationLimits) -> TypedPartitionBuild:
        artifact = self.relation_build.stage_root / "partitions"
        with self.ingestor._build_lock(self.ingestor.lock_path(self.relation_build.inventory)):
            self._reopen_task4_locked()
            self._clean_task6_scratch_locked()
            if artifact.exists():
                try:
                    assignments, identity, stored_limits = _read_subtree(
                        self.topology_context,
                        artifact,
                    )
                except (
                    _ingestion().ArtifactValidationError,
                    KeyError,
                    TypeError,
                    ValueError,
                ):
                    os.replace(
                        artifact,
                        self.relation_build.stage_root
                        / f".partitions-quarantine-{uuid.uuid4().hex}",
                    )
                else:
                    book = _qualified_book(
                        self.topology_context,
                        assignments,
                        limits,
                        backend=identity["backend"],
                        estimated_resources=identity[
                            "estimated_resources"
                        ],
                        measured_resources=identity[
                            "measured_resources"
                        ],
                    )
                    if stored_limits.fingerprint != limits.fingerprint:
                        publish_parent = (
                            self.ingestor.store_root
                            / ".topobench-partition-work"
                            / self.relation_build.stage_root.name
                        )
                        _republish_partition_metadata(
                            artifact,
                            book,
                            limits,
                            work_parent=publish_parent,
                        )
                        assignments, identity, stored_limits = _read_subtree(
                            self.topology_context,
                            artifact,
                        )
                        book = _qualified_book(
                            self.topology_context,
                            assignments,
                            stored_limits,
                            backend=identity["backend"],
                            estimated_resources=identity[
                                "estimated_resources"
                            ],
                            measured_resources=identity[
                                "measured_resources"
                            ],
                        )
                    return TypedPartitionBuild(
                        inventory=self.relation_build.inventory,
                        stage_root=self.relation_build.stage_root,
                        artifact_root=artifact,
                        book=book,
                        limits=stored_limits,
                        binding=_immutable_json(identity),
                        evidence=_partition_evidence(book, stored_limits),
                        resumed=True,
                    )
            try: book = self.generate(limits)
            except _ingestion().ArtifactValidationError:
                configured = self.ingestor.source.spec.partition.external_partition_map
                if configured is None: raise
                assignments = _external_assignments(self.topology_context, self.ingestor.source.spec.source_root, configured)
                book = _qualified_book(self.topology_context, assignments, limits, backend="external", estimated_resources=self.estimate_resources(), measured_resources={})
            publish_parent = (
                self.ingestor.store_root
                / ".topobench-partition-work"
                / self.relation_build.stage_root.name
            )
            _publish_subtree(
                self.topology_context,
                artifact,
                book,
                limits,
                work_parent=publish_parent,
            )
            assignments, identity, stored_limits = _read_subtree(
                self.topology_context,
                artifact,
            )
            reopened_book = _qualified_book(
                self.topology_context,
                assignments,
                stored_limits,
                backend=identity["backend"],
                estimated_resources=identity["estimated_resources"],
                measured_resources=identity["measured_resources"],
            )
            return TypedPartitionBuild(
                inventory=self.relation_build.inventory,
                stage_root=self.relation_build.stage_root,
                artifact_root=artifact,
                book=reopened_book,
                limits=stored_limits,
                binding=_immutable_json(identity),
                evidence=_partition_evidence(reopened_book, stored_limits),
                resumed=False,
            )


def _partition_worker(request_name: str, response_name: str) -> None:
    """Materialize topology and invoke PyG in one clean spawned process."""
    request_path = Path(request_name)
    response_path = Path(response_name)
    try:
        request = _safe_json(request_path, "PARTITION-BACKEND-001")
        work_root = request_path.parent.resolve(strict=True)
        output_root = Path(request["output_root"])
        if (
            request.get("format_version")
            != "topology-only-pyg-worker-v1"
            or output_root != request_path.parent / "output"
            or request["fingerprint"] not in work_root.parts
        ):
            raise _error(
                "PARTITION-BACKEND-001",
                "worker request is not fingerprint-confined",
            )
        nodes = request.get("nodes")
        relations = request.get("relations")
        if not isinstance(nodes, list) or not isinstance(relations, list):
            raise _error(
                "PARTITION-BACKEND-001",
                "worker request lacks topology counts",
            )
        materialized: list[dict[str, Any]] = []
        for record in relations:
            if not isinstance(record, dict):
                raise _error(
                    "PARTITION-BACKEND-001",
                    "worker relation record is malformed",
                )
            row_path = Path(record["row_path"])
            colptr_path = Path(record["colptr_path"])
            for path in (row_path, colptr_path):
                if path.is_symlink() or not path.is_file():
                    raise _error(
                        "PARTITION-BACKEND-001",
                        "worker received an unsafe canonical array",
                    )
            source = np.load(
                row_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            colptr = np.load(
                colptr_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            destination = np.repeat(
                np.arange(
                    int(record["destination_count"]),
                    dtype=np.int64,
                ),
                np.diff(colptr),
            )
            if (
                source.dtype != np.int64
                or colptr.dtype != np.int64
                or len(source) != record["edge_count"]
                or len(destination) != record["edge_count"]
            ):
                raise _error(
                    "PARTITION-BACKEND-001",
                    "worker canonical array metadata differs",
                )
            materialized.append(
                {
                    **record,
                    "relation": tuple(record["relation"]),
                    "source": source,
                    "destination": destination,
                }
            )
        data = HeteroData()
        for node in nodes:
            if (
                not isinstance(node, dict)
                or not isinstance(node.get("internal_key"), str)
                or not isinstance(node.get("count"), int)
            ):
                raise _error(
                    "PARTITION-BACKEND-001",
                    "worker node record is malformed",
                )
            data[node["internal_key"]].num_nodes = node["count"]
        for record in materialized:
            edge_type = (
                record["source_internal_key"],
                record["internal_key"],
                record["destination_internal_key"],
            )
            data[edge_type].edge_index = torch.from_numpy(
                np.stack(
                    (record["source"], record["destination"]),
                )
            )
            reverse_candidates = [
                (candidate["source"], candidate["destination"])
                for candidate in materialized
                if (
                    candidate["source_internal_key"]
                    == record["destination_internal_key"]
                    and candidate["destination_internal_key"]
                    == record["source_internal_key"]
                )
            ]
            missing_source, missing_destination = _missing_reverse_arcs(
                record["source"],
                record["destination"],
                reverse_candidates,
            )
            if len(missing_source):
                synthetic_type = (
                    record["destination_internal_key"],
                    f"srev_{record['internal_key']}",
                    record["source_internal_key"],
                )
                data[synthetic_type].edge_index = torch.from_numpy(
                    np.stack((missing_source, missing_destination))
                )
        Partitioner(
            data,
            num_parts=request["num_partitions"],
            root=str(output_root),
            recursive=request["recursive"],
        ).generate_partition()
        _atomic_json(
            response_path,
            {
                "status": "ok",
                "peak_rss_bytes": _rss_bytes(),
                "temporary_disk_bytes": _tree_bytes(work_root),
            },
        )
    except Exception as error:
        try:
            _atomic_json(
                response_path,
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "detail": str(error) or repr(error),
                },
            )
        except Exception:
            pass


def _missing_reverse_arcs(
    source: np.ndarray,
    destination: np.ndarray,
    available_relations: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return only reverse arc multiplicities absent from canonical topology."""
    desired_source = np.asarray(destination, dtype=np.int64)
    desired_destination = np.asarray(source, dtype=np.int64)
    count = len(desired_source)
    if count == 0:
        return desired_source, desired_destination
    available = [
        (
            np.asarray(candidate_source, dtype=np.int64),
            np.asarray(candidate_destination, dtype=np.int64),
        )
        for candidate_source, candidate_destination in available_relations
        if len(candidate_source)
    ]
    if not available:
        return desired_source, desired_destination
    if len(available) == 1:
        available_source, available_destination = available[0]
    else:
        available_source = np.concatenate(
            [candidate[0] for candidate in available]
        )
        available_destination = np.concatenate(
            [candidate[1] for candidate in available]
        )
    desired_order = np.lexsort(
        (desired_destination, desired_source)
    )
    available_order = np.lexsort(
        (available_destination, available_source)
    )
    missing = np.ones(count, dtype=np.bool_)
    desired_index = 0
    available_index = 0
    while desired_index < count:
        desired_start = desired_index
        desired_row = int(desired_source[desired_order[desired_start]])
        desired_col = int(
            desired_destination[desired_order[desired_start]]
        )
        desired_index += 1
        while desired_index < count:
            index = desired_order[desired_index]
            if (
                int(desired_source[index]),
                int(desired_destination[index]),
            ) != (desired_row, desired_col):
                break
            desired_index += 1
        while available_index < len(available_order):
            index = available_order[available_index]
            available_pair = (
                int(available_source[index]),
                int(available_destination[index]),
            )
            if available_pair >= (desired_row, desired_col):
                break
            available_index += 1
        available_start = available_index
        while available_index < len(available_order):
            index = available_order[available_index]
            if (
                int(available_source[index]),
                int(available_destination[index]),
            ) != (desired_row, desired_col):
                break
            available_index += 1
        matched = min(
            desired_index - desired_start,
            available_index - available_start,
        )
        if matched:
            missing[
                desired_order[desired_start : desired_start + matched]
            ] = False
    return desired_source[missing], desired_destination[missing]


def _derive_ownership(topology: TopologyContext, assignments: Mapping[str, np.ndarray]) -> dict[CanonicalRelation, np.ndarray]:
    return {relation: np.asarray(assignments[relation[2]], dtype=np.int64)[item.destination] for relation, item in topology.relations.items()}


def _qualified_book(topology: TopologyContext, assignments: Mapping[str, np.ndarray], limits: PartitionQualificationLimits, *, backend: str, estimated_resources: Mapping[str, Any], measured_resources: Mapping[str, Any]) -> TypedPartitionBook:
    if backend not in {"pyg", "external"}: raise _error("PARTITION-BACKEND-001", f"unsupported backend {backend!r}")
    if set(assignments) != set(topology.node_types): raise _error("PARTITION-ID-001", "typed assignment keys incomplete")
    normalized = {}
    for name in topology.node_types:
        value = np.asarray(assignments[name])
        if value.dtype != np.int64 or value.ndim != 1 or len(value) != topology.node_counts[name] or np.any(value < 0) or np.any(value >= topology.num_partitions): raise _error("PARTITION-ID-001", f"assignment {name!r} malformed")
        normalized[name] = value
    ownership = _derive_ownership(topology, normalized); statistics = _statistics(topology, normalized, ownership); checks = _checks(statistics, topology, limits, backend)
    for check in checks:
        if not check.passed: raise _error(check.check_id, check.detail or "qualification failed")
    version = torch.__version__
    if backend == "pyg":
        import torch_geometric
        version = torch_geometric.__version__
    return TypedPartitionBook.from_assignments(num_partitions=topology.num_partitions, node_assignments=normalized, edge_ownership=ownership, topology_fingerprint=topology.fingerprint, source_binding=topology.source_binding, backend=backend, backend_version=version, options={"num_partitions": topology.num_partitions, "recursive": topology.recursive, "scoring_view": "canonical-plus-missing-reverse-v1"}, provenance={"producer": "TopologyOnlyPyGPartitioner", "trusted_local_pyg": backend == "pyg"}, estimated_resources=estimated_resources, measured_resources=measured_resources, statistics=statistics, qualification_checks=checks)


def _statistics(topology: TopologyContext, assignments: Mapping[str, np.ndarray], ownership: Mapping[CanonicalRelation, np.ndarray]) -> PartitionStatistics:
    parts = topology.num_partitions; node_counts = {name: tuple(int(x) for x in np.bincount(assignments[name], minlength=parts)) for name in topology.node_types}; feature = np.zeros(parts, dtype=np.int64)
    for node in topology.arrays_metadata["nodes"].values(): feature += np.asarray(node_counts[node["node_type"]]) * np.dtype(node["storage_dtype"]).itemsize * int(node["feature_width"])
    total = feature.copy(); supervision = topology.arrays_metadata["supervision"]; total += np.asarray(node_counts[supervision["target_node_type"]]) * np.dtype(supervision["storage_dtype"]).itemsize
    phases = {}; target = assignments[supervision["target_node_type"]]
    for tag, record in sorted(topology.arrays_metadata["splits"].items()):
        if record["qualified"] is not True: continue
        phases[tag] = {}
        for phase, phase_record in sorted(record["phases"].items()):
            ids = np.load(topology.stage_root / "arrays" / phase_record["relative_path"], mmap_mode="r", allow_pickle=False); phases[tag][phase] = tuple(int(x) for x in np.bincount(target[ids], minlength=parts))
    relation_counts = {}; owned = np.zeros(parts, dtype=np.int64); cut = np.zeros(parts, dtype=np.int64)
    for relation, item in topology.relations.items():
        owner = ownership[relation]; counts = np.bincount(owner, minlength=parts); owned += counts; relation_counts[_relation_label(relation)] = tuple(int(x) for x in counts)
        different = assignments[relation[0]][item.source] != owner; cut += np.bincount(owner[different], minlength=parts)
        edge_bytes = 16 + (np.dtype(item.metadata["edge_id"]["storage_dtype"]).itemsize if item.metadata.get("edge_id") else 0)
        for field in item.metadata["fields"].values(): edge_bytes += np.dtype(field["storage_dtype"]).itemsize * int(np.prod(field.get("value_shape", []) or [1]))
        total += counts * edge_bytes
    fraction = np.divide(cut, owned, out=np.zeros(parts, dtype=np.float64), where=owned != 0)
    return PartitionStatistics(node_counts, phases, relation_counts, tuple(map(int, feature)), tuple(map(int, total)), tuple(map(int, owned)), tuple(map(int, cut)), tuple(map(float, fraction)), tuple(map(float, 1 - fraction)))


def _checks(stats: PartitionStatistics, topology: TopologyContext, limits: PartitionQualificationLimits, backend: str) -> tuple[QualificationCheck, ...]:
    unknown_node_types = (
        set(limits.max_nodes_per_type) - set(stats.node_counts)
    )
    if unknown_node_types:
        raise _error(
            "PARTITION-TYPE-BALANCE-001",
            f"unknown node-type limit {min(unknown_node_types)!r}",
        )
    unknown_tags = set(limits.max_phase_nodes) - set(
        topology.arrays_metadata["splits"]
    )
    if unknown_tags:
        raise _error(
            "PARTITION-PHASE-BALANCE-001",
            f"unknown split tag {min(unknown_tags)!r}",
        )
    relation_labels = set(stats.relation_edge_counts)
    unknown_relations = [
        relation
        for relation in limits.max_edges_per_relation
        if _relation_label(relation) not in relation_labels
    ]
    if unknown_relations:
        raise _error(
            "PARTITION-RELATION-BALANCE-001",
            f"unknown canonical relation {unknown_relations[0]!r}",
        )
    total_nodes = [sum(values[i] for values in stats.node_counts.values()) for i in range(topology.num_partitions)]
    type_pass = all(max(stats.node_counts[name]) <= limit for name, limit in limits.max_nodes_per_type.items())
    phase_pass = all(max(stats.phase_counts[tag][phase]) <= limit for tag, phase_limits in limits.max_phase_nodes.items() if tag in stats.phase_counts for phase, limit in phase_limits.items())
    relation_pass = all(max(stats.relation_edge_counts[_relation_label(key)]) <= limit for key, limit in limits.max_edges_per_relation.items())
    cut_pass = (limits.max_cut_fraction is None or max(stats.cut_fraction) <= limits.max_cut_fraction) and (limits.min_locality is None or min(stats.locality) >= limits.min_locality)
    return (
        QualificationCheck("PARTITION-ID-001", True, {"nodes": sum(topology.node_counts.values())}), QualificationCheck("PARTITION-EMPTY-001", all(x > 0 for x in total_nodes), {"total_nodes": total_nodes}, detail="empty partition"),
        QualificationCheck("PARTITION-TYPE-BALANCE-001", type_pass, stats.node_counts, limits.max_nodes_per_type, "type maximum exceeded"), QualificationCheck("PARTITION-PHASE-BALANCE-001", phase_pass, stats.phase_counts, limits.max_phase_nodes, "phase maximum exceeded"),
        QualificationCheck("PARTITION-RELATION-BALANCE-001", relation_pass, stats.relation_edge_counts, {_relation_label(k): v for k, v in limits.max_edges_per_relation.items()}, "relation maximum exceeded"),
        QualificationCheck("PARTITION-FEATURE-BYTES-001", limits.max_feature_bytes is None or max(stats.feature_bytes) <= limits.max_feature_bytes, {"per_partition": stats.feature_bytes}, limits.max_feature_bytes, "feature bytes exceeded"),
        QualificationCheck("PARTITION-TOTAL-SIZE-001", limits.max_total_size_bytes is None or max(stats.total_size_bytes) <= limits.max_total_size_bytes, {"per_partition": stats.total_size_bytes}, limits.max_total_size_bytes, "total size exceeded"), QualificationCheck("PARTITION-CUT-001", cut_pass, {"cut_fraction": stats.cut_fraction, "locality": stats.locality}, {"max_cut_fraction": limits.max_cut_fraction, "min_locality": limits.min_locality}, "cut/locality exceeded"),
        QualificationCheck("PARTITION-MEMORY-001", True, {}), QualificationCheck("PARTITION-TEMP-DISK-001", True, {}), QualificationCheck("PARTITION-BACKEND-001", True, {"backend": backend}), QualificationCheck("PARTITION-OUTPUT-001", True, {}), QualificationCheck("PARTITION-FINGERPRINT-001", True, {"topology": topology.fingerprint}), QualificationCheck("PARTITION-EXTERNAL-MAP-001", True, {"used": backend == "external"}),
    )


def _external_assignments(
    topology: TopologyContext,
    source_root: Path,
    configured: str,
) -> dict[str, np.ndarray]:
    manifest = _safe_json(
        _safe_relative(
            source_root,
            configured,
            "PARTITION-EXTERNAL-MAP-001",
        ),
        "PARTITION-EXTERNAL-MAP-001",
    )
    expected_manifest_keys = {
        "format_version",
        "topology_fingerprint",
        "num_partitions",
        "node_type_offsets",
        "assignment",
    }
    if (
        set(manifest) != expected_manifest_keys
        or manifest.get("format_version")
        != "typed-external-partition-map-v1"
        or manifest.get("num_partitions")
        != topology.num_partitions
    ):
        raise _error(
            "PARTITION-EXTERNAL-MAP-001",
            "invalid external manifest",
        )
    if manifest.get("topology_fingerprint") != topology.fingerprint:
        raise _error(
            "PARTITION-FINGERPRINT-001",
            "external topology differs",
        )
    record = manifest.get("assignment")
    if (
        not isinstance(record, dict)
        or set(record)
        != {"relative_path", "sha256", "dtype", "shape"}
        or not isinstance(record.get("relative_path"), str)
        or not record["relative_path"].endswith(".npy")
    ):
        raise _error(
            "PARTITION-EXTERNAL-MAP-001",
            "assignment must be an exact checksum-pinned .npy record",
        )
    path = _safe_relative(
        source_root,
        record["relative_path"],
        "PARTITION-EXTERNAL-MAP-001",
    )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise _error(
            "PARTITION-EXTERNAL-MAP-001",
            "cannot securely open external assignment",
        ) from error
    try:
        path_stat = path.lstat()
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or (path_stat.st_dev, path_stat.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise _error(
                "PARTITION-EXTERNAL-MAP-001",
                "external assignment is not a pinned regular file",
            )
        digest = hashlib.sha256()
        copied_bytes = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            copied_bytes += len(chunk)
        if (
            copied_bytes != before.st_size
            or digest.hexdigest() != record.get("sha256")
        ):
            raise _error(
                "PARTITION-EXTERNAL-MAP-001",
                "external assignment changed while hashing",
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            with os.fdopen(
                descriptor,
                mode="rb",
                closefd=False,
            ) as stream:
                loaded = np.load(stream, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise _error(
                "PARTITION-EXTERNAL-MAP-001",
                "unsafe NumPy assignment",
            ) from error
        if not isinstance(loaded, np.ndarray):
            close = getattr(loaded, "close", None)
            if callable(close):
                close()
            raise _error(
                "PARTITION-EXTERNAL-MAP-001",
                "external assignment is not one NumPy array",
            )
        values = np.array(loaded, dtype=loaded.dtype, copy=True)
        after = os.fstat(descriptor)
        if (
            (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
        ):
            raise _error(
                "PARTITION-EXTERNAL-MAP-001",
                "external assignment changed while loading",
            )
    finally:
        os.close(descriptor)
    offsets: dict[str, list[int]] = {}
    offset = 0
    for internal, name in topology.internal_node_types.items():
        count = topology.node_counts[name]
        offsets[internal] = [offset, offset + count]
        offset += count
    if (
        manifest.get("node_type_offsets") != offsets
        or values.dtype != np.int64
        or list(values.shape) != record.get("shape")
        or record.get("dtype") != "int64"
        or len(values) != offset
    ):
        raise _error(
            "PARTITION-EXTERNAL-MAP-001",
            "assignment metadata differs",
        )
    return {
        topology.internal_node_types[key]: values[start:end]
        for key, (start, end) in offsets.items()
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping): return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [_json_value(item) for item in value]
    if isinstance(value, np.generic): return value.item()
    return value


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _save_npy(path: Path, value: np.ndarray) -> None:
    with path.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_tree_directories(root: Path) -> None:
    directories = [
        candidate
        for candidate in root.rglob("*")
        if candidate.is_dir()
    ]
    directories.sort(
        key=lambda candidate: len(
            candidate.relative_to(root).parts
        ),
        reverse=True,
    )
    for directory in directories:
        _fsync_directory(directory)
    _fsync_directory(root)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            value,
            stream,
            sort_keys=True,
            separators=(",", ":"),
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _identity_record(
    book: TypedPartitionBook,
    limits: PartitionQualificationLimits,
) -> dict[str, Any]:
    return {
        "format_version": "typed-partition-book-v1",
        "num_partitions": book.num_partitions,
        "topology_fingerprint": book.topology_fingerprint,
        "source_binding": _json_value(book.source_binding),
        "backend": book.backend,
        "backend_version": book.backend_version,
        "options": _json_value(book.options),
        "provenance": _json_value(book.provenance),
        "estimated_resources": _json_value(
            book.estimated_resources
        ),
        "measured_resources": _json_value(
            book.measured_resources
        ),
        "content_identity": book.content_identity,
        "limits_fingerprint": limits.fingerprint,
    }


def _immutable_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _immutable_json(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_json(item) for item in value)
    return value


def _partition_evidence(
    book: TypedPartitionBook,
    limits: PartitionQualificationLimits,
) -> Mapping[str, Any]:
    return _immutable_json(
        {
            "statistics": book.statistics.as_record(),
            "qualification": {
                "limits": limits.as_record(),
                "checks": tuple(
                    check.as_record()
                    for check in book.qualification_checks
                ),
            },
        }
    )


def _publish_subtree(
    topology: TopologyContext,
    artifact: Path,
    book: TypedPartitionBook,
    limits: PartitionQualificationLimits,
    *,
    work_parent: Path,
) -> None:
    work_parent.mkdir(parents=True, exist_ok=True)
    temporary = work_parent / uuid.uuid4().hex
    temporary.mkdir()
    outputs: dict[str, str] = {}
    try:
        for index, name in enumerate(topology.node_types):
            root = temporary / "node_types" / f"n{index:04d}"; root.mkdir(parents=True)
            for filename, value in (("assignment", book.node_assignments[name]), ("permutation", book.node_permutations[name]), ("inverse", book.node_inverse_permutations[name]), ("partptr", book.node_partptr[name])):
                path = root / f"{filename}.npy"; _save_npy(path, value); outputs[path.relative_to(temporary).as_posix()] = _file_sha(path)
        for index, relation in enumerate(sorted(topology.relations)):
            root = temporary / "relations" / f"r{index:04d}"; root.mkdir(parents=True); path = root / "edge_partition.npy"; _save_npy(path, book.edge_ownership[relation]); outputs[path.relative_to(temporary).as_posix()] = _file_sha(path)
        identity = _identity_record(book, limits)
        _atomic_json(temporary / "partition_book.json", identity); _atomic_json(temporary / "statistics.json", book.statistics.as_record()); _atomic_json(temporary / "qualification.json", {"limits": limits.as_record(), "checks": [x.as_record() for x in book.qualification_checks]})
        for name in ("partition_book.json", "statistics.json", "qualification.json"): outputs[name] = _file_sha(temporary / name)
        _atomic_json(temporary / "partitions.complete.json", {"format_version": "typed-partitions-completion-v1", "content_identity": book.content_identity, "outputs": dict(sorted(outputs.items()))})
        _fsync_tree_directories(temporary)
        os.replace(temporary, artifact)
        _fsync_directory(artifact.parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        if work_parent.is_dir() and not any(work_parent.iterdir()):
            work_parent.rmdir()


def _republish_partition_metadata(
    artifact: Path,
    book: TypedPartitionBook,
    limits: PartitionQualificationLimits,
    *,
    work_parent: Path,
) -> None:
    work_parent.mkdir(parents=True, exist_ok=True)
    temporary = work_parent / uuid.uuid4().hex
    backup = (
        artifact.parent
        / f".partitions-tmp-previous-{uuid.uuid4().hex}"
    )
    try:
        shutil.copytree(
            artifact,
            temporary,
            copy_function=os.link,
        )
        _atomic_json(
            temporary / "partition_book.json",
            _identity_record(book, limits),
        )
        _atomic_json(
            temporary / "qualification.json",
            {
                "limits": limits.as_record(),
                "checks": [
                    check.as_record()
                    for check in book.qualification_checks
                ],
            },
        )
        completion = _safe_json(
            temporary / "partitions.complete.json",
            "PARTITION-OUTPUT-001",
        )
        outputs = completion.get("outputs")
        if not isinstance(outputs, dict):
            raise _error(
                "PARTITION-OUTPUT-001",
                "published completion is malformed",
            )
        outputs["partition_book.json"] = _file_sha(
            temporary / "partition_book.json"
        )
        outputs["qualification.json"] = _file_sha(
            temporary / "qualification.json"
        )
        _atomic_json(
            temporary / "partitions.complete.json",
            {
                "format_version": "typed-partitions-completion-v1",
                "content_identity": book.content_identity,
                "outputs": dict(sorted(outputs.items())),
            },
        )
        _fsync_tree_directories(temporary)
        os.replace(artifact, backup)
        try:
            os.replace(temporary, artifact)
        except Exception:
            os.replace(backup, artifact)
            raise
        _fsync_directory(artifact.parent)
        shutil.rmtree(backup)
        _fsync_directory(artifact.parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        if work_parent.is_dir() and not any(work_parent.iterdir()):
            work_parent.rmdir()


def _load_published_array(path: Path, role: str) -> np.ndarray:
    try:
        value = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False,
        )
    except (OSError, ValueError) as error:
        raise _error(
            "PARTITION-OUTPUT-001",
            f"{role} is malformed",
        ) from error
    if not isinstance(value, np.ndarray):
        close = getattr(value, "close", None)
        if callable(close):
            close()
        raise _error(
            "PARTITION-OUTPUT-001",
            f"{role} is not one NumPy array",
        )
    return value


def _limits_from_record(value: Any) -> PartitionQualificationLimits:
    expected = {
        "max_nodes_per_type",
        "max_phase_nodes",
        "max_edges_per_relation",
        "max_feature_bytes",
        "max_total_size_bytes",
        "max_cut_fraction",
        "min_locality",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise _error(
            "PARTITION-OUTPUT-001",
            "stored qualification limits are malformed",
        )
    relation_limits = value["max_edges_per_relation"]
    if not isinstance(relation_limits, dict):
        raise _error(
            "PARTITION-OUTPUT-001",
            "stored relation limits are malformed",
        )
    parsed_relations: dict[CanonicalRelation, Any] = {}
    try:
        for encoded, limit in relation_limits.items():
            relation = json.loads(encoded)
            if (
                not isinstance(relation, list)
                or len(relation) != 3
                or any(not isinstance(item, str) for item in relation)
            ):
                raise ValueError("invalid canonical relation")
            parsed_relations[tuple(relation)] = limit
        return PartitionQualificationLimits(
            max_nodes_per_type=value["max_nodes_per_type"],
            max_phase_nodes=value["max_phase_nodes"],
            max_edges_per_relation=parsed_relations,
            max_feature_bytes=value["max_feature_bytes"],
            max_total_size_bytes=value["max_total_size_bytes"],
            max_cut_fraction=value["max_cut_fraction"],
            min_locality=value["min_locality"],
        )
    except (
        json.JSONDecodeError,
        TypeError,
        ValueError,
        _ingestion().ArtifactValidationError,
    ) as error:
        raise _error(
            "PARTITION-OUTPUT-001",
            "stored qualification limits are malformed",
        ) from error


def _read_subtree(
    topology: TopologyContext,
    artifact: Path,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Any],
    PartitionQualificationLimits,
]:
    if artifact.is_symlink() or not artifact.is_dir():
        raise _error(
            "PARTITION-OUTPUT-001",
            "unsafe partition subtree",
        )
    expected_outputs = {
        "partition_book.json",
        "statistics.json",
        "qualification.json",
    }
    for index, _ in enumerate(topology.node_types):
        expected_outputs.update(
            {
                f"node_types/n{index:04d}/assignment.npy",
                f"node_types/n{index:04d}/permutation.npy",
                f"node_types/n{index:04d}/inverse.npy",
                f"node_types/n{index:04d}/partptr.npy",
            }
        )
    for index, _ in enumerate(sorted(topology.relations)):
        expected_outputs.add(
            f"relations/r{index:04d}/edge_partition.npy"
        )
    completion = _safe_json(
        artifact / "partitions.complete.json",
        "PARTITION-OUTPUT-001",
    )
    outputs = completion.get("outputs")
    if (
        completion.get("format_version")
        != "typed-partitions-completion-v1"
        or not isinstance(outputs, dict)
        or set(outputs) != expected_outputs
    ):
        raise _error(
            "PARTITION-OUTPUT-001",
            "partition completion does not declare the exact output set",
        )
    observed_outputs: set[str] = set()
    for path in artifact.rglob("*"):
        if path.is_symlink():
            raise _error(
                "PARTITION-OUTPUT-001",
                "partition subtree contains a symlink",
            )
        if path.is_file():
            observed_outputs.add(
                path.relative_to(artifact).as_posix()
            )
    if observed_outputs != expected_outputs | {
        "partitions.complete.json"
    }:
        raise _error(
            "PARTITION-OUTPUT-001",
            "partition subtree contains missing or unknown files",
        )
    for relative, digest in outputs.items():
        path = _safe_relative(
            artifact,
            relative,
            "PARTITION-OUTPUT-001",
        )
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or not path.is_file()
            or _file_sha(path) != digest
        ):
            raise _error(
                "PARTITION-OUTPUT-001",
                f"corrupt {relative}",
            )
    identity = _safe_json(
        artifact / "partition_book.json",
        "PARTITION-OUTPUT-001",
    )
    qualification = _safe_json(
        artifact / "qualification.json",
        "PARTITION-OUTPUT-001",
    )
    if set(qualification) != {"limits", "checks"}:
        raise _error(
            "PARTITION-OUTPUT-001",
            "published qualification evidence is malformed",
        )
    stored_limits = _limits_from_record(qualification["limits"])
    expected_identity_keys = {
        "format_version",
        "num_partitions",
        "topology_fingerprint",
        "source_binding",
        "backend",
        "backend_version",
        "options",
        "provenance",
        "estimated_resources",
        "measured_resources",
        "content_identity",
        "limits_fingerprint",
    }
    if (
        set(identity) != expected_identity_keys
        or identity.get("format_version")
        != "typed-partition-book-v1"
        or identity.get("num_partitions")
        != topology.num_partitions
        or identity.get("topology_fingerprint")
        != topology.fingerprint
        or identity.get("source_binding")
        != _json_value(topology.source_binding)
        or identity.get("limits_fingerprint") != stored_limits.fingerprint
        or completion.get("content_identity")
        != identity.get("content_identity")
        or not isinstance(identity.get("backend"), str)
        or not isinstance(identity.get("backend_version"), str)
        or not isinstance(identity.get("options"), dict)
        or not isinstance(identity.get("provenance"), dict)
        or not isinstance(identity.get("estimated_resources"), dict)
        or not isinstance(identity.get("measured_resources"), dict)
        or not isinstance(identity.get("content_identity"), str)
        or len(identity["content_identity"]) != 64
    ):
        raise _error(
            "PARTITION-FINGERPRINT-001",
            "published partition identity evidence is malformed",
        )
    assignments: dict[str, np.ndarray] = {}
    derived: dict[str, dict[str, np.ndarray]] = {}
    for index, name in enumerate(topology.node_types):
        root = artifact / "node_types" / f"n{index:04d}"
        assignments[name] = _load_published_array(
            root / "assignment.npy",
            f"published assignment for {name!r}",
        )
        derived[name] = {
            role: _load_published_array(
                root / f"{filename}.npy",
                f"published {role} for {name!r}",
            )
            for role, filename in (
                ("permutation", "permutation"),
                ("inverse", "inverse"),
                ("partptr", "partptr"),
            )
        }
        assignment = assignments[name]
        if (
            assignment.dtype != np.int64
            or assignment.ndim != 1
            or len(assignment) != topology.node_counts[name]
        ):
            raise _error(
                "PARTITION-ID-001",
                f"published assignment for {name!r} is malformed",
            )
        ordinal = np.arange(len(assignment), dtype=np.int64)
        permutation = np.lexsort((ordinal, assignment))
        inverse = np.empty(len(assignment), dtype=np.int64)
        inverse[permutation] = ordinal
        partptr = np.concatenate(
            (
                [0],
                np.cumsum(
                    np.bincount(
                        assignment,
                        minlength=topology.num_partitions,
                    )
                ),
            )
        )
        if (
            not np.array_equal(
                derived[name]["permutation"],
                permutation,
            )
            or not np.array_equal(
                derived[name]["inverse"],
                inverse,
            )
            or not np.array_equal(
                derived[name]["partptr"],
                partptr,
            )
        ):
            raise _error(
                "PARTITION-ID-001",
                f"published derived arrays for {name!r} are inconsistent",
            )
    ownership = _derive_ownership(topology, assignments)
    for index, relation in enumerate(sorted(topology.relations)):
        observed = _load_published_array(
            artifact
            / "relations"
            / f"r{index:04d}"
            / "edge_partition.npy",
            f"published ownership for {relation!r}",
        )
        if not np.array_equal(observed, ownership[relation]):
            raise _error(
                "PARTITION-OUTPUT-001",
                f"published ownership for {relation!r} differs",
            )
    statistics = _safe_json(
        artifact / "statistics.json",
        "PARTITION-OUTPUT-001",
    )
    expected_statistics = _statistics(
        topology,
        assignments,
        ownership,
    ).as_record()
    if statistics != expected_statistics:
        raise _error(
            "PARTITION-OUTPUT-001",
            "published partition statistics are inconsistent",
        )
    book = _qualified_book(
        topology,
        assignments,
        stored_limits,
        backend=identity["backend"],
        estimated_resources=identity["estimated_resources"],
        measured_resources=identity["measured_resources"],
    )
    if identity != _identity_record(book, stored_limits):
        raise _error(
            "PARTITION-FINGERPRINT-001",
            "published partition identity fields are inconsistent",
        )
    expected_qualification = {
        "limits": stored_limits.as_record(),
        "checks": [
            check.as_record()
            for check in book.qualification_checks
        ],
    }
    if qualification != expected_qualification:
        raise _error(
            "PARTITION-OUTPUT-001",
            "published qualification evidence is inconsistent",
        )
    return assignments, identity, stored_limits

__all__ = ["CanonicalRelationTopology", "PyGPartitionArtifactAdapter", "TopologyContext", "TopologyOnlyPyGPartitioner", "TypedPartitionBuild"]
