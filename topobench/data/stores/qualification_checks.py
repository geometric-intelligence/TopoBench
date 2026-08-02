"""Canonical qualification for immutable typed graph stores.

This module deliberately depends only on the Python standard library and NumPy.
Parquet schema restoration is an explicit, lazy operation owned by
:mod:`typed_graph_store`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
_FileIdentity = tuple[int, int, int, int, int]
from numpy.lib import format as npy_format

STORE_FORMAT_VERSION = "typed-graph-store-v1"
CONTENT_HASH_VERSION = "typed-graph-store-content-v1"
REPORT_FORMAT_VERSION = "typed-graph-qualification-v1"
_PHASES = ("train", "val", "test")
_CHUNK_ROWS = 64 * 1024
_MANIFEST_KEYS = {
    "format_version",
    "content_hash_version",
    "qualification_check_set",
    "content_sha256",
    "metadata_binding_sha256",
    "output_kind",
    "target_node_type",
    "target_internal_key",
    "active_split_tag",
    "supported_capabilities",
    "task_bindings",
    "source_binding",
    "nodes",
    "relations",
    "splits",
    "partition",
    "build_environment",
    "qualification_report",
    "files",
}
_FILE_RECORD_KEYS = {
    "relative_path",
    "role",
    "dtype",
    "shape",
    "byte_size",
    "sha256",
    "finite",
}
_CAPABILITIES = [
    "homogeneous-cluster",
    "heterogeneous-cluster",
    "heterogeneous-neighbor",
    "pyg-feature-store",
    "pyg-graph-store-csc",
]
_METADATA_BINDING_KEYS = (
    "output_kind",
    "target_node_type",
    "target_internal_key",
    "active_split_tag",
    "task_bindings",
    "source_binding",
    "nodes",
    "relations",
    "qualification_check_set",
    "splits",
    "partition",
)


@dataclass(frozen=True, slots=True)
class QualificationCheckResult:
    """One stable qualification result with actionable local evidence."""

    check_id: str
    passed: bool
    observed: Any
    expected: Any = None
    limit: Any = None
    evidence: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    remediation: str = ""

    def as_record(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "passed": self.passed,
            "observed": _json_value(self.observed),
            "expected": _json_value(self.expected),
            "limit": _json_value(self.limit),
            "evidence": _json_value(self.evidence),
            "remediation": self.remediation,
        }


@dataclass(frozen=True, slots=True)
class QualificationReport:
    """A versioned non-executable report for one local store path."""

    passed: bool
    checks: tuple[QualificationCheckResult, ...]
    report_path: Path
    store_path: Path
    format_version: str = REPORT_FORMAT_VERSION

    @property
    def failures(self) -> tuple[QualificationCheckResult, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def as_record(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "passed": self.passed,
            "store_path": str(self.store_path),
            "report_path": str(self.report_path),
            "checks": [check.as_record() for check in self.checks],
        }


class QualificationFailure(RuntimeError):
    """Raised after a structured qualification report records a failure."""

    def __init__(self, result: QualificationCheckResult, report_path: Path) -> None:
        self.result = result
        self.check_id = result.check_id
        self.report_path = report_path
        super().__init__(
            f"{result.check_id}: store qualification failed; report={report_path}; "
            f"observed={result.observed!r}; expected={result.expected!r}"
        )


@dataclass(frozen=True, slots=True)
class ValidatedStore:
    """Canonical manifest returned only after every store invariant passes."""

    root: Path
    manifest: Mapping[str, Any]
    report: QualificationReport
    file_identities: Mapping[str, _FileIdentity]


class _CheckError(Exception):
    def __init__(self, result: QualificationCheckResult) -> None:
        self.result = result
        super().__init__(result.check_id)


class _Checker:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.results: list[QualificationCheckResult] = []
        self.handles: list[_ArrayHandle] = []
        self.file_identities: dict[str, _FileIdentity] = {}

    def track(self, handle: _ArrayHandle) -> _ArrayHandle:
        self.handles.append(handle)
        return handle

    def passed(
        self,
        check_id: str,
        *,
        observed: Any,
        expected: Any,
        evidence: Mapping[str, Any],
    ) -> None:
        self.results.append(
            QualificationCheckResult(
                check_id,
                True,
                observed,
                expected,
                evidence=MappingProxyType(dict(evidence)),
                remediation="none",
            )
        )

    def fail(
        self,
        check_id: str,
        *,
        observed: Any,
        expected: Any = None,
        limit: Any = None,
        evidence: Mapping[str, Any] | None = None,
        remediation: str,
    ) -> None:
        result = QualificationCheckResult(
            check_id,
            False,
            observed,
            expected,
            limit,
            MappingProxyType(dict(evidence or {"store_path": str(self.root)})),
            remediation,
        )
        self.results.append(result)
        raise _CheckError(result)


@dataclass(slots=True)
class _ArrayHandle:
    array: np.ndarray
    stream: Any
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        mmap = getattr(self.array, "_mmap", None)
        if mmap is not None:
            mmap.close()
        self.stream.close()


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _frame(digest: Any, role: str, payload: bytes) -> None:
    encoded_role = role.encode("utf-8")
    digest.update(len(encoded_role).to_bytes(8, "big"))
    digest.update(encoded_role)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def compute_content_identity(manifest: Mapping[str, Any]) -> str:
    """Compute the collision-framed semantic identity of a manifest."""
    semantic = dict(manifest)
    semantic.pop("content_sha256", None)
    digest = hashlib.sha256()
    _frame(digest, "format", CONTENT_HASH_VERSION.encode("ascii"))
    _frame(digest, "manifest", _canonical_json(semantic))
    return digest.hexdigest()


def compute_metadata_binding(manifest: Mapping[str, Any]) -> str:
    """Bind metadata records to the store semantics without a hash cycle."""
    semantic = {key: manifest[key] for key in _METADATA_BINDING_KEYS}
    digest = hashlib.sha256()
    _frame(digest, "format", b"typed-graph-store-metadata-binding-v1")
    _frame(digest, "semantic-manifest", _canonical_json(semantic))
    return digest.hexdigest()


def qualification_check_set_fingerprint(
    checks: Sequence[Mapping[str, Any]],
) -> str:
    """Hash the exact ordered upstream Task6 qualification evidence."""
    digest = hashlib.sha256()
    _frame(digest, "format", b"typed-graph-qualification-check-set-v1")
    _frame(digest, "count", str(len(checks)).encode("ascii"))
    for ordinal, check in enumerate(checks):
        _frame(
            digest,
            f"check:{ordinal}",
            _canonical_json(check),
        )
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_manifest_schema(
    manifest: Mapping[str, Any],
    checker: _Checker,
) -> None:
    if set(manifest) != _MANIFEST_KEYS:
        checker.fail(
            "MANIFEST-SCHEMA-001",
            observed=sorted(manifest),
            expected=sorted(_MANIFEST_KEYS),
            remediation="rebuild with the exact supported manifest schema",
        )
    source = manifest.get("source_binding")
    source_keys = {
        "source_fingerprint",
        "config_fingerprint",
        "dependency_versions",
        "task3_content_sha256",
        "task4_content_sha256",
        "task6_source_binding",
        "partition_book_identity",
    }
    scalar_sha_keys = (
        "source_fingerprint",
        "config_fingerprint",
        "task3_content_sha256",
        "task4_content_sha256",
        "partition_book_identity",
    )
    qualification_set = manifest.get("qualification_check_set")
    if (
        not _is_sha256(manifest.get("content_sha256"))
        or not _is_sha256(manifest.get("metadata_binding_sha256"))
        or manifest.get("output_kind") not in {"homogeneous", "heterogeneous"}
        or not isinstance(manifest.get("target_node_type"), str)
        or not manifest["target_node_type"]
        or not isinstance(manifest.get("target_internal_key"), str)
        or not isinstance(manifest.get("active_split_tag"), str)
        or manifest.get("supported_capabilities") != _CAPABILITIES
        or not isinstance(manifest.get("task_bindings"), dict)
        or not isinstance(manifest.get("nodes"), dict)
        or not isinstance(manifest.get("relations"), dict)
        or not isinstance(manifest.get("splits"), dict)
        or not isinstance(manifest.get("partition"), dict)
        or manifest.get("build_environment") != "build_environment.json"
        or manifest.get("qualification_report") != "qualification_report.json"
        or not isinstance(source, dict)
        or set(source) != source_keys
        or any(not _is_sha256(source.get(key)) for key in scalar_sha_keys)
        or not isinstance(source.get("task6_source_binding"), dict)
        or not isinstance(source.get("dependency_versions"), dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in source.get("dependency_versions", {}).items()
        )
        or not isinstance(qualification_set, dict)
        or set(qualification_set) != {"count", "sha256"}
        or isinstance(qualification_set.get("count"), bool)
        or not isinstance(qualification_set.get("count"), int)
        or qualification_set["count"] <= 0
        or not _is_sha256(qualification_set.get("sha256"))
    ):
        checker.fail(
            "MANIFEST-SCHEMA-001",
            observed="invalid manifest scalar, enum, or source identity",
            expected="exact typed manifest values and SHA256 identities",
            remediation="rebuild the manifest from one qualified Task1-6 stage",
        )


def _expected_file_role(relative: str) -> tuple[str, str] | None:
    exact = {
        "build_environment.json": ("build-environment", "json"),
        "qualification_report.json": ("qualification-report", "json"),
        "partitions/partition_book.json": ("partition-book", "json"),
        "partitions/statistics.json": ("partition-statistics", "json"),
    }
    if relative in exact:
        return exact[relative]
    patterns = (
        (r"nodes/(n\d{4})/x\.npy", "node-feature:{0}", "npy"),
        (r"nodes/(n\d{4})/y\.npy", "node-label:{0}", "npy"),
        (
            r"nodes/(n\d{4})/node_ids\.parquet",
            "external-node-ids:{0}",
            "parquet",
        ),
        (
            r"relations/(r\d{4})/(colptr|row)\.npy",
            "relation-{1}:{0}",
            "npy",
        ),
        (
            r"relations/(r\d{4})/edge_id\.npy",
            "relation-edge-id:{0}",
            "npy",
        ),
        (
            r"relations/(r\d{4})/fields/([^/]+)\.npy",
            "relation-field:{0}:{1}",
            "npy",
        ),
        (
            r"splits/([^/]+)/(train|val|test)_ids\.npy",
            "split:{0}:{1}",
            "npy",
        ),
        (
            r"partitions/node_types/(n\d{4})/(assignment|permutation|inverse|partptr)\.npy",
            "partition-{1}:{0}",
            "npy",
        ),
        (
            r"partitions/relations/(r\d{4})/edge_partition\.npy",
            "partition-edge-ownership:{0}",
            "npy",
        ),
    )
    for pattern, role, kind in patterns:
        match = re.fullmatch(pattern, relative)
        if match is not None:
            return role.format(*match.groups()), kind
    return None

def _safe_numpy_dtype(value: str) -> bool:
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError):
        return False
    return dtype.fields is None and dtype.subdtype is None and dtype.kind in {
        "b",
        "i",
        "u",
        "f",
    }



def split_fingerprint(tag: str, split: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    """Hash split semantics and exact phase IDs with stable framing."""
    digest = hashlib.sha256()
    _frame(digest, "format", b"typed-store-split-v1")
    _frame(digest, "tag", tag.encode("utf-8"))
    _frame(digest, "coverage", str(split["coverage"]).encode("ascii"))
    _frame(digest, "qualified", b"1" if split["qualified"] else b"0")
    _frame(digest, "population", str(split["supervision_population"]).encode("ascii"))
    for phase in _PHASES:
        _array_frame(digest, phase, arrays[phase])
    return digest.hexdigest()


def _array_frame(digest: Any, role: str, value: np.ndarray) -> None:
    array = np.asarray(value)
    _frame(digest, f"{role}:dtype", array.dtype.str.encode("ascii"))
    _frame(digest, f"{role}:shape", _canonical_json(list(array.shape)))
    encoded_role = role.encode("utf-8")
    digest.update(len(encoded_role).to_bytes(8, "big"))
    digest.update(encoded_role)
    digest.update(array.nbytes.to_bytes(8, "big"))
    if array.ndim == 0:
        digest.update(np.asarray(array, order="C").tobytes(order="C"))
        return
    for start in range(0, array.shape[0], _CHUNK_ROWS):
        digest.update(np.asarray(array[start : start + _CHUNK_ROWS], order="C").tobytes(order="C"))


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError("unsafe relative path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in posix.parts
        or value in {".", ""}
        or posix.as_posix() != value
    ):
        raise ValueError("unsafe relative path")
    return value


def _secure_descriptor(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        path_stat = path.lstat()
        descriptor_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or not stat.S_ISREG(descriptor_stat.st_mode)
            or path_stat.st_nlink != 1
            or descriptor_stat.st_nlink != 1
            or (path_stat.st_dev, path_stat.st_ino)
            != (descriptor_stat.st_dev, descriptor_stat.st_ino)
        ):
            raise OSError("not a uniquely linked regular file")
        return descriptor, descriptor_stat
    except BaseException:
        os.close(descriptor)
        raise


def _read_json_secure(
    path: Path,
    expected_identity: _FileIdentity | None = None,
) -> dict[str, Any]:
    descriptor, before = _secure_descriptor(path)
    if (
        expected_identity is not None
        and _stat_identity(before) != expected_identity
    ):
        os.close(descriptor)
        raise OSError("JSON artifact changed after checksum validation")
    try:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        if total != before.st_size or _stat_identity(before) != _stat_identity(after):
            raise OSError("file changed while reading")
        value = json.loads(b"".join(chunks).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON root must be an object")
        return value
    finally:
        os.close(descriptor)


def _stat_identity(value: os.stat_result) -> _FileIdentity:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _hash_file(path: Path) -> tuple[str, int, _FileIdentity]:
    descriptor, before = _secure_descriptor(path)
    digest = hashlib.sha256()
    total = 0
    try:
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        if total != before.st_size or _stat_identity(before) != _stat_identity(after):
            raise OSError("file changed while hashing")
        return digest.hexdigest(), total, _stat_identity(before)
    finally:
        os.close(descriptor)


def _open_npy(
    path: Path,
    expected_identity: _FileIdentity | None = None,
) -> _ArrayHandle:
    descriptor, before = _secure_descriptor(path)
    if (
        expected_identity is not None
        and _stat_identity(before) != expected_identity
    ):
        os.close(descriptor)
        raise OSError("array changed after checksum validation")
    stream = os.fdopen(descriptor, "rb", closefd=True)
    try:
        version = npy_format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = npy_format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran_order, dtype = npy_format.read_array_header_2_0(stream)
        elif version == (3, 0):
            shape, fortran_order, dtype = npy_format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported npy version {version!r}")
        offset = stream.tell()
        array = np.memmap(
            stream,
            dtype=dtype,
            mode="r",
            offset=offset,
            shape=shape,
            order="F" if fortran_order else "C",
        )
        after = os.fstat(stream.fileno())
        if _stat_identity(before) != _stat_identity(after):
            raise OSError("array changed while opening")
        array.flags.writeable = False
        return _ArrayHandle(array, stream)
    except BaseException:
        stream.close()
        raise


def _file_records(manifest: Mapping[str, Any], checker: _Checker) -> dict[str, dict[str, Any]]:
    raw = manifest.get("files")
    if not isinstance(raw, list):
        checker.fail(
            "MANIFEST-001",
            observed=type(raw).__name__,
            expected="list of exact file records",
            remediation="rebuild the store from the qualified Task1-6 stage",
        )
    records: dict[str, dict[str, Any]] = {}
    for ordinal, item in enumerate(raw):
        try:
            relative = _safe_relative(item["relative_path"])
        except (KeyError, TypeError, ValueError) as error:
            checker.fail(
                "MANIFEST-001",
                observed=f"invalid file record {ordinal}: {error}",
                expected="safe unique relative path",
                remediation="rebuild the manifest",
            )
        if (
            not isinstance(item, dict)
            or set(item) != _FILE_RECORD_KEYS
            or relative in records
            or not isinstance(item.get("role"), str)
            or isinstance(item.get("byte_size"), bool)
            or not isinstance(item.get("byte_size"), int)
            or item["byte_size"] < 0
            or not _is_sha256(item.get("sha256"))
            or not isinstance(item.get("dtype"), str)
            or not isinstance(item.get("shape"), list)
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 0
                for size in item["shape"]
            )
            or not isinstance(item.get("finite"), bool)
        ):
            checker.fail(
                "MANIFEST-SCHEMA-001",
                observed=item,
                expected="exact unique typed file record",
                remediation="rebuild the manifest",
            )
        policy = _expected_file_role(relative)
        kind = policy[1] if policy is not None else None
        if (
            policy is None
            or item["role"] != policy[0]
            or (
                kind == "json"
                and (item["dtype"] != "json" or item["shape"] != [])
            )
            or (
                kind == "parquet"
                and (
                    not item["dtype"].startswith("parquet:")
                    or len(item["shape"]) != 2
                )
            )
            or (
                kind == "npy"
                and (
                    not item["shape"]
                    or not _safe_numpy_dtype(item["dtype"])
                )
            )
        ):
            checker.fail(
                "FILE-POLICY-001",
                observed={"path": relative, "role": item["role"]},
                expected="one declared non-executable Task7 artifact role/path/type",
                remediation="remove executable or undeclared payloads and rebuild",
            )
        records[relative] = item
    return records


def _scan_file_set(root: Path, checker: _Checker) -> tuple[set[str], set[str]]:
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        status = path.lstat()
        if stat.S_ISDIR(status.st_mode):
            if path.is_symlink():
                checker.fail(
                    "ARTIFACT-TYPE-001",
                    observed=relative,
                    expected="real directory",
                    remediation="remove links and rebuild the immutable store",
                )
            observed_directories.add(relative)
            continue
        if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1 or path.is_symlink():
            checker.fail(
                "ARTIFACT-TYPE-001",
                observed={"path": relative, "mode": status.st_mode, "links": status.st_nlink},
                expected="one uniquely linked regular file",
                remediation="remove special/link artifacts and rebuild",
            )
        observed_files.add(relative)
    return observed_files, observed_directories


def _same_record(nested: Any, file_record: Mapping[str, Any]) -> bool:
    if not isinstance(nested, Mapping):
        return False
    return all(
        nested.get(key) == file_record.get(key)
        for key in ("relative_path", "role", "dtype", "shape")
    )


def _validate_array_record(
    root: Path,
    record: Mapping[str, Any],
    files: Mapping[str, Mapping[str, Any]],
    checker: _Checker,
) -> _ArrayHandle:
    relative = record.get("relative_path")
    if not isinstance(relative, str) or relative not in files or not _same_record(record, files[relative]):
        checker.fail(
            "MANIFEST-001",
            observed=record,
            expected="array record identical to exact file record",
            remediation="rebuild the manifest",
        )
    try:
        handle = checker.track(
            _open_npy(
                root / relative,
                checker.file_identities.get(relative),
            )
        )
    except (OSError, ValueError) as error:
        checker.fail(
            "ARRAY-SHAPE-001",
            observed=f"{relative}: {error}",
            expected={"dtype": record.get("dtype"), "shape": record.get("shape")},
            remediation="rebuild the array from source",
        )
    actual_dtype = handle.array.dtype
    expected_dtype = np.dtype(record.get("dtype"))
    if actual_dtype.itemsize > 1 and actual_dtype.byteorder == ">":
        handle.close()
        checker.fail(
            "ARRAY-ENDIANNESS-001",
            observed=actual_dtype.str,
            expected="little-endian storage",
            evidence={"relative_path": relative},
            remediation="rewrite the array in canonical little-endian form",
        )
    if actual_dtype != expected_dtype:
        handle.close()
        checker.fail(
            "ARRAY-DTYPE-001",
            observed=actual_dtype.str,
            expected=expected_dtype.str,
            evidence={"relative_path": relative},
            remediation="rewrite the array with the declared exact dtype",
        )
    expected_shape = tuple(record.get("shape", ()))
    if handle.array.shape != expected_shape:
        actual = handle.array.shape
        handle.close()
        checker.fail(
            "ARRAY-SHAPE-001",
            observed=list(actual),
            expected=list(expected_shape),
            evidence={"relative_path": relative},
            remediation="rebuild the complete aligned array",
        )
    if record.get("finite") is True and np.issubdtype(actual_dtype, np.floating):
        for start in range(0, handle.array.shape[0], _CHUNK_ROWS):
            if not np.isfinite(handle.array[start : start + _CHUNK_ROWS]).all():
                handle.close()
                checker.fail(
                    "ARRAY-FINITE-001",
                    observed="non-finite value",
                    expected="all values finite",
                    evidence={"relative_path": relative, "row_start": start},
                    remediation="repair the source values and rebuild",
                )
    return handle


def _strictly_increasing(value: np.ndarray) -> bool:
    previous: int | None = None
    for start in range(0, len(value), _CHUNK_ROWS):
        chunk = value[start : start + _CHUNK_ROWS]
        if len(chunk) > 1 and np.any(chunk[1:] <= chunk[:-1]):
            return False
        if previous is not None and len(chunk) and int(chunk[0]) <= previous:
            return False
        if len(chunk):
            previous = int(chunk[-1])
    return True


def _arrays_overlap(left: np.ndarray, right: np.ndarray) -> bool:
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_value = int(left[left_index])
        right_value = int(right[right_index])
        if left_value == right_value:
            return True
        if left_value < right_value:
            left_index += 1
        else:
            right_index += 1
    return False


def _validate_nodes(
    root: Path,
    manifest: Mapping[str, Any],
    files: Mapping[str, Mapping[str, Any]],
    checker: _Checker,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    nodes = manifest.get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        checker.fail(
            "MANIFEST-001",
            observed=nodes,
            expected="non-empty stable node map",
            remediation="rebuild the store",
        )
    expected_keys = [f"n{ordinal:04d}" for ordinal in range(len(nodes))]
    if list(nodes) != expected_keys:
        checker.fail(
            "MANIFEST-001",
            observed=list(nodes),
            expected=expected_keys,
            remediation="restore collision-safe stable internal keys",
        )
    names: dict[str, str] = {}
    for key, node in nodes.items():
        if (
            not isinstance(node, dict)
            or set(node)
            != {
                "internal_key",
                "name",
                "id_dtype",
                "count",
                "feature_width",
                "x",
                "y",
                "node_ids",
                "fields",
            }
            or node.get("internal_key") != key
            or not isinstance(node.get("name"), str)
            or not node["name"]
            or node["name"] in names
            or node.get("id_dtype") not in {"int64", "uint64", "string"}
            or isinstance(node.get("count"), bool)
            or not isinstance(node.get("count"), int)
            or node["count"] < 0
            or isinstance(node.get("feature_width"), bool)
            or not isinstance(node.get("feature_width"), int)
            or node["feature_width"] <= 0
            or not isinstance(node.get("fields"), dict)
        ):
            checker.fail(
                "MANIFEST-001",
                observed=node,
                expected="unique named node record with non-negative count",
                remediation="rebuild node metadata",
            )
        names[node["name"]] = key
        feature = _validate_array_record(root, node["x"], files, checker)
        if feature.array.shape != (node["count"], node["feature_width"]):
            actual = list(feature.array.shape)
            feature.close()
            checker.fail(
                "ARRAY-SHAPE-001",
                observed=actual,
                expected=[node["count"], node.get("feature_width")],
                evidence={"node_type": node["name"]},
                remediation="rebuild aligned node features",
            )
        feature.close()
        label = node.get("y")
        if label is not None:
            handle = _validate_array_record(root, label, files, checker)
            if handle.array.shape != (node["count"],):
                actual = list(handle.array.shape)
                handle.close()
                checker.fail(
                    "ARRAY-SHAPE-001",
                    observed=actual,
                    expected=[node["count"]],
                    evidence={"node_type": node["name"], "role": "label"},
                    remediation="rebuild aligned labels",
                )
            handle.close()
        node_ids = node.get("node_ids")
        if not isinstance(node_ids, dict) or not _same_record(node_ids, files.get(node_ids.get("relative_path"), {})):
            checker.fail(
                "MANIFEST-001",
                observed=node_ids,
                expected="checksum-pinned external-ID Parquet record",
                remediation="rebuild type-local external IDs",
            )
    return nodes, names


def _validate_relations(
    root: Path,
    manifest: Mapping[str, Any],
    files: Mapping[str, Mapping[str, Any]],
    nodes: Mapping[str, Mapping[str, Any]],
    checker: _Checker,
) -> dict[str, dict[str, Any]]:
    relations = manifest.get("relations")
    if not isinstance(relations, dict) or not relations:
        checker.fail(
            "MANIFEST-001",
            observed=relations,
            expected="non-empty stable relation map",
            remediation="rebuild relation metadata",
        )
    expected_keys = [f"r{ordinal:04d}" for ordinal in range(len(relations))]
    if list(relations) != expected_keys:
        checker.fail(
            "MANIFEST-001",
            observed=list(relations),
            expected=expected_keys,
            remediation="restore stable relation keys",
        )
    seen: set[tuple[str, str, str]] = set()
    for key, relation in relations.items():
        if (
            not isinstance(relation, dict)
            or set(relation)
            != {
                "internal_key",
                "relation",
                "source_internal_key",
                "destination_internal_key",
                "source_count",
                "destination_count",
                "edge_count",
                "canonical_order",
                "colptr",
                "row",
                "edge_id",
                "fields",
            }
            or isinstance(relation.get("edge_count"), bool)
            or not isinstance(relation.get("edge_count"), int)
            or relation["edge_count"] < 0
            or isinstance(relation.get("source_count"), bool)
            or not isinstance(relation.get("source_count"), int)
            or isinstance(relation.get("destination_count"), bool)
            or not isinstance(relation.get("destination_count"), int)
            or not isinstance(relation.get("canonical_order"), list)
            or any(
                not isinstance(value, str)
                for value in relation.get("canonical_order", ())
            )
            or not isinstance(relation.get("fields"), dict)
        ):
            checker.fail(
                "MANIFEST-SCHEMA-001",
                observed=relation,
                expected="exact typed canonical relation record",
                remediation="rebuild relation metadata",
            )
        triple_raw = relation.get("relation") if isinstance(relation, dict) else None
        if (
            not isinstance(triple_raw, list)
            or len(triple_raw) != 3
            or any(not isinstance(item, str) for item in triple_raw)
        ):
            checker.fail(
                "MANIFEST-001",
                observed=triple_raw,
                expected="canonical relation triple",
                remediation="rebuild relation metadata",
            )
        triple = tuple(triple_raw)
        if (
            triple in seen
            or relation.get("internal_key") != key
            or relation.get("source_internal_key") not in nodes
            or relation.get("destination_internal_key") not in nodes
        ):
            checker.fail(
                "MANIFEST-001",
                observed=relation,
                expected="unique relation with valid node cross-references",
                remediation="rebuild relation metadata",
            )
        seen.add(triple)
        source_count = nodes[relation["source_internal_key"]]["count"]
        destination_count = nodes[relation["destination_internal_key"]]["count"]
        if relation.get("source_count") != source_count or relation.get("destination_count") != destination_count:
            checker.fail(
                "MANIFEST-001",
                observed={"source": relation.get("source_count"), "destination": relation.get("destination_count")},
                expected={"source": source_count, "destination": destination_count},
                remediation="rebuild relation cross-references",
            )
        colptr = _validate_array_record(root, relation["colptr"], files, checker)
        row = _validate_array_record(root, relation["row"], files, checker)
        edge_count = relation.get("edge_count")
        if (
            colptr.array.dtype != np.dtype("<i8")
            or colptr.array.shape != (destination_count + 1,)
            or row.array.dtype != np.dtype("<i8")
            or row.array.shape != (edge_count,)
        ):
            actual = {
                "colptr": list(colptr.array.shape),
                "row": list(row.array.shape),
            }
            colptr.close()
            row.close()
            checker.fail(
                "ARRAY-SHAPE-001",
                observed=actual,
                expected={"colptr": [destination_count + 1], "row": [edge_count]},
                evidence={"relation": list(triple)},
                remediation="rebuild canonical CSC arrays",
            )
        pointers = colptr.array
        rows = row.array
        if (
            len(pointers) == 0
            or int(pointers[0]) != 0
            or int(pointers[-1]) != edge_count
            or not np.all(pointers[1:] >= pointers[:-1])
        ):
            colptr.close()
            row.close()
            checker.fail(
                "CSC-COLPTR-001",
                observed="non-monotone or wrong CSC endpoints",
                expected={"first": 0, "last": edge_count, "monotone": True},
                evidence={"relation": list(triple)},
                remediation="rebuild relation CSC",
            )
        for start in range(0, len(rows), _CHUNK_ROWS):
            chunk = rows[start : start + _CHUNK_ROWS]
            if len(chunk):
                minimum = int(chunk.min())
                maximum = int(chunk.max())
                if minimum < 0 or maximum >= source_count:
                    colptr.close()
                    row.close()
                    checker.fail(
                        "CSC-ROW-BOUNDS-001",
                        observed={"minimum": minimum, "maximum": maximum},
                        limit={"minimum": 0, "exclusive_maximum": source_count},
                        evidence={"relation": list(triple), "edge_start": start},
                        remediation="rebuild endpoint mappings and CSC",
                    )
        edge_id_handle: _ArrayHandle | None = None
        edge_ids: np.ndarray | None = None
        if relation.get("edge_id") is not None:
            edge_id_handle = _validate_array_record(root, relation["edge_id"], files, checker)
            edge_ids = edge_id_handle.array
            if edge_ids.shape[0] != edge_count:
                actual_length = edge_ids.shape[0]
                edge_id_handle.close()
                colptr.close()
                row.close()
                checker.fail(
                    "EDGE-FIELD-LENGTH-001",
                    observed=actual_length,
                    expected=edge_count,
                    evidence={"relation": list(triple), "field": "edge_id"},
                    remediation="rebuild aligned edge IDs",
                )
        for field_name, field_record in relation.get("fields", {}).items():
            field = _validate_array_record(root, field_record, files, checker)
            if field.array.ndim == 0 or field.array.shape[0] != edge_count:
                actual = list(field.array.shape)
                field.close()
                if edge_id_handle is not None:
                    edge_id_handle.close()
                colptr.close()
                row.close()
                checker.fail(
                    "EDGE-FIELD-LENGTH-001",
                    observed=actual,
                    expected={"first_dimension": edge_count},
                    evidence={"relation": list(triple), "field": field_name},
                    remediation="rebuild aligned edge fields",
                )
            field.close()
        ordered = True
        for destination in range(destination_count):
            start = int(pointers[destination])
            end = int(pointers[destination + 1])
            segment = rows[start:end]
            if len(segment) > 1:
                if edge_ids is None:
                    if np.any(segment[1:] <= segment[:-1]):
                        ordered = False
                        break
                else:
                    decreasing = segment[1:] < segment[:-1]
                    equal = segment[1:] == segment[:-1]
                    if np.any(decreasing) or np.any(equal & (edge_ids[start + 1 : end] <= edge_ids[start : end - 1])):
                        ordered = False
                        break
        if edge_id_handle is not None:
            edge_id_handle.close()
        colptr.close()
        row.close()
        if not ordered:
            checker.fail(
                "CSC-ORDER-001",
                observed="non-canonical source/edge-ID order within a destination",
                expected="strict canonical order",
                evidence={"relation": list(triple)},
                remediation="rebuild the relation in canonical CSC order",
            )
    return relations


def _validate_splits(
    root: Path,
    manifest: Mapping[str, Any],
    files: Mapping[str, Mapping[str, Any]],
    nodes: Mapping[str, Mapping[str, Any]],
    checker: _Checker,
) -> dict[str, dict[str, Any]]:
    splits = manifest.get("splits")
    target_key = manifest.get("target_internal_key")
    if not isinstance(splits, dict) or not splits or target_key not in nodes:
        checker.fail(
            "MANIFEST-001",
            observed={"splits": splits, "target": target_key},
            expected="registered splits and valid target node key",
            remediation="rebuild split metadata",
        )
    population = nodes[target_key]["count"]
    for tag, split in splits.items():
        if (
            not isinstance(tag, str)
            or not tag
            or not isinstance(split, dict)
            or set(split)
            != {
                "coverage",
                "qualified",
                "supervision_population",
                "phases",
                "fingerprint",
            }
            or split.get("coverage") not in {"complete", "partial"}
            or not isinstance(split.get("qualified"), bool)
            or isinstance(split.get("supervision_population"), bool)
            or not isinstance(split.get("supervision_population"), int)
            or split["supervision_population"] != population
            or not _is_sha256(split.get("fingerprint"))
        ):
            checker.fail(
                "MANIFEST-SCHEMA-001",
                observed=split,
                expected="exact typed split registry record",
                evidence={"tag": tag},
                remediation="rebuild split metadata",
            )
        phases = split.get("phases") if isinstance(split, dict) else None
        if not isinstance(phases, dict) or set(phases) != set(_PHASES):
            checker.fail(
                "SPLIT-PHASES-001",
                observed=list(phases) if isinstance(phases, dict) else phases,
                expected=list(_PHASES),
                evidence={"tag": tag},
                remediation="publish the exact train/val/test phase triplet",
            )
        handles: dict[str, _ArrayHandle] = {}
        try:
            for phase in _PHASES:
                handle = _validate_array_record(root, phases[phase], files, checker)
                handles[phase] = handle
                values = handle.array
                if (
                    values.dtype != np.dtype("<i8")
                    or values.ndim != 1
                    or (len(values) and (int(values[0]) < 0 or int(values[-1]) >= population))
                    or not _strictly_increasing(values)
                ):
                    checker.fail(
                        "SPLIT-ID-001",
                        observed={"dtype": values.dtype.str, "shape": list(values.shape)},
                        limit={"minimum": 0, "exclusive_maximum": population},
                        evidence={"tag": tag, "phase": phase},
                        remediation="rebuild sorted unique type-local split IDs",
                    )
            if any(
                _arrays_overlap(handles[left].array, handles[right].array)
                for index, left in enumerate(_PHASES)
                for right in _PHASES[index + 1 :]
            ):
                checker.fail(
                    "SPLIT-DISJOINT-001",
                    observed="phase overlap",
                    expected="pairwise-disjoint phase IDs",
                    evidence={"tag": tag},
                    remediation="repair the split registry and rebuild",
                )
            union_count = sum(len(handles[phase].array) for phase in _PHASES)
            if split.get("coverage") == "complete" and union_count != population:
                checker.fail(
                    "SPLIT-COVERAGE-001",
                    observed=union_count,
                    expected=population,
                    evidence={"tag": tag},
                    remediation="publish the exact complete target population",
                )
            expected_fingerprint = split_fingerprint(
                tag,
                split,
                {phase: handles[phase].array for phase in _PHASES},
            )
            if split.get("fingerprint") != expected_fingerprint:
                checker.fail(
                    "SPLIT-FINGERPRINT-001",
                    observed=split.get("fingerprint"),
                    expected=expected_fingerprint,
                    evidence={"tag": tag},
                    remediation="restore the accepted split artifact",
                )
        finally:
            for handle in handles.values():
                handle.close()
    if manifest.get("active_split_tag") not in splits:
        checker.fail(
            "SPLIT-PHASES-001",
            observed=manifest.get("active_split_tag"),
            expected=list(splits),
            remediation="select a registered active split",
        )
    return splits


def _validate_partitions(
    root: Path,
    manifest: Mapping[str, Any],
    files: Mapping[str, Mapping[str, Any]],
    nodes: Mapping[str, Mapping[str, Any]],
    relations: Mapping[str, Mapping[str, Any]],
    checker: _Checker,
) -> None:
    partition = manifest.get("partition")
    if (
        not isinstance(partition, dict)
        or set(partition)
        != {
            "format_version",
            "num_partitions",
            "topology_fingerprint",
            "source_binding",
            "backend",
            "backend_version",
            "options",
            "content_identity",
            "node_types",
            "relations",
        }
        or partition.get("format_version") != "typed-partition-book-v1"
        or not _is_sha256(partition.get("topology_fingerprint"))
        or not isinstance(partition.get("source_binding"), dict)
        or not isinstance(partition.get("backend"), str)
        or not isinstance(partition.get("backend_version"), str)
        or not isinstance(partition.get("options"), dict)
        or not _is_sha256(partition.get("content_identity"))
    ):
        checker.fail(
            "PARTITION-FINGERPRINT-001",
            observed=partition,
            expected="exact typed-partition-book-v1 schema",
            remediation="publish the qualified Task6 partition book",
        )
    count = partition.get("num_partitions")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        checker.fail(
            "PARTITION-ID-001",
            observed=count,
            limit={"minimum": 2},
            remediation="rebuild a valid partition book",
        )
    node_records = partition.get("node_types")
    relation_records = partition.get("relations")
    if not isinstance(node_records, dict) or set(node_records) != set(nodes):
        checker.fail(
            "PARTITION-ID-001",
            observed=list(node_records) if isinstance(node_records, dict) else node_records,
            expected=list(nodes),
            remediation="publish assignments for every node type",
        )
    assignment_handles: dict[str, _ArrayHandle] = {}
    ownership_handles: dict[str, _ArrayHandle] = {}
    try:
        for key, node in nodes.items():
            record = node_records[key]
            if (
                not isinstance(record, dict)
                or set(record)
                != {"assignment", "permutation", "inverse", "partptr"}
            ):
                checker.fail(
                    "MANIFEST-SCHEMA-001",
                    observed=record,
                    expected="exact partition node array map",
                    evidence={"node_type": node["name"]},
                    remediation="rebuild partition metadata",
                )
            assignment = _validate_array_record(root, record["assignment"], files, checker)
            permutation = _validate_array_record(root, record["permutation"], files, checker)
            inverse = _validate_array_record(root, record["inverse"], files, checker)
            partptr = _validate_array_record(root, record["partptr"], files, checker)
            assignment_handles[key] = assignment
            expected_length = node["count"]
            values = assignment.array
            if (
                values.dtype != np.dtype("<i8")
                or values.shape != (expected_length,)
                or (len(values) and (int(values.min()) < 0 or int(values.max()) >= count))
            ):
                permutation.close()
                inverse.close()
                partptr.close()
                checker.fail(
                    "PARTITION-ID-001",
                    observed={"dtype": values.dtype.str, "shape": list(values.shape)},
                    limit={"minimum": 0, "exclusive_maximum": count},
                    evidence={"node_type": node["name"]},
                    remediation="restore the qualified assignment",
                )
            perm = permutation.array
            inv = inverse.array
            if perm.dtype != np.dtype("<i8") or perm.shape != (expected_length,) or (len(perm) and (int(perm.min()) < 0 or int(perm.max()) >= expected_length)):
                actual = {"dtype": perm.dtype.str, "shape": list(perm.shape)}
                permutation.close()
                inverse.close()
                partptr.close()
                checker.fail(
                    "PARTITION-PERMUTATION-001",
                    observed=actual,
                    expected={"dtype": "<i8", "shape": [expected_length], "bijection": True},
                    evidence={"node_type": node["name"]},
                    remediation="rebuild derived partition permutations",
                )
            ordered_assignments = values[perm]
            if len(ordered_assignments) > 1 and (
                np.any(ordered_assignments[1:] < ordered_assignments[:-1])
                or np.any((ordered_assignments[1:] == ordered_assignments[:-1]) & (perm[1:] <= perm[:-1]))
            ):
                permutation.close()
                inverse.close()
                partptr.close()
                checker.fail(
                    "PARTITION-PERMUTATION-001",
                    observed="not stable assignment order",
                    expected="partition then original ordinal order",
                    evidence={"node_type": node["name"]},
                    remediation="rebuild derived partition permutations",
                )
            if inv.dtype != np.dtype("<i8") or inv.shape != (expected_length,) or not np.array_equal(inv[perm], np.arange(expected_length, dtype=np.int64)):
                permutation.close()
                inverse.close()
                partptr.close()
                checker.fail(
                    "PARTITION-INVERSE-001",
                    observed="inverse does not round-trip permutation",
                    expected="inverse[permutation] == arange(node_count)",
                    evidence={"node_type": node["name"]},
                    remediation="rebuild derived inverse permutations",
                )
            counts = np.bincount(values, minlength=count)
            expected_ptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
            if partptr.array.dtype != np.dtype("<i8") or not np.array_equal(partptr.array, expected_ptr):
                actual = partptr.array.tolist()
                permutation.close()
                inverse.close()
                partptr.close()
                checker.fail(
                    "PARTITION-PARTPTR-001",
                    observed=actual,
                    expected=expected_ptr.tolist(),
                    evidence={"node_type": node["name"]},
                    remediation="rebuild derived partition pointers",
                )
            permutation.close()
            inverse.close()
            partptr.close()
        if not isinstance(relation_records, dict) or set(relation_records) != set(relations):
            checker.fail(
                "PARTITION-EDGE-OWNERSHIP-001",
                observed=list(relation_records) if isinstance(relation_records, dict) else relation_records,
                expected=list(relations),
                remediation="publish edge ownership for every relation",
            )
        for key, relation in relations.items():
            relation_record = relation_records[key]
            if (
                not isinstance(relation_record, dict)
                or set(relation_record) != {"edge_partition"}
            ):
                checker.fail(
                    "MANIFEST-SCHEMA-001",
                    observed=relation_record,
                    expected="exact relation edge-ownership map",
                    evidence={"relation": relation["relation"]},
                    remediation="rebuild partition metadata",
                )
            ownership = _validate_array_record(root, relation_records[key]["edge_partition"], files, checker)
            ownership_handles[key] = ownership
            if (
                ownership.array.dtype != np.dtype("<i8")
                or ownership.array.shape != (relation["edge_count"],)
                or (len(ownership.array) and (int(ownership.array.min()) < 0 or int(ownership.array.max()) >= count))
            ):
                checker.fail(
                    "PARTITION-EDGE-OWNERSHIP-001",
                    observed={"dtype": ownership.array.dtype.str, "shape": list(ownership.array.shape)},
                    limit={"minimum": 0, "exclusive_maximum": count},
                    evidence={"relation": relation["relation"]},
                    remediation="restore qualified edge ownership",
                )
        expected_identity = _partition_identity(partition, nodes, relations, assignment_handles, ownership_handles)
        if partition.get("content_identity") != expected_identity:
            checker.fail(
                "PARTITION-FINGERPRINT-001",
                observed=partition.get("content_identity"),
                expected=expected_identity,
                remediation="restore the accepted Task6 partition book",
            )
    finally:
        for handle in (*assignment_handles.values(), *ownership_handles.values()):
            handle.close()


def _partition_identity(
    partition: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
    relations: Mapping[str, Mapping[str, Any]],
    assignments: Mapping[str, _ArrayHandle],
    ownership: Mapping[str, _ArrayHandle],
) -> str:
    digest = hashlib.sha256()
    _frame(digest, "format", b"typed-partition-book-v1")
    _frame(digest, "parts", str(partition["num_partitions"]).encode())
    _frame(digest, "topology", str(partition["topology_fingerprint"]).encode())
    for key, node in sorted(nodes.items(), key=lambda item: item[1]["name"]):
        _frame(digest, "node_type", node["name"].encode())
        _partition_array_frame(digest, "assignment", assignments[key].array)
    relation_items = sorted(relations.items(), key=lambda item: tuple(item[1]["relation"]))
    for key, relation in relation_items:
        _frame(digest, "relation", json.dumps(relation["relation"], separators=(",", ":")).encode())
        _partition_array_frame(digest, "ownership", ownership[key].array)
    _frame(digest, "options", _canonical_json(partition["options"]))
    return digest.hexdigest()


def _partition_array_frame(digest: Any, role: str, value: np.ndarray) -> None:
    _frame(digest, role + ":dtype", b"<i8")
    _frame(digest, role + ":shape", json.dumps(list(value.shape)).encode())
    name = role.encode()
    digest.update(len(name).to_bytes(8, "big"))
    digest.update(name)
    digest.update((value.size * 8).to_bytes(8, "big"))
    for start in range(0, value.size, 128 * 1024):
        chunk = np.asarray(value[start : start + 128 * 1024], dtype=np.dtype("<i8"), order="C")
        digest.update(memoryview(chunk).cast("B"))


def _validate_output_contract(
    manifest: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
    relations: Mapping[str, Mapping[str, Any]],
    checker: _Checker,
) -> None:
    target_key = manifest["target_internal_key"]
    target = nodes.get(target_key)
    output_kind = manifest["output_kind"]
    if (
        target is None
        or target["name"] != manifest["target_node_type"]
        or (
            output_kind == "homogeneous"
            and (
                len(nodes) != 1
                or len(relations) != 1
                or any(
                    relation["source_internal_key"] != target_key
                    or relation["destination_internal_key"] != target_key
                    for relation in relations.values()
                )
            )
        )
        or (output_kind == "heterogeneous" and len(nodes) < 2)
    ):
        checker.fail(
            "OUTPUT-KIND-001",
            observed={
                "output_kind": output_kind,
                "target_node_type": manifest["target_node_type"],
                "target_internal_key": target_key,
                "node_count": len(nodes),
                "relation_count": len(relations),
            },
            expected="output kind consistent with target and typed topology",
            remediation="rebuild using the matching homogeneous/heterogeneous spec",
        )


def _validate_metadata_records(
    root: Path,
    manifest: Mapping[str, Any],
    checker: _Checker,
) -> None:
    expected_binding = compute_metadata_binding(manifest)
    if manifest.get("metadata_binding_sha256") != expected_binding:
        checker.fail(
            "MANIFEST-SCHEMA-001",
            observed=manifest.get("metadata_binding_sha256"),
            expected=expected_binding,
            remediation="rebuild metadata from the exact semantic manifest",
        )
    split_fingerprints = {
        tag: split["fingerprint"] for tag, split in manifest["splits"].items()
    }
    environment_keys = {
        "format_version",
        "python",
        "python_implementation",
        "pytorch",
        "pyg",
        "partition_backend",
        "partition_backend_version",
        "dependency_versions",
        "dependency_lock_sha256",
        "source_state_sha256",
        "config_sha256",
        "os",
        "os_release",
        "architecture",
        "processor",
        "cpu_count",
        "cuda_available",
        "cuda_version",
        "cuda_device",
        "container_image",
        "store_filesystem",
        "metadata_binding_sha256",
        "task_bindings",
        "partition_book_identity",
        "split_fingerprints",
    }
    try:
        environment = _read_json_secure(
            root / "build_environment.json",
            checker.file_identities.get("build_environment.json"),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        checker.fail(
            "BUILD-ENVIRONMENT-001",
            observed=str(error),
            expected="safe typed-graph-build-environment-v1 JSON",
            remediation="rebuild the environment record",
        )
    nullable_strings = (
        "cuda_version",
        "cuda_device",
        "container_image",
    )
    required_strings = (
        "python",
        "python_implementation",
        "pytorch",
        "pyg",
        "partition_backend",
        "partition_backend_version",
        "os",
        "os_release",
        "architecture",
        "processor",
        "store_filesystem",
    )
    if (
        set(environment) != environment_keys
        or environment.get("format_version")
        != "typed-graph-build-environment-v1"
        or any(not isinstance(environment.get(key), str) for key in required_strings)
        or any(
            environment.get(key) is not None
            and not isinstance(environment.get(key), str)
            for key in nullable_strings
        )
        or (
            environment.get("dependency_lock_sha256") is not None
            and not _is_sha256(environment.get("dependency_lock_sha256"))
        )
        or not isinstance(environment.get("dependency_versions"), dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in environment.get("dependency_versions", {}).items()
        )
        or environment.get("dependency_versions")
        != manifest["source_binding"]["dependency_versions"]
        or (
            environment.get("cpu_count") is not None
            and (
                isinstance(environment.get("cpu_count"), bool)
                or not isinstance(environment.get("cpu_count"), int)
                or environment["cpu_count"] <= 0
            )
        )
        or not isinstance(environment.get("cuda_available"), bool)
        or environment.get("source_state_sha256")
        != manifest["source_binding"]["source_fingerprint"]
        or environment.get("config_sha256")
        != manifest["source_binding"]["config_fingerprint"]
        or environment.get("partition_backend")
        != manifest["partition"]["backend"]
        or environment.get("partition_backend_version")
        != manifest["partition"]["backend_version"]
        or environment.get("metadata_binding_sha256") != expected_binding
        or environment.get("task_bindings") != manifest["task_bindings"]
        or environment.get("partition_book_identity")
        != manifest["partition"]["content_identity"]
        or environment.get("split_fingerprints") != split_fingerprints
    ):
        checker.fail(
            "BUILD-ENVIRONMENT-001",
            observed=environment,
            expected="exact versioned environment cross-bound to store identities",
            remediation="rebuild the environment record from the qualified store",
        )

    report_keys = {
        "format_version",
        "passed",
        "report_path",
        "checks",
        "metadata_binding_sha256",
        "task_bindings",
        "partition_book_identity",
        "split_fingerprints",
    }
    try:
        report = _read_json_secure(
            root / "qualification_report.json",
            checker.file_identities.get("qualification_report.json"),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        checker.fail(
            "QUALIFICATION-REPORT-001",
            observed=str(error),
            expected=REPORT_FORMAT_VERSION,
            remediation="rebuild the qualification report",
        )
    checks = report.get("checks")
    check_keys = {
        "check_id",
        "passed",
        "observed",
        "expected",
        "limit",
        "evidence",
        "remediation",
    }
    if (
        set(report) != report_keys
        or report.get("format_version") != REPORT_FORMAT_VERSION
        or report.get("passed") is not True
        or report.get("report_path") != "qualification_report.json"
        or not isinstance(checks, list)
        or len(checks) != manifest["qualification_check_set"]["count"]
        or qualification_check_set_fingerprint(checks)
        != manifest["qualification_check_set"]["sha256"]
        or any(
            not isinstance(check, dict)
            or set(check) != check_keys
            or not isinstance(check.get("check_id"), str)
            or check.get("passed") is not True
            or not isinstance(check.get("evidence"), dict)
            or not isinstance(check.get("remediation"), str)
            for check in checks
        )
        or len({check["check_id"] for check in checks}) != len(checks)
        or report.get("metadata_binding_sha256") != expected_binding
        or report.get("task_bindings") != manifest["task_bindings"]
        or report.get("partition_book_identity")
        != manifest["partition"]["content_identity"]
        or report.get("split_fingerprints") != split_fingerprints
    ):
        checker.fail(
            "QUALIFICATION-REPORT-001",
            observed=report,
            expected="exact passed report cross-bound to store identities",
            remediation="regenerate qualification evidence from Task6",
        )


def _validate(
    root: Path,
    checker: _Checker,
    *,
    expected_bindings: Mapping[str, Any] | None,
    require_directory_identity: bool,
) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        checker.fail(
            "MANIFEST-001",
            observed=str(root),
            expected="real store directory",
            remediation="select a complete promoted store",
        )
    try:
        manifest_descriptor, manifest_stat = _secure_descriptor(
            root / "manifest.json"
        )
        os.close(manifest_descriptor)
        manifest_identity = _stat_identity(manifest_stat)
        manifest = _read_json_secure(
            root / "manifest.json",
            manifest_identity,
        )
        checker.file_identities["manifest.json"] = manifest_identity
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        checker.fail(
            "MANIFEST-001",
            observed=str(error),
            expected="safe versioned manifest.json",
            remediation="rebuild or redownload the complete store",
        )
    if manifest.get("format_version") != STORE_FORMAT_VERSION or manifest.get("content_hash_version") != CONTENT_HASH_VERSION:
        checker.fail(
            "VERSION-001",
            observed={"format": manifest.get("format_version"), "content": manifest.get("content_hash_version")},
            expected={"format": STORE_FORMAT_VERSION, "content": CONTENT_HASH_VERSION},
            remediation="use a producer and consumer supporting the same store version",
        )
    _validate_manifest_schema(manifest, checker)
    files = _file_records(manifest, checker)
    observed_files, observed_directories = _scan_file_set(root, checker)
    expected_files = {"manifest.json", *files}
    expected_directories = {
        parent.as_posix()
        for relative in expected_files
        for parent in PurePosixPath(relative).parents
        if parent.as_posix() != "."
    }
    if (
        observed_files != expected_files
        or observed_directories != expected_directories
    ):
        checker.fail(
            "FILE-SET-001",
            observed={
                "missing": sorted(expected_files - observed_files),
                "extra": sorted(observed_files - expected_files),
                "missing_directories": sorted(
                    expected_directories - observed_directories
                ),
                "extra_directories": sorted(
                    observed_directories - expected_directories
                ),
            },
            expected={
                "files": sorted(expected_files),
                "directories": sorted(expected_directories),
            },
            remediation="restore the exact checksum-declared file set",
        )
    for relative, record in files.items():
        try:
            digest, size, identity = _hash_file(root / relative)
        except OSError as error:
            checker.fail(
                "ARTIFACT-TYPE-001",
                observed=f"{relative}: {error}",
                expected="unchanged uniquely linked regular file",
                remediation="rebuild or redownload the store",
            )
        checker.file_identities[relative] = identity
        if digest != record["sha256"] or size != record["byte_size"]:
            checker.fail(
                "CHECKSUM-001",
                observed={"path": relative, "sha256": digest, "byte_size": size},
                expected={"sha256": record["sha256"], "byte_size": record["byte_size"]},
                remediation="restore the checksum-pinned artifact",
            )
    nodes, _ = _validate_nodes(root, manifest, files, checker)
    relations = _validate_relations(root, manifest, files, nodes, checker)
    _validate_output_contract(manifest, nodes, relations, checker)
    _validate_splits(root, manifest, files, nodes, checker)
    _validate_partitions(root, manifest, files, nodes, relations, checker)
    _validate_metadata_records(root, manifest, checker)
    source_binding = manifest.get("source_binding")
    partition_binding = manifest["partition"].get("source_binding")
    if not isinstance(source_binding, dict) or partition_binding != source_binding.get("task6_source_binding"):
        checker.fail(
            "STALE-BINDING-001",
            observed=partition_binding,
            expected=source_binding.get("task6_source_binding") if isinstance(source_binding, dict) else source_binding,
            remediation="finalize again from one exact Task1-6 build",
        )
    actual_bindings = manifest.get("task_bindings")
    expected = dict(expected_bindings or {})
    if expected_bindings is not None and actual_bindings != expected:
        checker.fail(
            "STALE-BINDING-001",
            observed=actual_bindings,
            expected=expected,
            remediation="select the store built for the requested task bindings",
        )
    computed = compute_content_identity(manifest)
    if manifest.get("content_sha256") != computed:
        checker.fail(
            "CONTENT-IDENTITY-001",
            observed=manifest.get("content_sha256"),
            expected=computed,
            remediation="restore the immutable content-addressed store",
        )
    if require_directory_identity and root.name != computed:
        checker.fail(
            "CONTENT-IDENTITY-001",
            observed=root.name,
            expected=computed,
            remediation="promote under the exact content SHA256 directory name",
        )
    checker.passed(
        "STORE-QUALIFIED-001",
        observed="all canonical checks passed",
        expected="all canonical checks passed",
        evidence={"store_path": str(root), "content_sha256": computed},
    )
    return manifest


def _default_report_path(root: Path) -> Path:
    parent = root.parent / ".qualification-reports"
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        parent = Path(tempfile.gettempdir()) / "topobench-store-qualification"
        parent.mkdir(parents=True, exist_ok=True)
    return parent / f"{root.name}-{uuid.uuid4().hex}.json"


def _write_report(path: Path, report: QualificationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        payload = _canonical_json(report.as_record()) + b"\n"
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _run_qualification(
    root: Path,
    *,
    expected_bindings: Mapping[str, Any] | None,
    report_path: str | Path | None,
    require_directory_identity: bool,
) -> tuple[
    QualificationReport,
    dict[str, Any] | None,
    Mapping[str, _FileIdentity],
]:
    checker = _Checker(root)
    validated_manifest: dict[str, Any] | None = None
    try:
        validated_manifest = _validate(
            root,
            checker,
            expected_bindings=expected_bindings,
            require_directory_identity=require_directory_identity,
        )
    except _CheckError:
        pass
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        json.JSONDecodeError,
    ) as error:
        try:
            checker.fail(
                "MANIFEST-001",
                observed=f"{type(error).__name__}: {error}",
                expected="complete internally consistent manifest",
                remediation="rebuild or redownload the qualified store",
            )
        except _CheckError:
            pass
    finally:
        for handle in checker.handles:
            handle.close()
    destination = (
        Path(report_path)
        if report_path is not None
        else _default_report_path(root)
    )
    report = QualificationReport(
        not any(not result.passed for result in checker.results),
        tuple(checker.results),
        destination,
        root,
    )
    _write_report(destination, report)
    return report, validated_manifest, MappingProxyType(
        dict(checker.file_identities)
    )


def qualify_store(
    path: str | Path,
    *,
    expected_bindings: Mapping[str, Any] | None = None,
    report_path: str | Path | None = None,
    require_directory_identity: bool = True,
) -> QualificationReport:
    """Run canonical checks and always return/write a structured report."""
    report, _, _ = _run_qualification(
        Path(path),
        expected_bindings=expected_bindings,
        report_path=report_path,
        require_directory_identity=require_directory_identity,
    )
    return report


def validate_store(
    path: str | Path,
    *,
    expected_bindings: Mapping[str, Any] | None = None,
    report_path: str | Path | None = None,
    require_directory_identity: bool = True,
) -> ValidatedStore:
    """Return the exact manifest instance accepted by canonical validation."""
    root = Path(path)
    report, manifest, file_identities = _run_qualification(
        root,
        expected_bindings=expected_bindings,
        report_path=report_path,
        require_directory_identity=require_directory_identity,
    )
    if not report.passed or manifest is None:
        raise QualificationFailure(report.failures[0], report.report_path)
    return ValidatedStore(
        root,
        MappingProxyType(manifest),
        report,
        file_identities,
    )


__all__ = [
    "CONTENT_HASH_VERSION",
    "QualificationCheckResult",
    "QualificationFailure",
    "QualificationReport",
    "REPORT_FORMAT_VERSION",
    "STORE_FORMAT_VERSION",
    "ValidatedStore",
    "compute_content_identity",
    "qualification_check_set_fingerprint",
    "compute_metadata_binding",
    "qualify_store",
    "split_fingerprint",
    "validate_store",
]
