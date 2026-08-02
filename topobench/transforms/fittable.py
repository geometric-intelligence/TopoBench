"""Bounded fitted-transform contracts and immutable non-executable state."""

from __future__ import annotations

import errno
import dataclasses
import hashlib
import inspect
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from torch_geometric.data import Data, HeteroData

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ARRAY_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_STATE_FORMAT = "topobench-fitted-transform-v1"
NativeGraph = Data | HeteroData


class FitStateError(RuntimeError):
    """Report an invalid, corrupt, or incompatible fitted state."""


class FitStateNotFoundError(FitStateError):
    """Report that an exact fitted-state identity has not been published."""


class FitStatus(str, Enum):
    """One-way lifecycle for a fitted transform instance."""

    UNFITTED = "unfitted"
    FITTING = "fitting"
    FITTED = "fitted"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TransformSpec:
    """Immutable runtime declaration for one native graph transform."""

    input_kinds: tuple[str, ...]
    output_kind: str
    deterministic: bool
    device: str
    preserves_node_identity: bool
    preserves_supervision: bool
    feature_width_behavior: str
    edge_effects: str
    accesses_labels: bool
    target_node_type: str | None
    target_field: str
    input_dtype: str
    output_dtype: str
    accumulation_dtype: str

    def __post_init__(self) -> None:
        if not self.input_kinds or any(
            kind not in {"Data", "HeteroData"} for kind in self.input_kinds
        ):
            raise ValueError("input_kinds must declare Data and/or HeteroData")
        if len(set(self.input_kinds)) != len(self.input_kinds):
            raise ValueError("input_kinds must not contain duplicates")
        if self.output_kind not in {"same", "Data", "HeteroData"}:
            raise ValueError("output_kind must be same, Data, or HeteroData")
        if self.device != "cpu":
            raise ValueError("fitted transforms currently require CPU execution")
        if not self.deterministic:
            raise ValueError("fitted transforms must be deterministic")
        if not self.preserves_node_identity or not self.preserves_supervision:
            raise ValueError("fitted transforms must preserve identity and supervision")
        if self.edge_effects != "none":
            raise ValueError("fitted transforms must declare no edge effects")
        for name, value in (
            ("feature_width_behavior", self.feature_width_behavior),
            ("target_field", self.target_field),
            ("input_dtype", self.input_dtype),
            ("output_dtype", self.output_dtype),
            ("accumulation_dtype", self.accumulation_dtype),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.target_node_type is not None and not self.target_node_type:
            raise ValueError("target_node_type must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class FitContext:
    """Complete path-independent identity of one canonical training fit view."""

    content_sha256: str
    active_split_tag: str
    train_ids_sha256: str
    train_source_sha256: str
    target_node_type: str
    target_field: str
    input_shape: tuple[int, int]
    input_width: int
    input_dtype: str
    input_schema_sha256: str
    package_versions: tuple[tuple[str, str], ...]
    numeric_precision: str

    def __post_init__(self) -> None:
        for name in (
            "content_sha256",
            "train_ids_sha256",
            "train_source_sha256",
            "input_schema_sha256",
        ):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        for name in (
            "active_split_tag",
            "target_node_type",
            "target_field",
            "input_dtype",
            "numeric_precision",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if (
            len(self.input_shape) != 2
            or self.input_shape[0] < 0
            or self.input_shape[1] < 1
            or self.input_width != self.input_shape[1]
        ):
            raise ValueError("input_shape and input_width are inconsistent")
        normalized_versions = tuple(sorted(self.package_versions))
        if normalized_versions != self.package_versions or len(
            {name for name, _ in self.package_versions}
        ) != len(self.package_versions):
            raise ValueError("package_versions must be uniquely sorted by name")
        if any(not name or not version for name, version in self.package_versions):
            raise ValueError("package_versions require non-empty names and values")
        dtype = np.dtype(self.input_dtype)
        if dtype.hasobject:
            raise ValueError("input_dtype must be non-executable")


@dataclass(frozen=True, slots=True)
class PublishedFitState:
    """One strictly validated immutable fitted-state directory."""

    key: str
    path: Path
    manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]


@runtime_checkable
class FittableTransform(Protocol):
    """Runtime-checkable bounded training fit followed by native transformation."""

    spec: TransformSpec
    status: FitStatus
    @property
    def state_key(self) -> str | None:
        """Immutable published-state identity, available only while fitted."""
        ...


    def canonical_config(self) -> Mapping[str, object]: ...

    def implementation_versions(self) -> Mapping[str, str]: ...

    def begin_fit(self, context: FitContext) -> None: ...

    def update_fit(
        self, features: np.ndarray, labels: np.ndarray | None = None
    ) -> None: ...

    def finalize_fit(self, state_root: str | Path) -> PublishedFitState: ...

    def load_state(
        self, state_root: str | Path, context: FitContext
    ) -> PublishedFitState: ...

    def transform(self, batch: NativeGraph) -> NativeGraph: ...


def _json_value(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical JSON mapping keys must be strings")
        return {key: _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or type(value) in {bool, int}:
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("canonical JSON values must be finite")
        return value
    raise TypeError(f"value of type {type(value).__name__} is not canonical JSON")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _json_value(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_fingerprint(transform: FittableTransform) -> str:
    digest = hashlib.sha256()
    for transform_type in type(transform).__mro__:
        if transform_type is object:
            continue
        digest.update(transform_type.__module__.encode("utf-8"))
        digest.update(b"\0")
        digest.update(transform_type.__qualname__.encode("utf-8"))
        digest.update(b"\0")
        try:
            source = inspect.getsource(transform_type).encode("utf-8")
        except (OSError, TypeError):
            source = b""
            for name in (
                "canonical_config",
                "implementation_versions",
                "begin_fit",
                "update_fit",
                "finalize_fit",
                "load_state",
                "transform",
            ):
                method = transform_type.__dict__.get(name)
                code = getattr(method, "__code__", None)
                if code is not None:
                    source += code.co_code
                    source += repr(code.co_consts).encode("utf-8")
        digest.update(source)
        digest.update(b"\0")
    return digest.hexdigest()


def build_fit_state_key(
    context: FitContext, transform: FittableTransform
) -> str:
    """Hash every scientific and implementation input to fitted state."""

    if not isinstance(transform, FittableTransform):
        raise TypeError("transform must implement FittableTransform")
    transform_type = type(transform)
    payload = {
        "format": _STATE_FORMAT,
        "context": context,
        "transform": {
            "module": transform_type.__module__,
            "qualname": transform_type.__qualname__,
            "code_sha256": _code_fingerprint(transform),
            "config": transform.canonical_config(),
            "spec": transform.spec,
            "implementation_versions": transform.implementation_versions(),
        },
    }
    return _sha_bytes(_canonical_json(payload))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _readonly_tree(path: Path) -> None:
    for child in path.iterdir():
        child.chmod(0o444)
    path.chmod(0o555)


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


class FitStatePublisher:
    """Publish/load strict JSON plus ``.npy`` state with atomic promotion."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)


    def publish(
        self,
        key: str,
        *,
        metadata: Mapping[str, object],
        arrays: Mapping[str, np.ndarray],
    ) -> PublishedFitState:
        """Atomically publish an exact state, never overwriting an existing key."""

        if not _SHA256.fullmatch(key):
            raise ValueError("state key must be a lowercase SHA-256 digest")
        normalized_metadata = _json_value(metadata)
        if not isinstance(normalized_metadata, dict):
            raise TypeError("state metadata must be a JSON mapping")
        if not arrays:
            raise ValueError("fitted state must contain at least one array")
        self.root.mkdir(parents=True, exist_ok=True)
        _fsync_directory(self.root.parent)
        if self.root.is_symlink() or not self.root.is_dir():
            raise FitStateError("fitted state root must be a real directory")
        staging = self.root / f".staging-{key}-{uuid.uuid4().hex}"
        target = self.root / key
        staging.mkdir(mode=0o700)
        try:
            records: dict[str, dict[str, object]] = {}
            for name in sorted(arrays):
                if not _ARRAY_NAME.fullmatch(name):
                    raise ValueError(f"unsafe fitted-state array name: {name!r}")
                value = np.asarray(arrays[name])
                if value.dtype.hasobject:
                    raise TypeError(
                        "object arrays are executable/pickle-backed and are forbidden"
                    )
                if value.dtype.kind in "fc" and not np.isfinite(value).all():
                    raise ValueError(f"fitted-state array {name!r} must be finite")
                value = np.ascontiguousarray(value)
                path = staging / f"{name}.npy"
                with path.open("xb") as stream:
                    np.save(stream, value, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                records[name] = {
                    "dtype": value.dtype.str,
                    "shape": list(value.shape),
                    "sha256": _sha_file(path),
                }
            manifest = {
                "format": _STATE_FORMAT,
                "state_key": key,
                "metadata": normalized_metadata,
                "arrays": records,
            }
            manifest["content_sha256"] = _sha_bytes(_canonical_json(manifest))
            manifest_path = staging / "manifest.json"
            with manifest_path.open("xb") as stream:
                stream.write(_canonical_json(manifest))
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            _fsync_directory(staging)
            if target.exists():
                existing = self.load(key)
                proposed = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                if _canonical_json(existing.manifest) != _canonical_json(proposed):
                    raise FitStateError(
                        "existing fitted-state identity cannot be overwritten"
                    )
                for name, value in arrays.items():
                    if not np.array_equal(existing.arrays[name], np.asarray(value)):
                        raise FitStateError(
                            "existing fitted-state identity cannot be overwritten"
                        )
                return existing
            try:
                os.rename(staging, target)
            except OSError as error:
                if error.errno not in {errno.EEXIST, errno.ENOTEMPTY} or not target.exists():
                    raise
                existing = self.load(
                    key,
                    expected_metadata=normalized_metadata,
                )
                if _canonical_json(existing.manifest) != _canonical_json(manifest):
                    raise FitStateError(
                        "concurrent fitted-state identity differs"
                    ) from error
                for name, value in arrays.items():
                    if not np.array_equal(existing.arrays[name], np.asarray(value)):
                        raise FitStateError(
                            "concurrent fitted-state arrays differ"
                        ) from error
                return existing
            _readonly_tree(target)
            _fsync_directory(target)
            _fsync_directory(self.root)
            return self.load(key, expected_metadata=normalized_metadata)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def load(
        self,
        key: str,
        *,
        expected_metadata: Mapping[str, object] | None = None,
    ) -> PublishedFitState:
        """Strictly validate and memory-map one immutable fitted state."""

        if not _SHA256.fullmatch(key):
            raise ValueError("state key must be a lowercase SHA-256 digest")
        target = self.root / key
        if not target.exists():
            raise FitStateNotFoundError(
                f"fitted state not found for exact identity {key}"
            )
        if target.is_symlink() or not target.is_dir():
            raise FitStateError("fitted state path is not a real directory")
        manifest_path = target / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise FitStateError("fitted state manifest is missing or unsafe")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise FitStateError("fitted state manifest is corrupt") from error
        if (
            not isinstance(manifest, dict)
            or set(manifest)
            != {
                "format",
                "state_key",
                "content_sha256",
                "metadata",
                "arrays",
            }
            or manifest.get("format") != _STATE_FORMAT
            or manifest.get("state_key") != key
            or not _SHA256.fullmatch(str(manifest.get("content_sha256")))
            or not isinstance(manifest.get("metadata"), dict)
            or not isinstance(manifest.get("arrays"), dict)
            or not manifest["arrays"]
        ):
            raise FitStateError("fitted state manifest semantics are invalid")
        content_payload = dict(manifest)
        content_sha256 = content_payload.pop("content_sha256", None)
        if content_sha256 != _sha_bytes(_canonical_json(content_payload)):
            raise FitStateError("fitted state manifest checksum is corrupt")
        if expected_metadata is not None and manifest["metadata"] != _json_value(
            expected_metadata
        ):
            raise FitStateError("fitted state metadata identity mismatch")
        expected_files = {"manifest.json"}
        loaded: dict[str, np.ndarray] = {}
        for name, record in manifest["arrays"].items():
            if not isinstance(name, str) or not _ARRAY_NAME.fullmatch(name):
                raise FitStateError("fitted state contains an unsafe array name")
            if not isinstance(record, dict) or set(record) != {
                "dtype",
                "shape",
                "sha256",
            }:
                raise FitStateError(f"fitted state array {name!r} has invalid metadata")
            filename = f"{name}.npy"
            expected_files.add(filename)
            path = target / filename
            if path.is_symlink() or not path.is_file():
                raise FitStateError(f"fitted state array {name!r} is missing or unsafe")
            if not _SHA256.fullmatch(str(record["sha256"])) or _sha_file(path) != record[
                "sha256"
            ]:
                raise FitStateError(f"fitted state array {name!r} checksum is corrupt")
            try:
                value = np.load(path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as error:
                raise FitStateError(
                    f"fitted state array {name!r} is not a safe NumPy array"
                ) from error
            if value.dtype.hasobject:
                raise FitStateError("fitted state object arrays are forbidden")
            if value.dtype.str != record["dtype"] or list(value.shape) != record["shape"]:
                raise FitStateError(f"fitted state array {name!r} dtype/shape mismatch")
            if value.dtype.kind in "fc" and not np.isfinite(value).all():
                raise FitStateError(f"fitted state array {name!r} must be finite")
            value.flags.writeable = False
            loaded[name] = value
        actual_files = {path.name for path in target.iterdir()}
        if actual_files != expected_files:
            raise FitStateError("fitted state file set is corrupt or incomplete")
        return PublishedFitState(
            key=key,
            path=target,
            manifest=_freeze_json(manifest),  # type: ignore[arg-type]
            arrays=MappingProxyType(loaded),
        )


__all__ = [
    "FitContext",
    "FitStateError",
    "FitStateNotFoundError",
    "FitStatePublisher",
    "FitStatus",
    "FittableTransform",
    "PublishedFitState",
    "TransformSpec",
    "build_fit_state_key",
]
