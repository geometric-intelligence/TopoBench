"""Immutable schemas for bounded, non-secret execution evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import PurePath, PureWindowsPath
from types import MappingProxyType
from typing import Any, TypeAlias
from urllib.parse import urlsplit

EVENT_SCHEMA_VERSION = "execution-event-v1"
SUMMARY_SCHEMA_VERSION = "execution-summary-v1"
_MAX_STRING = 512
_MAX_MAP_KEYS = 64
_MAX_MAP_DEPTH = 4
_IDENTITY_REPR_LIMIT = 16_384
_CHECK_ID = re.compile(r"^[A-Z][A-Z0-9]*(?:[-.][A-Z0-9]+)+$")
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_KEY_SEPARATOR = re.compile(r"[^A-Za-z0-9]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SECRET_TOKENS = frozenset(
    {
        "auth",
        "authorization",
        "bearer",
        "cookie",
        "credential",
        "key",
        "password",
        "secret",
        "token",
    }
)
_SECRET_COMPACT_NAMES = frozenset(
    {
        "accesskey",
        "accesstoken",
        "apikey",
        "authtoken",
        "bearercredential",
        "bearertoken",
        "clientsecret",
        "secretkey",
    }
)
_SECRET_VALUE = re.compile(
    r"(?:authorization|bearer|access[\s_-]*token|api[\s_-]*key|"
    r"client[\s_-]*secret|secret[\s_-]*key|token|password|secret|key|"
    r"credentials?|auth|cookie)(?:\s+|[\s_-]*[=:]\s*)[^/&?#\s]+",
    re.IGNORECASE,
)
_RAW_DATA_KEY = re.compile(
    r"(?:external_?ids?|partition_?ids?|seed_?lists?|seed_?ids?|predictions?|labels?|features?|tensors?|raw_?rows?|environment(?:_values?)?)$",
    re.IGNORECASE,
)
_IDENTITY_KEY = re.compile(r"(?:descriptor|store)[_-]?identity$", re.IGNORECASE)

Primitive: TypeAlias = None | bool | int | float | str
PrimitiveMap: TypeAlias = Mapping[str, Any]


class ExecutionOperation(str, Enum):
    """Stable operation identifiers spanning conversion through artifacts."""

    CONVERSION = "conversion"
    PARTITION = "partition"
    VALIDATION = "validation"
    FITTED_TRANSFORM = "fitted_transform"
    SELECTED_READ = "selected_read"
    NATIVE_ASSEMBLY = "native_assembly"
    HOST_WAIT = "host_wait"
    HOST_PIN = "host_pin"
    H2D_QUEUE = "h2d_queue"
    H2D_COPY = "h2d_copy"
    MODEL_COMPUTE = "model_compute"
    OPTIMIZER = "optimizer"
    EVALUATOR = "evaluator"
    CHECKPOINT = "checkpoint"
    ARTIFACT = "artifact"


class ExecutionStatus(str, Enum):
    """Stable terminal states for one execution operation."""

    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


def _bounded_string(value: object, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if (not value and not allow_empty) or len(value) > _MAX_STRING or "\x00" in value:
        raise ValueError(f"{name} must be a bounded non-NUL string")
    return value


def _integer(
    value: object,
    name: str,
    *,
    minimum: int | None = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _optional_integer(
    value: object,
    name: str,
    *,
    minimum: int | None = 0,
) -> int | None:
    return None if value is None else _integer(value, name, minimum=minimum)


def _safe_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _freeze_primitive(
    value: object,
    name: str,
    *,
    depth: int = 0,
) -> Primitive | PrimitiveMap:
    if value is None or type(value) in {bool, int, str}:
        if isinstance(value, str):
            return _bounded_string(value, name, allow_empty=True)
        return value
    if type(value) is float:
        return _safe_float(value, name)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must contain only strict primitive maps")
    if depth >= _MAX_MAP_DEPTH:
        raise ValueError(f"{name} exceeds maximum map depth {_MAX_MAP_DEPTH}")
    if len(value) > _MAX_MAP_KEYS:
        raise ValueError(f"{name} exceeds maximum map keys {_MAX_MAP_KEYS}")
    frozen: dict[str, Primitive | PrimitiveMap] = {}
    for raw_key, item in value.items():
        key = _bounded_string(raw_key, f"{name} key")
        if len(key) > 64:
            raise ValueError(f"{name} key exceeds 64 characters")
        frozen[key] = _freeze_primitive(item, f"{name}.{key}", depth=depth + 1)
    return MappingProxyType(dict(sorted(frozen.items())))


def _plain(value: Primitive | PrimitiveMap) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _canonical_identity(value: object, depth: int = 0) -> object:
    if depth > 8:
        return {"type": type(value).__name__}
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_identity(item, depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_identity(item, depth + 1) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_identity(getattr(value, field.name), depth + 1)
            for field in fields(value)
        }
    representation = repr(value)
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": representation[:_IDENTITY_REPR_LIMIT],
    }


def descriptor_digest(identity: object) -> str:
    """Hash an opaque descriptor or store identity without exposing its content."""

    canonical = json.dumps(
        _canonical_identity(identity),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _secret_key(key: str) -> bool:
    separated = _CAMEL_BOUNDARY.sub("_", key)
    tokens = tuple(
        token.lower()
        for token in _KEY_SEPARATOR.sub("_", separated).split("_")
        if token
    )
    normalized = {
        token[:-1] if token.endswith("s") else token
        for token in tokens
    }
    compact = "".join(tokens).lower()
    return (
        bool(normalized & _SECRET_TOKENS)
        or compact in _SECRET_COMPACT_NAMES
        or compact.endswith(("password", "credential"))
    )


def _redacted_uri_or_path(value: str) -> str | None:
    if PurePath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        return "[redacted-path]"
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "[redacted-uri:uri]" if "://" in value else None
    if (
        parsed.scheme
        or parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
    ):
        scheme = parsed.scheme.lower()
        if not re.fullmatch(r"[a-z][a-z0-9+.-]{0,15}", scheme):
            scheme = "uri"
        return f"[redacted-uri:{scheme}]"
    return None



def _redacted_value(key: str, value: object, depth: int) -> Primitive | PrimitiveMap:
    if _secret_key(key) or _RAW_DATA_KEY.search(key):
        return "[redacted]"
    if _IDENTITY_KEY.search(key):
        return descriptor_digest(value)
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is float:
        return _safe_float(value, key)
    if isinstance(value, str):
        if _SECRET_VALUE.search(value):
            return "[redacted]"
        redacted_location = _redacted_uri_or_path(value)
        if redacted_location is not None:
            return redacted_location
        return value[:_MAX_STRING]
    if isinstance(value, Mapping):
        if depth >= _MAX_MAP_DEPTH:
            return MappingProxyType({"truncated": True})
        return redact_mapping(value, _depth=depth + 1)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return MappingProxyType({"count": min(len(value), 2**63 - 1)})
    return f"[{type(value).__name__}]"[:_MAX_STRING]


def redact_mapping(
    value: Mapping[str, object],
    *,
    _depth: int = 0,
) -> PrimitiveMap:
    """Recursively bound and redact evidence before schema construction."""

    if not isinstance(value, Mapping):
        raise TypeError("redaction input must be a mapping")
    redacted: dict[str, Primitive | PrimitiveMap] = {}
    for raw_key, item in list(value.items())[:_MAX_MAP_KEYS]:
        key = str(raw_key)[:64] or "unnamed"
        redacted[key] = _redacted_value(key, item, _depth)
    return MappingProxyType(dict(sorted(redacted.items())))
def _redact_strict_primitive(
    value: object,
    name: str,
) -> Primitive | PrimitiveMap:
    frozen = _freeze_primitive(value, name)
    if isinstance(frozen, Mapping):
        return redact_mapping(frozen)
    if isinstance(frozen, str):
        return _redacted_value(name, frozen, 0)
    return frozen





def _validate_wall_time(value: object) -> str:
    value = _bounded_string(value, "wall_time_utc")
    if not value.endswith("Z"):
        raise ValueError("wall_time_utc must use the UTC Z suffix")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("wall_time_utc must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("wall_time_utc must be UTC")
    return value


def _optional_name(value: object, name: str) -> str | None:
    if value is None:
        return None
    result = _bounded_string(value, name)
    if not _NAME.fullmatch(result):
        raise ValueError(f"{name} must be a stable bounded identifier")
    return result


def _optional_reference(value: object) -> str | None:
    if value is None:
        return None
    result = _bounded_string(value, "report_reference")
    if PurePath(result).is_absolute() or "/" in result or "\\" in result:
        raise ValueError("report_reference must be one local basename")
    if _SECRET_VALUE.search(result) or _secret_key(result):
        return "redacted-report"
    return result


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    """One immutable bounded execution event containing primitives only."""

    operation: ExecutionOperation
    status: ExecutionStatus
    phase: str
    wall_time_utc: str
    monotonic_ns: int
    split: str | None = None
    duration_ns: int | None = None
    epoch: int | None = None
    global_step: int | None = None
    descriptor_sequence: int | None = None
    descriptor_digest: str | None = None
    node_count: int | None = None
    edge_count: int | None = None
    row_count: int | None = None
    example_count: int | None = None
    unique_storage_bytes: int | None = None
    host_queue_configured_depth: int | None = None
    host_queue_depth: int | None = None
    host_queue_configured_bytes: int | None = None
    host_queue_bytes: int | None = None
    device_queue_configured_depth: int | None = None
    device_queue_depth: int | None = None
    device_queue_configured_bytes: int | None = None
    device_queue_bytes: int | None = None
    rss_bytes: int | None = None
    rss_delta_bytes: int | None = None
    pinned_bytes: int | None = None
    pinned_delta_bytes: int | None = None
    gpu_bytes: int | None = None
    gpu_delta_bytes: int | None = None
    temp_disk_bytes: int | None = None
    temp_disk_delta_bytes: int | None = None
    final_disk_bytes: int | None = None
    final_disk_delta_bytes: int | None = None
    sampled: bool = False
    check_id: str | None = None
    check_passed: bool | None = None
    check_expected: Primitive | PrimitiveMap = None
    check_observed: Primitive | PrimitiveMap = None
    evidence: PrimitiveMap = field(
        default_factory=lambda: MappingProxyType({})
    )
    remediation: str | None = None
    report_reference: str | None = None
    schema_version: str = EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        try:
            operation = ExecutionOperation(self.operation)
        except (TypeError, ValueError) as error:
            raise ValueError("operation must be a declared execution operation") from error
        try:
            status = ExecutionStatus(self.status)
        except (TypeError, ValueError) as error:
            raise ValueError("status must be a declared execution status") from error
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "phase", _optional_name(self.phase, "phase"))
        object.__setattr__(self, "split", _optional_name(self.split, "split"))
        object.__setattr__(self, "wall_time_utc", _validate_wall_time(self.wall_time_utc))
        object.__setattr__(self, "monotonic_ns", _integer(self.monotonic_ns, "monotonic_ns"))
        for name in (
            "duration_ns",
            "epoch",
            "global_step",
            "descriptor_sequence",
            "node_count",
            "edge_count",
            "row_count",
            "example_count",
            "unique_storage_bytes",
            "host_queue_configured_depth",
            "host_queue_depth",
            "host_queue_configured_bytes",
            "host_queue_bytes",
            "device_queue_configured_depth",
            "device_queue_depth",
            "device_queue_configured_bytes",
            "device_queue_bytes",
            "rss_bytes",
            "pinned_bytes",
            "gpu_bytes",
            "temp_disk_bytes",
            "final_disk_bytes",
        ):
            object.__setattr__(self, name, _optional_integer(getattr(self, name), name))
        for name in (
            "rss_delta_bytes",
            "pinned_delta_bytes",
            "gpu_delta_bytes",
            "temp_disk_delta_bytes",
            "final_disk_delta_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _optional_integer(getattr(self, name), name, minimum=None),
            )
        if self.descriptor_digest is not None:
            digest = _bounded_string(self.descriptor_digest, "descriptor_digest")
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("descriptor_digest must be a lowercase SHA-256")
            object.__setattr__(self, "descriptor_digest", digest)
        if type(self.sampled) is not bool:
            raise TypeError("sampled must be bool")
        if self.check_id is not None:
            check_id = _bounded_string(self.check_id, "check_id")
            if len(check_id) > 64 or not _CHECK_ID.fullmatch(check_id):
                raise ValueError("check_id must be a stable uppercase identifier")
            object.__setattr__(self, "check_id", check_id)
        if self.check_passed is not None and type(self.check_passed) is not bool:
            raise TypeError("check_passed must be bool or None")
        object.__setattr__(
            self,
            "check_expected",
            _redact_strict_primitive(self.check_expected, "check_expected"),
        )
        object.__setattr__(
            self,
            "check_observed",
            _redact_strict_primitive(self.check_observed, "check_observed"),
        )
        evidence = _freeze_primitive(self.evidence, "evidence")
        assert isinstance(evidence, Mapping)
        object.__setattr__(self, "evidence", redact_mapping(evidence))
        if self.remediation is not None:
            object.__setattr__(
                self,
                "remediation",
                _redacted_value(
                    "remediation",
                    _bounded_string(self.remediation, "remediation"),
                    0,
                ),
            )
        object.__setattr__(self, "report_reference", _optional_reference(self.report_reference))
        if self.schema_version != EVENT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {EVENT_SCHEMA_VERSION!r}")

    def as_record(self) -> dict[str, Any]:
        """Return the exact canonical JSON-compatible schema record."""

        record: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Enum):
                value = value.value
            elif isinstance(value, Mapping):
                value = _plain(value)
            record[field.name] = value
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "ExecutionEvent":
        """Strictly load exactly one current-version record."""

        if not isinstance(record, Mapping):
            raise TypeError("execution event record must be a mapping")
        expected = {field.name for field in fields(cls)}
        if set(record) != expected:
            raise ValueError(
                "execution event record keys differ: "
                f"missing={sorted(expected - set(record))!r}, "
                f"extra={sorted(set(record) - expected)!r}"
            )
        return cls(**dict(record))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class OperationAggregate:
    """Nearest-rank duration statistics for one operation/status pair."""

    operation: ExecutionOperation
    status: ExecutionStatus
    count: int
    minimum_ns: int
    maximum_ns: int
    mean_ns: float
    p50_ns: int
    p95_ns: int
    p99_ns: int
    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", ExecutionOperation(self.operation))
        object.__setattr__(self, "status", ExecutionStatus(self.status))
        object.__setattr__(self, "count", _integer(self.count, "count", minimum=1))
        for name in (
            "minimum_ns",
            "maximum_ns",
            "p50_ns",
            "p95_ns",
            "p99_ns",
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name),
            )
        mean = _safe_float(self.mean_ns, "mean_ns")
        object.__setattr__(self, "mean_ns", mean)
        if not (
            self.minimum_ns
            <= self.p50_ns
            <= self.p95_ns
            <= self.p99_ns
            <= self.maximum_ns
        ) or not self.minimum_ns <= mean <= self.maximum_ns:
            raise ValueError("aggregate duration statistics are inconsistent")


    def as_record(self) -> dict[str, object]:
        return {
            "operation": self.operation.value,
            "status": self.status.value,
            "count": self.count,
            "minimum_ns": self.minimum_ns,
            "maximum_ns": self.maximum_ns,
            "mean_ns": self.mean_ns,
            "p50_ns": self.p50_ns,
            "p95_ns": self.p95_ns,
            "p99_ns": self.p99_ns,
        }
    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "OperationAggregate":
        expected = {
            "operation",
            "status",
            "count",
            "minimum_ns",
            "maximum_ns",
            "mean_ns",
            "p50_ns",
            "p95_ns",
            "p99_ns",
        }
        if not isinstance(record, Mapping) or set(record) != expected:
            raise ValueError("operation aggregate record has invalid keys")
        return cls(**dict(record))  # type: ignore[arg-type]





@dataclass(frozen=True, slots=True)
class ExecutionSummary:
    """Immutable aggregate evidence that never retains event payloads."""

    aggregates: tuple[OperationAggregate, ...]
    sample_every_n: int
    sample_offset: int
    dropped_event_count: int
    rotated_file_count: int
    evidence_digest: str
    schema_version: str = SUMMARY_SCHEMA_VERSION
    conversion_records_per_second: float | None = None
    conversion_bytes_per_second: float | None = None
    selected_read_records_per_second: float | None = None
    selected_read_bytes_per_second: float | None = None
    native_assembly_records_per_second: float | None = None
    native_assembly_bytes_per_second: float | None = None
    achieved_input_stall_fraction: float | None = None
    host_queue_peak_depth: int | None = None
    host_queue_peak_bytes: int | None = None
    device_queue_peak_depth: int | None = None
    device_queue_peak_bytes: int | None = None
    rss_peak_bytes: int | None = None
    rss_peak_delta_bytes: int | None = None
    pinned_peak_bytes: int | None = None
    pinned_peak_delta_bytes: int | None = None
    gpu_peak_bytes: int | None = None
    gpu_peak_delta_bytes: int | None = None
    temp_disk_peak_bytes: int | None = None
    temp_disk_peak_delta_bytes: int | None = None
    final_disk_peak_bytes: int | None = None
    final_disk_peak_delta_bytes: int | None = None
    def __post_init__(self) -> None:
        if not isinstance(self.aggregates, tuple) or any(
            not isinstance(aggregate, OperationAggregate)
            for aggregate in self.aggregates
        ):
            raise TypeError("aggregates must be a tuple of OperationAggregate")
        keys = tuple(
            (aggregate.operation.value, aggregate.status.value)
            for aggregate in self.aggregates
        )
        if keys != tuple(sorted(set(keys))):
            raise ValueError("aggregate operation/status pairs must be unique and sorted")
        every = _integer(self.sample_every_n, "sample_every_n", minimum=1)
        offset = _integer(self.sample_offset, "sample_offset")
        if offset >= every:
            raise ValueError("sample_offset must be smaller than sample_every_n")
        object.__setattr__(
            self,
            "dropped_event_count",
            _integer(self.dropped_event_count, "dropped_event_count"),
        )
        object.__setattr__(
            self,
            "rotated_file_count",
            _integer(self.rotated_file_count, "rotated_file_count"),
        )
        for name in (
            "conversion_records_per_second",
            "conversion_bytes_per_second",
            "selected_read_records_per_second",
            "selected_read_bytes_per_second",
            "native_assembly_records_per_second",
            "native_assembly_bytes_per_second",
            "achieved_input_stall_fraction",
        ):
            value = getattr(self, name)
            if value is not None:
                numeric = _safe_float(value, name)
                if numeric < 0:
                    raise ValueError(f"{name} must be non-negative")
                object.__setattr__(self, name, numeric)
        if (
            self.achieved_input_stall_fraction is not None
            and self.achieved_input_stall_fraction > 1
        ):
            raise ValueError("achieved_input_stall_fraction must not exceed one")
        for name in (
            "host_queue_peak_depth",
            "host_queue_peak_bytes",
            "device_queue_peak_depth",
            "device_queue_peak_bytes",
            "rss_peak_bytes",
            "rss_peak_delta_bytes",
            "pinned_peak_bytes",
            "pinned_peak_delta_bytes",
            "gpu_peak_bytes",
            "gpu_peak_delta_bytes",
            "temp_disk_peak_bytes",
            "temp_disk_peak_delta_bytes",
            "final_disk_peak_bytes",
            "final_disk_peak_delta_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _optional_integer(getattr(self, name), name),
            )
        digest = _bounded_string(self.evidence_digest, "evidence_digest")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("evidence_digest must be a lowercase SHA-256")
        if self.schema_version != SUMMARY_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {SUMMARY_SCHEMA_VERSION!r}"
            )


    def for_operation(
        self,
        operation: ExecutionOperation,
        status: ExecutionStatus,
    ) -> OperationAggregate:
        for aggregate in self.aggregates:
            if aggregate.operation is operation and aggregate.status is status:
                return aggregate
        raise KeyError((operation.value, status.value))

    def as_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "aggregates": [aggregate.as_record() for aggregate in self.aggregates],
            "sample_every_n": self.sample_every_n,
            "sample_offset": self.sample_offset,
            "dropped_event_count": self.dropped_event_count,
            "rotated_file_count": self.rotated_file_count,
            "evidence_digest": self.evidence_digest,
            "conversion_records_per_second": self.conversion_records_per_second,
            "conversion_bytes_per_second": self.conversion_bytes_per_second,
            "selected_read_records_per_second": self.selected_read_records_per_second,
            "selected_read_bytes_per_second": self.selected_read_bytes_per_second,
            "native_assembly_records_per_second": self.native_assembly_records_per_second,
            "native_assembly_bytes_per_second": self.native_assembly_bytes_per_second,
            "achieved_input_stall_fraction": self.achieved_input_stall_fraction,
            "host_queue_peak_depth": self.host_queue_peak_depth,
            "host_queue_peak_bytes": self.host_queue_peak_bytes,
            "device_queue_peak_depth": self.device_queue_peak_depth,
            "device_queue_peak_bytes": self.device_queue_peak_bytes,
            "rss_peak_bytes": self.rss_peak_bytes,
            "rss_peak_delta_bytes": self.rss_peak_delta_bytes,
            "pinned_peak_bytes": self.pinned_peak_bytes,
            "pinned_peak_delta_bytes": self.pinned_peak_delta_bytes,
            "gpu_peak_bytes": self.gpu_peak_bytes,
            "gpu_peak_delta_bytes": self.gpu_peak_delta_bytes,
            "temp_disk_peak_bytes": self.temp_disk_peak_bytes,
            "temp_disk_peak_delta_bytes": self.temp_disk_peak_delta_bytes,
            "final_disk_peak_bytes": self.final_disk_peak_bytes,
            "final_disk_peak_delta_bytes": self.final_disk_peak_delta_bytes,
        }
    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "ExecutionSummary":
        expected = {
            "schema_version",
            "aggregates",
            "sample_every_n",
            "sample_offset",
            "dropped_event_count",
            "rotated_file_count",
            "evidence_digest",
            "conversion_records_per_second",
            "conversion_bytes_per_second",
            "selected_read_records_per_second",
            "selected_read_bytes_per_second",
            "native_assembly_records_per_second",
            "native_assembly_bytes_per_second",
            "achieved_input_stall_fraction",
            "host_queue_peak_depth",
            "host_queue_peak_bytes",
            "device_queue_peak_depth",
            "device_queue_peak_bytes",
            "rss_peak_bytes",
            "rss_peak_delta_bytes",
            "pinned_peak_bytes",
            "pinned_peak_delta_bytes",
            "gpu_peak_bytes",
            "gpu_peak_delta_bytes",
            "temp_disk_peak_bytes",
            "temp_disk_peak_delta_bytes",
            "final_disk_peak_bytes",
            "final_disk_peak_delta_bytes",
        }
        if not isinstance(record, Mapping) or set(record) != expected:
            raise ValueError("execution summary record has invalid keys")
        raw_aggregates = record["aggregates"]
        if not isinstance(raw_aggregates, list):
            raise TypeError("execution summary aggregates must be a list")
        return cls(
            aggregates=tuple(
                OperationAggregate.from_record(aggregate)
                for aggregate in raw_aggregates
            ),
            sample_every_n=record["sample_every_n"],  # type: ignore[arg-type]
            sample_offset=record["sample_offset"],  # type: ignore[arg-type]
            dropped_event_count=record["dropped_event_count"],  # type: ignore[arg-type]
            rotated_file_count=record["rotated_file_count"],  # type: ignore[arg-type]
            evidence_digest=record["evidence_digest"],  # type: ignore[arg-type]
            conversion_records_per_second=record[
                "conversion_records_per_second"
            ],  # type: ignore[arg-type]
            conversion_bytes_per_second=record[
                "conversion_bytes_per_second"
            ],  # type: ignore[arg-type]
            selected_read_records_per_second=record[
                "selected_read_records_per_second"
            ],  # type: ignore[arg-type]
            selected_read_bytes_per_second=record[
                "selected_read_bytes_per_second"
            ],  # type: ignore[arg-type]
            native_assembly_records_per_second=record[
                "native_assembly_records_per_second"
            ],  # type: ignore[arg-type]
            native_assembly_bytes_per_second=record[
                "native_assembly_bytes_per_second"
            ],  # type: ignore[arg-type]
            achieved_input_stall_fraction=record[
                "achieved_input_stall_fraction"
            ],  # type: ignore[arg-type]
            host_queue_peak_depth=record["host_queue_peak_depth"],  # type: ignore[arg-type]
            host_queue_peak_bytes=record["host_queue_peak_bytes"],  # type: ignore[arg-type]
            device_queue_peak_depth=record["device_queue_peak_depth"],  # type: ignore[arg-type]
            device_queue_peak_bytes=record["device_queue_peak_bytes"],  # type: ignore[arg-type]
            rss_peak_bytes=record["rss_peak_bytes"],  # type: ignore[arg-type]
            rss_peak_delta_bytes=record["rss_peak_delta_bytes"],  # type: ignore[arg-type]
            pinned_peak_bytes=record["pinned_peak_bytes"],  # type: ignore[arg-type]
            pinned_peak_delta_bytes=record["pinned_peak_delta_bytes"],  # type: ignore[arg-type]
            gpu_peak_bytes=record["gpu_peak_bytes"],  # type: ignore[arg-type]
            gpu_peak_delta_bytes=record["gpu_peak_delta_bytes"],  # type: ignore[arg-type]
            temp_disk_peak_bytes=record["temp_disk_peak_bytes"],  # type: ignore[arg-type]
            temp_disk_peak_delta_bytes=record["temp_disk_peak_delta_bytes"],  # type: ignore[arg-type]
            final_disk_peak_bytes=record["final_disk_peak_bytes"],  # type: ignore[arg-type]
            final_disk_peak_delta_bytes=record["final_disk_peak_delta_bytes"],  # type: ignore[arg-type]
            schema_version=record["schema_version"],  # type: ignore[arg-type]
        )





def _nearest_rank(values: Sequence[int], percentile: float) -> int:
    rank = max(1, math.ceil(percentile * len(values)))
    return values[rank - 1]


def summarize_events(
    events: Iterable[ExecutionEvent],
    *,
    sample_every_n: int = 1,
    sample_offset: int = 0,
    dropped_event_count: int = 0,
    rotated_file_count: int = 0,
) -> ExecutionSummary:
    """Derive canonical immutable statistics without retaining input objects."""

    every = _integer(sample_every_n, "sample_every_n", minimum=1)
    offset = _integer(sample_offset, "sample_offset")
    if offset >= every:
        raise ValueError("sample_offset must be smaller than sample_every_n")
    dropped = _integer(dropped_event_count, "dropped_event_count")
    rotated = _integer(rotated_file_count, "rotated_file_count")
    canonical_records: list[dict[str, Any]] = []
    throughput: dict[ExecutionOperation, list[int]] = {
        operation: [0, 0, 0, 0]
        for operation in (
            ExecutionOperation.CONVERSION,
            ExecutionOperation.SELECTED_READ,
            ExecutionOperation.NATIVE_ASSEMBLY,
        )
    }
    host_wait_ns = 0
    model_compute_ns = 0
    peaks: dict[str, int | None] = {
        "host_queue_peak_depth": None,
        "host_queue_peak_bytes": None,
        "device_queue_peak_depth": None,
        "device_queue_peak_bytes": None,
        "rss_peak_bytes": None,
        "rss_peak_delta_bytes": None,
        "pinned_peak_bytes": None,
        "pinned_peak_delta_bytes": None,
        "gpu_peak_bytes": None,
        "gpu_peak_delta_bytes": None,
        "temp_disk_peak_bytes": None,
        "temp_disk_peak_delta_bytes": None,
        "final_disk_peak_bytes": None,
        "final_disk_peak_delta_bytes": None,
    }
    durations: dict[tuple[ExecutionOperation, ExecutionStatus], list[int]] = {}
    for event in events:
        if not isinstance(event, ExecutionEvent):
            raise TypeError("summary input must contain ExecutionEvent instances")
        canonical_records.append(event.as_record())
        if event.duration_ns is not None:
            durations.setdefault((event.operation, event.status), []).append(event.duration_ns)
        if event.status is ExecutionStatus.SUCCESS:
            if event.operation is ExecutionOperation.HOST_WAIT:
                host_wait_ns += event.duration_ns
            elif event.operation is ExecutionOperation.MODEL_COMPUTE:
                model_compute_ns += event.duration_ns
            values = throughput.get(event.operation)
            if values is not None:
                record_count = (
                    event.row_count
                    if event.row_count is not None
                    else event.example_count
                )
                if record_count is not None and event.duration_ns > 0:
                    values[0] += record_count
                    values[1] += event.duration_ns
                if (
                    event.unique_storage_bytes is not None
                    and event.duration_ns > 0
                ):
                    values[2] += event.unique_storage_bytes
                    values[3] += event.duration_ns
        observations = {
            "host_queue_peak_depth": event.host_queue_depth,
            "host_queue_peak_bytes": event.host_queue_bytes,
            "device_queue_peak_depth": event.device_queue_depth,
            "device_queue_peak_bytes": event.device_queue_bytes,
            "rss_peak_bytes": event.rss_bytes,
            "pinned_peak_bytes": event.pinned_bytes,
            "gpu_peak_bytes": event.gpu_bytes,
            "temp_disk_peak_bytes": event.temp_disk_bytes,
            "final_disk_peak_bytes": event.final_disk_bytes,
        }
        deltas = {
            "rss_peak_delta_bytes": event.rss_delta_bytes,
            "pinned_peak_delta_bytes": event.pinned_delta_bytes,
            "gpu_peak_delta_bytes": event.gpu_delta_bytes,
            "temp_disk_peak_delta_bytes": event.temp_disk_delta_bytes,
            "final_disk_peak_delta_bytes": event.final_disk_delta_bytes,
        }
        for name, value in observations.items():
            if value is not None:
                current = peaks[name]
                peaks[name] = value if current is None else max(current, value)
        for name, value in deltas.items():
            if value is not None:
                magnitude = abs(value)
                current = peaks[name]
                peaks[name] = (
                    magnitude if current is None else max(current, magnitude)
                )
    canonical_records.sort(
        key=lambda record: (
            record["operation"],
            record["status"],
            -1 if record["descriptor_sequence"] is None else record["descriptor_sequence"],
            record["monotonic_ns"],
            record["wall_time_utc"],
        )
    )
    digest = hashlib.sha256(
        json.dumps(
            canonical_records,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    aggregates: list[OperationAggregate] = []
    for (operation, status), raw_values in sorted(
        durations.items(), key=lambda item: (item[0][0].value, item[0][1].value)
    ):
        values = sorted(raw_values)
        aggregates.append(
            OperationAggregate(
                operation,
                status,
                len(values),
                values[0],
                values[-1],
                sum(values) / len(values),
                _nearest_rank(values, 0.50),
                _nearest_rank(values, 0.95),
                _nearest_rank(values, 0.99),
            )
        )
    def rate(operation: ExecutionOperation, value_index: int) -> float | None:
        values = throughput[operation]
        duration_ns = values[value_index + 1]
        return (
            None
            if duration_ns == 0
            else values[value_index] / (duration_ns / 1_000_000_000)
        )
    stall_denominator = host_wait_ns + model_compute_ns
    stall_fraction = (
        None if stall_denominator == 0 else host_wait_ns / stall_denominator
    )
    return ExecutionSummary(
        aggregates=tuple(aggregates),
        sample_every_n=every,
        sample_offset=offset,
        dropped_event_count=dropped,
        rotated_file_count=rotated,
        evidence_digest=digest,
        conversion_records_per_second=rate(ExecutionOperation.CONVERSION, 0),
        conversion_bytes_per_second=rate(ExecutionOperation.CONVERSION, 2),
        selected_read_records_per_second=rate(
            ExecutionOperation.SELECTED_READ,
            0,
        ),
        selected_read_bytes_per_second=rate(
            ExecutionOperation.SELECTED_READ,
            2,
        ),
        native_assembly_records_per_second=rate(
            ExecutionOperation.NATIVE_ASSEMBLY,
            0,
        ),
        native_assembly_bytes_per_second=rate(
            ExecutionOperation.NATIVE_ASSEMBLY,
            2,
        ),
        achieved_input_stall_fraction=stall_fraction,
        **peaks,
    )


__all__ = [
    "EVENT_SCHEMA_VERSION",
    "SUMMARY_SCHEMA_VERSION",
    "ExecutionEvent",
    "ExecutionOperation",
    "ExecutionStatus",
    "ExecutionSummary",
    "OperationAggregate",
    "Primitive",
    "PrimitiveMap",
    "descriptor_digest",
    "redact_mapping",
    "summarize_events",
]
