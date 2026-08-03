"""Checksummed, recoverable, bounded local JSONL execution evidence."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import stat
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path

from topobench.profiling.execution_events import ExecutionEvent

_FRAME_VERSION = "execution-log-frame-v1"
_FRAME_KEYS = frozenset(
    {"frame_version", "payload_bytes", "payload_sha256", "record"}
)
_PROCESS_LOCK_GUARD = threading.Lock()
_PROCESS_LOCKS: dict[str, threading.RLock] = {}


class EventLogCorruptionError(RuntimeError):
    """Report a complete frame that fails strict framing or schema checks."""


class UnsafeEventLogPathError(ValueError):
    """Reject evidence paths that escape their root or traverse symlinks."""


class EventLogOverflowError(ValueError):
    """Reject an individual event that cannot fit one bounded segment."""


class FsyncPolicy(StrEnum):
    """Declared durability point for authoritative local evidence."""

    ALWAYS = "always"
    ROTATION = "rotation"
    CLOSE = "close"


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _canonical(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _process_lock(path: Path) -> threading.RLock:
    key = str(path)
    with _PROCESS_LOCK_GUARD:
        return _PROCESS_LOCKS.setdefault(key, threading.RLock())


def _existing_components(path: Path) -> Iterator[Path]:
    components: list[Path] = []
    current = path
    while True:
        components.append(current)
        if current == current.parent:
            break
        current = current.parent
    yield from reversed(components)


def _reject_symlink_components(path: Path) -> None:
    for component in _existing_components(path):
        try:
            mode = component.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise UnsafeEventLogPathError(
                f"event log path traverses symlink component {component.name!r}"
            )


def _secure_open(path: Path, flags: int, mode: int = 0o600) -> int:
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as error:
        if path.is_symlink():
            raise UnsafeEventLogPathError(
                f"event log path {path.name!r} must not be a symlink"
            ) from error
        raise
    observed = os.fstat(descriptor)
    if not stat.S_ISREG(observed.st_mode):
        os.close(descriptor)
        raise UnsafeEventLogPathError("event log must be a regular file")
    return descriptor


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class LocalEventLog:
    """Authoritative append-only JSONL with deterministic bounded rotation."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_bytes: int,
        max_records: int,
        max_rotations: int = 0,
        fsync_policy: FsyncPolicy | str = FsyncPolicy.ALWAYS,
        allowed_root: str | Path | None = None,
    ) -> None:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        candidate = Path(os.path.abspath(candidate))
        root = (
            Path(os.path.abspath(Path(allowed_root).expanduser()))
            if allowed_root is not None
            else candidate.parent
        )
        _reject_symlink_components(root)
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise UnsafeEventLogPathError(
                "event log path escapes allowed_root"
            ) from error
        _reject_symlink_components(candidate)
        root.mkdir(parents=True, exist_ok=True)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        _reject_symlink_components(candidate)
        if (
            candidate.parent.resolve() != root.resolve()
            and allowed_root is None
        ):
            raise UnsafeEventLogPathError(
                "event log parent changed during creation"
            )
        try:
            policy = FsyncPolicy(fsync_policy)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "fsync_policy must be always, rotation, or close"
            ) from error
        self.path = candidate
        self.allowed_root = root.resolve()
        self.max_bytes = _integer(max_bytes, "max_bytes", minimum=256)
        self.max_records = _integer(max_records, "max_records", minimum=1)
        self.max_rotations = _integer(
            max_rotations, "max_rotations", minimum=0
        )
        self.fsync_policy = policy
        self.rotated_file_count = 0
        self.recovered_tail_bytes = 0
        self.evicted_event_count = 0
        self._segments: list[list[ExecutionEvent]] = []
        self._retained_events: tuple[ExecutionEvent, ...] = ()
        self._active_size = 0
        self._segment_fingerprint: tuple[
            tuple[str, int, int, int, int],
            ...,
        ] = ()
        self._closed = False
        self._process_lock = _process_lock(candidate)
        self._lock_path = candidate.with_name(f".{candidate.name}.lock")
        _reject_symlink_components(self._lock_path)
        self._lock_descriptor = _secure_open(
            self._lock_path,
            os.O_CREAT | os.O_RDWR,
        )
        with self._exclusive():
            self._ensure_active()
            self._recover_retained_locked()

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        with self._process_lock:
            fcntl.flock(self._lock_descriptor, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(self._lock_descriptor, fcntl.LOCK_UN)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("LocalEventLog is closed")

    def _ensure_active(self) -> None:
        _reject_symlink_components(self.path)
        descriptor = _secure_open(
            self.path,
            os.O_CREAT | os.O_APPEND | os.O_WRONLY,
        )
        os.close(descriptor)

    def _set_segments(
        self,
        segments: list[list[ExecutionEvent]],
    ) -> None:
        self._segments = segments or [[]]
        self._retained_events = tuple(
            event for segment in self._segments for event in segment
        )

    def _fingerprint_locked(
        self,
    ) -> tuple[tuple[str, int, int, int, int], ...]:
        fingerprint: list[tuple[str, int, int, int, int]] = []
        for path in (*self._rotation_paths(oldest_first=True), self.path):
            _reject_symlink_components(path)
            stat = path.stat(follow_symlinks=False)
            fingerprint.append(
                (
                    path.name,
                    stat.st_ino,
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                )
            )
        return tuple(fingerprint)

    def _refresh_if_changed_locked(self) -> None:
        if self._fingerprint_locked() != self._segment_fingerprint:
            self._recover_retained_locked()

    def _recover_retained_locked(self) -> None:
        segments: list[list[ExecutionEvent]] = []
        active_size = 0
        for path in (*self._rotation_paths(oldest_first=True), self.path):
            events, size, recovered = self._scan(
                path,
                recover_tail=path == self.path,
            )
            if path == self.path:
                active_size = size
            self.recovered_tail_bytes += recovered
            segments.append(list(events))
        self._active_size = active_size
        self._set_segments(segments)
        self._segment_fingerprint = self._fingerprint_locked()

    def _rotation_path(self, index: int) -> Path:
        return self.path.with_name(f"{self.path.name}.{index}")

    def _rotation_paths(self, *, oldest_first: bool) -> tuple[Path, ...]:
        indices = (
            range(self.max_rotations, 0, -1)
            if oldest_first
            else range(1, self.max_rotations + 1)
        )
        return tuple(
            path
            for path in (self._rotation_path(index) for index in indices)
            if path.exists()
        )

    @staticmethod
    def _encode(event: ExecutionEvent) -> bytes:
        if not isinstance(event, ExecutionEvent):
            raise TypeError("local event log accepts only ExecutionEvent")
        record = event.as_record()
        payload = _canonical(record)
        frame: dict[str, object] = {
            "frame_version": _FRAME_VERSION,
            "payload_bytes": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "record": record,
        }
        return _canonical(frame) + b"\n"

    @staticmethod
    def _decode(
        line: bytes, *, path: Path, line_number: int
    ) -> ExecutionEvent:
        try:
            frame = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise EventLogCorruptionError(
                f"invalid complete JSON frame at {path.name}:{line_number}"
            ) from error
        if not isinstance(frame, dict) or set(frame) != _FRAME_KEYS:
            raise EventLogCorruptionError(
                f"invalid frame keys at {path.name}:{line_number}"
            )
        if frame["frame_version"] != _FRAME_VERSION:
            raise EventLogCorruptionError(
                f"unsupported frame version at {path.name}:{line_number}"
            )
        record = frame["record"]
        if not isinstance(record, dict):
            raise EventLogCorruptionError(
                f"frame record is not a mapping at {path.name}:{line_number}"
            )
        payload = _canonical(record)
        payload_bytes = frame["payload_bytes"]
        if (
            isinstance(payload_bytes, bool)
            or not isinstance(payload_bytes, int)
            or payload_bytes != len(payload)
        ):
            raise EventLogCorruptionError(
                f"frame length mismatch at {path.name}:{line_number}"
            )
        checksum = frame["payload_sha256"]
        if checksum != hashlib.sha256(payload).hexdigest():
            raise EventLogCorruptionError(
                f"frame checksum mismatch at {path.name}:{line_number}"
            )
        try:
            event = ExecutionEvent.from_record(record)
            if event.as_record() != record:
                raise ValueError(
                    "event record is not canonical or safely redacted"
                )
            return event
        except (TypeError, ValueError) as error:
            raise EventLogCorruptionError(
                f"strict event schema failure at {path.name}:{line_number}"
            ) from error

    def _scan(
        self,
        path: Path,
        *,
        recover_tail: bool,
    ) -> tuple[tuple[ExecutionEvent, ...], int, int]:
        _reject_symlink_components(path)
        descriptor = _secure_open(path, os.O_RDONLY)
        try:
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                payload = stream.read()
        finally:
            os.close(descriptor)
        if not payload:
            return (), 0, 0
        lines = payload.splitlines(keepends=True)
        complete_count = len(lines)
        recovered = 0
        if lines and not lines[-1].endswith(b"\n"):
            if not recover_tail:
                raise EventLogCorruptionError(
                    f"truncated frame in rotated log {path.name}"
                )
            recovered = len(lines[-1])
            complete_count -= 1
            valid_size = len(payload) - recovered
            descriptor = _secure_open(path, os.O_WRONLY)
            try:
                os.ftruncate(descriptor, valid_size)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            payload = payload[:valid_size]
            lines = lines[:complete_count]
            _sync_directory(path.parent)
        events = tuple(
            self._decode(line[:-1], path=path, line_number=index)
            for index, line in enumerate(lines, 1)
        )
        return events, len(payload), recovered

    def _rotate(self) -> None:
        for path in (self.path, *self._rotation_paths(oldest_first=False)):
            _reject_symlink_components(path)
        if self.max_rotations:
            oldest = self._rotation_path(self.max_rotations)
            if oldest.exists():
                oldest.unlink()
            for index in range(self.max_rotations - 1, 0, -1):
                source = self._rotation_path(index)
                if source.exists():
                    os.replace(source, self._rotation_path(index + 1))
            os.replace(self.path, self._rotation_path(1))
        else:
            self.path.unlink(missing_ok=True)
        self._ensure_active()
        if self.max_rotations:
            self._segments.append([])
            while len(self._segments) > self.max_rotations + 1:
                discarded = self._segments.pop(0)
                self.evicted_event_count += len(discarded)
        else:
            self.evicted_event_count += sum(
                len(segment) for segment in self._segments
            )
            self._segments = [[]]
        self._set_segments(self._segments)
        self._active_size = 0
        self.rotated_file_count += 1
        if self.fsync_policy in {FsyncPolicy.ALWAYS, FsyncPolicy.ROTATION}:
            for path in (self.path, *self._rotation_paths(oldest_first=False)):
                descriptor = _secure_open(path, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            _sync_directory(self.path.parent)

    def append(self, event: ExecutionEvent) -> None:
        """Append one framed event, rotating before a segment exceeds bounds."""

        self._ensure_open()
        encoded = self._encode(event)
        if len(encoded) > self.max_bytes:
            raise EventLogOverflowError(
                f"framed event size {len(encoded)} exceeds max_bytes {self.max_bytes}"
            )
        with self._exclusive():
            self._ensure_active()
            self._refresh_if_changed_locked()
            events = tuple(self._segments[-1])
            size = self._active_size
            if (
                len(events) >= self.max_records
                or size + len(encoded) > self.max_bytes
            ):
                self._rotate()
            descriptor = _secure_open(self.path, os.O_APPEND | os.O_WRONLY)
            try:
                written = os.write(descriptor, encoded)
                if written != len(encoded):
                    raise OSError(
                        f"short event-log append: wrote {written} of {len(encoded)} bytes"
                    )
                if self.fsync_policy is FsyncPolicy.ALWAYS:
                    os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._active_size += len(encoded)
            self._segments[-1].append(event)
            self._set_segments(self._segments)
            self._segment_fingerprint = self._fingerprint_locked()

    def load(self) -> tuple[ExecutionEvent, ...]:
        """Strictly load retained rotations from oldest through active."""

        self._ensure_open()
        with self._exclusive():
            self._refresh_if_changed_locked()
            return self._retained_events

    @property
    def retained_events(self) -> tuple[ExecutionEvent, ...]:
        """Return the exact immutable retained window without disk replay."""

        self._ensure_open()
        with self._exclusive():
            self._refresh_if_changed_locked()
            return self._retained_events

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Apply the declared final durability boundary exactly once."""

        if self._closed:
            return
        with self._exclusive():
            if self.fsync_policy in {
                FsyncPolicy.ALWAYS,
                FsyncPolicy.ROTATION,
                FsyncPolicy.CLOSE,
            }:
                descriptor = _secure_open(self.path, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                _sync_directory(self.path.parent)
        self._closed = True
        os.close(self._lock_descriptor)

    def __enter__(self) -> LocalEventLog:
        self._ensure_open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


__all__ = [
    "EventLogCorruptionError",
    "EventLogOverflowError",
    "FsyncPolicy",
    "LocalEventLog",
    "UnsafeEventLogPathError",
]
