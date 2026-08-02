"""Contracts for the authoritative bounded local execution-event log."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from topobench.profiling.execution_events import (
    ExecutionEvent,
    ExecutionOperation,
    ExecutionStatus,
    descriptor_digest,
    summarize_events,
)
from topobench.profiling.local_event_log import (
    EventLogCorruptionError,
    FsyncPolicy,
    LocalEventLog,
    UnsafeEventLogPathError,
)


def _event(sequence: int, *, evidence_size: int = 0) -> ExecutionEvent:
    return ExecutionEvent(
        operation=ExecutionOperation.SELECTED_READ,
        status=ExecutionStatus.SUCCESS,
        phase="fit",
        split="train",
        wall_time_utc="2026-08-02T12:00:00Z",
        monotonic_ns=sequence,
        duration_ns=sequence * 10,
        descriptor_sequence=sequence,
        descriptor_digest=descriptor_digest({"sequence": sequence}),
        evidence={"padding": "x" * evidence_size},
    )


def test_frames_round_trip_with_checksum_length_and_strict_schema(tmp_path: Path) -> None:
    path = tmp_path / "evidence" / "events.jsonl"
    with LocalEventLog(
        path,
        max_bytes=64_000,
        max_records=8,
        max_rotations=1,
        fsync_policy=FsyncPolicy.ALWAYS,
        allowed_root=tmp_path,
    ) as event_log:
        event_log.append(_event(1))
        event_log.append(_event(2))
        assert event_log.load() == (_event(1), _event(2))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    frame = json.loads(lines[0])
    assert set(frame) == {"frame_version", "payload_bytes", "payload_sha256", "record"}
    assert len(frame["payload_sha256"]) == 64
    assert frame["payload_bytes"] > 0


def test_rotation_is_bounded_atomic_and_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    event_log = LocalEventLog(
        path,
        max_bytes=4096,
        max_records=2,
        max_rotations=2,
        fsync_policy="rotation",
    )
    for sequence in range(1, 8):
        event_log.append(_event(sequence, evidence_size=8))
    event_log.close()

    assert path.is_file()
    assert path.with_name("events.jsonl.1").is_file()
    assert path.with_name("events.jsonl.2").is_file()
    assert not path.with_name("events.jsonl.3").exists()
    assert event_log.rotated_file_count == 3
    assert [event.descriptor_sequence for event in LocalEventLog(
        path,
        max_bytes=4096,
        max_records=2,
        max_rotations=2,
    ).load()] == [3, 4, 5, 6, 7]
    assert all(candidate.stat().st_size <= 4096 for candidate in (
        path,
        path.with_name("events.jsonl.1"),
        path.with_name("events.jsonl.2"),
    ))


def test_reopen_recovers_only_a_truncated_tail_and_rejects_middle_corruption(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    with LocalEventLog(path, max_bytes=64_000, max_records=10) as event_log:
        event_log.append(_event(1))
        event_log.append(_event(2))
    with path.open("ab") as stream:
        stream.write(b'{"frame_version":"partial')

    recovered = LocalEventLog(path, max_bytes=64_000, max_records=10)
    assert recovered.recovered_tail_bytes > 0
    assert recovered.load() == (_event(1), _event(2))
    recovered.close()

    lines = path.read_bytes().splitlines(keepends=True)
    first = json.loads(lines[0])
    first["payload_sha256"] = "0" * 64
    lines[0] = json.dumps(first, sort_keys=True).encode() + b"\n"
    path.write_bytes(b"".join(lines))
    with pytest.raises(EventLogCorruptionError, match="checksum"):
        LocalEventLog(path, max_bytes=64_000, max_records=10)


def test_threaded_append_is_serialized_without_lost_or_partial_frames(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    left = LocalEventLog(path, max_bytes=1_000_000, max_records=128)
    right = LocalEventLog(path, max_bytes=1_000_000, max_records=128)

    def append_one(sequence: int) -> None:
        (left if sequence % 2 else right).append(_event(sequence))

    with ThreadPoolExecutor(max_workers=8) as pool:
        tuple(pool.map(append_one, range(1, 65)))
    left.close()
    right.close()

    loaded = LocalEventLog(path, max_bytes=1_000_000, max_records=128).load()
    assert sorted(event.descriptor_sequence for event in loaded) == list(range(1, 65))
    assert len(path.read_bytes().splitlines()) == 64


def test_two_instances_refresh_same_size_rotations_once_per_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events.jsonl"
    left = LocalEventLog(
        path,
        max_bytes=64_000,
        max_records=1,
        max_rotations=2,
    )
    right = LocalEventLog(
        path,
        max_bytes=64_000,
        max_records=1,
        max_rotations=2,
    )

    left.append(_event(1))
    active_size = path.stat().st_size
    assert right.retained_events == (_event(1),)
    right.append(_event(2))
    assert path.stat().st_size == active_size
    assert left.retained_events == (_event(1), _event(2))
    left.append(_event(3))
    assert path.stat().st_size == active_size
    right.append(_event(4))
    assert path.stat().st_size == active_size

    with LocalEventLog(
        path,
        max_bytes=64_000,
        max_records=1,
        max_rotations=2,
    ) as fresh:
        authoritative = fresh.load()
    assert [
        event.descriptor_sequence for event in authoritative
    ] == [2, 3, 4]
    authoritative_summary = summarize_events(authoritative)
    for event_log in (left, right):
        retained = event_log.retained_events
        summary = summarize_events(retained)
        assert retained == authoritative
        assert len(retained) == len(authoritative)
        assert summary.evidence_digest == authoritative_summary.evidence_digest

    scan_count = 0
    original_scan = left._scan

    def counted_scan(
        candidate: Path,
        *,
        recover_tail: bool,
    ) -> tuple[tuple[ExecutionEvent, ...], int, int]:
        nonlocal scan_count
        scan_count += 1
        return original_scan(candidate, recover_tail=recover_tail)

    monkeypatch.setattr(left, "_scan", counted_scan)
    assert left.retained_events == authoritative
    assert left.load() == authoritative
    assert scan_count == 0

    right.append(_event(5))
    refreshed = left.retained_events
    assert [event.descriptor_sequence for event in refreshed] == [3, 4, 5]
    changed_scan_count = scan_count
    assert changed_scan_count == 3
    assert left.retained_events == refreshed
    assert left.load() == refreshed
    assert scan_count == changed_scan_count
    left.close()
    right.close()



def test_symlink_and_allowed_root_escape_are_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir(exist_ok=True)
    symlink = tmp_path / "linked"
    symlink.symlink_to(outside, target_is_directory=True)
    with pytest.raises(UnsafeEventLogPathError):
        LocalEventLog(
            symlink / "events.jsonl",
            max_bytes=1000,
            max_records=2,
            allowed_root=tmp_path,
        )
    with pytest.raises(UnsafeEventLogPathError):
        LocalEventLog(
            outside / "events.jsonl",
            max_bytes=1000,
            max_records=2,
            allowed_root=tmp_path,
        )

    target = tmp_path / "target.jsonl"
    target.touch()
    path = tmp_path / "events.jsonl"
    path.symlink_to(target)
    with pytest.raises(UnsafeEventLogPathError):
        LocalEventLog(path, max_bytes=1000, max_records=2)
    os.unlink(path)
