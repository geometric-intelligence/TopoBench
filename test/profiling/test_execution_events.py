"""Contracts for schema-versioned structured execution evidence."""

from __future__ import annotations
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime

import pytest
import torch

from topobench.profiling.execution_events import (
    EVENT_SCHEMA_VERSION,
    ExecutionEvent,
    ExecutionOperation,
    ExecutionStatus,
    descriptor_digest,
    redact_mapping,
    summarize_events,
)


def _event(
    operation: ExecutionOperation,
    *,
    status: ExecutionStatus = ExecutionStatus.SUCCESS,
    sequence: int = 1,
    duration_ns: int = 10,
) -> ExecutionEvent:
    return ExecutionEvent(
        operation=operation,
        status=status,
        phase="fit",
        split="train",
        wall_time_utc="2026-08-02T12:00:00.000000Z",
        monotonic_ns=100 + sequence,
        duration_ns=duration_ns,
        epoch=2,
        global_step=7,
        descriptor_sequence=sequence,
        descriptor_digest=descriptor_digest({"external_id": sequence}),
        node_count=11,
        edge_count=13,
        row_count=17,
        example_count=19,
        unique_storage_bytes=23,
        host_queue_configured_depth=4,
        host_queue_depth=2,
        host_queue_configured_bytes=101,
        host_queue_bytes=53,
        device_queue_configured_depth=3,
        device_queue_depth=1,
        device_queue_configured_bytes=103,
        device_queue_bytes=47,
        rss_bytes=1000,
        rss_delta_bytes=100,
        pinned_bytes=500,
        pinned_delta_bytes=50,
        gpu_bytes=700,
        gpu_delta_bytes=70,
        temp_disk_bytes=900,
        temp_disk_delta_bytes=90,
        final_disk_bytes=800,
        final_disk_delta_bytes=80,
        sampled=True,
        evidence={"source_digest": "a" * 64},
    )


def test_every_declared_operation_round_trips_with_all_resource_fields() -> None:
    expected = {
        "conversion",
        "partition",
        "validation",
        "fitted_transform",
        "selected_read",
        "native_assembly",
        "host_wait",
        "host_pin",
        "h2d_queue",
        "h2d_copy",
        "model_compute",
        "optimizer",
        "evaluator",
        "checkpoint",
        "artifact",
    }
    assert {status.value for status in ExecutionStatus} == {
        "success",
        "warning",
        "error",
        "cancelled",
        "skipped",
    }
    assert {operation.value for operation in ExecutionOperation} == expected

    for sequence, operation in enumerate(ExecutionOperation, 1):
        event = _event(operation, sequence=sequence)
        record = event.as_record()
        assert record["schema_version"] == EVENT_SCHEMA_VERSION
        assert record["operation"] == operation.value
        assert record["status"] == "success"
        assert record["descriptor_digest"] != str(sequence)
        assert ExecutionEvent.from_record(record) == event


def test_schema_is_frozen_strict_and_rejects_unsafe_identifiers_or_values() -> None:
    event = _event(ExecutionOperation.VALIDATION)
    with pytest.raises(FrozenInstanceError):
        event.phase = "mutated"  # type: ignore[misc]

    with pytest.raises(ValueError, match="check_id"):
        ExecutionEvent(
            operation=ExecutionOperation.VALIDATION,
            status=ExecutionStatus.ERROR,
            phase="qualification",
            wall_time_utc="2026-08-02T12:00:00Z",
            monotonic_ns=1,
            check_id="manifest check",
        )
    with pytest.raises(TypeError, match="primitive"):
        ExecutionEvent(
            operation=ExecutionOperation.MODEL_COMPUTE,
            status=ExecutionStatus.SUCCESS,
            phase="fit",
            wall_time_utc="2026-08-02T12:00:00Z",
            monotonic_ns=1,
            evidence={"batch": torch.tensor([1])},
        )
    record = event.as_record()
    record["unexpected"] = 1
    with pytest.raises(ValueError, match="keys"):
        ExecutionEvent.from_record(record)


def test_descriptor_identity_and_recursive_redaction_never_disclose_raw_data() -> None:
    identity = {
        "store": "warehouse-alpha",
        "partition_ids": [8, 13],
        "seed_ids": [21],
    }
    digest = descriptor_digest(identity)
    assert len(digest) == 64
    assert "warehouse" not in digest
    assert digest == descriptor_digest(identity)

    redacted = redact_mapping(
        {
            "token": "top-secret-token",
            "authorization": "Bearer authorization-value",
            "apiKey": "camel-api-value",
            "AccessToken": "pascal-access-value",
            "secret-key": "separator-secret-value",
            "client_secret": "client-secret-value",
            "apikey": "lower-api-value",
            "authtoken": "lower-auth-value",
            "accesstoken": "lower-access-value",
            "accesskey": "lower-access-key-value",
            "secretkey": "lower-secret-key-value",
            "clientsecret": "lower-client-secret-value",
            "monkey": "banana",
            "nested": {
                "dbPassword": "hunter2",
                "bearerCredential": "bearer-value",
                "url": "https://user%3Aencoded:pass%40word@example.org/data?auth=yes#fragment",
                "database_uri": "postgres://db-user:db-pass@private.example/db",
                "object_uri": "s3://private-bucket/secret/key?token=value",
                "ssh_uri": "ssh://deploy@private.example/etc/config",
                "file_uri": "file:///Users/private/credentials.json",
                "absolute_path": "/Users/private/rows.parquet",
                "external_ids": ["patient-1", "patient-2"],
                "labels": [0, 1],
                "safe_count": 2,
            },
        }
    )
    serialized = repr(redacted)
    for forbidden in (
        "top-secret-token",
        "authorization-value",
        "camel-api-value",
        "pascal-access-value",
        "separator-secret-value",
        "client-secret-value",
        "hunter2",
        "lower-api-value",
        "lower-auth-value",
        "lower-access-value",
        "lower-access-key-value",
        "lower-secret-key-value",
        "lower-client-secret-value",
        "bearer-value",
        "patient-1",
        "user%3Aencoded",
        "pass%40word",
        "private.example",
        "private-bucket",
        "/Users/private",
        "auth=yes",
        "[0, 1]",
    ):
        assert forbidden not in serialized
    assert redacted["monkey"] == "banana"
    assert redacted["nested"]["safe_count"] == 2


def test_summary_uses_nearest_rank_percentiles_and_canonical_evidence() -> None:
    durations = (1, 2, 3, 4, 100)
    events = tuple(
        _event(
            ExecutionOperation.SELECTED_READ,
            sequence=index,
            duration_ns=duration,
        )
        for index, duration in enumerate(durations, 1)
    ) + (
        _event(
            ExecutionOperation.SELECTED_READ,
            status=ExecutionStatus.ERROR,
            sequence=6,
            duration_ns=9,
        ),
    )
    summary = summarize_events(
        reversed(events),
        sample_every_n=3,
        sample_offset=1,
        dropped_event_count=2,
        rotated_file_count=4,
    )
    aggregate = summary.for_operation(
        ExecutionOperation.SELECTED_READ,
        ExecutionStatus.SUCCESS,
    )
    assert aggregate.count == 5
    assert aggregate.minimum_ns == 1
    assert aggregate.maximum_ns == 100
    assert aggregate.mean_ns == 22
    assert aggregate.p50_ns == 3
    assert aggregate.p95_ns == 100
    assert aggregate.p99_ns == 100
    assert summary.sample_every_n == 3
    assert summary.sample_offset == 1
    assert summary.dropped_event_count == 2
    assert summary.rotated_file_count == 4
    assert len(summary.evidence_digest) == 64
    assert summary == summarize_events(
        events,
        sample_every_n=3,
        sample_offset=1,
        dropped_event_count=2,
        rotated_file_count=4,
    )
    assert "events" not in summary.as_record()
    assert type(summary).from_record(summary.as_record()) == summary
    assert summary.selected_read_records_per_second == pytest.approx(
        85 / (110 / 1_000_000_000)
    )
    assert summary.selected_read_bytes_per_second == pytest.approx(
        115 / (110 / 1_000_000_000)
    )
    assert summary.conversion_records_per_second is None
    assert summary.native_assembly_records_per_second is None
    conversion = summarize_events(
        (_event(ExecutionOperation.CONVERSION, duration_ns=10),),
    )
    assert conversion.conversion_records_per_second == pytest.approx(
        17 / (10 / 1_000_000_000)
    )
    assert conversion.conversion_bytes_per_second == pytest.approx(
        23 / (10 / 1_000_000_000)
    )
    native = summarize_events(
        (_event(ExecutionOperation.NATIVE_ASSEMBLY, duration_ns=20),),
    )
    assert native.native_assembly_records_per_second == pytest.approx(
        17 / (20 / 1_000_000_000)
    )
    assert native.native_assembly_bytes_per_second == pytest.approx(
        23 / (20 / 1_000_000_000)
    )
    profiled = summarize_events(
        (
            replace(
                _event(
                    ExecutionOperation.HOST_WAIT,
                    sequence=10,
                    duration_ns=50,
                ),
                row_count=None,
                example_count=None,
                unique_storage_bytes=None,
            ),
            replace(
                _event(
                    ExecutionOperation.MODEL_COMPUTE,
                    sequence=11,
                    duration_ns=100,
                ),
                row_count=None,
                example_count=None,
                unique_storage_bytes=None,
                host_queue_depth=3,
                host_queue_bytes=99,
                rss_bytes=1400,
                rss_delta_bytes=-300,
            ),
        ),
        sample_every_n=1,
        sample_offset=0,
    )
    assert profiled.achieved_input_stall_fraction == pytest.approx(1 / 3)
    assert profiled.host_queue_peak_depth == 3
    assert profiled.host_queue_peak_bytes == 99
    assert profiled.rss_peak_bytes == 1400
    assert profiled.rss_peak_delta_bytes == 300
    assert all(
        value is None or type(value) in {int, float, str, list}
        for value in profiled.as_record().values()
    )
    model_only = summarize_events(
        (_event(ExecutionOperation.MODEL_COMPUTE, duration_ns=10),),
    )
    assert model_only.conversion_records_per_second is None
    assert model_only.selected_read_records_per_second is None
    assert model_only.native_assembly_records_per_second is None
    assert datetime.fromisoformat(events[0].wall_time_utc.replace("Z", "+00:00")).tzinfo is UTC
