"""Atomic promotion, cache identity, and corruption quarantine."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import test.data.stores.test_typed_graph_store as typed_store_test_module
import topobench.data.stores.typed_graph_store as store_module
from test.data.stores.test_typed_graph_store import QualifiedStoreFixture
from topobench.data.stores.qualification_checks import QualificationFailure
from topobench.data.stores.typed_graph_ingestion import (
    ArtifactValidationError,
)
from topobench.data.stores.typed_graph_store import (
    TypedGraphStore,
    TypedGraphStoreWriter,
)


@pytest.fixture(name="qualified_store_fixtures", scope="session")
def _qualified_store_fixtures(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, QualifiedStoreFixture]:
    return typed_store_test_module.qualified_stores.__wrapped__(
        tmp_path_factory
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_checksums(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def test_validated_identity_is_cache_hit_and_changed_task_binding_is_miss(
    qualified_store_fixtures: dict[str, QualifiedStoreFixture],
) -> None:
    fixture = qualified_store_fixtures["heterogeneous"]
    hit = TypedGraphStoreWriter(
        fixture.ingestor,
        fixture.partition_build,
    ).build()
    assert hit.cache_hit is True
    assert hit.path == fixture.store_build.path

    changed = TypedGraphStoreWriter(
        fixture.ingestor,
        fixture.partition_build,
        task_bindings={"consumer_contract": "neighbor-v2"},
    ).build()
    assert changed.cache_hit is False
    assert changed.content_sha256 != fixture.store_build.content_sha256
    assert changed.path.is_dir()
    with TypedGraphStore.open(
        changed.path,
        expected_bindings={"consumer_contract": "neighbor-v2"},
    ) as store:
        assert store.task_bindings == {"consumer_contract": "neighbor-v2"}


def test_failed_finalization_is_invisible_and_never_mutates_task1_6_stage(
    qualified_store_fixtures: dict[str, QualifiedStoreFixture],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = qualified_store_fixtures["homogeneous"]
    source_stage = fixture.partition_build.stage_root
    before = _stage_checksums(source_stage)
    writer = TypedGraphStoreWriter(
        fixture.ingestor,
        fixture.partition_build,
        task_bindings={"fault_injection": "copy"},
    )
    original = writer._copy_file
    copied = 0

    def fail_after_first(source: Path, destination: Path) -> None:
        nonlocal copied
        copied += 1
        if copied == 2:
            raise OSError("injected copy failure")
        original(source, destination)

    monkeypatch.setattr(writer, "_copy_file", fail_after_first)
    with pytest.raises(OSError, match="injected copy failure"):
        writer.build()

    assert _stage_checksums(source_stage) == before
    assert fixture.store_build.path.is_dir()
    assert not list(fixture.ingestor.store_root.glob("fault_injection*"))
    assert not list(
        (fixture.ingestor.store_root / ".staging").glob("finalize-*")
    )


@pytest.mark.parametrize("returncode", [-9, 137])
def test_bounded_validation_rejects_worker_crash_and_cleans_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
) -> None:
    request_paths: list[Path] = []

    def crash(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        request_path = Path(command[-1])
        assert request_path.is_file()
        request_paths.append(request_path)
        return subprocess.CompletedProcess(command, returncode, "", "")

    monkeypatch.setattr(store_module.subprocess, "run", crash)
    with pytest.raises(ArtifactValidationError, match="VALIDATION-WORKER-001"):
        store_module._validate_store_bounded(
            tmp_path,
            expected_bindings={},
            require_directory_identity=False,
        )

    assert len(request_paths) == 1
    assert not request_paths[0].exists()
    assert not request_paths[0].parent.exists()


def test_bounded_validation_rejects_timeout_and_cleans_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_paths: list[Path] = []

    def timeout(command: list[str], **_kwargs: object) -> None:
        request_path = Path(command[-1])
        assert request_path.is_file()
        request_paths.append(request_path)
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(store_module.subprocess, "run", timeout)
    with pytest.raises(ArtifactValidationError, match="VALIDATION-TIMEOUT-001"):
        store_module._validate_store_bounded(
            tmp_path,
            expected_bindings={},
            require_directory_identity=False,
        )

    assert len(request_paths) == 1
    assert not request_paths[0].exists()
    assert not request_paths[0].parent.exists()


def test_bounded_validation_rejects_malformed_or_tampered_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(("not-json", "tampered"))

    def respond(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        mode = next(responses)
        if mode == "not-json":
            return subprocess.CompletedProcess(command, 0, "not-json", "")
        request = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        report_path = Path(request["report_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text('{"passed":true}\\n', encoding="utf-8")
        report_stat = report_path.stat()
        report_identity = [
            report_stat.st_dev,
            report_stat.st_ino,
            report_stat.st_size,
            report_stat.st_mtime_ns,
            report_stat.st_ctime_ns,
        ]
        evidence = {
            "format_version": "typed-store-bounded-validation-v1",
            "status": "passed",
            "failure": None,
            "validation_root_sha256": request["validation_root_sha256"],
            "report_path": str(report_path),
            "report_sha256": "0" * 64,
            "report_identity": report_identity,
            "manifest_sha256": "0" * 64,
            "manifest_identity": [0, 0, 0, 0, 0],
            "file_identities": {},
            "memory": {
                "measurement_scope": "isolated-validation-worker",
                "baseline_rss_bytes": 1,
                "peak_rss_bytes": 2,
                "peak_rss_delta_bytes": 1,
                "declared_peak_rss_delta_limit_bytes": 320 * 1024**2,
                "peak_rss_by_phase": {"canonical_validation": 2},
            },
        }
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(evidence),
            "",
        )

    monkeypatch.setattr(store_module.subprocess, "run", respond)
    for match in ("malformed JSON", "report digest"):
        with pytest.raises(ArtifactValidationError, match=match):
            store_module._validate_store_bounded(
                tmp_path,
                expected_bindings={},
                require_directory_identity=False,
            )


def test_bounded_validation_rejects_corrupt_candidate(
    tmp_path: Path,
    qualified_store_fixtures: dict[str, QualifiedStoreFixture],
) -> None:
    fixture = qualified_store_fixtures["homogeneous"]
    candidate = tmp_path / "candidate"
    shutil.copytree(fixture.store_build.path, candidate)
    payload = candidate / "nodes/n0000/x.npy"
    payload.chmod(0o600)
    payload.write_bytes(payload.read_bytes() + b"corrupt")

    with pytest.raises(QualificationFailure) as captured:
        store_module._validate_store_bounded(
            candidate,
            expected_bindings={},
            require_directory_identity=False,
        )

    assert captured.value.check_id == "CHECKSUM-001"


def test_recovered_partition_book_closes_before_promotion(
    qualified_store_fixtures: dict[str, QualifiedStoreFixture],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finalization releases its recovered maps without closing caller state."""
    fixture = qualified_store_fixtures["homogeneous"]
    original_book = fixture.partition_build.book
    recovered_books: list[object] = []
    writer = TypedGraphStoreWriter(
        fixture.ingestor,
        fixture.partition_build,
        task_bindings={"fault_injection": "promotion-boundary"},
    )

    def observe_promotion(_candidate: Path) -> None:
        recovered_book = writer.partition_build.book
        recovered_books.append(recovered_book)
        assert recovered_book is not original_book
        assert recovered_book.closed is True
        raise RuntimeError("injected promotion boundary")

    monkeypatch.setattr(writer, "_promote_candidate", observe_promotion)
    with pytest.raises(RuntimeError, match="injected promotion boundary"):
        writer.build()

    assert len(recovered_books) == 1
    assert recovered_books[0].closed is True
    assert original_book.closed is False


def test_corrupt_collision_is_quarantined_before_fresh_atomic_promotion(
    qualified_store_fixtures: dict[str, QualifiedStoreFixture],
) -> None:
    fixture = qualified_store_fixtures["heterogeneous"]
    path = fixture.store_build.path
    payload = path / "nodes/n0000/x.npy"
    os.chmod(payload, 0o600)
    payload.write_bytes(payload.read_bytes() + b"corrupt")

    rebuilt = TypedGraphStoreWriter(
        fixture.ingestor,
        fixture.partition_build,
    ).build()

    assert rebuilt.path == path
    assert rebuilt.cache_hit is False
    quarantines = list(
        fixture.ingestor.store_root.glob(
            f".quarantine-{fixture.store_build.content_sha256}-*"
        )
    )
    assert quarantines
    assert all(item.is_dir() for item in quarantines)
    with TypedGraphStore.open(rebuilt.path) as reopened:
        assert reopened.content_sha256 == fixture.store_build.content_sha256


def test_incomplete_and_unknown_staging_cannot_be_opened_or_promoted(
    qualified_store_fixtures: dict[str, QualifiedStoreFixture],
) -> None:
    fixture = qualified_store_fixtures["homogeneous"]
    staging = fixture.ingestor.store_root / ".staging" / "interrupted"
    staging.mkdir(parents=True)
    (staging / "nodes").mkdir()
    with pytest.raises(QualificationFailure) as captured:
        TypedGraphStore.open(staging)
    assert captured.value.check_id == "MANIFEST-001"
    assert staging.is_dir()
    assert fixture.store_build.path.is_dir()
