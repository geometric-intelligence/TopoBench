"""Atomic promotion, cache identity, and corruption quarantine."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from topobench.data.stores.qualification_checks import QualificationFailure
from topobench.data.stores.typed_graph_store import (
    TypedGraphStore,
    TypedGraphStoreWriter,
)
from test.data.stores.test_typed_graph_store import (
    QualifiedStoreFixture,
    qualified_stores,
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
    qualified_stores: dict[str, QualifiedStoreFixture],
) -> None:
    fixture = qualified_stores["heterogeneous"]
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
    qualified_stores: dict[str, QualifiedStoreFixture],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = qualified_stores["homogeneous"]
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


def test_corrupt_collision_is_quarantined_before_fresh_atomic_promotion(
    qualified_stores: dict[str, QualifiedStoreFixture],
) -> None:
    fixture = qualified_stores["heterogeneous"]
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
    qualified_stores: dict[str, QualifiedStoreFixture],
) -> None:
    fixture = qualified_stores["homogeneous"]
    staging = fixture.ingestor.store_root / ".staging" / "interrupted"
    staging.mkdir(parents=True)
    (staging / "nodes").mkdir()
    with pytest.raises(QualificationFailure) as captured:
        TypedGraphStore.open(staging)
    assert captured.value.check_id == "MANIFEST-001"
    assert staging.is_dir()
    assert fixture.store_build.path.is_dir()
