"""Structured immutable-store qualification and malformed-artifact checks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

import topobench.data.stores.qualification_checks as qualification_module
from topobench.data.stores.qualification_checks import (
    QualificationFailure,
    QualificationReport,
    qualify_store,
    validate_store,
)
from test.data.stores.test_typed_graph_store import (
    QualifiedStoreFixture,
    qualified_stores,
)


def _mutable_store(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    for path in sorted(destination.rglob("*"), reverse=True):
        if path.is_symlink():
            continue
        os.chmod(path, 0o700 if path.is_dir() else 0o600)
    os.chmod(destination, 0o700)
    return destination


def _manifest(root: Path) -> dict[str, object]:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(root: Path, manifest: dict[str, object]) -> None:
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _file_record(manifest: dict[str, object], relative: str) -> dict[str, object]:
    return next(
        record
        for record in manifest["files"]  # type: ignore[index]
        if record["relative_path"] == relative
    )


def _reseal_file(root: Path, relative: str) -> None:
    manifest = _manifest(root)
    path = root / relative
    record = _file_record(manifest, relative)
    record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    record["byte_size"] = path.stat().st_size
    _write_manifest(root, manifest)


def _array_path(manifest: dict[str, object], role_prefix: str) -> str:
    return next(
        record["relative_path"]
        for record in manifest["files"]  # type: ignore[index]
        if str(record["role"]).startswith(role_prefix)
    )


def _save_array(root: Path, relative: str, value: np.ndarray) -> None:
    with (root / relative).open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    _reseal_file(root, relative)


def _unknown_version(root: Path) -> None:
    manifest = _manifest(root)
    manifest["format_version"] = "typed-graph-store-v999"
    _write_manifest(root, manifest)


def _checksum_error(root: Path) -> None:
    manifest = _manifest(root)
    relative = _array_path(manifest, "node-feature:")
    path = root / relative
    path.write_bytes(path.read_bytes() + b"stale")


def _shape_error(root: Path) -> None:
    manifest = _manifest(root)
    relative = _array_path(manifest, "node-feature:")
    value = np.load(root / relative, allow_pickle=False)
    _save_array(root, relative, value[:1])


def _dtype_error(root: Path) -> None:
    manifest = _manifest(root)
    relative = _array_path(manifest, "node-feature:")
    value = np.load(root / relative, allow_pickle=False)
    _save_array(root, relative, value.astype(np.float16))


def _endianness_error(root: Path) -> None:
    manifest = _manifest(root)
    relative = _array_path(manifest, "node-feature:")
    value = np.load(root / relative, allow_pickle=False)
    _save_array(root, relative, value.astype(value.dtype.newbyteorder(">")))


def _finite_error(root: Path) -> None:
    manifest = _manifest(root)
    relative = _array_path(manifest, "node-feature:")
    value = np.array(np.load(root / relative, allow_pickle=False), copy=True)
    value.flat[0] = np.nan
    _save_array(root, relative, value)


@pytest.mark.parametrize(
    ("mutator", "check_id"),
    [
        (_unknown_version, "VERSION-001"),
        (_checksum_error, "CHECKSUM-001"),
        (_shape_error, "ARRAY-SHAPE-001"),
        (_dtype_error, "ARRAY-DTYPE-001"),
        (_endianness_error, "ARRAY-ENDIANNESS-001"),
        (_finite_error, "ARRAY-FINITE-001"),
    ],
)
def test_file_and_array_failures_have_exact_structured_reports(
    qualified_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
    mutator: Callable[[Path], None],
    check_id: str,
) -> None:
    root = _mutable_store(
        qualified_stores["heterogeneous"].store_build.path,
        tmp_path / check_id,
    )
    mutator(root)
    report_path = tmp_path / f"{check_id}.json"
    report = qualify_store(
        root,
        report_path=report_path,
        require_directory_identity=False,
    )
    assert isinstance(report, QualificationReport)
    assert report.passed is False
    failure = report.failures[0]
    assert failure.check_id == check_id
    assert failure.passed is False
    assert failure.observed is not None
    assert failure.expected is not None or failure.limit is not None
    assert failure.evidence

    assert failure.remediation
    assert report.report_path == report_path
    assert report_path.is_file()
    with pytest.raises(QualificationFailure) as captured:
        validate_store(root, require_directory_identity=False)
    assert captured.value.check_id == check_id
    assert captured.value.report_path is not None
def test_unknown_empty_directory_is_rejected_by_the_exact_store_layout(
    qualified_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
) -> None:
    root = _mutable_store(
        qualified_stores["homogeneous"].store_build.path,
        tmp_path / "unknown-directory",
    )
    (root / "unknown-empty-directory").mkdir()
    report = qualify_store(root, require_directory_identity=False)
    assert report.failures[0].check_id == "FILE-SET-001"



def test_exact_file_set_rejects_missing_extra_symlink_and_hardlink(
    qualified_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
) -> None:
    source = qualified_stores["homogeneous"].store_build.path

    missing = _mutable_store(source, tmp_path / "missing")
    manifest = _manifest(missing)
    (missing / _array_path(manifest, "node-feature:")).unlink()
    assert qualify_store(missing, require_directory_identity=False).failures[0].check_id == "FILE-SET-001"

    extra = _mutable_store(source, tmp_path / "extra")
    (extra / "unknown.bin").write_bytes(b"unknown")
    assert qualify_store(extra, require_directory_identity=False).failures[0].check_id == "FILE-SET-001"

    symlink = _mutable_store(source, tmp_path / "symlink")
    manifest = _manifest(symlink)
    relative = _array_path(manifest, "node-feature:")
    (symlink / relative).unlink()
    (symlink / relative).symlink_to(symlink / "manifest.json")
    assert qualify_store(symlink, require_directory_identity=False).failures[0].check_id == "ARTIFACT-TYPE-001"

    hardlink = _mutable_store(source, tmp_path / "hardlink")
    manifest = _manifest(hardlink)
    relative = _array_path(manifest, "node-feature:")
    original = hardlink / relative
    alias = hardlink / "alias"
    os.link(original, alias)
    try:
        report = qualify_store(hardlink, require_directory_identity=False)
        assert report.failures[0].check_id == "ARTIFACT-TYPE-001"
    finally:
        alias.unlink()


def _relation_record(manifest: dict[str, object], relation_name: str) -> dict[str, object]:
    return next(
        record
        for record in manifest["relations"].values()  # type: ignore[index,union-attr]
        if record["relation"][1] == relation_name
    )


@pytest.mark.parametrize(
    "check_id",
    ["CSC-COLPTR-001", "CSC-ROW-BOUNDS-001", "CSC-ORDER-001"],
)
def test_csc_semantics_are_qualified_after_checksum_validation(
    qualified_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
    check_id: str,
) -> None:
    root = _mutable_store(
        qualified_stores["heterogeneous"].store_build.path,
        tmp_path / check_id,
    )
    manifest = _manifest(root)
    relation = _relation_record(manifest, "writes")
    if check_id == "CSC-COLPTR-001":
        relative = relation["colptr"]["relative_path"]
        value = np.array(np.load(root / relative, allow_pickle=False), copy=True)
        value[1] = value[2] + 1
    elif check_id == "CSC-ROW-BOUNDS-001":
        relative = relation["row"]["relative_path"]
        value = np.array(np.load(root / relative, allow_pickle=False), copy=True)
        value[0] = int(relation["source_count"])
    else:
        relative = relation["edge_id"]["relative_path"]
        value = np.array(np.load(root / relative, allow_pickle=False), copy=True)
        value[0], value[1] = value[1], value[0]
    _save_array(root, relative, value)
    report = qualify_store(root, require_directory_identity=False)
    assert report.failures[0].check_id == check_id


def test_split_phase_disjointness_and_fingerprint_are_checked(
    qualified_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
) -> None:
    source = qualified_stores["heterogeneous"].store_build.path

    invalid_id = _mutable_store(source, tmp_path / "split-id")
    manifest = _manifest(invalid_id)
    relative = manifest["splits"]["primary"]["phases"]["train"]["relative_path"]
    value = np.array(np.load(invalid_id / relative, allow_pickle=False), copy=True)
    value[0] = 99
    _save_array(invalid_id, relative, value)
    assert qualify_store(invalid_id, require_directory_identity=False).failures[0].check_id == "SPLIT-ID-001"

    overlap = _mutable_store(source, tmp_path / "split-overlap")
    manifest = _manifest(overlap)
    train = manifest["splits"]["primary"]["phases"]["train"]["relative_path"]
    val = manifest["splits"]["primary"]["phases"]["val"]["relative_path"]
    train_value = np.load(overlap / train, allow_pickle=False)
    _save_array(overlap, val, np.array([train_value[0]], dtype=np.int64))
    assert qualify_store(overlap, require_directory_identity=False).failures[0].check_id == "SPLIT-DISJOINT-001"

    stale = _mutable_store(source, tmp_path / "split-fingerprint")
    manifest = _manifest(stale)
    manifest["splits"]["primary"]["fingerprint"] = "0" * 64
    _write_manifest(stale, manifest)
    assert qualify_store(stale, require_directory_identity=False).failures[0].check_id == "SPLIT-FINGERPRINT-001"


@pytest.mark.parametrize(
    ("role_prefix", "check_id", "mutation"),
    [
        ("partition-assignment:", "PARTITION-ID-001", lambda value: np.full_like(value, 99)),
        ("partition-permutation:", "PARTITION-PERMUTATION-001", lambda value: np.zeros_like(value)),
        ("partition-inverse:", "PARTITION-INVERSE-001", lambda value: np.zeros_like(value)),
        ("partition-partptr:", "PARTITION-PARTPTR-001", lambda value: np.zeros_like(value)),
        ("partition-edge-ownership:", "PARTITION-EDGE-OWNERSHIP-001", lambda value: np.full_like(value, 99)),
    ],
)
def test_partition_arrays_and_book_identity_are_cross_checked(
    qualified_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
    role_prefix: str,
    check_id: str,
    mutation: Callable[[np.ndarray], np.ndarray],
) -> None:
    root = _mutable_store(
        qualified_stores["heterogeneous"].store_build.path,
        tmp_path / check_id,
    )
    manifest = _manifest(root)
    relative = _array_path(manifest, role_prefix)
    value = np.load(root / relative, allow_pickle=False)
    _save_array(root, relative, mutation(value))
    assert qualify_store(root, require_directory_identity=False).failures[0].check_id == check_id


def test_stale_task_binding_and_content_directory_identity_are_last_gates(
    qualified_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
) -> None:
    source = qualified_stores["homogeneous"].store_build.path
    stale = _mutable_store(source, tmp_path / "stale")
    report = qualify_store(
        stale,
        expected_bindings={"consumer_contract": "required"},
        require_directory_identity=False,
    )
    assert report.failures[0].check_id == "STALE-BINDING-001"

    content = _mutable_store(source, tmp_path / "content")
    manifest = _manifest(content)
    manifest["content_sha256"] = "0" * 64
    _write_manifest(content, manifest)
    report = qualify_store(content, require_directory_identity=False)
    assert report.failures[0].check_id == "CONTENT-IDENTITY-001"


@pytest.mark.parametrize(
    ("relative", "mutation", "check_id"),
    [
        (
            "build_environment.json",
            lambda record: record.pop("format_version"),
            "BUILD-ENVIRONMENT-001",
        ),
        (
            "build_environment.json",
            lambda record: record.update(
                dependency_versions={"forged": "0"}
            ),
            "BUILD-ENVIRONMENT-001",
        ),
        (
            "qualification_report.json",
            lambda record: record.update(metadata_binding_sha256="0" * 64),
            "QUALIFICATION-REPORT-001",
        ),
    ],
)
def test_versioned_metadata_is_exact_and_cross_bound(
    qualified_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
    relative: str,
    mutation: Callable[[dict[str, object]], object],
    check_id: str,
) -> None:
    root = _mutable_store(
        qualified_stores["homogeneous"].store_build.path,
        tmp_path / check_id,
    )
    path = root / relative
    record = json.loads(path.read_text(encoding="utf-8"))
    mutation(record)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    _reseal_file(root, relative)
    report = qualify_store(root, require_directory_identity=False)
    assert report.failures[0].check_id == check_id


def test_validate_store_returns_the_manifest_instance_that_was_qualified(
    qualified_stores: dict[str, QualifiedStoreFixture],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = qualified_stores["homogeneous"].store_build.path
    original = qualification_module._read_json_secure
    manifest_reads = 0

    def tracked(
        path: Path,
        expected_identity: tuple[int, int, int, int, int] | None = None,
    ) -> dict[str, object]:
        nonlocal manifest_reads
        result = original(path, expected_identity)
        if path.name == "manifest.json":
            manifest_reads += 1
            if manifest_reads > 1:
                result["output_kind"] = "unchecked"
        return result

    monkeypatch.setattr(qualification_module, "_read_json_secure", tracked)
    validated = validate_store(source)
    assert manifest_reads == 1
    assert validated.manifest["output_kind"] == "homogeneous"


def test_top_manifest_schema_and_output_kind_are_exact(
    qualified_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
) -> None:
    source = qualified_stores["heterogeneous"].store_build.path
    extra = _mutable_store(source, tmp_path / "extra-manifest-key")
    manifest = _manifest(extra)
    manifest["future_field"] = True
    _write_manifest(extra, manifest)
    assert (
        qualify_store(extra, require_directory_identity=False).failures[0].check_id
        == "MANIFEST-SCHEMA-001"
    )

    inconsistent = _mutable_store(source, tmp_path / "inconsistent-output-kind")
    manifest = _manifest(inconsistent)
    manifest["output_kind"] = "homogeneous"
    _write_manifest(inconsistent, manifest)
    assert (
        qualify_store(
            inconsistent,
            require_directory_identity=False,
        ).failures[0].check_id
        == "OUTPUT-KIND-001"
    )


def test_qualification_report_preserves_exact_upstream_check_set(
    qualified_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
) -> None:
    source = qualified_stores["homogeneous"].store_build.path
    manifest = _manifest(source)
    report = json.loads(
        (source / "qualification_report.json").read_text(encoding="utf-8")
    )
    expected_set = manifest["qualification_check_set"]
    assert expected_set["count"] == len(report["checks"])
    assert expected_set["count"] > 0
    upstream = qualified_stores[
        "homogeneous"
    ].partition_build.book.qualification_checks
    assert [check["check_id"] for check in report["checks"]] == [
        check.check_id for check in upstream
    ]
    for stored, original in zip(report["checks"], upstream, strict=True):
        assert set(stored) == {
            "check_id",
            "passed",
            "observed",
            "expected",
            "limit",
            "evidence",
            "remediation",
        }
        assert stored["passed"] is original.passed
        assert stored["observed"] == qualification_module._json_value(
            original.observed
        )
        assert stored["expected"] == qualification_module._json_value(
            original.limit
        )
        assert stored["limit"] == qualification_module._json_value(
            original.limit
        )
        assert stored["evidence"] == {"detail": original.detail}
        assert stored["remediation"] == (
            "none" if original.passed else "repartition"
        )
    assert expected_set["sha256"] == (
        qualification_module.qualification_check_set_fingerprint(
            report["checks"]
        )
    )

    variants = {
        "empty": [],
        "truncated": report["checks"][:-1],
        "duplicate": [*report["checks"], report["checks"][0]],
        "reordered": list(reversed(report["checks"])),
    }
    for name, checks in variants.items():
        root = _mutable_store(source, tmp_path / name)
        path = root / "qualification_report.json"
        changed = dict(report)
        changed["checks"] = checks
        path.write_text(
            json.dumps(changed, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _reseal_file(root, "qualification_report.json")
        failure = qualify_store(
            root,
            require_directory_identity=False,
        ).failures[0]
        assert failure.check_id == "QUALIFICATION-REPORT-001"


@pytest.mark.parametrize(
    ("target_kind", "check_id"),
    [
        ("array", "ARRAY-SHAPE-001"),
        ("json", "BUILD-ENVIRONMENT-001"),
    ],
)
def test_same_size_rewrite_with_restored_mtime_is_rejected_on_reopen(
    qualified_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
    check_id: str,
) -> None:
    root = _mutable_store(
        qualified_stores["homogeneous"].store_build.path,
        tmp_path / target_kind,
    )
    manifest = _manifest(root)
    relative = (
        _array_path(manifest, "node-feature:")
        if target_kind == "array"
        else "build_environment.json"
    )
    target = root / relative
    original_hash = qualification_module._hash_file
    attacked = False

    def race(
        path: Path,
    ) -> tuple[str, int, tuple[int, int, int, int, int]]:
        nonlocal attacked
        result = original_hash(path)
        if path == target and not attacked:
            attacked = True
            before = path.stat()
            if target_kind == "array":
                array = np.load(path, mmap_mode="r+")
                array.flat[0] = array.flat[0] + 1
                array.flush()
                del array
            else:
                payload = path.read_bytes()
                changed = payload.replace(
                    b'"os":"Darwin"',
                    b'"os":"Xarwin"',
                    1,
                )
                assert changed != payload and len(changed) == len(payload)
                path.write_bytes(changed)
            os.utime(
                path,
                ns=(before.st_atime_ns, before.st_mtime_ns),
            )
            after = path.stat()
            assert after.st_size == before.st_size
            assert after.st_mtime_ns == before.st_mtime_ns
            assert after.st_ctime_ns != before.st_ctime_ns
        return result

    monkeypatch.setattr(qualification_module, "_hash_file", race)
    report = qualify_store(root, require_directory_identity=False)
    assert report.failures[0].check_id == check_id
