"""Digest-pinned, non-executable prepartitioned store bundles."""

from __future__ import annotations

import functools
import hashlib
import http.server
import json
import os
import stat
import threading
import zipfile
from pathlib import Path
from typing import Callable

import pytest

import topobench.data.stores.store_bundle as bundle_module
from topobench.data.stores.qualification_checks import (
    QualificationFailure,
    compute_content_identity,
)
from topobench.data.stores.store_bundle import BundleLimits, StoreBundle
from test.data.stores.test_typed_graph_store import (
    QualifiedStoreFixture,
    qualified_stores,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_archive(
    source: Path,
    destination: Path,
    transform: Callable[[str, bytes], tuple[str, bytes] | None],
) -> None:
    with zipfile.ZipFile(source, "r") as original, zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_STORED,
    ) as rewritten:
        for info in original.infolist():
            changed = transform(info.filename, original.read(info))
            if changed is None:
                continue
            name, payload = changed
            rewritten.writestr(name, payload)


def test_package_download_install_move_and_reopen_over_local_http(
    qualified_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
) -> None:
    fixture = qualified_stores["heterogeneous"]
    archive = tmp_path / "qualified-store.zip"
    artifact = StoreBundle.package(fixture.store_build.path, archive)
    assert artifact.path == archive
    assert artifact.sha256 == _sha256(archive)
    assert artifact.byte_size == archive.stat().st_size
    with zipfile.ZipFile(archive) as bundle:
        assert all(info.compress_type == zipfile.ZIP_STORED for info in bundle.infolist())
        assert not any(
            Path(info.filename).suffix in {".pt", ".pth", ".pkl", ".pickle"}
            for info in bundle.infolist()
        )

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(tmp_path),
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        downloaded = StoreBundle.download(
            f"http://127.0.0.1:{server.server_port}/{archive.name}",
            tmp_path / "downloaded.zip",
            expected_sha256=artifact.sha256,
            max_bytes=artifact.byte_size,
            timeout_seconds=5.0,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    installed = StoreBundle.install(
        downloaded,
        tmp_path / "installed-stores",
        expected_sha256=artifact.sha256,
    )
    assert installed.content_sha256 == fixture.store_build.content_sha256
    source_path = installed.path
    moved = StoreBundle.move(installed, tmp_path / "moved-stores")
    assert not source_path.exists()
    assert moved.path.name == fixture.store_build.content_sha256
    assert moved.external_ids("entity.kind", [0, 3]) == ["a", "d"]
    same_location = StoreBundle.move(moved, moved.path.parent)
    assert same_location.path.name == fixture.store_build.content_sha256
    same_location.close()


def test_download_requires_exact_digest_and_enforces_streaming_byte_cap(
    qualified_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
) -> None:
    archive = tmp_path / "store.zip"
    artifact = StoreBundle.package(
        qualified_stores["homogeneous"].store_build.path,
        archive,
    )
    uri = archive.as_uri()
    with pytest.raises(ValueError, match="BUNDLE-DIGEST-001"):
        StoreBundle.download(
            uri,
            tmp_path / "bad-digest.zip",
            expected_sha256="0" * 64,
            max_bytes=artifact.byte_size,
        )
    with pytest.raises(ValueError, match="BUNDLE-SIZE-001"):
        StoreBundle.download(
            uri,
            tmp_path / "too-large.zip",
            expected_sha256=artifact.sha256,
            max_bytes=artifact.byte_size - 1,
        )
    assert not (tmp_path / "bad-digest.zip").exists()
    assert not (tmp_path / "too-large.zip").exists()


@pytest.mark.parametrize(
    "member_name",
    ["../escape", "/absolute", "C:\\windows", "nested\\backslash"],
)
def test_install_rejects_untrusted_member_paths(
    tmp_path: Path,
    member_name: str,
) -> None:
    archive = tmp_path / f"unsafe-{abs(hash(member_name))}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        bundle.writestr(member_name, b"payload")
    with pytest.raises(ValueError, match="BUNDLE-PATH-001"):
        StoreBundle.install(
            archive,
            tmp_path / "stores",
            expected_sha256=_sha256(archive),
        )
    assert not (tmp_path / "escape").exists()


def test_install_rejects_duplicates_case_collisions_special_files_and_compression(
    tmp_path: Path,
) -> None:
    archives: list[tuple[Path, str]] = []

    duplicate = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate, "w") as bundle:
        bundle.writestr("manifest.json", b"one")
        bundle.writestr("manifest.json", b"two")
    archives.append((duplicate, "BUNDLE-DUPLICATE-001"))

    case_collision = tmp_path / "case.zip"
    with zipfile.ZipFile(case_collision, "w") as bundle:
        bundle.writestr("A", b"one")
        bundle.writestr("a", b"two")
    archives.append((case_collision, "BUNDLE-DUPLICATE-001"))

    symlink = tmp_path / "symlink.zip"
    with zipfile.ZipFile(symlink, "w") as bundle:
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        bundle.writestr(info, "target")
    archives.append((symlink, "BUNDLE-TYPE-001"))

    compressed = tmp_path / "compressed.zip"
    with zipfile.ZipFile(
        compressed,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as bundle:
        bundle.writestr("manifest.json", b"{}" * 100)
    archives.append((compressed, "BUNDLE-COMPRESSION-001"))

    for archive, check_id in archives:
        with pytest.raises(ValueError, match=check_id):
            StoreBundle.install(
                archive,
                tmp_path / f"stores-{archive.stem}",
                expected_sha256=_sha256(archive),
            )


def test_archive_limits_and_validated_file_set_are_enforced(
    qualified_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zip"
    artifact = StoreBundle.package(
        qualified_stores["homogeneous"].store_build.path,
        source,
    )
    with zipfile.ZipFile(source) as bundle:
        count = len(bundle.infolist())
        total = sum(info.file_size for info in bundle.infolist())
    with pytest.raises(ValueError, match="BUNDLE-COUNT-001"):
        StoreBundle.install(
            source,
            tmp_path / "count-stores",
            expected_sha256=artifact.sha256,
            limits=BundleLimits(max_members=count - 1),
        )
    with pytest.raises(ValueError, match="BUNDLE-SIZE-001"):
        StoreBundle.install(
            source,
            tmp_path / "size-stores",
            expected_sha256=artifact.sha256,
            limits=BundleLimits(max_uncompressed_bytes=total - 1),
        )

    missing = tmp_path / "missing.zip"
    _rewrite_archive(
        source,
        missing,
        lambda name, payload: None if name.endswith("x.npy") else (name, payload),
    )
    with pytest.raises(QualificationFailure) as captured:
        StoreBundle.install(
            missing,
            tmp_path / "missing-stores",
            expected_sha256=_sha256(missing),
        )
    assert captured.value.check_id == "FILE-SET-001"
    assert captured.value.report_path is not None
    assert captured.value.report_path.is_file()
    failure_report = json.loads(
        captured.value.report_path.read_text(encoding="utf-8")
    )
    assert failure_report["checks"][0]["evidence"]
    assert failure_report["checks"][0]["remediation"]

    extra = tmp_path / "extra.zip"
    _rewrite_archive(source, extra, lambda name, payload: (name, payload))
    with zipfile.ZipFile(extra, "a", compression=zipfile.ZIP_STORED) as bundle:
        bundle.writestr("unexpected.bin", b"unexpected")
    with pytest.raises(QualificationFailure) as captured:
        StoreBundle.install(
            extra,
            tmp_path / "extra-stores",
            expected_sha256=_sha256(extra),
        )
    assert captured.value.check_id == "FILE-SET-001"


def test_install_rejects_manifest_declared_executable_payload(
    qualified_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
) -> None:
    source = tmp_path / "safe.zip"
    StoreBundle.package(qualified_stores["homogeneous"].store_build.path, source)
    payload = b"not-a-safe-store-artifact"
    unsafe = tmp_path / "unsafe-payload.zip"

    def declare_payload(name: str, data: bytes) -> tuple[str, bytes]:
        if name == "manifest.json":
            manifest = json.loads(data)
            manifest["files"].append(
                {
                    "relative_path": "payload.pkl",
                    "role": "partition-statistics",
                    "dtype": "binary",
                    "shape": [],
                    "byte_size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "finite": False,
                }
            )
            manifest["files"].sort(key=lambda record: record["relative_path"])
            manifest["content_sha256"] = compute_content_identity(manifest)
            data = json.dumps(manifest, sort_keys=True).encode()
        return name, data

    _rewrite_archive(source, unsafe, declare_payload)
    with zipfile.ZipFile(unsafe, "a", compression=zipfile.ZIP_STORED) as bundle:
        bundle.writestr("payload.pkl", payload)
    with pytest.raises(QualificationFailure) as captured:
        StoreBundle.install(
            unsafe,
            tmp_path / "unsafe-stores",
            expected_sha256=_sha256(unsafe),
        )
    assert captured.value.check_id == "FILE-POLICY-001"


def test_install_pins_the_archive_descriptor_before_digest_validation(
    qualified_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "qualified.zip"
    expected_identity = qualified_stores["homogeneous"].store_build.content_sha256
    StoreBundle.package(
        qualified_stores["homogeneous"].store_build.path,
        archive,
    )
    expected_digest = _sha256(archive)
    replacement = tmp_path / "replacement.zip"
    with zipfile.ZipFile(
        replacement,
        "w",
        compression=zipfile.ZIP_STORED,
    ) as bundle:
        bundle.writestr("unexpected.bin", b"not the qualified archive")
    original_hash = bundle_module._sha256_file

    def replace_after_hash(path: Path) -> str:
        digest = original_hash(path)
        os.replace(replacement, path)
        return digest

    monkeypatch.setattr(bundle_module, "_sha256_file", replace_after_hash)
    installed = StoreBundle.install(
        archive,
        tmp_path / "pinned-stores",
        expected_sha256=expected_digest,
    )
    assert installed.path.name == expected_identity
    installed.close()


def test_install_rejects_unknown_version_and_cross_filesystem_promotion(
    qualified_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.zip"
    StoreBundle.package(
        qualified_stores["homogeneous"].store_build.path,
        source,
    )
    unknown = tmp_path / "unknown.zip"

    def change_version(name: str, payload: bytes) -> tuple[str, bytes]:
        if name == "manifest.json":
            manifest = json.loads(payload)
            manifest["format_version"] = "future-store-v99"
            payload = json.dumps(manifest, sort_keys=True).encode()
        return name, payload

    _rewrite_archive(source, unknown, change_version)
    with pytest.raises(QualificationFailure) as captured:
        StoreBundle.install(
            unknown,
            tmp_path / "unknown-stores",
            expected_sha256=_sha256(unknown),
        )
    assert captured.value.check_id == "VERSION-001"

    monkeypatch.setattr(bundle_module, "_same_filesystem", lambda _a, _b: False)
    with pytest.raises(ValueError, match="BUNDLE-FILESYSTEM-001"):
        StoreBundle.install(
            source,
            tmp_path / "cross-filesystem-stores",
            expected_sha256=_sha256(source),
        )
