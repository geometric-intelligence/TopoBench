"""Safe non-executable packaging and promotion of typed graph stores."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO

from topobench.data.stores.qualification_checks import (
    QualificationFailure,
    _secure_descriptor,
    _stat_identity,
    validate_store,
)
from topobench.data.stores.typed_graph_store import (
    TypedGraphStore,
    _fsync_directory,
    _fsync_tree,
    _make_read_only,
)


@dataclass(frozen=True, slots=True)
class BundleArtifact:
    path: Path
    sha256: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class BundleLimits:
    max_members: int = 100_000
    max_uncompressed_bytes: int = 1 << 40
    max_member_bytes: int = 1 << 38
    max_archive_bytes: int = 1 << 40
    max_compression_ratio: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "max_members",
            "max_uncompressed_bytes",
            "max_member_bytes",
            "max_archive_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_compression_ratio < 1.0:
            raise ValueError("max_compression_ratio must be at least one")


class StoreBundle:
    """Package, digest-pin, safely extract, promote, and move stores."""

    @staticmethod
    def package(store_path: str | Path, archive_path: str | Path) -> BundleArtifact:
        validated = validate_store(store_path)
        root = validated.root
        archive = Path(archive_path)
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_name(f".{archive.name}.{uuid.uuid4().hex}.tmp")
        members = [
            "manifest.json",
            *sorted(record["relative_path"] for record in validated.manifest["files"]),
        ]
        try:
            with zipfile.ZipFile(
                temporary,
                "w",
                compression=zipfile.ZIP_STORED,
                allowZip64=True,
            ) as bundle:
                for relative in members:
                    info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_STORED
                    info.create_system = 3
                    info.external_attr = (stat.S_IFREG | 0o444) << 16
                    _copy_validated_member(
                        root / relative,
                        validated.file_identities[relative],
                        bundle,
                        info,
                    )
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, archive)
            _fsync_directory(archive.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return BundleArtifact(archive, _sha256_file(archive), archive.stat().st_size)

    @staticmethod
    def download(
        url: str,
        destination: str | Path,
        *,
        expected_sha256: str,
        max_bytes: int,
        timeout_seconds: float = 30.0,
    ) -> Path:
        expected = _expected_digest(expected_sha256)
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("BUNDLE-SIZE-001: max_bytes must be positive")
        if timeout_seconds <= 0:
            raise ValueError("BUNDLE-TIMEOUT-001: timeout must be positive")
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.download")
        digest = hashlib.sha256()
        total = 0
        try:
            with urllib.request.urlopen(url, timeout=timeout_seconds) as response, temporary.open("xb") as stream:
                while True:
                    chunk = response.read(min(1024 * 1024, max_bytes - total + 1))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(
                            f"BUNDLE-SIZE-001: download exceeded byte cap {max_bytes}"
                        )
                    digest.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            observed = digest.hexdigest()
            if observed != expected:
                raise ValueError(
                    f"BUNDLE-DIGEST-001: observed {observed}, expected {expected}"
                )
            os.replace(temporary, target)
            _fsync_directory(target.parent)
            return target
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def install(
        archive_path: str | Path,
        stores_root: str | Path,
        *,
        expected_sha256: str,
        limits: BundleLimits | None = None,
    ) -> TypedGraphStore:
        archive = Path(archive_path)
        expected = _expected_digest(expected_sha256)
        active_limits = limits or BundleLimits()
        descriptor, before = _secure_descriptor(archive)
        archive_stream = os.fdopen(descriptor, "rb", closefd=True)
        stage: Path | None = None
        try:
            size = before.st_size
            if size > active_limits.max_archive_bytes:
                raise ValueError(
                    f"BUNDLE-SIZE-001: archive has {size} bytes, limit is {active_limits.max_archive_bytes}"
                )
            observed = _sha256_stream(archive_stream)
            if _stat_identity(before) != _stat_identity(
                os.fstat(archive_stream.fileno())
            ):
                raise ValueError(
                    "BUNDLE-MUTATION-001: archive changed during digest validation"
                )
            if observed != expected:
                raise ValueError(
                    f"BUNDLE-DIGEST-001: observed {observed}, expected {expected}"
                )
            archive_stream.seek(0)
            root = Path(stores_root)
            root.mkdir(parents=True, exist_ok=True)
            staging_parent = root / ".staging"
            staging_parent.mkdir(parents=True, exist_ok=True)
            stage = staging_parent / f"bundle-{uuid.uuid4().hex}"
            report_path = (
                root
                / ".qualification-reports"
                / f"bundle-install-{uuid.uuid4().hex}.json"
            )
            stage.mkdir(parents=False, exist_ok=False)
            if not _same_filesystem(stage, root):
                raise ValueError(
                    "BUNDLE-FILESYSTEM-001: extraction staging and final store differ"
                )
            with zipfile.ZipFile(
                archive_stream,
                "r",
                allowZip64=True,
            ) as bundle:
                for info, relative in _validated_members(bundle, active_limits):
                    destination = stage.joinpath(
                        *PurePosixPath(relative).parts
                    )
                    if info.is_dir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    _extract_member(bundle, info, destination)
            if _stat_identity(before) != _stat_identity(
                os.fstat(archive_stream.fileno())
            ):
                raise ValueError(
                    "BUNDLE-MUTATION-001: archive changed during extraction"
                )
            validated = validate_store(
                stage,
                report_path=report_path,
                require_directory_identity=False,
            )
            identity = validated.manifest["content_sha256"]
            target = root / identity
            if os.path.lexists(target):
                try:
                    validate_store(target, require_directory_identity=True)
                except QualificationFailure:
                    if target.is_symlink() or not target.is_dir():
                        raise ValueError(
                            "BUNDLE-TYPE-001: colliding final path is unsafe"
                        )
                    quarantine = (
                        root
                        / f".quarantine-{identity}-{uuid.uuid4().hex}"
                    )
                    os.replace(target, quarantine)
                    _fsync_directory(root)
                else:
                    shutil.rmtree(stage)
                    return TypedGraphStore.open(target)
            _make_read_only(stage, movable_root=True)
            _fsync_tree(stage)
            os.replace(stage, target)
            target.chmod(0o555)
            _fsync_directory(target)
            _fsync_directory(root)
            return TypedGraphStore.open(target)
        finally:
            archive_stream.close()
            if stage is not None and stage.exists():
                shutil.rmtree(stage, ignore_errors=True)

    @staticmethod
    def move(
        store: TypedGraphStore | str | Path,
        destination_root: str | Path,
    ) -> TypedGraphStore:
        if isinstance(store, TypedGraphStore):
            source = store.path
            identity = store.content_sha256
            store.close()
        else:
            opened = TypedGraphStore.open(store)
            source = opened.path
            identity = opened.content_sha256
            opened.close()
        destination_parent = Path(destination_root)
        destination_parent.mkdir(parents=True, exist_ok=True)
        if not _same_filesystem(source, destination_parent):
            raise ValueError(
                "BUNDLE-FILESYSTEM-001: moving a store requires one filesystem"
            )
        target = destination_parent / identity
        if os.path.lexists(target) and os.path.samefile(source, target):
            return TypedGraphStore.open(source)
        if os.path.lexists(target):
            validate_store(target, require_directory_identity=True)
            _remove_read_only_tree(source)
        else:
            source.chmod(0o755)
            try:
                os.replace(source, target)
            except BaseException:
                source.chmod(0o555)
                raise
            target.chmod(0o555)
        _fsync_directory(source.parent)
        _fsync_directory(destination_parent)
        return TypedGraphStore.open(target)


def _validated_members(
    bundle: zipfile.ZipFile,
    limits: BundleLimits,
) -> list[tuple[zipfile.ZipInfo, str]]:
    infos = bundle.infolist()
    if len(infos) > limits.max_members:
        raise ValueError(
            f"BUNDLE-COUNT-001: archive has {len(infos)} members, limit is {limits.max_members}"
        )
    names: set[str] = set()
    folded: set[str] = set()
    total = 0
    result: list[tuple[zipfile.ZipInfo, str]] = []
    for info in infos:
        relative = _member_path(info.filename)
        folded_name = relative.casefold()
        if relative in names or folded_name in folded:
            raise ValueError(
                f"BUNDLE-DUPLICATE-001: duplicate or case-colliding member {relative!r}"
            )
        names.add(relative)
        folded.add(folded_name)
        if info.flag_bits & 0x1:
            raise ValueError("BUNDLE-TYPE-001: encrypted members are unsupported")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(
                f"BUNDLE-COMPRESSION-001: member {relative!r} is not ZIP_STORED"
            )
        mode = info.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        if info.is_dir():
            if file_type not in {0, stat.S_IFDIR}:
                raise ValueError(
                    f"BUNDLE-TYPE-001: directory member {relative!r} has unsafe type"
                )
        elif file_type not in {0, stat.S_IFREG}:
            raise ValueError(
                f"BUNDLE-TYPE-001: member {relative!r} is not a regular file"
            )
        if info.file_size > limits.max_member_bytes:
            raise ValueError(
                f"BUNDLE-SIZE-001: member {relative!r} exceeds per-member limit"
            )
        total += info.file_size
        if total > limits.max_uncompressed_bytes:
            raise ValueError(
                "BUNDLE-SIZE-001: total uncompressed bytes exceed the archive limit"
            )
        ratio = info.file_size / max(1, info.compress_size)
        if ratio > limits.max_compression_ratio:
            raise ValueError(
                f"BUNDLE-COMPRESSION-001: member {relative!r} ratio {ratio} exceeds limit"
            )
        result.append((info, relative))
    return result


def _member_path(value: str) -> str:
    if not value or "\x00" in value or "\\" in value:
        raise ValueError(f"BUNDLE-PATH-001: unsafe member path {value!r}")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in posix.parts
        or posix.as_posix() != value
        or value in {".", ""}
    ):
        raise ValueError(f"BUNDLE-PATH-001: unsafe member path {value!r}")
    return value


def _extract_member(
    bundle: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination: Path,
) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
    copied = 0
    try:
        with bundle.open(info, "r") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                copied += len(chunk)
                if copied > info.file_size:
                    raise ValueError(
                        f"BUNDLE-SIZE-001: member {info.filename!r} exceeded declared size"
                    )
                view = memoryview(chunk)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
        if copied != info.file_size:
            raise ValueError(
                f"BUNDLE-SIZE-001: member {info.filename!r} was truncated"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_read_only_tree(root: Path) -> None:
    root.chmod(0o755)
    for path in root.rglob("*"):
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o755)
    shutil.rmtree(root)


def _copy_validated_member(
    source_path: Path,
    expected_identity: tuple[int, int, int, int, int],
    bundle: zipfile.ZipFile,
    info: zipfile.ZipInfo,
) -> None:
    descriptor, before = _secure_descriptor(source_path)
    if _stat_identity(before) != expected_identity:
        os.close(descriptor)
        raise ValueError(
            f"BUNDLE-SOURCE-001: validated member changed: {info.filename}"
        )
    with os.fdopen(descriptor, "rb", closefd=True) as source, bundle.open(
        info,
        "w",
        force_zip64=True,
    ) as destination:
        _copy_stream(source, destination)
        if _stat_identity(before) != _stat_identity(
            os.fstat(source.fileno())
        ):
            raise ValueError(
                f"BUNDLE-SOURCE-001: member changed while packaging: {info.filename}"
            )


def _copy_stream(source: BinaryIO, destination: BinaryIO) -> None:
    while True:
        chunk = source.read(1024 * 1024)
        if not chunk:
            return
        destination.write(chunk)


def _expected_digest(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("BUNDLE-DIGEST-001: expected SHA256 must be lowercase hex")
    return value


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _same_filesystem(left: Path, right: Path) -> bool:
    return left.stat().st_dev == right.stat().st_dev


__all__ = ["BundleArtifact", "BundleLimits", "StoreBundle"]
