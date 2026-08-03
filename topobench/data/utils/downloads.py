"""Authenticated, bounded, and atomic remote archive acquisition."""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import posixpath
import re
import shutil
import stat
import tarfile
import tempfile
import threading
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO, Literal
from urllib.parse import urljoin, urlparse

import requests

CONNECT_TIMEOUT_SECONDS = 10.0
READ_TIMEOUT_SECONDS = 30.0
MAX_REDIRECTS = 5
DOWNLOAD_CHUNK_BYTES = 64 * 1024
EXTRACTION_CHUNK_BYTES = 64 * 1024
_ASSET_RECEIPT = ".topobench-remote-archive.json"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_PROCESS_ACQUISITION_LOCK = threading.Lock()
_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})


@dataclass(frozen=True, slots=True)
class ArchiveLimits:
    """Exact resource ceilings for one remote archive."""

    max_compressed_bytes: int
    max_members: int
    max_member_bytes: int
    max_total_bytes: int
    max_expansion_ratio: int

    def __post_init__(self) -> None:
        for field_name in (
            "max_compressed_bytes",
            "max_members",
            "max_member_bytes",
            "max_total_bytes",
            "max_expansion_ratio",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{field_name} must be a positive integer")
        if self.max_member_bytes > self.max_total_bytes:
            raise ValueError("max_member_bytes cannot exceed max_total_bytes")


def _validate_https_url(url: str) -> None:
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
    except (TypeError, ValueError) as error:
        raise ValueError(
            "remote archive URL must be an exact HTTPS URL"
        ) from error
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError("remote archive URL must be an exact HTTPS URL")


@dataclass(frozen=True, slots=True)
class RemoteArchive:
    """Immutable trust and resource manifest for a remote archive."""

    url: str
    sha256: str
    size_bytes: int
    limits: ArchiveLimits
    archive_format: Literal["zip", "tar"] = "zip"

    def __post_init__(self) -> None:
        _validate_https_url(self.url)
        if _SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError(
                "sha256 must be exactly 64 lowercase hexadecimal digits"
            )
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            raise ValueError("size_bytes must be a positive integer")
        if self.size_bytes > self.limits.max_compressed_bytes:
            raise ValueError(
                "size_bytes cannot exceed the compressed-byte limit"
            )
        if self.archive_format not in {"zip", "tar"}:
            raise ValueError("archive_format must be exactly 'zip' or 'tar'")


@dataclass(frozen=True, slots=True)
class _ArchiveMember:
    name: str
    relative_path: PurePosixPath
    size: int
    is_directory: bool
    source: zipfile.ZipInfo | tarfile.TarInfo


def acquire_verified_archive(
    asset: RemoteArchive,
    destination: str | Path,
    validate: Callable[[Path], None],
) -> Path:
    """Authenticate, safely extract, validate, and atomically publish an archive."""
    destination_path = Path(destination)
    if destination_path.name in {"", ".", ".."}:
        raise ValueError("cache destination must name a child directory")
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with _exclusive_directory_lock(destination_path.parent):
        if _reuse_authenticated_destination(asset, destination_path, validate):
            return destination_path

        with tempfile.TemporaryDirectory(
            prefix=f".{destination_path.name}.",
            dir=destination_path.parent,
        ) as private_name:
            private_root = Path(private_name)
            os.chmod(private_root, 0o700)
            archive_path = private_root / "archive"
            extraction_root = private_root / "extracted"

            _download_authenticated_archive(asset, archive_path)
            extraction_root.mkdir(mode=0o700)
            _extract_bounded_archive(asset, archive_path, extraction_root)
            validate(extraction_root)
            _write_asset_receipt(asset, extraction_root)
            _fsync_tree(extraction_root)

            if destination_path.exists():
                raise FileExistsError(
                    f"cache destination appeared during acquisition: {destination_path}"
                )
            os.replace(extraction_root, destination_path)
            _fsync_directory(destination_path.parent)

    return destination_path


def _reuse_authenticated_destination(
    asset: RemoteArchive,
    destination: Path,
    validate: Callable[[Path], None],
) -> bool:
    if not destination.exists():
        return False
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("cache destination must be a real directory")

    receipt_path = destination / _ASSET_RECEIPT
    if _authenticated_receipt_matches(asset, receipt_path):
        try:
            validate(destination)
        except Exception:
            shutil.rmtree(destination)
            return False
        return True

    shutil.rmtree(destination)
    return False


def _authenticated_receipt_matches(
    asset: RemoteArchive,
    receipt_path: Path,
) -> bool:
    expected = _asset_receipt_bytes(asset)
    try:
        metadata = receipt_path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != len(expected):
        return False

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(receipt_path, flags)
    except OSError:
        return False
    with os.fdopen(descriptor, "rb", closefd=True) as stream:
        actual = stream.read(len(expected) + 1)
    return hmac.compare_digest(actual, expected)


@contextmanager
def _exclusive_directory_lock(directory: Path) -> Iterator[None]:
    with _PROCESS_ACQUISITION_LOCK:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        descriptor = os.open(directory, flags)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


@contextmanager
def _private_binary_output(path: Path) -> Iterator[BinaryIO]:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        stream = os.fdopen(descriptor, "wb", closefd=True)
    except BaseException:
        os.close(descriptor)
        raise
    try:
        yield stream
    finally:
        stream.close()


def _resolve_redirect_url(current_url: str, location: object) -> str:
    if not isinstance(location, str) or not location.strip():
        raise ValueError(
            "redirect response is missing a valid Location header"
        )
    try:
        redirected_url = urljoin(current_url, location.strip())
    except (TypeError, ValueError) as error:
        raise ValueError(
            "remote archive URL must be an exact HTTPS URL"
        ) from error
    _validate_https_url(redirected_url)
    return redirected_url


@contextmanager
def _streaming_response(asset: RemoteArchive) -> Iterator[requests.Response]:
    current_url = asset.url
    visited_urls = {current_url}
    redirects_followed = 0

    while True:
        response = requests.get(
            current_url,
            stream=True,
            headers={"Accept-Encoding": "identity"},
            timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
            allow_redirects=False,
        )
        try:
            status_code = response.status_code
            if status_code in _REDIRECT_STATUS_CODES:
                if redirects_followed >= MAX_REDIRECTS:
                    raise ValueError("remote archive redirect limit exceeded")
                next_url = _resolve_redirect_url(
                    current_url,
                    response.headers.get("Location"),
                )
                if next_url in visited_urls:
                    raise ValueError("remote archive redirect cycle detected")
                visited_urls.add(next_url)
                redirects_followed += 1
                current_url = next_url
                continue
            if 300 <= status_code < 400:
                raise ValueError(
                    f"unsupported remote archive redirect status {status_code}"
                )
            response.raise_for_status()
            yield response
            return
        finally:
            response.close()


def _download_authenticated_archive(
    asset: RemoteArchive,
    archive_path: Path,
) -> None:
    digest = hashlib.sha256()
    received = 0

    with _streaming_response(asset) as response:
        _validate_content_length(response.headers.get("Content-Length"), asset)

        with _private_binary_output(archive_path) as stream:
            for chunk in response.iter_content(
                chunk_size=DOWNLOAD_CHUNK_BYTES
            ):
                if not chunk:
                    continue
                if len(chunk) > asset.limits.max_compressed_bytes - received:
                    raise ValueError(
                        "download crossed the compressed-byte limit"
                    )
                stream.write(chunk)
                digest.update(chunk)
                received += len(chunk)
            stream.flush()
            os.fsync(stream.fileno())

    if received != asset.size_bytes:
        raise ValueError(
            f"downloaded size {received} does not match expected size "
            f"{asset.size_bytes}"
        )
    if not hmac.compare_digest(digest.hexdigest(), asset.sha256):
        raise ValueError("downloaded archive SHA-256 does not match manifest")


def _validate_content_length(
    declared_value: str | None,
    asset: RemoteArchive,
) -> None:
    if declared_value is None:
        return
    try:
        declared_size = int(declared_value)
    except ValueError as error:
        raise ValueError(
            "Content-Length must be a non-negative integer"
        ) from error
    if declared_size < 0:
        raise ValueError("Content-Length must be a non-negative integer")
    if declared_size > asset.limits.max_compressed_bytes:
        raise ValueError("declared body exceeds the compressed-byte limit")
    if declared_size != asset.size_bytes:
        raise ValueError(
            "declared body size does not match the exact manifest size"
        )


def _extract_bounded_archive(
    asset: RemoteArchive,
    archive_path: Path,
    extraction_root: Path,
) -> None:
    if asset.archive_format == "zip":
        with zipfile.ZipFile(archive_path) as archive:
            members = _validated_zip_members(archive, asset)
            _extract_zip_members(
                archive, members, extraction_root, asset.limits
            )
        return

    with tarfile.open(archive_path, mode="r:*") as archive:
        members = _validated_tar_members(archive, asset)
        _extract_tar_members(archive, members, extraction_root, asset.limits)


def _validated_zip_members(
    archive: zipfile.ZipFile,
    asset: RemoteArchive,
) -> tuple[_ArchiveMember, ...]:
    members: list[_ArchiveMember] = []
    total = 0
    seen: set[PurePosixPath] = set()

    for info in archive.infolist():
        if len(members) >= asset.limits.max_members:
            raise ValueError("archive member count exceeds manifest limit")
        is_directory = info.is_dir()
        _validate_zip_member_type(info, is_directory)
        relative_path = _validate_member_path(info.filename, is_directory)
        total = _bounded_expanded_total(total, info.file_size, asset.limits)
        _validate_unique_member(relative_path, is_directory, seen, members)
        seen.add(relative_path)
        members.append(
            _ArchiveMember(
                name=info.filename,
                relative_path=relative_path,
                size=info.file_size,
                is_directory=is_directory,
                source=info,
            )
        )

    _validate_expansion_ratio(total, asset.size_bytes, asset.limits)
    return tuple(members)


def _validated_tar_members(
    archive: tarfile.TarFile,
    asset: RemoteArchive,
) -> tuple[_ArchiveMember, ...]:
    members: list[_ArchiveMember] = []
    total = 0
    seen: set[PurePosixPath] = set()

    for info in archive:
        if len(members) >= asset.limits.max_members:
            raise ValueError("archive member count exceeds manifest limit")
        if not (info.isfile() or info.isdir()):
            raise ValueError(
                f"archive member {info.name!r} is not a regular file or directory"
            )
        relative_path = _validate_member_path(info.name, info.isdir())
        total = _bounded_expanded_total(total, info.size, asset.limits)
        _validate_unique_member(relative_path, info.isdir(), seen, members)
        seen.add(relative_path)
        members.append(
            _ArchiveMember(
                name=info.name,
                relative_path=relative_path,
                size=info.size,
                is_directory=info.isdir(),
                source=info,
            )
        )

    _validate_expansion_ratio(total, asset.size_bytes, asset.limits)
    return tuple(members)


def _validate_zip_member_type(
    info: zipfile.ZipInfo,
    is_directory: bool,
) -> None:
    unix_mode = info.external_attr >> 16
    file_type = stat.S_IFMT(unix_mode)
    expected_type = stat.S_IFDIR if is_directory else stat.S_IFREG
    if file_type not in {0, expected_type}:
        raise ValueError(
            f"archive member {info.filename!r} is not a regular file or directory"
        )
    if info.flag_bits & 0x1:
        raise ValueError(
            f"encrypted archive member {info.filename!r} is unsupported"
        )


def _validate_member_path(
    name: str,
    is_directory: bool,
) -> PurePosixPath:
    if not name or "\x00" in name or "\\" in name:
        raise ValueError(f"unsafe archive member path: {name!r}")
    normalized_name = (
        name[:-1] if is_directory and name.endswith("/") else name
    )
    if (
        normalized_name in {"", "."}
        or normalized_name.startswith("/")
        or posixpath.normpath(normalized_name) != normalized_name
        or PureWindowsPath(normalized_name).drive
    ):
        raise ValueError(f"unsafe archive member path: {name!r}")
    path = PurePosixPath(normalized_name)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    return path


def _bounded_expanded_total(
    current: int,
    member_size: int,
    limits: ArchiveLimits,
) -> int:
    if member_size < 0:
        raise ValueError("archive member size cannot be negative")
    if member_size > limits.max_member_bytes:
        raise ValueError("archive member exceeds the per-member byte limit")
    if member_size > limits.max_total_bytes - current:
        raise ValueError("archive exceeds the total expanded-byte limit")
    return current + member_size


def _validate_expansion_ratio(
    expanded_bytes: int,
    compressed_bytes: int,
    limits: ArchiveLimits,
) -> None:
    if expanded_bytes == 0:
        return
    rounded_up_ratio = ((expanded_bytes - 1) // compressed_bytes) + 1
    if rounded_up_ratio > limits.max_expansion_ratio:
        raise ValueError("archive exceeds the expansion ratio limit")


def _validate_unique_member(
    path: PurePosixPath,
    is_directory: bool,
    seen: set[PurePosixPath],
    members: list[_ArchiveMember],
) -> None:
    if path in seen:
        raise ValueError(f"duplicate normalized archive member path: {path}")
    for parent in path.parents:
        if parent == PurePosixPath("."):
            continue
        for member in members:
            if member.relative_path == parent and not member.is_directory:
                raise ValueError(
                    f"archive file member is an ancestor of {path}"
                )
    if not is_directory:
        for member in members:
            if path in member.relative_path.parents:
                raise ValueError(
                    f"archive file member is an ancestor of {member.relative_path}"
                )


def _extract_zip_members(
    archive: zipfile.ZipFile,
    members: tuple[_ArchiveMember, ...],
    root: Path,
    limits: ArchiveLimits,
) -> None:
    for member in members:
        target = root.joinpath(*member.relative_path.parts)
        if member.is_directory:
            target.mkdir(mode=0o700, parents=True, exist_ok=True)
            continue
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        source_info = member.source
        assert isinstance(source_info, zipfile.ZipInfo)
        with archive.open(source_info) as source:
            _write_extracted_file(source, target, member.size, limits)


def _extract_tar_members(
    archive: tarfile.TarFile,
    members: tuple[_ArchiveMember, ...],
    root: Path,
    limits: ArchiveLimits,
) -> None:
    for member in members:
        target = root.joinpath(*member.relative_path.parts)
        if member.is_directory:
            target.mkdir(mode=0o700, parents=True, exist_ok=True)
            continue
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        source_info = member.source
        assert isinstance(source_info, tarfile.TarInfo)
        source = archive.extractfile(source_info)
        if source is None:
            raise ValueError(
                f"regular archive member {member.name!r} has no data"
            )
        with source:
            _write_extracted_file(source, target, member.size, limits)


def _write_extracted_file(
    source: BinaryIO,
    target: Path,
    expected_size: int,
    limits: ArchiveLimits,
) -> None:
    written = 0
    with _private_binary_output(target) as output:
        while chunk := source.read(EXTRACTION_CHUNK_BYTES):
            if len(chunk) > expected_size - written:
                raise ValueError(
                    "archive member expanded beyond declared size"
                )
            if len(chunk) > limits.max_member_bytes - written:
                raise ValueError(
                    "archive member crossed the per-member byte limit"
                )
            output.write(chunk)
            written += len(chunk)
        if written != expected_size:
            raise ValueError(
                "archive member expanded size differs from metadata"
            )
        output.flush()
        os.fsync(output.fileno())


def _asset_receipt_bytes(asset: RemoteArchive) -> bytes:
    document = {
        "archive_format": asset.archive_format,
        "limits": {
            "max_compressed_bytes": asset.limits.max_compressed_bytes,
            "max_expansion_ratio": asset.limits.max_expansion_ratio,
            "max_member_bytes": asset.limits.max_member_bytes,
            "max_members": asset.limits.max_members,
            "max_total_bytes": asset.limits.max_total_bytes,
        },
        "sha256": asset.sha256,
        "size_bytes": asset.size_bytes,
        "url": asset.url,
    }
    return (
        json.dumps(
            document, allow_nan=False, separators=(",", ":"), sort_keys=True
        )
        + "\n"
    ).encode("utf-8")


def _write_asset_receipt(asset: RemoteArchive, root: Path) -> None:
    receipt_path = root / _ASSET_RECEIPT
    with _private_binary_output(receipt_path) as stream:
        stream.write(_asset_receipt_bytes(asset))
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_tree(root: Path) -> None:
    for directory, _, _ in os.walk(root, topdown=False):
        _fsync_directory(Path(directory))


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "CONNECT_TIMEOUT_SECONDS",
    "MAX_REDIRECTS",
    "READ_TIMEOUT_SECONDS",
    "ArchiveLimits",
    "RemoteArchive",
    "acquire_verified_archive",
]
