"""Shared filesystem primitives for trusted local artifact boundaries."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO


def trusted_relative_path(
    path: Path,
    trusted_root: Path,
    *,
    boundary: str,
) -> tuple[Path, Path]:
    """Resolve a path and require it to be below a non-symlinked trusted root."""
    if trusted_root.is_symlink():
        raise PermissionError(f"trusted {boundary} root must not be a symlink")
    root = trusted_root.resolve(strict=False)
    resolved = path.resolve(strict=False)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise PermissionError(
            f"{boundary} path is outside the trusted root"
        ) from error
    if not relative.parts:
        raise ValueError(f"{boundary} path must be below the trusted root")
    return resolved, relative


def validate_directory_chain(
    path: Path,
    root: Path,
    *,
    boundary: str,
) -> None:
    """Require each existing directory from root through path to be private."""
    relative = path.relative_to(root)
    directories = (
        root,
        *(root / parent for parent in relative.parents if parent.parts),
        path,
    )
    for directory in directories:
        directory_stat = directory.stat(follow_symlinks=False)
        if not stat.S_ISDIR(directory_stat.st_mode) or directory.is_symlink():
            raise PermissionError(
                f"trusted {boundary} path contains a symlink or non-directory"
            )
        if directory_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise PermissionError(
                f"trusted {boundary} directories must not be group/world writable"
            )
        if hasattr(os, "geteuid") and directory_stat.st_uid != os.geteuid():
            raise PermissionError(
                f"trusted {boundary} directory owner UID mismatch"
            )


def open_trusted_file(
    path: Path,
    trusted_root: Path,
    *,
    boundary: str,
    label: str,
    require_private: bool = False,
) -> BinaryIO:
    """Open one regular artifact without following links or accepting path races."""
    resolved, _ = trusted_relative_path(path, trusted_root, boundary=boundary)
    root = trusted_root.resolve(strict=True)
    validate_directory_chain(resolved.parent, root, boundary=boundary)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise PermissionError(f"unable to securely open {label}") from error
    try:
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise PermissionError(f"{label} must be a regular file")
        forbidden_mode = (
            0o077 if require_private else (stat.S_IWGRP | stat.S_IWOTH)
        )
        if descriptor_stat.st_mode & forbidden_mode:
            expectation = (
                "private" if require_private else "not group/world writable"
            )
            raise PermissionError(f"{label} mode must be {expectation}")
        if hasattr(os, "geteuid") and descriptor_stat.st_uid != os.geteuid():
            raise PermissionError(
                f"{label} owner UID does not match the effective UID"
            )
        if descriptor_stat.st_nlink != 1:
            raise PermissionError(
                f"{label} must have exactly one filesystem link"
            )
        path_stat = path.stat(follow_symlinks=False)
        if (path_stat.st_dev, path_stat.st_ino) != (
            descriptor_stat.st_dev,
            descriptor_stat.st_ino,
        ):
            raise PermissionError(f"{label} changed while it was opened")
        return os.fdopen(descriptor, "rb", closefd=True)
    except BaseException:
        os.close(descriptor)
        raise


def digest_open_file(file: BinaryIO) -> tuple[str, int]:
    """Hash a binary file, return its size, and rewind it."""
    digest = hashlib.sha256()
    size = 0
    while chunk := file.read(1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
    file.seek(0)
    return digest.hexdigest(), size


def atomic_replace(path: Path, write: Callable[[BinaryIO], object]) -> None:
    """Write a private sibling temporary file and durably replace the target."""
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as file:
            write(file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_replace_with_digest(
    path: Path,
    write: Callable[[BinaryIO], object],
) -> tuple[str, int]:
    """Durably replace a target and hash the exact temporary file published."""
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w+b", closefd=True) as file:
            write(file)
            file.flush()
            os.fsync(file.fileno())
            file.seek(0)
            sha256, byte_size = digest_open_file(file)
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return sha256, byte_size
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_replace_bytes(path: Path, content: bytes) -> None:
    """Durably replace a target with fixed bytes."""
    atomic_replace(path, lambda file: file.write(content))
