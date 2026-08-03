"""Trusted local checkpoint storage and safe selected-weight loading."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO

import torch
from lightning.pytorch.plugins.io import TorchCheckpointIO

from topobench.utils.trusted_io import (
    atomic_replace,
    digest_open_file,
    open_trusted_file,
    trusted_relative_path,
    validate_directory_chain,
)

_CHECKPOINT_MANIFEST_SCHEMA = "topobench.checkpoint-manifest"
_SCHEMA_VERSION = 1

_FileIdentity = tuple[int, int, int, int, int]


def _file_identity(status: os.stat_result) -> _FileIdentity:
    """Return one mutation-sensitive identity for an opened checkpoint file."""
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _same_open_file(
    before: _FileIdentity,
    after: os.stat_result,
) -> bool:
    """Return whether descriptor bytes stayed on the same file version."""
    observed = _file_identity(after)
    return observed[:4] == before[:4]


def _close_descriptors(descriptors: Mapping[str, int]) -> None:
    for descriptor in descriptors.values():
        try:
            os.close(descriptor)
        except OSError:
            pass


@dataclass(slots=True)
class LoadedSelectedCheckpoint:
    """Selected state and open identities of the exact loaded artifacts."""

    checkpoint: Mapping[str, object]
    checkpoint_id: str
    artifact_identities: Mapping[str, _FileIdentity]
    _descriptors: dict[str, int]

    def matches_artifact(self, role: str, path: Path) -> bool:
        """Return whether path is the exact filesystem object kept open."""
        descriptor = self._descriptors.get(role)
        if descriptor is None:
            return False
        opened = os.fstat(descriptor)
        candidate = path.stat(follow_symlinks=False)
        return (candidate.st_dev, candidate.st_ino) == (
            opened.st_dev,
            opened.st_ino,
        )

    def close(self) -> None:
        """Release retained descriptors after cleanup no longer needs them."""
        _close_descriptors(self._descriptors)
        self._descriptors.clear()

    def __del__(self) -> None:
        self.close()


def checkpoint_manifest_path(path: str | Path) -> Path:
    """Return the digest-manifest sidecar for a checkpoint."""
    return Path(f"{Path(path)}.manifest.json")


def checkpoint_state_path(path: str | Path) -> Path:
    """Return the closed-schema selected-state companion path."""
    return Path(f"{Path(path)}.state.pt")


def _manifest(
    checkpoint_path: Path,
    *,
    output_root: Path,
) -> dict[str, object]:
    resolved_output = output_root.resolve(strict=True)
    resolved_checkpoint = checkpoint_path.resolve(strict=True)
    with checkpoint_path.open("rb") as checkpoint_file:
        sha256, byte_size = digest_open_file(checkpoint_file)
    return {
        "schema": _CHECKPOINT_MANIFEST_SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "run_root_sha256": hashlib.sha256(
            str(resolved_output).encode("utf-8")
        ).hexdigest(),
        "relative_path": resolved_checkpoint.relative_to(
            resolved_output
        ).as_posix(),
        "sha256": sha256,
        "byte_size": byte_size,
    }


def _write_manifest(checkpoint_path: Path, *, output_root: Path) -> None:
    manifest = _manifest(checkpoint_path, output_root=output_root)
    content = json.dumps(
        manifest,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    atomic_replace(
        checkpoint_manifest_path(checkpoint_path),
        lambda file: file.write(content),
    )


@contextmanager
def _validated_trusted_resume(
    path: str | Path,
    *,
    output_root: str | Path,
    checkpoint_root: str | Path,
) -> Iterator[
    tuple[
        Path,
        BinaryIO,
        str,
        Mapping[str, _FileIdentity],
    ]
]:
    """Yield the validated path and the exact descriptor whose bytes were hashed."""
    checkpoint_path = Path(path)
    output_path = Path(output_root)
    checkpoint_root_path = Path(checkpoint_root)
    resolved_checkpoint, checkpoint_relative = trusted_relative_path(
        checkpoint_path, checkpoint_root_path, boundary="checkpoint"
    )
    resolved_output = output_path.resolve(strict=True)
    try:
        output_relative = resolved_checkpoint.relative_to(resolved_output)
    except ValueError as error:
        raise PermissionError(
            "checkpoint root must be inside the current run root"
        ) from error

    manifest_file = open_trusted_file(
        checkpoint_manifest_path(checkpoint_path),
        checkpoint_root_path,
        boundary="checkpoint",
        label="checkpoint manifest",
        require_private=True,
    )
    with manifest_file:
        manifest_identity = _file_identity(os.fstat(manifest_file.fileno()))
        try:
            manifest = json.load(manifest_file)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "checkpoint manifest is not valid JSON"
            ) from error
        if not _same_open_file(
            manifest_identity,
            os.fstat(manifest_file.fileno()),
        ):
            raise RuntimeError("checkpoint manifest changed during validation")
    expected_keys = {
        "schema",
        "schema_version",
        "run_root_sha256",
        "relative_path",
        "sha256",
        "byte_size",
    }
    if type(manifest) is not dict or set(manifest) != expected_keys:
        raise ValueError("checkpoint manifest has an invalid schema")
    if (
        manifest["schema"] != _CHECKPOINT_MANIFEST_SCHEMA
        or manifest["schema_version"] != _SCHEMA_VERSION
    ):
        raise ValueError("checkpoint manifest schema or version is invalid")
    expected_root_id = hashlib.sha256(
        str(resolved_output).encode("utf-8")
    ).hexdigest()
    if manifest["run_root_sha256"] != expected_root_id:
        raise ValueError("checkpoint manifest belongs to a different run root")
    if manifest["relative_path"] != output_relative.as_posix():
        raise ValueError(
            "checkpoint manifest path does not match the requested checkpoint"
        )
    if type(manifest["sha256"]) is not str or len(manifest["sha256"]) != 64:
        raise ValueError("checkpoint manifest SHA-256 is invalid")
    if type(manifest["byte_size"]) is not int or manifest["byte_size"] < 0:
        raise ValueError("checkpoint manifest byte size is invalid")

    checkpoint_file = open_trusted_file(
        checkpoint_path,
        checkpoint_root_path,
        boundary="checkpoint",
        label="checkpoint",
        require_private=True,
    )
    with checkpoint_file:
        checkpoint_identity = _file_identity(
            os.fstat(checkpoint_file.fileno())
        )
        sha256, byte_size = digest_open_file(checkpoint_file)
        if not _same_open_file(
            checkpoint_identity,
            os.fstat(checkpoint_file.fileno()),
        ):
            raise RuntimeError("checkpoint changed during digest validation")
        if sha256 != manifest["sha256"] or byte_size != manifest["byte_size"]:
            raise ValueError(
                "checkpoint digest or size does not match its manifest"
            )
        if (
            checkpoint_relative.as_posix()
            != resolved_checkpoint.relative_to(
                checkpoint_root_path.resolve(strict=True)
            ).as_posix()
        ):
            raise RuntimeError("checkpoint identity changed during validation")
        yield (
            resolved_checkpoint,
            checkpoint_file,
            sha256,
            MappingProxyType(
                {
                    "checkpoint": checkpoint_identity,
                    "manifest": manifest_identity,
                }
            ),
        )


def validate_trusted_resume(
    path: str | Path,
    *,
    output_root: str | Path,
    checkpoint_root: str | Path,
) -> Path:
    """Validate one exact private same-run checkpoint before full resume."""
    with _validated_trusted_resume(
        path,
        output_root=output_root,
        checkpoint_root=checkpoint_root,
    ) as (resolved_checkpoint, _, _, _):
        return resolved_checkpoint


def load_selected_checkpoint(
    path: str | Path,
    *,
    output_root: str | Path,
    checkpoint_root: str | Path,
) -> LoadedSelectedCheckpoint:
    """Safely load selected state and retain exact cleanup identities."""
    checkpoint_path = Path(path)
    root_path = Path(checkpoint_root)
    manifest_path = checkpoint_manifest_path(checkpoint_path)
    state_path = checkpoint_state_path(checkpoint_path)
    has_manifest = manifest_path.is_file()
    has_state = state_path.is_file()
    if has_manifest != has_state:
        raise ValueError(
            "selected checkpoint manifest and state must both be present"
        )

    if not has_manifest:
        checkpoint_file = open_trusted_file(
            checkpoint_path,
            root_path,
            boundary="checkpoint",
            label="selected checkpoint",
            require_private=False,
        )
        with checkpoint_file:
            checkpoint_identity = _file_identity(
                os.fstat(checkpoint_file.fileno())
            )
            checkpoint_id, _ = digest_open_file(checkpoint_file)
            checkpoint = torch.load(
                checkpoint_file,
                map_location="cpu",
                weights_only=True,
            )
            if not _same_open_file(
                checkpoint_identity,
                os.fstat(checkpoint_file.fileno()),
            ):
                raise RuntimeError(
                    "selected checkpoint changed during deserialization"
                )
            retained_descriptor = os.dup(checkpoint_file.fileno())
        if not isinstance(checkpoint, Mapping):
            os.close(retained_descriptor)
            raise TypeError("selected checkpoint must contain a mapping")
        return LoadedSelectedCheckpoint(
            checkpoint=checkpoint,
            checkpoint_id=checkpoint_id,
            artifact_identities=MappingProxyType(
                {"checkpoint": checkpoint_identity}
            ),
            _descriptors={"checkpoint": retained_descriptor},
        )

    retained_descriptors: dict[str, int] = {}
    try:
        with _validated_trusted_resume(
            checkpoint_path,
            output_root=output_root,
            checkpoint_root=root_path,
        ) as (
            _,
            checkpoint_file,
            checkpoint_id,
            artifact_identities,
        ):
            retained_descriptors["checkpoint"] = os.dup(
                checkpoint_file.fileno()
            )
            state_file = open_trusted_file(
                state_path,
                root_path,
                boundary="checkpoint",
                label="selected checkpoint state",
                require_private=True,
            )
            with state_file:
                state_identity = _file_identity(os.fstat(state_file.fileno()))
                selected_state = torch.load(
                    state_file,
                    map_location="cpu",
                    weights_only=True,
                )
                if not _same_open_file(
                    state_identity,
                    os.fstat(state_file.fileno()),
                ):
                    raise RuntimeError(
                        "selected checkpoint state changed during "
                        "deserialization"
                    )
                retained_descriptors["state"] = os.dup(state_file.fileno())

            manifest_file = open_trusted_file(
                manifest_path,
                root_path,
                boundary="checkpoint",
                label="checkpoint manifest",
                require_private=True,
            )
            with manifest_file:
                observed_manifest_identity = _file_identity(
                    os.fstat(manifest_file.fileno())
                )
                if (
                    observed_manifest_identity
                    != artifact_identities["manifest"]
                ):
                    raise RuntimeError(
                        "checkpoint manifest changed after validation"
                    )
                retained_descriptors["manifest"] = os.dup(
                    manifest_file.fileno()
                )

        expected_keys = {
            "schema",
            "schema_version",
            "checkpoint_sha256",
            "state_dict",
            "epoch",
            "global_step",
        }
        if (
            type(selected_state) is not dict
            or set(selected_state) != expected_keys
        ):
            raise ValueError("selected checkpoint state has an invalid schema")
        if (
            selected_state["schema"] != "topobench.selected-checkpoint-state"
            or selected_state["schema_version"] != _SCHEMA_VERSION
        ):
            raise ValueError(
                "selected checkpoint state schema or version is invalid"
            )
        if selected_state["checkpoint_sha256"] != checkpoint_id:
            raise ValueError(
                "selected checkpoint state does not match checkpoint bytes"
            )
        state_dict = selected_state["state_dict"]
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise ValueError(
                "selected checkpoint state_dict must be non-empty"
            )
        return LoadedSelectedCheckpoint(
            checkpoint=selected_state,
            checkpoint_id=checkpoint_id,
            artifact_identities=MappingProxyType(
                {
                    **artifact_identities,
                    "state": state_identity,
                }
            ),
            _descriptors=retained_descriptors,
        )
    except BaseException:
        _close_descriptors(retained_descriptors)
        raise


class TrustedCheckpointIO(TorchCheckpointIO):
    """Atomic local CheckpointIO with a private same-run digest manifest."""

    def __init__(
        self, *, output_root: str | Path, checkpoint_root: str | Path
    ) -> None:
        super().__init__()
        self.output_root = Path(output_root)
        self.checkpoint_root = Path(checkpoint_root)

    def save_checkpoint(
        self,
        checkpoint: dict[str, Any],
        path: str | Path,
        storage_options: Any | None = None,
    ) -> None:
        if storage_options is not None:
            raise TypeError(
                "TrustedCheckpointIO does not support storage_options"
            )
        checkpoint_path = Path(path)
        trusted_relative_path(
            checkpoint_path, self.checkpoint_root, boundary="checkpoint"
        )
        self.output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        resolved_output = self.output_root.resolve(strict=True)
        resolved_checkpoint_root = self.checkpoint_root.resolve(strict=True)
        resolved_checkpoint_root.relative_to(resolved_output)
        validate_directory_chain(
            checkpoint_path.parent.resolve(),
            resolved_checkpoint_root,
            boundary="checkpoint",
        )
        atomic_replace(
            checkpoint_path, lambda file: torch.save(checkpoint, file)
        )
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise ValueError("checkpoint must contain a non-empty state_dict")
        selected_state = {
            "schema": "topobench.selected-checkpoint-state",
            "schema_version": _SCHEMA_VERSION,
            "checkpoint_sha256": _manifest(
                checkpoint_path,
                output_root=self.output_root,
            )["sha256"],
            "state_dict": state_dict,
            "epoch": checkpoint.get("epoch", 0),
            "global_step": checkpoint.get("global_step", 0),
        }
        atomic_replace(
            checkpoint_state_path(checkpoint_path),
            lambda file: torch.save(selected_state, file),
        )
        _write_manifest(checkpoint_path, output_root=self.output_root)

    def load_checkpoint(
        self,
        path: str | Path,
        map_location: Callable | None = lambda storage, loc: storage,
    ) -> dict[str, Any]:
        with _validated_trusted_resume(
            path,
            output_root=self.output_root,
            checkpoint_root=self.checkpoint_root,
        ) as (_, checkpoint_file, _, _):
            checkpoint = torch.load(
                checkpoint_file,
                map_location=map_location,
                weights_only=False,
            )
        if not isinstance(checkpoint, dict):
            raise TypeError(
                "trusted checkpoint must deserialize to a dictionary"
            )
        return checkpoint

    def remove_checkpoint(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        trusted_relative_path(
            checkpoint_path, self.checkpoint_root, boundary="checkpoint"
        )
        checkpoint_path.unlink(missing_ok=True)
        checkpoint_manifest_path(checkpoint_path).unlink(missing_ok=True)
        checkpoint_state_path(checkpoint_path).unlink(missing_ok=True)
