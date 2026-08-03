"""Closed-schema, digest-pinned storage for trusted local PyG caches."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.collate import collate

from topobench.data.hypergraph import HypergraphData
from topobench.utils.trusted_io import (
    atomic_replace_bytes,
    atomic_replace_with_digest,
    digest_open_file,
    open_trusted_file,
    trusted_relative_path,
    validate_directory_chain,
)

_CACHE_SCHEMA = "topobench.pyg-cache"
_CACHE_MANIFEST_SCHEMA = "topobench.pyg-cache-manifest"
_SCHEMA_VERSION = 1
_FAMILY_CLASSES = {
    "data": Data,
    "heterodata": HeteroData,
    "hypergraph": HypergraphData,
}


def cache_manifest_path(path: str | Path) -> Path:
    """Return the sidecar manifest path for one cache payload."""
    payload_path = Path(path)
    return payload_path.with_name(f"{payload_path.name}.manifest.json")


def _validate_closed_value(value: object, *, path: str = "payload") -> None:
    if (
        type(value) in {type(None), bool, int, float, str}
        or type(value) is torch.Tensor
    ):
        return
    if type(value) in {list, tuple}:
        for index, item in enumerate(value):
            _validate_closed_value(item, path=f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            _validate_closed_value(key, path=f"{path}.key")
            _validate_closed_value(item, path=f"{path}[{key!r}]")
        return
    raise TypeError(
        f"cache {path} contains unsupported type {type(value).__name__}"
    )


def _require_exact_mapping(
    value: object,
    *,
    keys: set[str],
    label: str,
) -> Mapping[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be a built-in dict")
    if set(value) != keys:
        raise ValueError(f"{label} has an invalid schema")
    return value


def _validate_family(
    family: str, data_cls: type[Data] | type[HeteroData]
) -> None:
    expected_cls = _FAMILY_CLASSES.get(family)
    if expected_cls is None:
        raise ValueError(f"unsupported PyG cache family {family!r}")
    if data_cls is not expected_cls:
        raise TypeError(
            f"cache family {family!r} requires static {expected_cls.__name__} reconstruction"
        )


def write_pyg_cache(
    data_list: Sequence[Data | HeteroData],
    path: str | Path,
    *,
    trusted_root: str | Path,
    family: str,
    cache_identity: str,
) -> None:
    """Atomically write one tensor/primitive-only PyG cache and its manifest."""
    payload_path = Path(path)
    root_path = Path(trusted_root)
    trusted_relative_path(payload_path, root_path, boundary="cache")
    expected_cls = _FAMILY_CLASSES.get(family)
    if expected_cls is None:
        raise ValueError(f"unsupported PyG cache family {family!r}")
    if type(cache_identity) is not str or not cache_identity:
        raise ValueError("cache_identity must be a non-empty string")
    if not data_list:
        raise ValueError("cannot write an empty PyG cache")
    if any(type(item) is not expected_cls for item in data_list):
        raise TypeError(
            f"cache family {family!r} received the wrong PyG data class"
        )

    root_path.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved_root = root_path.resolve(strict=True)
    resolved_payload, relative_path = trusted_relative_path(
        payload_path, root_path, boundary="cache"
    )
    validate_directory_chain(
        resolved_payload.parent, resolved_root, boundary="cache"
    )

    if len(data_list) == 1:
        data_record = data_list[0].to_dict()
        slices: dict = {}
    else:
        collated, slices, _ = collate(
            expected_cls,
            list(data_list),
            increment=False,
            add_batch=False,
        )
        data_record = collated.to_dict()
    payload = {
        "schema": _CACHE_SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "family": family,
        "cache_identity": cache_identity,
        "data": data_record,
        "slices": slices,
    }
    _validate_closed_value(payload)

    sha256, byte_size = atomic_replace_with_digest(
        payload_path,
        lambda file: torch.save(payload, file),
    )
    manifest = {
        "schema": _CACHE_MANIFEST_SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "family": family,
        "cache_identity": cache_identity,
        "payload": {
            "relative_path": relative_path.as_posix(),
            "sha256": sha256,
            "byte_size": byte_size,
        },
    }
    manifest_content = json.dumps(
        manifest,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    atomic_replace_bytes(cache_manifest_path(payload_path), manifest_content)


def load_pyg_cache(
    path: str | Path,
    *,
    trusted_root: str | Path,
    family: str,
    cache_identity: str,
    data_cls: type[Data] | type[HeteroData],
) -> tuple[Data | HeteroData, dict]:
    """Verify and safely reconstruct one tensor/primitive-only PyG cache."""
    payload_path = Path(path)
    root_path = Path(trusted_root)
    _, relative_path = trusted_relative_path(
        payload_path, root_path, boundary="cache"
    )
    _validate_family(family, data_cls)
    if type(cache_identity) is not str or not cache_identity:
        raise ValueError("cache_identity must be a non-empty string")

    manifest_file = open_trusted_file(
        cache_manifest_path(payload_path),
        root_path,
        boundary="cache",
        label="cache manifest",
    )
    with manifest_file:
        try:
            manifest_value = json.load(manifest_file)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("cache manifest is not valid JSON") from error
    manifest = _require_exact_mapping(
        manifest_value,
        keys={
            "schema",
            "schema_version",
            "family",
            "cache_identity",
            "payload",
        },
        label="cache manifest",
    )
    if (
        manifest["schema"] != _CACHE_MANIFEST_SCHEMA
        or manifest["schema_version"] != _SCHEMA_VERSION
    ):
        raise ValueError("cache manifest schema or version is invalid")
    if (
        manifest["family"] != family
        or manifest["cache_identity"] != cache_identity
    ):
        raise ValueError("cache manifest family or identity is stale")
    descriptor = _require_exact_mapping(
        manifest["payload"],
        keys={"relative_path", "sha256", "byte_size"},
        label="cache payload descriptor",
    )
    if descriptor["relative_path"] != relative_path.as_posix():
        raise ValueError(
            "cache manifest payload path does not match the requested cache"
        )
    if (
        type(descriptor["sha256"]) is not str
        or len(descriptor["sha256"]) != 64
    ):
        raise ValueError("cache payload SHA-256 is invalid")
    if type(descriptor["byte_size"]) is not int or descriptor["byte_size"] < 0:
        raise ValueError("cache payload byte size is invalid")

    payload_file = open_trusted_file(
        payload_path, root_path, boundary="cache", label="cache payload"
    )
    with payload_file:
        sha256, byte_size = digest_open_file(payload_file)
        if (
            sha256 != descriptor["sha256"]
            or byte_size != descriptor["byte_size"]
        ):
            raise ValueError(
                "cache payload digest or size does not match its manifest"
            )
        try:
            payload_value = torch.load(
                payload_file, map_location="cpu", weights_only=True
            )
        except Exception as error:
            raise ValueError(
                "cache payload failed safe weights-only loading"
            ) from error

    _validate_closed_value(payload_value)
    payload = _require_exact_mapping(
        payload_value,
        keys={
            "schema",
            "schema_version",
            "family",
            "cache_identity",
            "data",
            "slices",
        },
        label="cache payload",
    )
    if (
        payload["schema"] != _CACHE_SCHEMA
        or payload["schema_version"] != _SCHEMA_VERSION
    ):
        raise ValueError("cache payload schema or version is invalid")
    if payload["family"] != family:
        raise ValueError("cache payload family does not match its manifest")
    if payload["cache_identity"] != cache_identity:
        raise ValueError("cache payload identity does not match its manifest")
    if (
        type(payload["data"]) is not dict
        or type(payload["slices"]) is not dict
    ):
        raise TypeError("cache data and slices must be built-in dictionaries")
    return data_cls.from_dict(payload["data"]), payload["slices"]
