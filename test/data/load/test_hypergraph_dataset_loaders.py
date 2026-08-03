"""Offline tests for exact, non-executable hypergraph ingestion."""

from __future__ import annotations

import hashlib
import json
import pickle
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from topobench.data import (
    HYPERGRAPH_REPRESENTATION_VERSION,
    HypergraphData,
    validate_hypergraph_structure,
)
from topobench.data.datasets import CitationHypergraphDataset, HypergraphDataset
from topobench.data.utils.cache_io import cache_manifest_path
from topobench.data.utils.hypergraph_io import (
    SAFE_HYPERGRAPH_CONVERTER_VERSION,
    SAFE_HYPERGRAPH_FORMAT,
    SAFE_HYPERGRAPH_FORMAT_VERSION,
    _SAFE_METADATA_BYTE_LIMIT,
    ContentRoleSpec,
    load_hypergraph_content_dataset,
    load_hypergraph_npz_dataset,
    validate_hypergraph_npz_assets,
)


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_safe_fixture(
    raw_dir: Path,
    name: str,
    *,
    features: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    incidence: np.ndarray | None = None,
    sparse_shape: tuple[int, int] | None = None,
    compressed: bool = False,
) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray([0, 2, 1], dtype=np.int64) if labels is None else np.asarray(labels)
    incidence = (
        np.asarray([[1, 1, 2], [1, 1, 0]], dtype=np.int64)
        if incidence is None
        else np.asarray(incidence)
    )
    if sparse_shape is None:
        features = (
            np.asarray([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            if features is None
            else np.asarray(features)
        )
        arrays = {"features": features, "incidence": incidence, "labels": labels}
        feature_storage = "dense"
        num_nodes = int(features.shape[0])
    else:
        arrays = {
            "feature_indices": np.asarray([[1, 2], [7, 999_999]], dtype=np.int64),
            "feature_shape": np.asarray(sparse_shape, dtype=np.int64),
            "feature_values": np.asarray([2.0, 4.0], dtype=np.float32),
            "incidence": incidence,
            "labels": labels,
        }
        feature_storage = "coo"
        num_nodes = sparse_shape[0]

    npz_path = raw_dir / f"{name}.npz"
    save = np.savez_compressed if compressed else np.savez
    save(npz_path, **arrays)
    metadata = {
        "arrays": {
            key: {"dtype": value.dtype.str, "shape": list(value.shape)}
            for key, value in arrays.items()
        },
        "converter_version": SAFE_HYPERGRAPH_CONVERTER_VERSION,
        "feature_storage": feature_storage,
        "format": SAFE_HYPERGRAPH_FORMAT,
        "format_version": SAFE_HYPERGRAPH_FORMAT_VERSION,
        "incidence_roles": ["node", "hyperedge"],
        "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "num_hyperedges": 2,
        "num_nodes": num_nodes,
        "padding_count": 0,
        "padding_sentinel": None,
        "raw_sha256": "a" * 64,
    }
    (raw_dir / f"{name}.json").write_bytes(_canonical_json(metadata))
    return metadata


def _rewrite_metadata(raw_dir: Path, name: str, metadata: dict[str, Any]) -> None:
    (raw_dir / f"{name}.json").write_bytes(_canonical_json(metadata))


def _refresh_npz_digest(
    raw_dir: Path,
    name: str,
    metadata: dict[str, Any],
) -> None:
    metadata["npz_sha256"] = hashlib.sha256(
        (raw_dir / f"{name}.npz").read_bytes()
    ).hexdigest()
    _rewrite_metadata(raw_dir, name, metadata)


def _write_content_fixture(
    raw_dir: Path,
    name: str,
    *,
    first_label: str = "0",
    ambiguous_roles: bool = False,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{name}.content").write_text(
        "\n".join(
            [
                f"node-zero 0.0 0.0 {first_label}",
                "10 1.0 1.5 3",
                "node-z 2.0 2.5 5",
                "700 0.0 0.0 0",
                "edge-a 0.0 0.0 0",
                "padding 0.0 0.0 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    target = "10" if ambiguous_roles else "edge-a"
    (raw_dir / f"{name}.edges").write_text(
        f"10 {target}\n10 {target}\nnode-z 700\n",
        encoding="utf-8",
    )


def _content_roles() -> ContentRoleSpec:
    return ContentRoleSpec(num_node_rows=3, num_padding_rows=1, padding_sentinel="0")


def _assert_native(data: HypergraphData) -> None:
    assert validate_hypergraph_structure(data) is data
    assert data.representation_version == HYPERGRAPH_REPRESENTATION_VERSION
    assert type(data["representation_version"]) is int
    assert data.hyperedge_index.dtype == torch.long
    assert data.hyperedge_index.shape[0] == 2
    assert torch.equal(
        torch.unique(data.hyperedge_index[1]), torch.arange(data.num_hyperedges)
    )
    for legacy_field in ("x_0", "edge_index", "incidence_hyperedges", "n_x"):
        assert legacy_field not in data


@pytest.mark.parametrize(
    "bad_label",
    ["1.5", "nan", "inf", "-inf", "9223372036854775808"],
)
def test_content_parser_rejects_nonintegral_or_nonfinite_labels_before_long(
    tmp_path: Path,
    bad_label: str,
) -> None:
    name = "bad-content-label"
    _write_content_fixture(tmp_path, name, first_label=bad_label)
    with pytest.raises(ValueError, match="labels.*finite integer"):
        load_hypergraph_content_dataset(tmp_path, name, role_spec=_content_roles())


def test_content_parser_preserves_integer_labels_above_float_precision(
    tmp_path: Path,
) -> None:
    name = "exact-content-label"
    _write_content_fixture(
        tmp_path,
        name,
        first_label="9007199254740993",
    )

    data, _ = load_hypergraph_content_dataset(
        tmp_path,
        name,
        role_spec=_content_roles(),
    )

    assert torch.equal(data.y, torch.tensor([2, 0, 1], dtype=torch.long))


def test_content_parser_maps_int64_boundary_labels_without_overflow(
    tmp_path: Path,
) -> None:
    name = "boundary-content-labels"
    _write_content_fixture(tmp_path, name)
    (tmp_path / f"{name}.content").write_text(
        "\n".join(
            [
                f"node-zero 0.0 0.0 {2**63 - 1}",
                f"10 1.0 1.5 {-2**63}",
                "node-z 2.0 2.5 0",
                "700 0.0 0.0 0",
                "edge-a 0.0 0.0 0",
                "padding 0.0 0.0 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    data, _ = load_hypergraph_content_dataset(
        tmp_path,
        name,
        role_spec=_content_roles(),
    )

    assert torch.equal(data.y, torch.tensor([2, 0, 1], dtype=torch.long))


def test_content_parser_uses_declared_roles_without_zero_filtering(tmp_path: Path) -> None:
    name = "content-roles"
    _write_content_fixture(tmp_path, name)
    data, resolved_dir = load_hypergraph_content_dataset(
        tmp_path, name, role_spec=_content_roles()
    )
    _assert_native(data)
    assert resolved_dir == str(tmp_path)
    assert data.num_nodes == 3
    assert data.num_hyperedges == 2
    assert torch.equal(
        data.x, torch.tensor([[0.0, 0.0], [1.0, 1.5], [2.0, 2.5]])
    )
    assert torch.equal(data.y, torch.tensor([0, 1, 2]))
    assert torch.equal(
        data.hyperedge_index,
        torch.tensor([[1, 1, 2], [1, 1, 0]], dtype=torch.long),
    )
    assert 0 not in data.hyperedge_index[0]
    assert torch.equal(data.hyperedge_index[:, 0], data.hyperedge_index[:, 1])


def test_content_parser_rejects_undeclared_padding_rows(tmp_path: Path) -> None:
    name = "undeclared-padding"
    _write_content_fixture(tmp_path, name)
    with pytest.raises(ValueError, match="ambiguous content row roles"):
        load_hypergraph_content_dataset(
            tmp_path, name, role_spec=ContentRoleSpec(num_node_rows=3)
        )


def test_content_parser_rejects_node_hyperedge_role_overlap(tmp_path: Path) -> None:
    name = "ambiguous-roles"
    _write_content_fixture(tmp_path, name, ambiguous_roles=True)
    with pytest.raises(ValueError, match="node and hyperedge roles.*disjoint"):
        load_hypergraph_content_dataset(tmp_path, name, role_spec=_content_roles())


@pytest.mark.parametrize(
    "labels",
    [
        np.asarray([0.0, 1.5, 2.0]),
        np.asarray([0.0, np.nan, 2.0]),
        np.asarray([0.0, np.inf, 2.0]),
        np.asarray([0.0, float(2**63), 2.0]),
    ],
    ids=["fractional", "nan", "inf", "out_of_long_range"],
)
def test_safe_loader_rejects_invalid_labels_before_long(
    tmp_path: Path, labels: np.ndarray
) -> None:
    name = "bad-safe-labels"
    _write_safe_fixture(tmp_path, name, labels=labels)
    with pytest.raises(ValueError, match="labels.*finite integer"):
        load_hypergraph_npz_dataset(tmp_path, name)


def test_safe_loader_preserves_isolated_zero_label_and_duplicate_incidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "safe-hypergraph"
    _write_safe_fixture(tmp_path / name, name)

    def reject_pickle(*args: object, **kwargs: object) -> object:
        raise AssertionError("safe runtime must never execute pickle.load")

    monkeypatch.setattr(pickle, "load", reject_pickle)
    data, resolved_dir = load_hypergraph_npz_dataset(tmp_path, name)
    _assert_native(data)
    assert resolved_dir == str(tmp_path / name)
    assert torch.equal(data.x[0], torch.tensor([0.0, 0.0]))
    assert int(data.y[0]) == 0
    assert 0 not in data.hyperedge_index[0]
    assert torch.equal(
        data.hyperedge_index,
        torch.tensor([[1, 1, 2], [1, 1, 0]], dtype=torch.long),
    )
    validate_hypergraph_npz_assets(tmp_path, name)


def test_safe_loader_accepts_valid_nested_cache(tmp_path: Path) -> None:
    name = "nested-safe-cache"
    nested = tmp_path / name
    _write_safe_fixture(nested, name)

    data, resolved_dir = load_hypergraph_npz_dataset(tmp_path, name)

    _assert_native(data)
    assert resolved_dir == str(nested)


@pytest.mark.parametrize("component", ["root", "nested"])
def test_safe_loader_rejects_symlinked_data_directory_components(
    tmp_path: Path,
    component: str,
) -> None:
    name = "linked-cache"
    outside = tmp_path / "outside"
    _write_safe_fixture(outside / name, name)
    cache_root = tmp_path / "cache"
    if component == "root":
        cache_root.symlink_to(outside, target_is_directory=True)
    else:
        cache_root.mkdir()
        (cache_root / name).symlink_to(
            outside / name,
            target_is_directory=True,
        )

    with pytest.raises(PermissionError, match="symlink"):
        load_hypergraph_npz_dataset(cache_root, name)


@pytest.mark.parametrize("suffix", [".json", ".npz"])
def test_safe_loader_rejects_symlinked_asset_files(
    tmp_path: Path,
    suffix: str,
) -> None:
    name = "linked-asset"
    cache_root = tmp_path / "cache"
    _write_safe_fixture(cache_root, name)
    asset = cache_root / f"{name}{suffix}"
    outside = tmp_path / f"outside{suffix}"
    asset.replace(outside)
    asset.symlink_to(outside)

    with pytest.raises(PermissionError, match="symlink|regular file"):
        load_hypergraph_npz_dataset(cache_root, name)


def test_safe_loader_rejects_out_of_root_dataset_name(tmp_path: Path) -> None:
    name = "escape"
    _write_safe_fixture(tmp_path, name)
    (tmp_path / name).mkdir()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    with pytest.raises(ValueError, match="single path component"):
        load_hypergraph_npz_dataset(cache_root, f"../{name}")


def test_safe_loader_rejects_oversized_metadata_before_json_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "oversized-metadata"
    _write_safe_fixture(tmp_path, name)
    metadata_path = tmp_path / f"{name}.json"
    with metadata_path.open("wb") as stream:
        stream.truncate(_SAFE_METADATA_BYTE_LIMIT + 1)

    def unexpected_decode(*args: object, **kwargs: object) -> object:
        raise AssertionError("oversized metadata reached json.loads")

    monkeypatch.setattr(json, "loads", unexpected_decode)
    with pytest.raises(ValueError, match="metadata.*byte limit"):
        load_hypergraph_npz_dataset(tmp_path, name)

@pytest.mark.parametrize(
    "dataset_class",
    [CitationHypergraphDataset, HypergraphDataset],
    ids=["citation", "general"],
)
def test_processed_cache_second_hit_uses_static_hypergraph_reconstruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_class: type[CitationHypergraphDataset] | type[HypergraphDataset],
) -> None:
    name = dataset_class.__name__.lower()
    raw_dir = tmp_path / name / "raw"
    _write_safe_fixture(raw_dir, name)
    parameters = OmegaConf.create({"data_name": name})
    original_parameters = OmegaConf.to_container(parameters, resolve=True)
    raw_snapshot = {
        path.name: path.read_bytes()
        for path in raw_dir.iterdir()
        if path.is_file()
    }

    first = dataset_class(
        root=str(tmp_path),
        name=name,
        parameters=parameters,
    )
    processed_path = Path(first.processed_paths[0])
    manifest_path = cache_manifest_path(processed_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest) == {
        "schema",
        "schema_version",
        "family",
        "cache_identity",
        "payload",
    }
    assert manifest["schema"] == "topobench.pyg-cache-manifest"
    assert manifest["schema_version"] == 1
    assert manifest["family"] == "hypergraph"
    assert isinstance(manifest["cache_identity"], str)
    assert manifest["cache_identity"]
    assert set(manifest["payload"]) == {
        "relative_path",
        "sha256",
        "byte_size",
    }
    assert Path(manifest["payload"]["relative_path"]).name == processed_path.name
    assert manifest["payload"]["sha256"] == hashlib.sha256(
        processed_path.read_bytes()
    ).hexdigest()
    assert manifest["payload"]["byte_size"] == processed_path.stat().st_size

    def reject_reprocessing(*args: object, **kwargs: object) -> None:
        raise AssertionError("valid processed cache was reprocessed")

    monkeypatch.setattr(dataset_class, "process", reject_reprocessing)
    second = dataset_class(
        root=str(tmp_path),
        name=name,
        parameters=parameters,
    )

    assert type(first._data) is HypergraphData
    assert type(second._data) is HypergraphData
    _assert_native(second[0])
    assert torch.equal(second[0].x, first[0].x)
    assert torch.equal(second[0].hyperedge_index, first[0].hyperedge_index)
    assert OmegaConf.to_container(parameters, resolve=True) == original_parameters
    assert {
        path.name: path.read_bytes()
        for path in raw_dir.iterdir()
        if path.is_file()
    } == raw_snapshot



def test_safe_loader_preserves_high_shape_sparse_coo_features(tmp_path: Path) -> None:
    name = "safe-sparse"
    _write_safe_fixture(tmp_path, name, sparse_shape=(3, 1_000_000))
    data, _ = load_hypergraph_npz_dataset(tmp_path, name)
    assert data.x.layout == torch.sparse_coo
    assert data.x.shape == (3, 1_000_000)
    assert data.x._nnz() == 2
    assert torch.equal(data.x.coalesce().values(), torch.tensor([2.0, 4.0]))


def test_safe_loader_rejects_legacy_pickle_only_assets_without_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "features.pickle").write_bytes(b"hostile")
    called = False

    def observe_pickle(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("pickle.load was executed")

    monkeypatch.setattr(pickle, "load", observe_pickle)
    with pytest.raises(FileNotFoundError, match="safe hypergraph"):
        load_hypergraph_npz_dataset(tmp_path, "legacy")
    assert not called


def test_safe_loader_rejects_compressed_npz_before_numpy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "compressed-bomb"
    _write_safe_fixture(tmp_path, name, compressed=True)

    def unexpected_load(*args: object, **kwargs: object) -> object:
        raise AssertionError("np.load ran before raw ZIP preflight")

    monkeypatch.setattr(np, "load", unexpected_load)
    with pytest.raises(ValueError, match="ZIP_STORED"):
        load_hypergraph_npz_dataset(tmp_path, name)


def test_safe_loader_rejects_extra_zip_members_before_numpy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "extra-member"
    metadata = _write_safe_fixture(tmp_path, name)
    with zipfile.ZipFile(
        tmp_path / f"{name}.npz",
        mode="a",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        archive.writestr("nested/", b"")
    _refresh_npz_digest(tmp_path, name, metadata)

    def unexpected_load(*args: object, **kwargs: object) -> object:
        raise AssertionError("np.load ran before raw ZIP preflight")

    monkeypatch.setattr(np, "load", unexpected_load)
    with pytest.raises(ValueError, match="exact array members"):
        load_hypergraph_npz_dataset(tmp_path, name)


def test_safe_loader_applies_total_expanded_limit_before_numpy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "expanded-limit"
    _write_safe_fixture(tmp_path, name)

    def unexpected_load(*args: object, **kwargs: object) -> object:
        raise AssertionError("np.load ran before raw ZIP preflight")

    monkeypatch.setattr(np, "load", unexpected_load)
    with pytest.raises(ValueError, match="expanded bytes"):
        load_hypergraph_npz_dataset(
            tmp_path,
            name,
            max_total_array_bytes=16,
        )


def test_safe_loader_rejects_object_dtype_npz(tmp_path: Path) -> None:
    name = "object-array"
    metadata = _write_safe_fixture(
        tmp_path,
        name,
        features=np.asarray([[object()]], dtype=object),
        labels=np.asarray([0]),
        incidence=np.asarray([[0], [0]]),
    )
    metadata["num_nodes"] = 1
    metadata["num_hyperedges"] = 1
    _rewrite_metadata(tmp_path, name, metadata)
    with pytest.raises(TypeError, match="object dtype"):
        load_hypergraph_npz_dataset(tmp_path, name)


def test_safe_loader_rejects_unknown_schema_version(tmp_path: Path) -> None:
    name = "unknown-version"
    metadata = _write_safe_fixture(tmp_path, name)
    metadata["format_version"] = SAFE_HYPERGRAPH_FORMAT_VERSION + 1
    _rewrite_metadata(tmp_path, name, metadata)
    with pytest.raises(ValueError, match="format_version"):
        load_hypergraph_npz_dataset(tmp_path, name)


def test_safe_loader_rejects_digest_mismatch(tmp_path: Path) -> None:
    name = "digest-mismatch"
    metadata = _write_safe_fixture(tmp_path, name)
    metadata["npz_sha256"] = "0" * 64
    _rewrite_metadata(tmp_path, name, metadata)
    with pytest.raises(ValueError, match="NPZ SHA-256 mismatch"):
        load_hypergraph_npz_dataset(tmp_path, name)


def test_safe_loader_rejects_noncanonical_or_extra_metadata(tmp_path: Path) -> None:
    name = "extra-schema"
    metadata = _write_safe_fixture(tmp_path, name)
    metadata["unexpected"] = True
    _rewrite_metadata(tmp_path, name, metadata)
    with pytest.raises(ValueError, match="metadata schema"):
        load_hypergraph_npz_dataset(tmp_path, name)


def test_safe_loader_rejects_declared_dtype_or_shape_drift(tmp_path: Path) -> None:
    name = "schema-drift"
    metadata = _write_safe_fixture(tmp_path, name)
    metadata["arrays"]["features"]["shape"] = [3, 3]
    _rewrite_metadata(tmp_path, name, metadata)
    with pytest.raises(ValueError, match="features.*shape"):
        load_hypergraph_npz_dataset(tmp_path, name)


def test_safe_loader_rejects_ambiguous_incidence_roles(tmp_path: Path) -> None:
    name = "wrong-roles"
    metadata = _write_safe_fixture(tmp_path, name)
    metadata["incidence_roles"] = ["hyperedge", "node"]
    _rewrite_metadata(tmp_path, name, metadata)
    with pytest.raises(ValueError, match="incidence_roles"):
        load_hypergraph_npz_dataset(tmp_path, name)


def test_safe_loader_rejects_malformed_incidence(tmp_path: Path) -> None:
    name = "bad-incidence"
    _write_safe_fixture(
        tmp_path,
        name,
        incidence=np.asarray([[1, 2], [0, 1], [0, 0]], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="incidence.*shape"):
        load_hypergraph_npz_dataset(tmp_path, name)
