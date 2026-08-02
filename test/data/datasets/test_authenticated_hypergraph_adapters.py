"""Authentication contracts shared by the retained hypergraph adapters."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, ModuleType

import pytest

import topobench.data.datasets.citation_hypergraph_dataset as citation_module
import topobench.data.datasets.hypergraph_datasets as content_module
from topobench.data.datasets import CitationHypergraphDataset, HypergraphDataset
from topobench.data.utils.downloads import ArchiveLimits, RemoteArchive

ADAPTERS = (
    (citation_module, CitationHypergraphDataset),
    (content_module, HypergraphDataset),
)
DATASET_TYPES = (CitationHypergraphDataset, HypergraphDataset)


def _fixture_asset() -> RemoteArchive:
    return RemoteArchive(
        url="https://assets.example.invalid/fixture.zip",
        sha256="a" * 64,
        size_bytes=100,
        limits=ArchiveLimits(
            max_compressed_bytes=100,
            max_members=2,
            max_member_bytes=100,
            max_total_bytes=200,
            max_expansion_ratio=2,
        ),
    )


@pytest.mark.parametrize(("module", "dataset_type"), ADAPTERS)
def test_remote_adapters_expose_only_an_immutable_exact_asset_manifest(
    module: ModuleType,
    dataset_type: type,
) -> None:
    assert isinstance(dataset_type.ASSETS, MappingProxyType)
    assert dict(dataset_type.ASSETS) == {}
    assert not hasattr(dataset_type, "URLS")
    assert not hasattr(dataset_type, "FILE_FORMAT")
    assert not hasattr(module, "download_file_from_drive")


@pytest.mark.parametrize("dataset_type", DATASET_TYPES)
def test_remote_adapters_require_safe_npz_and_json_raw_names(
    dataset_type: type,
) -> None:
    adapter = object.__new__(dataset_type)
    adapter.name = "fixture"

    assert dataset_type.raw_file_names.fget(adapter) == [
        "fixture.json",
        "fixture.npz",
    ]


@pytest.mark.parametrize("dataset_type", DATASET_TYPES)
def test_remote_adapters_reject_names_without_an_authenticated_asset(
    dataset_type: type,
    tmp_path: Path,
) -> None:
    adapter = object.__new__(dataset_type)
    adapter.name = "unpublished"
    adapter.root = str(tmp_path)

    with pytest.raises(ValueError, match="no authenticated safe archive"):
        dataset_type.download(adapter)

    assert not (tmp_path / "unpublished" / "raw").exists()


@pytest.mark.parametrize(("module", "dataset_type"), ADAPTERS)
def test_remote_adapters_acquire_then_validate_the_exact_manifest_asset(
    module: ModuleType,
    dataset_type: type,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = _fixture_asset()
    monkeypatch.setattr(
        dataset_type,
        "ASSETS",
        MappingProxyType({"fixture": asset}),
    )
    adapter = object.__new__(dataset_type)
    adapter.name = "fixture"
    adapter.root = str(tmp_path)
    validation_calls: list[tuple[Path, str]] = []
    acquisition_calls: list[tuple[RemoteArchive, Path]] = []

    def validate(root: str | Path, name: str) -> None:
        validation_calls.append((Path(root), name))

    def acquire(
        selected_asset: RemoteArchive,
        destination: str | Path,
        callback,
    ) -> Path:
        staging = tmp_path / "private-staging"
        acquisition_calls.append((selected_asset, Path(destination)))
        callback(staging)
        return Path(destination)

    monkeypatch.setattr(module, "validate_hypergraph_npz_assets", validate)
    monkeypatch.setattr(module, "acquire_verified_archive", acquire)

    dataset_type.download(adapter)

    assert acquisition_calls == [
        (asset, tmp_path / "fixture" / "raw"),
    ]
    assert validation_calls == [(tmp_path / "private-staging", "fixture")]


@pytest.mark.parametrize(("module", "dataset_type"), ADAPTERS)
def test_remote_adapters_process_only_the_safe_npz_loader(
    module: ModuleType,
    dataset_type: type,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = object()
    raw_dir = tmp_path / "raw"
    processed_path = tmp_path / "processed" / "hypergraph.npz"
    loader_calls: list[tuple[Path, str]] = []
    validation_calls: list[object] = []
    save_calls: list[tuple[list[object], Path]] = []

    class _Adapter:
        name = "fixture"
        processed_paths = [str(processed_path)]

        @property
        def raw_dir(self) -> str:
            return str(raw_dir)

        def save(self, rows: list[object], path: str) -> None:
            save_calls.append((rows, Path(path)))

    def load(*, data_dir: str | Path, data_name: str):
        loader_calls.append((Path(data_dir), data_name))
        return data, "npz"

    def validate(row: object) -> object:
        validation_calls.append(row)
        return row

    monkeypatch.setattr(module, "load_hypergraph_npz_dataset", load)
    monkeypatch.setattr(module, "validate_hypergraph_structure", validate)

    dataset_type.process(_Adapter())

    assert loader_calls == [(raw_dir, "fixture")]
    assert validation_calls == [data]
    assert save_calls == [([data], processed_path)]
