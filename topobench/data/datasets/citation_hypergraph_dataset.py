"""Authenticated NPZ hypergraph dataset adapter."""

import os.path as osp
import warnings
from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar

from omegaconf import DictConfig
from torch_geometric.data import InMemoryDataset

from topobench.data import (
    HYPERGRAPH_CACHE_FILENAME,
    HypergraphData,
    validate_hypergraph_structure,
)
from topobench.data.utils.cache_io import (
    cache_manifest_path,
    load_pyg_cache,
    write_pyg_cache,
)
from topobench.data.utils.downloads import (
    RemoteArchive,
    acquire_verified_archive,
)
from topobench.data.utils.hypergraph_io import (
    hypergraph_asset_identity,
    load_hypergraph_npz_dataset,
    validate_hypergraph_npz_assets,
)


class CitationHypergraphDataset(InMemoryDataset):
    r"""Load a citation hypergraph only from authenticated safe NPZ assets.

    Parameters
    ----------
    root : str
        Root directory where the dataset will be saved.
    name : str
        Name of the dataset.
    parameters : DictConfig
        Configuration parameters for the dataset.

    Attributes
    ----------
    ASSETS:
        Immutable exact asset manifest. An entry is publishable only with an
        HTTPS URL, exact byte size and SHA-256, and explicit extraction limits.
    """

    ASSETS: ClassVar[Mapping[str, RemoteArchive]] = MappingProxyType({})

    def __init__(
        self,
        root: str,
        name: str,
        parameters: DictConfig,
    ) -> None:
        self.name = name
        self.parameters = parameters
        self.trusted_cache_root = root
        legacy_path = osp.join(root, name, "processed", "data.pt")
        native_path = osp.join(
            root,
            name,
            "processed",
            HYPERGRAPH_CACHE_FILENAME,
        )
        if osp.isfile(legacy_path) and not osp.isfile(native_path):
            warnings.warn(
                "Ignoring legacy processed cache data.pt; regenerating native "
                "hypergraph data.",
                UserWarning,
                stacklevel=2,
            )
        super().__init__(root)
        self.cache_identity = hypergraph_asset_identity(
            self.raw_dir,
            self.name,
        )
        self._data, slices = load_pyg_cache(
            self.processed_paths[0],
            trusted_root=self.trusted_cache_root,
            family="hypergraph",
            cache_identity=self.cache_identity,
            data_cls=HypergraphData,
        )
        self.slices = slices or None
        validate_hypergraph_structure(self.get(0))

    def __repr__(self) -> str:
        return f"{self.name}(self.root={self.root}, self.name={self.name}, self.parameters={self.parameters}, self.force_reload={self.force_reload})"

    @property
    def raw_dir(self) -> str:
        """Return the path to the raw directory of the dataset.

        Returns
        -------
        str
            Path to the raw directory.
        """
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory of the dataset.

        Returns
        -------
        str
            Path to the processed directory.
        """

        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw file names for the dataset.

        Returns
        -------
        list[str]
            List of raw file names.
        """
        return [f"{self.name}.json", f"{self.name}.npz"]

    @property
    def processed_file_names(self) -> list[str]:
        """Return the complete native processed-cache publication set."""
        return [
            HYPERGRAPH_CACHE_FILENAME,
            cache_manifest_path(HYPERGRAPH_CACHE_FILENAME).name,
        ]

    def download(self) -> None:
        """Acquire one authenticated safe-format archive atomically."""
        try:
            asset = self.ASSETS[self.name]
        except KeyError as error:
            raise ValueError(
                f"{self.name}: no authenticated safe archive is published"
            ) from error
        acquire_verified_archive(
            asset,
            self.raw_dir,
            lambda root: validate_hypergraph_npz_assets(root, self.name),
        )

    def process(self) -> None:
        """Parse safe NPZ assets and save one native hypergraph."""
        data, _ = load_hypergraph_npz_dataset(
            data_dir=self.raw_dir,
            data_name=self.name,
        )
        validate_hypergraph_structure(data)
        self.cache_identity = hypergraph_asset_identity(
            self.raw_dir,
            self.name,
        )
        write_pyg_cache(
            [data],
            self.processed_paths[0],
            trusted_root=self.trusted_cache_root,
            family="hypergraph",
            cache_identity=self.cache_identity,
        )
