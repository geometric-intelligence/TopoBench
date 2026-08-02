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
from topobench.data.utils.downloads import (
    RemoteArchive,
    acquire_verified_archive,
)
from topobench.data.utils.hypergraph_io import (
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
        self.load(self.processed_paths[0], data_cls=HypergraphData)
        data = self.get(0)
        validate_hypergraph_structure(data)

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
    def processed_file_names(self) -> str:
        """Return the processed file name for the dataset.

        Returns
        -------
        str
            Processed file name.
        """
        return HYPERGRAPH_CACHE_FILENAME

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
        self.save([data], self.processed_paths[0])
