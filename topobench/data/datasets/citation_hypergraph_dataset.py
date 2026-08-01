"""Dataset class for US County Demographics dataset."""

import os
import os.path as osp
import shutil
import warnings
from typing import ClassVar

from omegaconf import DictConfig
from torch_geometric.data import InMemoryDataset, extract_zip

from topobench.data import (
    HYPERGRAPH_CACHE_FILENAME,
    HypergraphData,
    validate_hypergraph_structure,
)
from topobench.data.utils.downloads import download_file_from_drive
from topobench.data.utils.hypergraph_io import load_hypergraph_pickle_dataset


class CitationHypergraphDataset(InMemoryDataset):
    r"""Dataset class for US County Demographics dataset.

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
    URLS (dict): Dictionary containing the URLs for downloading the dataset.
    FILE_FORMAT (dict): Dictionary containing the file formats for the dataset.
    RAW_FILE_NAMES (dict): Dictionary containing the raw file names for the dataset.
    """

    URLS: ClassVar = {
        "coauthorship_cora": "https://drive.google.com/file/d/1J5fLPABWrM9SH_7m85n7--oHDVmwJeib/view?usp=sharing",
        "coauthorship_dblp": "https://drive.google.com/file/d/16ryf4Ve-t0_nAla0VfjtSxSAG8Sye8TZ/view?usp=sharing",
        "cocitation_cora": "https://drive.google.com/file/d/1WVRx5yDxSdZpvL6FK5Ji8H3lOnyYlraN/view?usp=sharing",
        "cocitation_citeseer": "https://drive.google.com/file/d/1XWfu1jtijsmHmfCP6UQxyLsuPM8GBNJb/view?usp=sharing",
        "cocitation_pubmed": "https://drive.google.com/file/d/1XbqDJnHnV0HYvie3fcM8rquamnQsLTpK/view?usp=sharing",
    }

    FILE_FORMAT: ClassVar = {
        "coauthorship_cora": "zip",
        "coauthorship_dblp": "zip",
        "cocitation_cora": "zip",
        "cocitation_citeseer": "zip",
        "cocitation_pubmed": "zip",
    }

    RAW_FILE_NAMES: ClassVar = {}

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
        return []  # ["county_graph.csv", f"county_stats_{self.year}.csv"]

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
        r"""Download the dataset from a URL and saves it to the raw directory.

        Raises:
            FileNotFoundError: If the dataset URL is not found.
        """
        # Step 1: Download data from the source
        self.url = self.URLS[self.name]
        self.file_format = self.FILE_FORMAT[self.name]

        download_file_from_drive(
            file_link=self.url,
            path_to_save=self.raw_dir,
            dataset_name=self.name,
            file_format=self.file_format,
        )
        # Extract zip file
        folder = self.raw_dir
        filename = f"{self.name}.{self.file_format}"
        path = osp.join(folder, filename)
        extract_zip(path, folder)
        # Delete zip file
        os.unlink(path)

        # Move files from osp.join(folder, name_download) to folder
        for file in os.listdir(osp.join(folder, self.name)):
            shutil.move(
                osp.join(folder, self.name, file), osp.join(folder, file)
            )
        # Delete osp.join(folder, self.name) dir
        shutil.rmtree(osp.join(folder, self.name))

    def process(self) -> None:
        """Parse raw pickle files and save one native v2 hypergraph."""
        data, _ = load_hypergraph_pickle_dataset(
            data_dir=self.raw_dir,
            data_name=self.name,
        )
        validate_hypergraph_structure(data)
        self.save([data], self.processed_paths[0])
