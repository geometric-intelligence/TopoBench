r"""Dataset class for BREC dataset."""

import os
import os.path as osp
from typing import ClassVar

import networkx as nx
import numpy as np
import torch
import torch_geometric
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.io import fs
from torch_geometric.utils.convert import from_networkx

from topobench.data.utils import (
    download_file_from_link,
)

torch_geometric.seed_everything(2022)


NAME_TO_FILE_IDX_MAP = {
    "full": 0,
    "3wl": 1,
    "cut": 2,
}


def graph6_to_pyg(x):
    r"""Convert graph6 format to PyTorch Geometric Data.

    Parameters
    ----------
    x : bytes
        Graph6 encoded graph.

    Returns
    -------
    Data
        PyTorch Geometric Data object.
    """
    return from_networkx(nx.from_graph6_bytes(x))


class BRECDataset(InMemoryDataset):
    r"""Dataset class for BREC dataset.

    Parameters
    ----------
    root : str
        Root directory where the dataset will be saved.
    name : str
        Name of the dataset (full, 3wl, cut).
    parameters : DictConfig
        Configuration parameters for the dataset.
    **kwargs : dict
        Additional keyword arguments.

    Attributes
    ----------
    URL (dict): URL for downloading the dataset.
    FILE_FORMAT (dict): File format for the dataset.
    """

    URL: ClassVar = "https://github.com/GraphPKU/BREC/raw/refs/heads/Release/BREC_data_all.zip"
    FILE_FORMAT: ClassVar = "zip"

    def __init__(self, root: str, name: str, parameters: DictConfig, **kwargs):
        self.parameters = parameters
        self.name = name
        self.root = root
        super().__init__(
            root,
        )
        print(self.processed_paths)
        out = fs.torch_load(self.processed_paths[0])
        assert len(out) == 3 or len(out) == 4
        data, self.slices, self.sizes, data_cls = out
        self.data = data_cls.from_dict(data)
        assert isinstance(self._data, Data)

    def __repr__(self) -> str:
        return f"{self.name}(self.root={self.root}, self.name={self.name}, self.parameters={self.parameters}, self.force_reload={self.force_reload})"

    @property
    def raw_dir(self) -> str:
        r"""Return the path to the raw directory of the dataset.

        Returns
        -------
        str
            Path to the raw directory.
        """
        return osp.join(
            self.root,
            self.name,
            "raw",
        )

    @property
    def processed_dir(self) -> str:
        r"""Return the path to the processed directory of the dataset.

        Returns
        -------
        str
            Path to the processed directory.
        """
        self.processed_root = osp.join(
            self.root,
            self.name,
        )
        return osp.join(self.processed_root, "processed")

    @property
    def raw_file_names(self):
        r"""Return the names of the raw files in the dataset.

        Returns
        -------
        list
            List of raw file names.
        """
        return [
            "brec_v3_3wl.npy",
            "brec_v3_no4v_60cfi.npy",
            "brec_v3.npy",
        ]

    @property
    def processed_file_names(self):
        r"""Return the names of the processed files in the dataset.

        Returns
        -------
        list
            List of processed file names.
        """
        return ["data.pt"]

    def download(self) -> None:
        r"""Download the dataset from a URL and saves it to the raw directory.

        Raises:
            FileNotFoundError: If the dataset URL is not found.
        """
        # Step 1: Download data from the source
        self.url = self.URL
        self.file_format = self.FILE_FORMAT
        dataset_name = "BREC"

        download_file_from_link(
            file_link=self.url,
            path_to_save=self.raw_dir,
            dataset_name=dataset_name,
            file_format=self.file_format,
        )

        # Extract zip file
        folder = self.raw_dir
        filename = f"{dataset_name}.{self.file_format}"
        path = osp.join(folder, filename)
        extract_zip(path, folder)

        # Delete zip file
        os.unlink(path)

        # # Move files from osp.join(folder, name_download) to folder
        # for file in os.listdir(osp.join(folder, self.name)):
        #     shutil.move(osp.join(folder, self.name, file), folder)
        # # Delete osp.join(folder, self.name) dir
        # shutil.rmtree(osp.join(folder, self.name))

    def process(self):
        r"""Handle the data for the dataset.

        This method loads the already processed variation of the BREC dataset
        and converst the underlying graph6 format to PyTorch Geometric Data format.
        """
        file_name_path = osp.join(
            self.raw_dir, self.raw_file_names[NAME_TO_FILE_IDX_MAP[self.name]]
        )
        data_list = np.load(file_name_path, allow_pickle=True)
        data_list = [graph6_to_pyg(data) for data in data_list]

        for idx, data in enumerate(data_list):
            data["y"] = torch.tensor([idx], dtype=torch.long)
            data["x"] = torch.ones(1, dtype=torch.float)

        self.data, self.slices = self.collate(data_list)
        self._data_list = None  # Reset cache.
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )
