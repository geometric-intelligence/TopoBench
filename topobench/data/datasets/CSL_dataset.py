r"""Dataset class for CSL  dataset."""

import os
import random
import os.path as osp
from typing import ClassVar
import dgl
import pickle

import networkx as nx
import numpy as np
import torch
import torch_geometric
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.io import fs
from torch_geometric.utils.convert import from_networkx, to_networkx, from_dgl

from topobench.data.utils import (
    download_file_from_link,
)
from topobench.utils.utils import extras

class CSLDataset(InMemoryDataset):
    r"""Dataset class for CSL dataset.

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

    URL: ClassVar = "https://www.dropbox.com/s/rnbkp5ubgk82ocu/CSL.zip?dl=1"
    FILE_FORMAT: ClassVar = "zip"

    def __init__(self, root: str, name: str, parameters: DictConfig, **kwargs):
        self.parameters = parameters
        self.name = name
        self.root = root
        self.num_node_type = 1
        self.num_edge_type = 1
        super().__init__(
            root,
        )
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
        self.processed_root = osp.join(self.root, self.name)
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
            "graphs_Kary_Deterministic_Graphs.pkl",
            "y_Kary_Deterministic_Graphs.pt",
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
        dataset_name = self.name

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

        # # Delete zip file
        # os.unlink(path)

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
        graph_list_file_path = osp.join(
            self.raw_dir,
            self.raw_file_names[0],
        )
        graph_label_list_file_path = osp.join(
            self.raw_dir,
            self.raw_file_names[1],
        )

        data_list = []
        with open(graph_list_file_path, 'rb') as f:
            adj_list = pickle.load(f)
        label_list = torch.load(graph_label_list_file_path)

        for i, sample in enumerate(adj_list):
            _g = dgl.from_scipy(sample)
            g = dgl.remove_self_loop(_g)
            graph = from_dgl(g)
            graph.y = label_list[i]
            graph.x = torch.ones(
                (graph.num_nodes, 10),
                dtype=torch.float,
            )
            graph.edge_attr = torch.ones(
                (graph.num_edges, 1),
                dtype=torch.float,
            )
            data_list.append(graph)

        self.data, self.slices = self.collate(data_list)
        self._data_list = None  # Reset cache.
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )
