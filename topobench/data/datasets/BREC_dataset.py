r"""Dataset class for BREC dataset."""

import os
import random
import os.path as osp
from typing import ClassVar

import networkx as nx
import numpy as np
import torch
import torch_geometric
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs
from torch_geometric.utils.convert import from_networkx, to_networkx

from topobench.data.utils import (
    download_file_from_link,
)

torch_geometric.seed_everything(2022)


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

# TODO change
# def get_ranges(subset_name):
#     r"""Get the range of indices for a given subset name.

#     Parameters
#     ----------
#     subset_name : str
#         Name of the subset.
#     Returns

#     """
#     NUM_RELABEL=32
#     SAMPLE_NUM=400
#     start, end  = part_dict[subset_name]
#     train_range = (start * NUM_RELABEL * 2, (end) * NUM_RELABEL * 2)
#     reliability_range = ((start+SAMPLE_NUM) * NUM_RELABEL * 2, (end+SAMPLE_NUM) * NUM_RELABEL * 2)
#     return train_range, reliability_range


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

    URL: ClassVar = "https://github.com/GraphPKU/BREC/raw/refs/heads/Release/customize/Data/raw/{subset_name}.npy"
    FILE_FORMAT: ClassVar = "npy"

    def __init__(self, root: str, name: str, parameters: DictConfig, **kwargs):
        self.parameters = parameters
        self.subset = parameters.subset
        self.num_relabel = parameters.num_relabel
        self.name = name
        self.root = root
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
            f"{self.subset}.npy",
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
        self.url = self.URL.format(subset_name=self.subset)
        self.file_format = self.FILE_FORMAT
        dataset_name = self.subset

        download_file_from_link(
            file_link=self.url,
            path_to_save=self.raw_dir,
            dataset_name=dataset_name,
            file_format=self.file_format,
        )

        # # Extract zip file
        # folder = self.raw_dir
        # filename = f"{dataset_name}.{self.file_format}"
        # path = osp.join(folder, filename)
        # extract_zip(path, folder)

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
        file_name_path = osp.join(
            self.raw_dir,
            self.raw_file_names[0],
        )

        g6_list = []

        # Basic graphs: 0 - 60
        if self.subset == 'basic':
            basic = np.load(file_name_path)
            for i in range(0, basic.size, 2):
                g6_tuple = (basic[i].encode(), basic[i + 1].encode())
                g6_relabel_tuple = (self.generate_relabel(g6_tuple[0]), self.generate_relabel(g6_tuple[1]))
                g6_list.extend(self.reindex_to_interlacing(g6_relabel_tuple))

            for i in range(0, basic.size, 2):
                flag = random.randint(0, 1)
                g6_tuple = (basic[i].encode(), basic[i + 1].encode())
                g6_relabel = self.generate_relabel(g6_tuple[flag], num=self.num_relabel * 2)
                g6_list.extend(g6_relabel)
        # Simple regular graphs: 60 - 110
        elif self.subset == 'regular':
            regular = np.load(file_name_path)
            for i in range(regular.size // 2):
                g6_tuple = regular[i]
                g6_relabel_tuple = (self.generate_relabel(g6_tuple[0]), self.generate_relabel(g6_tuple[1]))
                g6_list.extend(self.reindex_to_interlacing(g6_relabel_tuple))

            for i in range(regular.size // 2):
                flag = random.randint(0, 1)
                g6_tuple = regular[i]
                g6_relabel = self.generate_relabel(g6_tuple[flag], num=self.num_relabel * 2)
                g6_list.extend(g6_relabel)
        # Strongly regular graphs: 110 - 160
        elif self.subset == 'str':
            strongly_regular = np.load(file_name_path)
            for i in range(0, strongly_regular.size, 2):
                g6_tuple = (strongly_regular[i].encode(), strongly_regular[i + 1].encode())
                g6_relabel_tuple = (self.generate_relabel(g6_tuple[0]), self.generate_relabel(g6_tuple[1]))
                g6_list.extend(self.reindex_to_interlacing(g6_relabel_tuple))

            for i in range(0, strongly_regular.size, 2):
                flag = random.randint(0, 1)
                g6_tuple = (strongly_regular[i].encode(), strongly_regular[i + 1].encode())
                g6_relabel = self.generate_relabel(g6_tuple[flag], num=self.num_relabel * 2)
                g6_list.extend(g6_relabel)
        # Extension graphs: 160 - 260
        elif self.subset == 'extension':
            extension = np.load(file_name_path)
            for i in range(extension.size // 2):
                g6_tuple = (extension[i][0].encode(), extension[i][1].encode())
                g6_relabel_tuple = (self.generate_relabel(g6_tuple[0]), self.generate_relabel(g6_tuple[1]))
                g6_list.extend(self.reindex_to_interlacing(g6_relabel_tuple))

            for i in range(extension.size // 2):
                flag = random.randint(0, 1)
                g6_tuple = (extension[i][0].encode(), extension[i][1].encode())
                g6_relabel = self.generate_relabel(g6_tuple[flag], num=self.num_relabel * 2)
                g6_list.extend(g6_relabel)
        # CFI graphs: 260 - 360
        elif self.subset == 'cfi':
            cfi = np.load(file_name_path)
            for i in range(cfi.size // 2):
                g6_tuple = cfi[i]
                g6_relabel_tuple = (self.generate_relabel(g6_tuple[0]), self.generate_relabel(g6_tuple[1]))
                g6_list.extend(self.reindex_to_interlacing(g6_relabel_tuple))

            for i in range(cfi.size // 2):
                flag = random.randint(0, 1)
                g6_tuple = cfi[i]
                g6_relabel = self.generate_relabel(g6_tuple[flag], num=self.num_relabel * 2)
                g6_list.extend(g6_relabel)
        # 4-vertex condition graphs: 360 - 380
        elif self.subset == '4vtx':
            vtx_4 = np.load(file_name_path)
            for i in range(0, vtx_4.size, 2):
                g6_tuple = (vtx_4[i].encode(), vtx_4[i + 1].encode())
                g6_relabel_tuple = (self.generate_relabel(g6_tuple[0]), self.generate_relabel(g6_tuple[1]))
                g6_list.extend(self.reindex_to_interlacing(g6_relabel_tuple))

            for i in range(0, vtx_4.size, 2):
                flag = random.randint(0, 1)
                g6_tuple = (vtx_4[i].encode(), vtx_4[i + 1].encode())
                g6_relabel = self.generate_relabel(g6_tuple[flag], num=self.num_relabel * 2)
                g6_list.extend(g6_relabel)
        # Distance regular graphs: 380 - 400
        elif self.subset == 'dr':
            distance_regular = np.load(file_name_path)
            for i in range(0, distance_regular.size, 2):
                g6_tuple = (distance_regular[i], distance_regular[i + 1])
                g6_relabel_tuple = (self.generate_relabel(g6_tuple[0]), self.generate_relabel(g6_tuple[1]))
                g6_list.extend(self.reindex_to_interlacing(g6_relabel_tuple))

            for i in range(0, distance_regular.size, 2):
                flag = random.randint(0, 1)
                g6_tuple = (distance_regular[i], distance_regular[i + 1])
                g6_relabel = self.generate_relabel(g6_tuple[flag], num=self.num_relabel * 2)
                g6_list.extend(g6_relabel)
        else:
            raise ValueError(f"Unknown subset: {self.subset}")

        # Convert to PyG format  
        data_list = [graph6_to_pyg(data) for data in g6_list]

        for idx, data in enumerate(data_list):
            data["y"] = torch.tensor([idx], dtype=torch.long)
            data["x"] = torch.ones((data.num_nodes, 10), dtype=torch.float)
            assert data.x.shape[1] == 10
            assert data.x.shape[0] == data.num_nodes

        self.data, self.slices = self.collate(data_list)
        self._data_list = None  # Reset cache.
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )

    def relabel(self, g6):
        pyg_graph = from_networkx(nx.from_graph6_bytes(g6))
        n = pyg_graph.num_nodes
        edge_index_relabel = pyg_graph.edge_index.clone().detach()
        index_mapping = dict(zip(list(range(n)), np.random.permutation(n), strict=False))
        for i in range(edge_index_relabel.shape[0]):
            for j in range(edge_index_relabel.shape[1]):
                edge_index_relabel[i, j] = index_mapping[edge_index_relabel[i, j].item()]
        edge_index_relabel = edge_index_relabel[
            :, torch.randperm(edge_index_relabel.shape[1])
        ]
        pyg_graph_relabel = torch_geometric.data.Data(
            edge_index=edge_index_relabel, num_nodes=n
        )
        g6_relabel = nx.to_graph6_bytes(
            to_networkx(pyg_graph_relabel, to_undirected=True), header=False
        ).strip()
        return g6_relabel


    def generate_relabel(self, g6, num=0):
        if num == 0:
            num = self.num_relabel 
        g6_list = [g6]
        g6_set = set(g6_list)
        for id in range(num - 1):
            g6_relabel = self.relabel(g6)
            g6_set.add(g6_relabel)
            while len(g6_set) == len(g6_list):
                g6_relabel = self.relabel(g6)
                g6_set.add(g6_relabel)
            g6_list.append(g6_relabel)
        return g6_list

    def reindex_to_interlacing(self, g6_relabel_tuple):
        return_list = []
        for (x, y) in zip(g6_relabel_tuple[0], g6_relabel_tuple[1], strict=False):
            return_list.extend([x, y])
        return return_list
