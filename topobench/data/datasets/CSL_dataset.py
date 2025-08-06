r"""Dataset class for CSL  dataset."""

import os.path as osp
from typing import ClassVar
import dgl
import pickle

import torch
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

    TEST_SPLIT_CSL = [
            [9,10,14,23,26,27,35,39,40,47,50,57,66,70,71,75,82,83,91,93,94,107,110,117,130,132,133,136,140,142],
            [1,4,5,16,21,24,32,33,38,46,48,52,62,72,73,79,80,85,97,101,102,108,109,111,123,127,129,143,147,149],
            [7,11,12,17,20,25,34,43,44,45,54,59,64,65,69,76,77,86,95,99,100,105,116,118,124,126,128,138,139,144],
            [0,2,13,15,19,22,31,36,41,55,56,58,61,67,68,78,81,88,92,96,104,106,113,115,121,125,131,137,145,146],
            [3,6,8,18,28,29,30,37,42,49,51,53,60,63,74,84,87,89,90,98,103,112,114,119,120,122,134,135,141,148]
    ]
    VAL_SPLIT_CSL =  [
        [9,10,14,23,26,27,35,39,40,47,50,57,66,70,71,75,82,83,91,93,94,107,110,117,130,132,133,136,140,142],
        [1,4,5,16,21,24,32,33,38,46,48,52,62,72,73,79,80,85,97,101,102,108,109,111,123,127,129,143,147,149],
        [7,11,12,17,20,25,34,43,44,45,54,59,64,65,69,76,77,86,95,99,100,105,116,118,124,126,128,138,139,144],
        [0,2,13,15,19,22,31,36,41,55,56,58,61,67,68,78,81,88,92,96,104,106,113,115,121,125,131,137,145,146],
        [3,6,8,18,28,29,30,37,42,49,51,53,60,63,74,84,87,89,90,98,103,112,114,119,120,122,134,135,141,148],

     ]
    TRAIN_SPLIT_CSL = [
        [62,69,88,115,43,128,11,55,108,84,137,74,48,104,72,81,24,34,90,12,134,37,99,58,63,25,22,7,4,30,103,21,87,28,148,86,68,44,118,145,36,114,95,54,42,144,18,20,124,8,85,126,122,138,49,73,5,92,141,113,45,109,120,146,51,59,149,89,41,15,46,139,2,116,123,78,61,97,121,112,125,76,101,119,6,1,67,100,31,29],
        [63,65,95,49,55,141,135,105,134,122,61,66,132,89,118,37,99,92,91,142,18,0,93,139,15,116,9,148,3,30,53,43,138,35,84,67,12,51,117,22,125,25,69,74,81,39,120,70,10,17,45,137,124,75,11,87,94,131,140,29,107,58,2,20,82,96,119,83,23,28,112,36,8,130,60,56,115,42,50,59,31,76,100,128,78,44,114,103,145,6],
        [42,137,47,57,10,87,148,83,61,23,107,32,109,73,14,123,46,72,38,37,51,48,24,127,21,92,141,36,132,6,13,136,84,19,98,91,63,120,134,16,106,18,60,50,112,80,125,66,22,135,131,94,4,3,1,79,70,102,110,145,96,33,56,146,108,93,58,143,119,97,15,129,27,8,81,40,149,115,55,75,9,30,78,74,104,89,35,62,133,114],
        [70,90,65,46,44,87,128,114,98,136,33,80,116,149,84,118,7,47,5,103,21,34,57,37,134,25,100,142,3,60,79,97,69,135,8,66,144,45,139,12,148,102,107,35,95,1,108,124,18,82,16,59,143,76,111,10,126,43,77,29,109,9,26,94,73,120,49,6,140,20,83,74,48,110,133,51,72,32,39,130,24,127,91,54,123,85,30,27,62,112],
        [55,26,13,23,46,16,101,136,91,125,111,80,25,9,14,128,56,15,109,100,133,20,40,82,143,21,88,113,38,77,45,139,62,67,130,124,93,10,35,110,73,144,116,44,58,142,107,85,36,83,131,106,64,7,70,138,105,127,108,66,97,79,34,0,147,5,52,61,47,72,92,48,17,1,33,68,24,129,32,78,123,102,95,149,145,11,41,76,57,99],
    ]
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
