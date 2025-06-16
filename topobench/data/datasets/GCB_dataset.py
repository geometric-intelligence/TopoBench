r"""Dataset class for GCB  dataset."""

import os.path as osp
from typing import ClassVar

import numpy as np
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.io import fs


from topobench.data.utils import (
    download_file_from_link,
)




class GCBDataset(InMemoryDataset):
    """The synthetic dataset from `"Pyramidal Reservoir Graph Neural Network"
    <https://arxiv.org/abs/2104.04710>`_ paper.

    Args:
        root (string): Root directory where the dataset should be saved.
        split (string): If `"train"`, loads the training dataset.
            If `"val"`, loads the validation dataset.
            If `"test"`, loads the test dataset. Defaults to `"train"`.
        easy (bool, optional): If `True`, use the easy version of the dataset.
            Defaults to `True`.
        small (bool, optional): If `True`, use the small version of the
            dataset. Defaults to `True`.
        transform (callable, optional): A function/transform that takes in an
            `torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            Defaults to `None`.
        pre_transform (callable, optional): A function/transform that takes in
            an `torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. Defaults to `None`.
        pre_filter (callable, optional): A function that takes in an
            `torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. Defaults to `None`.
    """

    URL: ClassVar = "http://github.com/FilippoMB/Benchmark_dataset_for_graph_classification/raw/master/datasets/"
    FILE_FORMAT: ClassVar = "npz"

    def __init__(self, root: str, name: str, split: str, parameters: DictConfig, **kwargs):
        self.parameters = parameters
        self.name = name
        self.root = root
        self.split = split
        self.filename = self.parameters['version'] + ('_small' if self.parameters['small'] else '')

        assert self.split in {'train', 'valid', 'test'}
        if self.split != 'valid':
            self.split = self.split[:2]
        else:
            self.split = self.split[:3]

        super().__init__(
            root,
        )
        out = fs.torch_load(self.processed_paths[0])
        assert len(out) == 3 or len(out) == 4
        data, self.slices, self.sizes, data_cls = out
        self.data = data_cls.from_dict(data)
        assert isinstance(self._data, Data)


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
        return [f'{self.filename}.npz']
    
    @property
    def processed_file_names(self):
        return f'{self.filename}_{self.split}.pt'

    def download(self) -> None:
        r"""Download the dataset from a URL and saves it to the raw directory.

        Raises:
            FileNotFoundError: If the dataset URL is not found.
        """
        # Step 1: Download data from the source
        self.url = self.URL + self.filename
        self.file_format = self.FILE_FORMAT
        self.url += '.' + self.file_format

        download_url(self.url, self.raw_dir)

    def process(self):
        npz_file_path = osp.join(
            self.raw_dir,
            self.raw_file_names[0],
        )
        npz = np.load(npz_file_path, allow_pickle=True)
        raw_data = (npz[f'{self.split}_{key}'] for key in ['feat', 'adj', 'class'])
        data_list = [Data(x=torch.FloatTensor(x), 
                        edge_index=torch.LongTensor(np.stack(adj.nonzero())), 
                        y=torch.LongTensor(y.nonzero()[0])) for x, adj, y in zip(*raw_data)]

        self.data, self.slices = self.collate(data_list)
        self._data_list = None  # Reset cache.
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )

