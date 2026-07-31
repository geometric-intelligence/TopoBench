"""Loader for the native heterogeneous PyG DBLP dataset."""

import torch_geometric.datasets as pyg_datasets
from omegaconf import DictConfig
from torch_geometric.data import Dataset

from topobench.data.loaders.base import AbstractLoader


class DBLPDatasetLoader(AbstractLoader):
    """Load DBLP without changing its native graph or official masks."""

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Return PyG's native DBLP dataset at the canonical data root."""
        return pyg_datasets.DBLP(root=str(self.get_data_dir()))


__all__ = ["DBLPDatasetLoader"]
