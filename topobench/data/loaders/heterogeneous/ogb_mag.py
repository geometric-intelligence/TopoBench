"""Loader for native heterogeneous OGB-MAG with complete node features."""

import torch_geometric.datasets as pyg_datasets
from omegaconf import DictConfig
from torch_geometric.data import Dataset

from topobench.data.loaders.base import AbstractLoader


class OGBMAGDatasetLoader(AbstractLoader):
    """Load OGB-MAG without changing native stores or official masks."""

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Return PyG OGB-MAG with metapath2vec features for every node type."""
        return pyg_datasets.OGB_MAG(
            root=str(self.get_data_dir()),
            preprocess=str(self.parameters.preprocess),
        )


__all__ = ["OGBMAGDatasetLoader"]
