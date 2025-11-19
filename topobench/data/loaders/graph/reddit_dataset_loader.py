"""Loaders for Reddit dataset."""

from pathlib import Path

from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import Reddit

from topobench.data.loaders.base import AbstractLoader


class RedditDatasetLoader(AbstractLoader):
    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        dataset = self._initialize_dataset()
        self.data_dir = self._redefine_data_dir(dataset)
        return dataset

    def _initialize_dataset(self) -> Dataset:
        return Reddit(
            root=str(self.root_data_dir),
        )

    def _redefine_data_dir(self, dataset: Dataset) -> Path:
        return str(Path(dataset.processed_dir))





