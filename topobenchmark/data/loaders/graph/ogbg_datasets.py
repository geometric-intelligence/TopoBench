"""Loaders for  Graph Property Prediction datasets."""

import os
from pathlib import Path

from ogb.graphproppred import PygGraphPropPredDataset
from omegaconf import DictConfig
from torch_geometric.data import Dataset

from topobenchmark.data.loaders.base import AbstractLoader


class OGBGDatasetLoader(AbstractLoader):
    """Load molecule datasets (molhiv, molpcba, ppa) with predefined splits.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - data_type: Type of the dataset (e.g., "molecule")
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)
        self.datasets: list[Dataset] = []

    def load_dataset(self) -> Dataset:
        """Load the molecule dataset with predefined splits.

        Returns
        -------
        Dataset
            The combined dataset with predefined splits.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        split_idx = self._load_splits()
        combined_dataset = self._combine_splits()
        combined_dataset.split_idx = split_idx
        return combined_dataset

    def _load_splits(self) -> None:
        """Load the dataset splits for the specified dataset.

        Returns
        -------
        dict
            The split indices for the dataset.
        """
        dataset = PygGraphPropPredDataset(
            name="ogbg-" + self.parameters.data_name.lower()
        )
        split_idx = dataset.get_idx_split()

        for split in ["train", "valid", "test"]:
            ds = dataset[split_idx[split]]
            ds.x = ds.x.long()
            self.datasets.append(ds)
        return split_idx

    def _combine_splits(self) -> Dataset:
        """Combine the dataset splits into a single dataset.

        Returns
        -------
        Dataset
            The combined dataset containing all splits.
        """
        return self.datasets[0] + self.datasets[1] + self.datasets[2]

    def get_data_dir(self) -> Path:
        """Get the data directory.

        Returns
        -------
        Path
            The path to the dataset directory.
        """
        return os.path.join(self.root_data_dir, self.parameters.data_name)
