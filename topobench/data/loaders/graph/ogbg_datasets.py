"""Loaders for  Graph Property Prediction datasets."""

import os
from pathlib import Path

from ogb.graphproppred import PygGraphPropPredDataset
from omegaconf import DictConfig
from torch_geometric.data import Dataset

from topobench.data.features import (
    OGB_ATOM_FEATURE_CARDINALITIES,
    encode_categorical_columns,
)
from topobench.data.loaders.base import AbstractLoader


class OGBGDatasetLoader(AbstractLoader):
    """Load OGB molecular property datasets with predefined splits.

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

        dataset = PygGraphPropPredDataset(
            name=self.parameters.data_name, root=self.root_data_dir
        )
        dataset._data.x = encode_categorical_columns(
            dataset._data.x,
            OGB_ATOM_FEATURE_CARDINALITIES,
        )
        # Squeeze the target tensor
        dataset._data.y = dataset._data.y.squeeze(1)
        dataset.split_idx = dataset.get_idx_split()

        return dataset

    def get_data_dir(self) -> Path:
        """Get the data directory.

        Returns
        -------
        Path
            The path to the dataset directory.
        """
        return os.path.join(self.root_data_dir, self.parameters.data_name)
