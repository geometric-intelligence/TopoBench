"""Loaders for Protein-Protein interactions datasets (PPI)."""

from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import PPI

from topobench.data.loaders.base import AbstractLoader


class PPIDatasetLoader(AbstractLoader):
    """Loader for the inductive PPI dataset."""

    def __init__(self, parameters: DictConfig) -> None:
        """Initializes the loader."""
        super().__init__(parameters)
        self.datasets: list[Dataset] = []

    def load_dataset(self) -> Dataset:
        """Load the molecule dataset with predefined splits.

        Returns
        -------
        Dataset
            The combined dataset with predefined splits.
        """
        self._load_splits()
        split_idx = self._prepare_split_idx()
        combined_dataset = self._combine_splits()
        combined_dataset.split_idx = split_idx
        return combined_dataset

    def _load_splits(self) -> None:
        """Load the dataset splits for the specified dataset."""
        for split in ["train", "val", "test"]:
            self.datasets.append(
                PPI(
                    root=str(self.root_data_dir),
                    split=split,
                )
            )

    def _prepare_split_idx(self) -> dict[str, np.ndarray]:
        """Prepare the split indices for the dataset.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary mapping split names to index arrays.
        """
        num_train = len(self.datasets[0])
        num_val = len(self.datasets[1])
        num_test = len(self.datasets[2])

        split_idx: dict[str, np.ndarray] = {}
        split_idx["train"] = np.arange(num_train)
        split_idx["valid"] = np.arange(num_train, num_train + num_val)
        split_idx["test"] = np.arange(
            num_train + num_val, num_train + num_val + num_test
        )

        return split_idx

    def _combine_splits(self) -> Dataset:
        """Combine the dataset splits into a single in-memory dataset.

        Returns
        -------
        Dataset
            A single PPI dataset containing all graphs from train/val/test.
        """
        # Use the train split as a base InMemoryDataset
        base = self.datasets[0]

        # Collect all graphs from train, val, test
        data_list = []
        for ds in self.datasets:
            for i in range(len(ds)):
                data_list.append(ds[i]) # noqa

        # Collate into a single `data` / `slices` structure
        data, slices = base.collate(data_list)

        # Overwrite the internal storage of `base`
        base._data = data
        base.data = data
        base.slices = slices

        return base

    def get_data_dir(self) -> Path:
        """Get the data directory.

        Returns
        -------
        Path
            The path to the dataset directory.
        """
        # This assumes self.parameters.data_name is 'PPI'
        return Path(self.root_data_dir) / self.parameters.data_name
