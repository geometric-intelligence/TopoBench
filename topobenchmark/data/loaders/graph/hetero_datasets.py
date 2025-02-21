"""Loaders for Heterophilous Graph datasets."""

from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import HeterophilousGraphDataset

from topobenchmark.data.loaders.base import AbstractLoader


class HeterophilousGraphDatasetLoader(AbstractLoader):
    """Load Heterophilous Graph datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - data_type: Type of the dataset (e.g., "heterophilous")
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load Heterophilous Graph dataset.

        Returns
        -------
        Dataset
            The loaded Heterophilous Graph dataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = HeterophilousGraphDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
        )
        return dataset
