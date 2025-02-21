"""Loaders for TU datasets."""

from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset

from topobenchmark.data.loaders.base import AbstractLoader


class TUDatasetLoader(AbstractLoader):
    """Load TU datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - data_type: Type of the dataset (e.g., "graph_classification")
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load TU dataset.

        Returns
        -------
        Dataset
            The loaded TU dataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = TUDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            use_node_attr=False,
        )
        return dataset
