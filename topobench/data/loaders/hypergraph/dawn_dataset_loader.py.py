"""Loaders for Dawn Temporal Hypergraph dataset."""

from omegaconf import DictConfig

from topobench.data.datasets.dawn_hypergraph_dataset import (
    DawnHypergraphDataset as HypergraphDataset,
)
from topobench.data.loaders.base import AbstractLoader


class DawnDatasetLoader(AbstractLoader):
    """Load Citation Hypergraph dataset with configurable parameters.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - other relevant parameters
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> HypergraphDataset:
        """Load the Citation Hypergraph dataset.

        Returns
        -------
        HypergraphDataset
            The loaded Citation Hypergraph dataset with the appropriate `data_dir`.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = self._initialize_dataset()
        self.data_dir = self.get_data_dir()
        return dataset

    def _initialize_dataset(self) -> HypergraphDataset:
        """Initialize the Citation Hypergraph dataset.

        Returns
        -------
        HypergraphDataset
            The initialized dataset instance.
        """
        return HypergraphDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            parameters=self.parameters,
        )
