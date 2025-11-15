"""Loaders for Citation Hypergraph dataset."""

from omegaconf import DictConfig

from topobench.data.datasets import SemanticDataset
from topobench.data.loaders.base import AbstractLoader


class SemanticDatasetLoader(AbstractLoader):
    """Load Semantic dataset with configurable parameters.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_name: Name of the dataset
            - other relevant parameters
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> SemanticDataset:
        """Load the Citation Hypergraph dataset.

        Returns
        -------
        SemanticDataset
            The loaded a Semantic dataset with the appropriate `data_name`.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = self._initialize_dataset()
        self.data_dir = self.get_data_dir()
        return dataset

    def _initialize_dataset(self) -> SemanticDatasetLoader:
        """Initialize the Citation Hypergraph dataset.

        Returns
        -------
        HypergraphDataset
            The initialized dataset instance.
        """
        return SemanticDatasetLoader(
            name=self.parameters.data_name,
            parameters=self.parameters,
        )
