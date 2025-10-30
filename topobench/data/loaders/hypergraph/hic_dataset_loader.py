"""Loaders for Citation Hypergraph dataset."""

from omegaconf import DictConfig

from topobench.data.datasets import HICDataset
from topobench.data.loaders.base import AbstractLoader


class HICDatasetLoader(AbstractLoader):
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
        self.data_dir = None

    def load_dataset(self) -> HICDataset:
        """Load the Citation Hypergraph dataset.

        Returns
        -------
        HICDataset
        """
        self.data_dir = self.get_data_dir()
        dataset = self._initialize_dataset()
        return dataset

    def _initialize_dataset(self) -> HICDataset:
        """Initialize the Citation Hypergraph dataset.

        Returns
        -------
        HICDataset
            The initialized dataset instance.
        """
        use_degree_as_tag = getattr(self.parameters, "use_degree_as_tag", False)

        return HICDataset(
            root=str(self.data_dir),
            name=self.parameters.data_name,
            use_degree_as_tag=use_degree_as_tag,
        )
