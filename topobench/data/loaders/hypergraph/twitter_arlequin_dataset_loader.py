"""Loaders for Citation Hypergraph dataset."""

from omegaconf import DictConfig

from topobench.data.datasets import TwitterArlequinDataset
from topobench.data.loaders.base import AbstractLoader


class TwitterArlequinDatasetLoader(AbstractLoader):
    """Load Twitter Arlequin dataset with configurable parameters.

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

    def load_dataset(self) -> TwitterArlequinDataset:
        """Load the Twitter Arlequin dataset.

        Returns
        -------
        TwitterArlequinDataset
            The loaded Twitter Arlequin dataset with the appropriate `data_dir`.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """
        dataset = self._initialize_dataset()
        self.data_dir = self.get_data_dir()
        return dataset

    def _initialize_dataset(self) -> TwitterArlequinDataset:
        """Initialize the Twitter Arlequin dataset.

        Returns
        -------
        TwitterArlequinDataset
            The initialized dataset instance.
        """
        return TwitterArlequinDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            parameters=self.parameters,
        )
