"""Loaders for Reddit dataset."""

from omegaconf import DictConfig
from torch_geometric.datasets import Reddit

from topobench.data.loaders.base import AbstractLoader


class RedditDatasetLoader(AbstractLoader):
    """Loader for the Reddit graph dataset.

    Parameters
    ----------
    parameters : DictConfig
        Configuration with data directory and dataset name.
    """

    def __init__(self, parameters: DictConfig) -> None:
        """Initialize the Reddit dataset loader."""
        super().__init__(parameters)

    def load_dataset(self) -> Reddit:
        """Load the Reddit dataset.

        Returns
        -------
        Reddit
            Loaded Reddit dataset instance.
        """
        dataset = self._initialize_dataset()
        self.data_dir = self.get_data_dir()
        return dataset

    def _initialize_dataset(self) -> Reddit:
        """Initialize the Reddit dataset instance.

        Returns
        -------
        Reddit
            Initialized Reddit dataset.
        """
        # root = <base_dir>/<data_name>
        root = str(self.root_data_dir / self.parameters.data_name)
        return Reddit(root=root)
