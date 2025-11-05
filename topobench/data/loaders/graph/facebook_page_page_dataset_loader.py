"""Loader for FacebookPagePage Graph dataset."""


from omegaconf import DictConfig
from torch_geometric.data import Dataset

from topobench.data.datasets.facebook_page_page_dataset import (
    FacebookPagePageDataset,
)
from topobench.data.loaders.base import AbstractLoader


class FacebookPagePageDatasetLoader(AbstractLoader):
    """Load FacebookPagePage Graph dataset.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load FacebookPagePage Graph dataset.

        Returns
        -------
        Dataset
            The loaded FacebookPagePage Graph dataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = FacebookPagePageDataset(
            root=str(self.root_data_dir),
                name=self.parameters.data_name,
                parameters=self.parameters,
        )
        return dataset
