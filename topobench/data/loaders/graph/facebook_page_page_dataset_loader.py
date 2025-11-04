"""Loader for FacebookPagePage Graph dataset."""

from torch_geometric.data import Dataset
from torch_geometric.datasets import FacebookPagePage

from topobench.data.loaders.base import AbstractLoader


class FacebookPagePageDatasetLoader(AbstractLoader):
    """Load FacebookPagePage Graph dataset.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
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

        dataset = FacebookPagePage(
            root=str(self.root_data_dir),
        )
        return dataset
