"""Loader for MUSAE GitHub dataset."""


from omegaconf import DictConfig

from topobench.data.datasets import MusaeGitHubDataset
from topobench.data.loaders.base import AbstractLoader


class MusaeGitHubDatasetLoader(AbstractLoader):
    """Load MUSAE GitHub dataset.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> MusaeGitHubDataset:
        """Load MUSAE GitHub dataset.

        Returns
        -------
        Dataset
            The loaded MUSAE GitHub dataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = MusaeGitHubDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            parameters=self.parameters,
        )
        return dataset
