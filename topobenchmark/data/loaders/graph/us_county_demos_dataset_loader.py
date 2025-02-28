"""Loaders for US County Demos dataset."""

from pathlib import Path

from omegaconf import DictConfig

from topobenchmark.data.datasets import USCountyDemosDataset
from topobenchmark.data.loaders.base import AbstractLoader


class USCountyDemosDatasetLoader(AbstractLoader):
    """Load US County Demos dataset with configurable year and task variable.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - year: Year of the dataset (if applicable)
            - task_variable: Task variable for the dataset
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> USCountyDemosDataset:
        """Load the US County Demos dataset.

        Returns
        -------
        USCountyDemosDataset
            The loaded US County Demos dataset with the appropriate `data_dir`.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = self._initialize_dataset()
        self.data_dir = self._redefine_data_dir(dataset)
        return dataset

    def _initialize_dataset(self) -> USCountyDemosDataset:
        """Initialize the US County Demos dataset.

        Returns
        -------
        USCountyDemosDataset
            The initialized dataset instance.
        """
        return USCountyDemosDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            parameters=self.parameters,
        )

    def _redefine_data_dir(self, dataset: USCountyDemosDataset) -> Path:
        """Redefine the data directory based on the chosen (year, task_variable) pair.

        Parameters
        ----------
        dataset : USCountyDemosDataset
            The dataset instance.

        Returns
        -------
        Path
            The redefined data directory path.
        """
        return dataset.processed_root
