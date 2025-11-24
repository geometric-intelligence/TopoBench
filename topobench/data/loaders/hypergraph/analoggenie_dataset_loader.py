"""Loader for AnalogGenie dataset."""

from omegaconf import DictConfig

from topobench.data.datasets.analoggenie_datasets import AnalogGenieDataset
from topobench.data.loaders.base import AbstractLoader


class AnalogGenieDatasetLoader(AbstractLoader):
    """Load AnalogGenie dataset with configurable parameters.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - other relevant parameters
    cfg : DictConfig, optional
        A DictConfig object containing configuration for the dataset itself,
        including parameters for the dataset and split_params.
    """

    def __init__(
        self, parameters: DictConfig, cfg: DictConfig | None = None
    ) -> None:
        super().__init__(parameters)
        self.cfg = cfg  # Store the cfg for dataset initialization

    def load_dataset(self) -> AnalogGenieDataset:
        """Load the AnalogGenie dataset.

        Returns
        -------
        AnalogGenieDataset
            The loaded AnalogGenie dataset with the appropriate `data_dir`.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = self._initialize_dataset()
        self.data_dir = self.get_data_dir()
        return dataset

    def _initialize_dataset(self) -> AnalogGenieDataset:
        """Initialize the AnalogGenie dataset.

        Returns
        -------
        AnalogGenieDataset
            The initialized dataset instance.
        """
        return AnalogGenieDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            parameters=self.cfg.parameters if self.cfg is not None else None,
        )
