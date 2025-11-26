"""Loaders for MIPLIB dataset."""

from omegaconf import DictConfig

from topobench.data.datasets import MIPLIBDataset
from topobench.data.loaders.base import AbstractLoader


class MIPLIBDatasetLoader(AbstractLoader):
    """Load MIPLIB dataset with configurable parameters.

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

    def load_dataset(self, **kwargs) -> MIPLIBDataset:
        """Load the MIPLIB dataset.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments (e.g., slice for testing).

        Returns
        -------
        MIPLIBDataset
            The loaded MIPLIB dataset with the appropriate `data_dir`.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = self._initialize_dataset(**kwargs)
        self.data_dir = self.get_data_dir()
        return dataset

    def _initialize_dataset(self, **kwargs) -> MIPLIBDataset:
        """Initialize the MIPLIB dataset.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments (e.g., slice for testing).

        Returns
        -------
        MIPLIBDataset
            The initialized dataset instance.
        """
        return MIPLIBDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            parameters=self.parameters,
            **kwargs,
        )
