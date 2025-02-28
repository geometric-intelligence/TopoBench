"""Abstract Loader class."""

import os
from abc import ABC, abstractmethod
from pathlib import Path

import torch_geometric
from omegaconf import DictConfig


class AbstractLoader(ABC):
    """Abstract class that provides an interface to load data.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters.
    """

    def __init__(self, parameters: DictConfig):
        self.parameters = parameters
        self.root_data_dir = Path(parameters["data_dir"])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(parameters={self.parameters})"

    def get_data_dir(self) -> Path:
        """Get the data directory.

        Returns
        -------
        Path
            The path to the dataset directory.
        """
        return os.path.join(self.root_data_dir, self.parameters.data_name)

    @abstractmethod
    def load_dataset(self) -> torch_geometric.data.Data:
        """Load data into Data.

        Raises
        ------
        NotImplementedError
            If the method is not implemented.
        """
        raise NotImplementedError

    def load(self, **kwargs) -> tuple[torch_geometric.data.Data, str]:
        """Load data.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        tuple[torch_geometric.data.Data, str]
            Tuple containing the loaded data and the data directory.
        """
        dataset = self.load_dataset(**kwargs)
        data_dir = self.get_data_dir()

        return dataset, data_dir
