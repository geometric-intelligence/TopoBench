from abc import ABC, abstractmethod

import torch_geometric
from omegaconf import DictConfig

# logger = logging.getLogger(__name__)


class AbstractLoader(ABC):
    """Abstract class that provides an interface to load data.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters.
    """

    def __init__(self, parameters: DictConfig):
        self.cfg = parameters

    @abstractmethod
    def load(
        self,
    ) -> torch_geometric.data.Data:
        """Load data into Data.

        Parameters
        ----------
        None

        Returns
        -------
        Data
            Data object containing the loaded data.
        """
        raise NotImplementedError
