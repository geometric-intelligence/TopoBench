"""CityNetwork dataset loader implementation."""

from pathlib import Path

from omegaconf import DictConfig

from topobench.data.datasets.citynetwork_dataset import CityNetworkDataset
from topobench.data.loaders.base import AbstractLoader


class CityNetworkDatasetLoader(AbstractLoader):
    """
    Loader for CityNetwork datasets.

    Parameters
    ----------
    config : DictConfig
        Configuration object containing dataset parameters.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _initialize_dataset(self) -> CityNetworkDataset:
        """
        Initialize the CityNetwork dataset.

        Returns
        -------
        CityNetworkDataset
            Initialized CityNetwork dataset instance.
        """
        return CityNetworkDataset(
            root=str(self.config.data_dir),
            name=self.config.get("name", "paris"),
            augmented=self.config.get("augmented", True),
        )

    def _redefine_data_dir(self) -> None:
        """Redefine the data directory path."""
        self.config.data_dir = (
            Path(self.config.data_dir)
            / "CityNetwork"
            / self.config.get("name", "paris")
        )
