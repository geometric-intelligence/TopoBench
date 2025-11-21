"""Loader for OC20 IS2RE dataset."""

import logging
from pathlib import Path

from omegaconf import DictConfig
from torch_geometric.data import Dataset

from topobench.data.datasets.is2re_dataset import IS2REDataset
from topobench.data.loaders.base import AbstractLoader
from topobench.data.utils.oc20_download import download_is2re_dataset

logger = logging.getLogger(__name__)


class IS2REDatasetLoader(AbstractLoader):
    """Load OC20 IS2RE dataset.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - download: Whether to download if not present (default: False)
            - dtype: Data type for tensors (default: "float32")
            - legacy_format: Use legacy PyG Data format (default: False)
            - max_samples: Limit dataset size for testing (default: None)
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load the IS2RE dataset.

        Returns
        -------
        Dataset
            The loaded IS2RE dataset with the appropriate configuration.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """
        # Download if requested
        if self.parameters.get("download", False):
            self._download_dataset()

        # Initialize LMDB dataset
        dataset = self._initialize_dataset()
        self.data_dir = self._redefine_data_dir(dataset)
        return dataset

    def _download_dataset(self):
        """Download the IS2RE dataset."""
        root = Path(self.get_data_dir())
        download_is2re_dataset(root=root, task="is2re")

    def _initialize_dataset(self) -> IS2REDataset:
        """Initialize the IS2RE dataset.

        Returns
        -------
        IS2REDataset
            The initialized IS2RE dataset.

        Raises
        ------
        RuntimeError
            If dataset initialization fails.
        """
        try:
            dataset = IS2REDataset(
                root=str(self.get_data_dir()),
                name=self.parameters.data_name,
                parameters=self.parameters,
            )
            return dataset
        except Exception as e:
            msg = f"Error initializing IS2RE dataset: {e}"
            raise RuntimeError(msg) from e

    def _redefine_data_dir(self, dataset: Dataset) -> Path:
        """Redefine the data directory based on dataset configuration.

        Parameters
        ----------
        dataset : Dataset
            The IS2RE dataset instance.

        Returns
        -------
        Path
            The redefined data directory path.
        """
        return self.get_data_dir()
