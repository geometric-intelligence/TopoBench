"""Loader for OC22 IS2RE dataset."""

import logging
from pathlib import Path

from omegaconf import DictConfig
from torch_geometric.data import Dataset

from topobench.data.datasets.oc22_is2re_dataset import OC22IS2REDataset
from topobench.data.loaders.base import AbstractLoader
from topobench.data.utils.oc20_download import download_is2re_dataset

logger = logging.getLogger(__name__)


class OC22IS2REDatasetLoader(AbstractLoader):
    """Load OC22 IS2RE dataset.

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
        """Load the OC22 IS2RE dataset.

        Returns
        -------
        Dataset
            The loaded OC22 IS2RE dataset with the appropriate configuration.

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
        """Download the OC22 IS2RE dataset."""
        root = Path(self.get_data_dir())
        download_is2re_dataset(root=root, task="oc22_is2re")

    def _initialize_dataset(self) -> OC22IS2REDataset:
        """Initialize the OC22 IS2RE dataset.

        Returns
        -------
        OC22IS2REDataset
            The initialized OC22 IS2RE dataset.

        Raises
        ------
        RuntimeError
            If dataset initialization fails.
        """
        try:
            dataset = OC22IS2REDataset(
                root=str(self.get_data_dir()),
                name=self.parameters.data_name,
                parameters=self.parameters,
            )
            return dataset
        except Exception as e:
            msg = f"Error initializing OC22 IS2RE dataset: {e}"
            raise RuntimeError(msg) from e

    def _redefine_data_dir(self, dataset: Dataset) -> Path:
        """Redefine the data directory based on dataset configuration.

        Parameters
        ----------
        dataset : Dataset
            The OC22 IS2RE dataset instance.

        Returns
        -------
        Path
            The redefined data directory path.
        """
        return self.get_data_dir()
