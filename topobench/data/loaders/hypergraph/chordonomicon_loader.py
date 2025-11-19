"""Loader for Chordonomicon dataset."""

from topobench.data.datasets import ChordonomiconDataset
from topobench.data.loaders.base import AbstractLoader


class ChordonomiconDatasetLoader(AbstractLoader):
    """Loader class for Chordonomicon dataset.

    Args:
        - parameters (DictConfig): Loader parameters.
            - data_dir (str): Root directory where the dataset folder is stored.
            - data_name (str): Name of the dataset.
    """

    def load_dataset(self) -> ChordonomiconDataset:
        """Load the Chordonomicon dataset.

        Returns
        -------
        ChordonomiconDataset
            The loaded Chordonomicon dataset.
        """
        return ChordonomiconDataset(
            data_dir=self.root_data_dir, data_name=self.parameters.data_name
        )
