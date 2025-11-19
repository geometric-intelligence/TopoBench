"""Loader for MUSAE Deezer Europe dataset."""


from omegaconf import DictConfig

from topobench.data.datasets import MusaeDeezerEuropeDataset
from topobench.data.loaders.base import AbstractLoader


class MusaeDeezerEuropeDatasetLoader(AbstractLoader):
    """Load MUSAE Deezer Europe dataset.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> MusaeDeezerEuropeDataset:
        """Load MUSAE Deezer Europe dataset.

        Returns
        -------
        Dataset
            The loaded MUSAE Deezer Europe dataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = MusaeDeezerEuropeDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            parameters=self.parameters,
        )
        return dataset
