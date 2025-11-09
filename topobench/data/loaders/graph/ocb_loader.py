"""OCB Circuit Dataset Loaders for TopoBench."""

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from topobench.data.datasets.ocb_dataset import OCB101Dataset, OCB301Dataset
from topobench.data.loaders.base import AbstractLoader


class OCBDatasetLoader(AbstractLoader):
    """Base loader for OCB datasets.

    Parameters
    ----------
    parameters : DictConfig
        A dictionary-like object containing parameters for the dataset loader.
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)
        self.dataset_class = None  # To be defined in subclasses

    def load_dataset(self) -> Any:
        """Load and return the specified OCB dataset.

        Returns
        -------
        Any
            The loaded OCB dataset.

        Raises
        ------
        RuntimeError
            If there is an error loading the dataset.
        """
        try:
            dataset = self._initialize_dataset()
            self.data_dir = self._redefine_data_dir(dataset)
            return dataset
        except Exception as e:
            raise RuntimeError(
                f"Error loading {self.parameters.data_name} dataset: {e}"
            ) from e

    def _initialize_dataset(self) -> Any:
        """Initialize the dataset instance.

        Raises
        ------
        NotImplementedError
            If `dataset_class` is not set in the subclass.

        Returns
        -------
        Any
            An instance of the dataset class.
        """
        if self.dataset_class is None:
            raise NotImplementedError(
                "dataset_class must be set in the subclass."
            )

        # The root directory will be .../data/graph/circuits/
        # The OCB101Dataset will create a .../circuits/OCB101/ folder within it
        return self.dataset_class(
            root=str(self.get_data_dir()), parameters=self.parameters
        )

    def _redefine_data_dir(self, dataset: Any) -> Path:
        """Redefine data directory to the processed data path.

        Parameters
        ----------
        dataset : Any
            The dataset instance.

        Returns
        -------
        Path
            The redefined data directory path.
        """
        # This ensures the framework looks for data in the correct processed folder
        # e.g., .../data/graph/circuits/OCB101/processed
        return Path(dataset.processed_dir)


class OCB101DatasetLoader(OCBDatasetLoader):
    """Loader for OCB101 circuit graph dataset.

    Parameters
    ----------
    parameters : DictConfig
        A dictionary-like object containing parameters for the dataset loader.
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)
        self.dataset_class = OCB101Dataset


class OCB301DatasetLoader(OCBDatasetLoader):
    """Loader for OCB301 circuit graph dataset.

    Parameters
    ----------
    parameters : DictConfig
        A dictionary-like object containing parameters for the dataset loader.
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)
        self.dataset_class = OCB301Dataset
