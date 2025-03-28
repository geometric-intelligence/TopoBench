"""Loader for manually loaded graph datasets."""

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from topobench.data.loaders.base import AbstractLoader
from topobench.data.utils import load_manual_graph
from topobench.dataloader import DataloadDataset


class ManualGraphDatasetLoader(AbstractLoader):
    """Load manually provided graph datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_name: Name of the dataset
            - data_dir: Root directory for data
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)
        assert self.parameters.data_dir, (
            "The 'data_dir' parameter must be provided."
        )

    def load_dataset(self) -> Any:
        """Load the manual graph dataset.

        Returns
        -------
        DataloadDataset
            The dataset object containing the manually loaded graph.
        """

        # Load the graph data using the manual graph loader function
        data = load_manual_graph()
        data["num_classes"] = 2

        # Create and return the dataset object
        dataset = DataloadDataset([data])
        return dataset

    def get_data_dir(self) -> Path:
        """Get the data directory.

        Returns
        -------
        Path
            The path to the dataset directory.
        """
        return Path(
            self.parameters.data_dir
        )  # Assuming 'data_dir' is in the config
