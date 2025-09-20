"""Loaders for TU datasets."""

import torch_geometric.transforms as T
from graph_universe import GraphUniverseDataset
from omegaconf import DictConfig
from torch_geometric.data import Data, Dataset

from topobench.data.loaders.base import AbstractLoader


class GraphUniverseDatasetLoader(AbstractLoader):
    """Load Graph Universe datasets.

    Parameters
    ----------
    task : str
        The task to be performed (e.g., "community_detection").
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - data_type: Type of the dataset (e.g., "graph_classification")
    """

    def __init__(self, task: str, parameters: DictConfig) -> None:
        self.task = task
        super().__init__(parameters)
        if self.parameters.get("num_nodes_range") is not None:
            self.parameters["generation_parameters"]["family_parameters"]["min_n_nodes"] = self.parameters.get("num_nodes_range")[0]
            self.parameters["generation_parameters"]["family_parameters"]["max_n_nodes"] = self.parameters.get("num_nodes_range")[1]

    def load_dataset(self) -> Dataset:
        """Load Graph Universe dataset.

        Returns
        -------
        Dataset
            The loaded Graph Universe dataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = GraphUniverseDataset(
            root=str(self.root_data_dir),
            parameters=self.parameters["generation_parameters"],
        )
        dataset.transform = DefineLabels(self.task)
        # dataset.y = getattr(dataset, self.task)
        # dataset.slices["y"] = dataset.slices[self.task]
        # for idx in range(len(dataset)):
        #     y = getattr(dataset[idx], self.task)
        #     setattr(dataset[idx], "y", y)
        #     #dataset[idx].y = y

        return dataset

    def load(self, **kwargs) -> tuple[Data, str]:
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
        data_dir = dataset.raw_dir

        return dataset, data_dir


class DefineLabels(T.BaseTransform):
    """Define labels for the dataset.

    Parameters
    ----------
    task : str
        The task to be performed (e.g., "community_detection").
    """

    def __init__(self, task):
        self.task = task

    def forward(self, data: Data) -> Data:
        """Forward pass to define labels.

        Parameters
        ----------
        data : Data
            The input data.

        Returns
        -------
        Data
            The transformed data with defined labels.
        """
        label = getattr(data, self.task)
        data.y = label
        return data
