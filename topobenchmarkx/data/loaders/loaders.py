"""Data loaders."""

import os

import numpy as np
import torch_geometric
from omegaconf import DictConfig

from topobenchmarkx.data.datasets import (
    FIXED_SPLITS_DATASETS,
    HETEROPHILIC_DATASETS,
    PLANETOID_DATASETS,
    TU_DATASETS,
    CitationHypergraphDataset,
    USCountyDemosDataset,
    H36MDataset,
)
from topobenchmarkx.data.loaders.base import AbstractLoader
from topobenchmarkx.data.utils import (
    load_cell_complex_dataset,
    load_manual_graph,
    load_simplicial_dataset,
)
from topobenchmarkx.dataloader import DataloadDataset


class GraphLoader(AbstractLoader):
    """Loader for graph datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters.
    **kwargs : dict
        Additional keyword arguments.

    Notes
    -----
    The parameters must contain the following keys:
    - data_dir (str): The directory where the dataset is stored.
    - data_name (str): The name of the dataset.
    - data_type (str): The type of the dataset.
    - split_type (str): The type of split to be used. It can be "fixed", "random", or "k-fold".
    If split_type is "random", the parameters must also contain the following keys:
    - data_seed (int): The seed for the split.
    - data_split_dir (str): The directory where the split is stored.
    - train_prop (float): The proportion of the training set.
    If split_type is "k-fold", the parameters must also contain the following keys:
    - data_split_dir (str): The directory where the split is stored.
    - k (int): The number of folds.
    - data_seed (int): The seed for the split.
    The parameters can be defined in a yaml file and then loaded using `omegaconf.OmegaConf.load('path/to/dataset/config.yaml')`.
    """

    def __init__(self, parameters: DictConfig, **kwargs):
        super().__init__(parameters)
        self.parameters = parameters

    def __repr__(self) -> str:
        """Return a string representation of the GraphLoader object.

        Returns
        -------
        str
            String representation of the GraphLoader object.
        """
        return f"{self.__class__.__name__}(parameters={self.parameters})"

    def load(self) -> tuple[torch_geometric.data.Dataset, str]:
        """Load graph dataset.

        Returns
        -------
        tuple[torch_geometric.data.Dataset, str]
            Tuple containing the loaded data and the data directory.
        """
        # Define the path to the data directory
        root_data_dir = self.parameters["data_dir"]
        data_dir = os.path.join(root_data_dir, self.parameters["data_name"])
        if (
            self.parameters.data_name in PLANETOID_DATASETS
            and self.parameters.data_type == "cocitation"
        ):
            dataset = torch_geometric.datasets.Planetoid(
                root=root_data_dir,
                name=self.parameters["data_name"],
            )

        elif self.parameters.data_name in TU_DATASETS:
            dataset = torch_geometric.datasets.TUDataset(
                root=root_data_dir,
                name=self.parameters["data_name"],
                use_node_attr=False,
            )

        elif self.parameters.data_name in FIXED_SPLITS_DATASETS:
            datasets = []
            for split in ["train", "val", "test"]:
                if self.parameters.data_name == "ZINC":
                    datasets.append(
                        torch_geometric.datasets.ZINC(
                            root=root_data_dir,
                            subset=True,
                            split=split,
                        )
                    )
                elif self.parameters.data_name == "AQSOL":
                    datasets.append(
                        torch_geometric.datasets.AQSOL(
                            root=root_data_dir,
                            split=split,
                        )
                    )
            # The splits are predefined
            # Extract and prepare split_idx
            split_idx = {"train": np.arange(len(datasets[0]))}
            split_idx["valid"] = np.arange(
                len(datasets[0]), len(datasets[0]) + len(datasets[1])
            )
            split_idx["test"] = np.arange(
                len(datasets[0]) + len(datasets[1]),
                len(datasets[0]) + len(datasets[1]) + len(datasets[2]),
            )
            # Join dataset to process it
            dataset = datasets[0] + datasets[1] + datasets[2]
            dataset.split_idx = split_idx
            data_dir = root_data_dir

        elif self.parameters.data_name in HETEROPHILIC_DATASETS:
            dataset = torch_geometric.datasets.HeterophilousGraphDataset(
                root=root_data_dir,
                name=self.parameters["data_name"],
            )

        elif self.parameters.data_name in ["US-county-demos"]:
            dataset = USCountyDemosDataset(
                root=root_data_dir,
                name=self.parameters["data_name"],
                parameters=self.parameters,
            )
            # Need to redefine data_dir for the (year, task_variable) pair chosen
            data_dir = dataset.processed_root

        elif self.parameters.data_name in ["H36MDataset"]:
            dataset = H36MDataset(
                root=root_data_dir,
                name=self.parameters["data_name"],
                parameters=self.parameters,
            )

            data_dir = dataset.processed_root

        elif self.parameters.data_name in ["manual"]:
            data = load_manual_graph()
            dataset = DataloadDataset([data], data_dir)

        else:
            raise NotImplementedError(
                f"Dataset {self.parameters.data_name} not implemented"
            )

        return dataset, data_dir


class HypergraphLoader(AbstractLoader):
    r"""Loader for hypergraph datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters.
    """

    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)
        self.parameters = parameters

    def load(
        self,
    ) -> torch_geometric.data.Dataset:
        r"""Load hypergraph dataset.

        Returns
        -------
        torch_geometric.data.Dataset
            A torch_geometric.data.Dataset object containing the loaded data.
        """

        root_data_dir = self.parameters["data_dir"]
        if self.parameters.data_name in [
            "coauthorship_cora",
            "coauthorship_dblp",
            "cocitation_citeseer",
            "cocitation_cora",
            "cocitation_pubmed",
        ]:
            dataset = CitationHypergraphDataset(
                root=root_data_dir,
                name=self.parameters["data_name"],
                parameters=self.parameters,
            )

            data_dir = os.path.join(
                root_data_dir, self.parameters["data_name"]
            )

        return dataset, data_dir


class CellComplexLoader(AbstractLoader):
    """Loader for cell complex datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters.
    """

    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)
        self.parameters = parameters

    def load(
        self,
    ) -> torch_geometric.data.Dataset:
        """Load cell complex dataset.

        Returns
        -------
        torch_geometric.data.Dataset
            Dataset object containing the loaded data.
        """
        return load_cell_complex_dataset(self.parameters)


class SimplicialLoader(AbstractLoader):
    """Loader for simplicial datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters.

    Returns
    -------
    torch_geometric.data.Dataset
        torch_geometric.data.Dataset object containing the loaded data.
    """

    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)
        self.parameters = parameters

    def load(
        self,
    ) -> torch_geometric.data.Dataset:
        """Load simplicial dataset.

        Returns
        -------
        torch_geometric.data.Dataset
            Dataset object containing the loaded data.
        """
        return load_simplicial_dataset(self.parameters)
