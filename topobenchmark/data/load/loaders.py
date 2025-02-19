import os
from collections.abc import Callable

import numpy as np
import rootutils
import torch
import torch_geometric
from omegaconf import DictConfig

# silent RDKit warnings
from rdkit import Chem, RDLogger

from modules.data.load.base import AbstractLoader
from modules.data.utils.concat2geometric_dataset import (
    ConcatToGeometricDataset,
)
from modules.data.utils.custom_dataset import CustomDataset
from modules.data.utils.utils import (
    load_cell_complex_dataset,
    load_gudhi_dataset,
    load_hypergraph_pickle_dataset,
    load_manual_graph,
    load_manual_mol,
    load_manual_points,
    load_point_cloud,
    load_random_points,
    load_simplicial_dataset,
)

RDLogger.DisableLog("rdApp.*")


class GraphLoader(AbstractLoader):
    r"""Loader for graph datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters.
    """

    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)
        self.parameters = parameters

    def is_valid_smiles(self, smiles):
        """Check if a SMILES string is valid using RDKit."""
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None

    def filter_qm9_dataset(self, dataset):
        """Filter the QM9 dataset to remove invalid SMILES strings."""
        return [data for data in dataset if self.is_valid_smiles(data.smiles)]

    def load(self) -> torch_geometric.data.Dataset:
        r"""Load graph dataset.

        Parameters
        ----------
        None

        Returns
        -------
        torch_geometric.data.Dataset
            torch_geometric.data.Dataset object containing the loaded data.
        """
        # Define the path to the data directory
        root_folder = rootutils.find_root()
        root_data_dir = os.path.join(root_folder, self.parameters["data_dir"])

        self.data_dir = os.path.join(
            root_data_dir, self.parameters["data_name"]
        )
        if (
            self.parameters.data_name.lower() in ["cora", "citeseer", "pubmed"]
            and self.parameters.data_type == "cocitation"
        ):
            dataset = torch_geometric.datasets.Planetoid(
                root=root_data_dir,
                name=self.parameters["data_name"],
            )

        elif self.parameters.data_name in [
            "MUTAG",
            "ENZYMES",
            "PROTEINS",
            "COLLAB",
            "IMDB-BINARY",
            "IMDB-MULTI",
            "REDDIT-BINARY",
            "NCI1",
            "NCI109",
        ]:
            dataset = torch_geometric.datasets.TUDataset(
                root=root_data_dir,
                name=self.parameters["data_name"],
                use_node_attr=False,
            )

        elif self.parameters.data_name in ["ZINC", "AQSOL"]:
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
            dataset = ConcatToGeometricDataset(dataset)

        elif self.parameters.data_name == "QM9":
            dataset = torch_geometric.datasets.QM9(root=root_data_dir)
            # Filter the QM9 dataset to remove invalid SMILES strings
            valid_dataset = self.filter_qm9_dataset(dataset)
            dataset = CustomDataset(valid_dataset, self.data_dir)

        elif self.parameters.data_name in ["manual"]:
            data = load_manual_graph()
            dataset = CustomDataset([data], self.data_dir)

        elif self.parameters.data_name in ["manual_rings"]:
            data = load_manual_mol()
            dataset = CustomDataset([data], self.data_dir)

        else:
            raise NotImplementedError(
                f"Dataset {self.parameters.data_name} not implemented"
            )

        return dataset


class CellComplexLoader(AbstractLoader):
    r"""Loader for cell complex datasets.

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
        r"""Load cell complex dataset.

        Parameters
        ----------
        None

        Returns
        -------
        torch_geometric.data.Dataset
            torch_geometric.data.Dataset object containing the loaded data.
        """
        return load_cell_complex_dataset(self.parameters)


class SimplicialLoader(AbstractLoader):
    r"""Loader for simplicial datasets.

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
        r"""Load simplicial dataset.

        Parameters
        ----------
        None

        Returns
        -------
        torch_geometric.data.Dataset
            torch_geometric.data.Dataset object containing the loaded data.
        """
        return load_simplicial_dataset(self.parameters)


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

        Parameters
        ----------
        None

        Returns
        -------
        torch_geometric.data.Dataset
            torch_geometric.data.Dataset object containing the loaded data.
        """
        return load_hypergraph_pickle_dataset(self.parameters)


class PointCloudLoader(AbstractLoader):
    r"""Loader for point cloud datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters.
    feature_generator: Optional[Callable[[torch.Tensor], torch.Tensor]]
        Function to generate the dataset features. If None, no features added.
    target_generator: Optional[Callable[[torch.Tensor], torch.Tensor]]
        Function to generate the target variable. If None, no target added.
    """

    def __init__(
        self,
        parameters: DictConfig,
        feature_generator: Callable[[torch.Tensor], torch.Tensor]
        | None = None,
        target_generator: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.feature_generator = feature_generator
        self.target_generator = target_generator
        super().__init__(parameters)
        self.parameters = parameters

    def load(self) -> torch_geometric.data.Dataset:
        r"""Load point cloud dataset.

        Parameters
        ----------
        None


        Returns
        -------
        torch_geometric.data.Dataset
            torch_geometric.data.Dataset object containing the loaded data.
        """
        # Define the path to the data directory
        root_folder = rootutils.find_root()
        root_data_dir = os.path.join(root_folder, self.parameters["data_dir"])
        self.data_dir = os.path.join(
            root_data_dir, self.parameters["data_name"]
        )

        if self.parameters.data_name.startswith("gudhi_"):
            data = load_gudhi_dataset(
                self.parameters,
                feature_generator=self.feature_generator,
                target_generator=self.target_generator,
            )
        elif self.parameters.data_name == "random_points":
            data = load_random_points(
                dim=self.parameters["dim"],
                num_classes=self.parameters["num_classes"],
                num_samples=self.parameters["num_samples"],
            )
        elif self.parameters.data_name == "toy_point_cloud":
            data = load_point_cloud(
                num_classes=self.parameters["num_classes"],
                num_samples=self.parameters["num_samples"],
            )
        elif self.parameters.data_name == "manual_points":
            data = load_manual_points()
        else:
            raise NotImplementedError(
                f"Dataset {self.parameters.data_name} not implemented"
            )

        return CustomDataset([data], self.data_dir)
