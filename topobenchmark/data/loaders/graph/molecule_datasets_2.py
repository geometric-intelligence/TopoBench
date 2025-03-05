"""Loaders for Molecule datasets (QM9)."""

from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import QM9

from topobenchmark.data.datasets.qm9_custom_dataset import QM9Custom
from topobenchmark.data.loaders.base import AbstractLoader


class MoleculeDatasetLoader2(AbstractLoader):
    """Load molecule datasets (QM9, ).

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - data_type: Type of the dataset (e.g., "molecule")
            - max_ring_size: Maximum ring size for the QM9Custom dataset
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load the molecule dataset with predefined splits.

        Returns
        -------
        Dataset
            The combined dataset with predefined splits.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """
        if self.parameters.data_name == "QM9":
            dataset = QM9(
                root=str(self.root_data_dir),
            )
        elif self.parameters.data_name == "QM9Custom":
            dataset = QM9Custom(
                root=str(self.root_data_dir),
                max_ring_size=self.parameters.max_ring_size,
            )
        else:
            raise ValueError(f"Unknown dataset: {self.parameters.data_name}")

        return dataset
