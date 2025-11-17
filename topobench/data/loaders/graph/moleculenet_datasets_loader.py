"""Loaders for MoleculeNet datasets."""

from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import MoleculeNet

from topobench.data.loaders.base import AbstractLoader


class MoleculeNetDatasetLoader(AbstractLoader):
    """Load MoleculeNet datasets.

    MoleculeNet is a benchmark collection of molecular property prediction datasets.
    This loader provides access to various molecular datasets including ESOL, FreeSolv,
    Lipophilicity, PCBA, MUV, HIV, BACE, BBBP, Tox21, ToxCast, SIDER, ClinTox, and others.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the MoleculeNet dataset
            - data_type: Type of the dataset (e.g., "molecular_property_prediction")
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load MoleculeNet dataset.

        Returns
        -------
        Dataset
            The loaded MoleculeNet dataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """
        dataset_name = self.parameters.data_name
        self.data_dir = self.parameters.data_dir
        print(
            f"Loading MoleculeNet dataset: {dataset_name} from {self.data_dir}"
        )
        dataset = MoleculeNet(root=self.data_dir, name=dataset_name)
        self._ensure_scalar_targets(dataset)
        return dataset

    @staticmethod
    def _ensure_scalar_targets(dataset: Dataset) -> None:
        """Squeeze singleton label dimensions for scalar targets.

        Some MoleculeNet tasks (e.g., single-target regression) return labels with
        an extra trailing singleton dimension (shape [num_graphs, 1]). Downstream
        training code expects 1D tensors so we squeeze that axis while leaving
        multi-target datasets untouched.

        Parameters
        ----------
        dataset : Dataset
            The dataset to process.
        """

        data_store = getattr(dataset, "_data", None)
        if data_store is None:
            return

        labels = getattr(data_store, "y", None)
        if labels is None or labels.ndim <= 1:
            return

        if labels.size(-1) == 1:
            dataset._data.y = labels.squeeze(-1)
