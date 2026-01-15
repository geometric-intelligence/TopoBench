"""Loaders for Citation Hypergraph dataset."""

from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import FakeDataset

from topobench.data.loaders.base import AbstractLoader


class FakeDatasetLoader(AbstractLoader):
    """Load FakeOnDisk datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration.
    """
    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load FakeOnDiskDataset.

        Returns
        -------
        Dataset
            The loaded FakeOnDiskDataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """
        try:
            dataset = FakeDataset(
                # root=str(self.root_data_dir),
                num_graphs=self.parameters.get("num_graphs", 1),
                avg_num_nodes=self.parameters.get("avg_num_nodes", 1000),
                avg_degree=self.parameters.get("avg_degree", 10.0),
                num_channels=self.parameters.get("num_channels", 64),
                edge_dim=self.parameters.get("edge_dim", 0),
                num_classes=self.parameters.get("num_classes", 10),
                task=self.parameters.get("task", "auto"),
                is_undirected=self.parameters.get("is_undirected", True),
            )
        except Exception as e:
            raise RuntimeError("Failed to load FakeOnDiskDataset") from e
            
        return dataset
