"""Loader for Metamath proof graph dataset."""

from pathlib import Path
from omegaconf import DictConfig
from torch_geometric.data import Dataset

from topobench.data.datasets import MetamathDataset
from topobench.data.loaders.base import AbstractLoader


class MetamathDatasetLoader(AbstractLoader):
    """Thin wrapper around MetamathDataset for TopoBench."""

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """
        Initialize the MetamathDataset and expose its processed_dir
        via self.data_dir (for split utils / logging).
        """
        dataset = MetamathDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            parameters=self.parameters,
        )

        # Point data_dir to processed folder for downstream utilities
        self.data_dir = Path(dataset.processed_dir)

        # No label collapsing or masking here; dataset is ready to use.
        # Splits are handled via dataset.split_idx + split_utils (fixed split).
        return dataset
