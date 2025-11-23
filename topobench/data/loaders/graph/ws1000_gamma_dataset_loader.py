# location: topobench/data/loaders/graph/ws1000_gamma_dataset_loader.py

from pathlib import Path
from omegaconf import DictConfig

from topobench.data.datasets import WS1000GammaDataset
from topobench.data.loaders.base import AbstractLoader


class WS1000GammaDatasetLoader(AbstractLoader):
    """Loader for the WS1000_gamma synthetic dataset.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing at least:
            - data_dir
            - data_name
            - num_nodes
            - feature_dim
            - mean_degree
            - beta
            - gamma
            - noise_scale
            - seed
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> WS1000GammaDataset:
        """Main method called by TopoBench: instantiate dataset and set data_dir."""
        dataset = self._initialize_dataset()
        self.data_dir = self._redefine_data_dir(dataset)

        return dataset

    def _initialize_dataset(self) -> WS1000GammaDataset:
        """Helper to instantiate the WS1000GammaDataset."""
        return WS1000GammaDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            parameters=self.parameters,
        )

    def _redefine_data_dir(self, dataset: WS1000GammaDataset) -> Path:
        """Return the processed root folder as dataset directory."""
        return Path(dataset.processed_dir)
