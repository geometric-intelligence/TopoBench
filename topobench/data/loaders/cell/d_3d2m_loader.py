"""Loader for 3D2M Cell dataset."""


from omegaconf import DictConfig
from torch_geometric.data import Dataset

from topobench.data.datasets.d_3d2m_dataset import (
    D3D2MDataset,
)
from topobench.data.loaders.base import AbstractLoader


class D3D2MDatasetLoader(AbstractLoader):
    """Load 3D2M Cell dataset.
    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load 3D2M Cell dataset.
        Returns
        -------
        Dataset
            The loaded 3D2M Cell dataset.
        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = D3D2MDataset(
            root=str(self.root_data_dir),
                name=self.parameters.data_name,
                parameters=self.parameters,
        )
        return dataset
