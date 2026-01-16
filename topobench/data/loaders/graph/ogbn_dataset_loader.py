"""Loader for OGB node property prediction datasets."""

from pathlib import Path

import torch
from ogb.nodeproppred import PygNodePropPredDataset
from omegaconf import DictConfig

from topobench.data.loaders.base import AbstractLoader


class OGBNDatasetLoader(AbstractLoader):
    """Load OGB node property prediction datasets (ogbn-arxiv, ogbn-products).

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing data_dir and data_name.
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self, **kwargs) -> PygNodePropPredDataset:
        """Load an OGB node property prediction dataset.

        Additional keyword arguments are accepted for API compatibility with
        other loaders (e.g. ``slice`` used in tests for long-running datasets),
        but are currently ignored because OGBN datasets are represented as a
        single large graph.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments accepted for API compatibility.

        Returns
        -------
        PygNodePropPredDataset
            The loaded OGBN dataset.
        """
        dataset = self._initialize_dataset()
        self.data_dir = self._redefine_data_dir(dataset)

        # Conver attributes to float
        dataset._data.x = dataset._data.x.to(torch.float)
        # Squeeze the target tensor
        dataset._data.y = dataset._data.y.squeeze(1)
        dataset.split_idx = dataset.get_idx_split()

        return dataset

    def _initialize_dataset(self) -> PygNodePropPredDataset:
        """Initialize the OGBN dataset specified by ``parameters.data_name``.

        Returns
        -------
        PygNodePropPredDataset
            The initialized dataset instance.
        """
        return PygNodePropPredDataset(
            name=self.parameters.data_name, root=str(self.root_data_dir)
        )

    def _redefine_data_dir(self, dataset: PygNodePropPredDataset) -> Path:
        """Redefine the data directory for the OGBN dataset.

        Parameters
        ----------
        dataset : PygNodePropPredDataset
            The dataset instance.

        Returns
        -------
        Path
            The processed root directory path.
        """
        return Path(dataset.root) / dataset.name / "processed"
