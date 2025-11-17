"""Loader for the OGBN-Arxiv dataset."""

from pathlib import Path

from ogb.nodeproppred import PygNodePropPredDataset
from omegaconf import DictConfig

from topobench.data.loaders.base import AbstractLoader


class OgbnArxivDatasetLoader(AbstractLoader):
    """Load the OGBN-Arxiv dataset.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing data_dir and data_name.
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> PygNodePropPredDataset:
        """Load the OGBN-Arxiv dataset.

        Returns
        -------
        PygNodePropPredDataset
            The loaded OGBN-Arxiv dataset.
        """
        dataset = self._initialize_dataset()
        self.data_dir = self._redefine_data_dir(dataset)
        return dataset

    def _initialize_dataset(self) -> PygNodePropPredDataset:
        """Initialize the OGBN-Arxiv dataset.

        Returns
        -------
        PygNodePropPredDataset
            The initialized dataset instance.
        """
        return PygNodePropPredDataset(
            name=self.parameters.data_name, root=str(self.root_data_dir)
        )

    def _redefine_data_dir(self, dataset: PygNodePropPredDataset) -> Path:
        """Redefine the data directory for OGBN-Arxiv dataset.

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
