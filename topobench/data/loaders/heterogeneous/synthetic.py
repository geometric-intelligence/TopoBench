"""Loader for the deterministic native heterogeneous debug dataset."""

from omegaconf import DictConfig
from torch_geometric.data import Dataset

from topobench.data.datasets import SyntheticHeterogeneousDataset
from topobench.data.loaders.base import AbstractLoader


class SyntheticHeterogeneousDatasetLoader(AbstractLoader):
    """Load the deterministic native heterogeneous debug graph.

    Parameters
    ----------
    parameters : DictConfig
        Loader configuration containing the data directory, dataset name,
        random seed, and optional synthetic node counts.
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Construct the one-graph synthetic heterogeneous dataset.

        Returns
        -------
        Dataset
            Deterministic native heterogeneous graph dataset.
        """
        return SyntheticHeterogeneousDataset(
            seed=int(self.parameters.seed),
            num_authors=int(self.parameters.get("num_authors", 36)),
            num_papers=int(self.parameters.get("num_papers", 24)),
            num_venues=int(self.parameters.get("num_venues", 6)),
        )
