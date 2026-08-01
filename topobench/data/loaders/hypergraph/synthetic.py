"""Loader for the deterministic native synthetic hypergraph dataset."""

from omegaconf import DictConfig
from torch_geometric.data import Dataset

from topobench.data.datasets.synthetic_hypergraph_dataset import (
    SyntheticHypergraphDataset,
)
from topobench.data.loaders.base import AbstractLoader


class SyntheticHypergraphDatasetLoader(AbstractLoader):
    """Load a packaged native hypergraph without downloads or cache files."""

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Construct the configured deterministic hypergraph fixture."""
        data_name = str(self.parameters.data_name)
        if data_name != "SyntheticHypergraph":
            raise ValueError(
                f"unsupported synthetic hypergraph dataset: {data_name!r}"
            )
        return SyntheticHypergraphDataset(
            seed=int(self.parameters.get("seed", 0)),
            num_nodes=int(self.parameters.get("num_nodes", 12)),
            num_hyperedges=int(self.parameters.get("num_hyperedges", 5)),
        )


__all__ = ["SyntheticHypergraphDatasetLoader"]
