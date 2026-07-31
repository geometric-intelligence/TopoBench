"""Loader for deterministic native homogeneous graph fixtures."""

from omegaconf import DictConfig
from torch_geometric.data import Dataset

from topobench.data.datasets.synthetic_graph_dataset import (
    SyntheticGraphDataset,
    SyntheticGraphTask,
)
from topobench.data.loaders.base import AbstractLoader

_DATASET_TASKS: dict[str, SyntheticGraphTask] = {
    "SyntheticGraph": "graph_classification",
    "SyntheticGraphRegression": "graph_regression",
    "SyntheticNodeGraph": "node_classification",
}


class SyntheticGraphDatasetLoader(AbstractLoader):
    """Load a packaged native homogeneous graph fixture."""

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Construct the fixture selected by the configured dataset name."""
        data_name = str(self.parameters.data_name)
        try:
            task = _DATASET_TASKS[data_name]
        except KeyError as error:
            raise ValueError(
                f"unsupported synthetic graph dataset: {data_name!r}"
            ) from error
        return SyntheticGraphDataset(
            task=task,
            seed=int(self.parameters.get("seed", 0)),
        )


__all__ = ["SyntheticGraphDatasetLoader"]
