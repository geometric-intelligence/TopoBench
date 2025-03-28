"""Loaders for GeometricShapes datasets."""

from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import GeometricShapes

from topobench.data.loaders.base import AbstractLoader


def rename_pos_to_x(data):
    """Rename the 'pos' attribute to 'x' in a PyG Data object.

    This function is needed as a pre_transform for the GeometricShapes dataset so that the 'pos' attribute is renamed to 'x' properly.

    Parameters
    ----------
    data : torch_geometric.data.Data
        The input data.

    Returns
    -------
    torch_geometric.data.Data
        The data with the 'pos' attribute renamed to 'x'.
    """
    if hasattr(data, "pos"):
        data.x = data.pos
        del data.pos
    return data


class GeometricShapesDatasetLoader(AbstractLoader):
    """Load GeometricShapes dataset.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load GeometricShapes dataset.

        Returns
        -------
        Dataset
            The loaded GeometricShapes dataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """
        train_split = [True, False]
        datasets = []

        for split in train_split:
            datasets.append(  # noqa: PERF401
                GeometricShapes(
                    root=str(self.root_data_dir),
                    train=split,
                    pre_transform=rename_pos_to_x,
                )
            )

        dataset = datasets[0] + datasets[1]

        return dataset
