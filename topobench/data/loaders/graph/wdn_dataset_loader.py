"""Loader for Water Distribution Network dataset."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig
from torch_geometric.data import Dataset

from topobench.data.datasets.wdn_dataset import (
    AnytownDataset,
    BalermanDataset,
    CTownDataset,
    DTownDataset,
    EXNDataset,
    KY1Dataset,
    KY6Dataset,
    KY8Dataset,
    LTownDataset,
    ModenaDataset,
)
from topobench.data.loaders.base import AbstractLoader


class WDNDatasetLoader(AbstractLoader):
    """
    Load WDN dataset.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_name: Name of the dataset
            - data_dir: Root directory for data
            - regressors: Observed variables
            - target: Target variables of the regression problem
            - temporal: Whether to consider the temporal dimension or not
            - num_scenarios: Number of scenarios to be considered
            - num_instants: Number of observations to be considered within each scenario.
    """

    # This map routes a data_name to a class of WDNDataset

    _DATASETS: dict[str, type[Any]] = {
        "anytown": AnytownDataset,
        "balerman": BalermanDataset,
        "ctown": CTownDataset,
        "dtown": DTownDataset,
        "exn": EXNDataset,
        "ky1": KY1Dataset,
        "ky6": KY6Dataset,
        "ky8": KY8Dataset,
        "ltown": LTownDataset,
        "modena": ModenaDataset,
    }

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """
        Load the chosen WDN dataset.

        Returns
        -------
        WDNDataset
            The loaded WDN dataset with the appropriate `data_dir`.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """
        name = self.parameters.data_name.lower()

        try:
            dataset_cls = type(self)._DATASETS[name]
        except KeyError as err:
            raise RuntimeError(
                f"Unknown dataset '{name}'. "
                f"Available datasets: {list(type(self)._DATASETS.keys())}"
            ) from err

        return dataset_cls(
            root=str(self.root_data_dir),
            parameters=self.parameters,
        )
