"""Loaders for GraphUniverse [1] datasets.

[1] "GraphUniverse: Enabling Systematic Evaluation of Inductive Generalization" by Louis Van Langendonck and Guillermo Bernardez and Nina Miolane and Pere Barlet-Ros
Accepted at The Fourteenth International Conference on Learning Representations, 2026},
https://openreview.net/forum?id=jRWxvQnqUt
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from graph_universe import GraphUniverseDataset
from omegaconf import DictConfig
from torch_geometric.data import Data, Dataset

from topobench.data.loaders.base import AbstractLoader, resolve_cache_config

_GRAPH_UNIVERSE_REPRESENTATION_VERSION = "pyg-data-v1"
_GRAPH_UNIVERSE_PARSER_VERSION = "graph-universe-v1"
_GENERATION_TIMEOUT_SECONDS = 3600
_GENERATION_SCRIPT = """
import json
import sys

from graph_universe import GraphUniverseDataset
import torch

with open(sys.argv[2], encoding="utf-8") as file:
    parameters = json.load(file)
dataset = GraphUniverseDataset(root=sys.argv[1], parameters=parameters)
torch.save(dataset, sys.argv[3])
"""


def _materialize_in_isolated_process(
    canonical_data_dir: Path,
    parameters: dict,
) -> GraphUniverseDataset:
    """Generate in a disposable root and return only the in-memory dataset."""
    canonical_data_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".graph-universe-",
        dir=canonical_data_dir.parent,
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        source_root = temporary_root / "source"
        parameters_path = temporary_root / "parameters.json"
        result_path = temporary_root / "result.pt"
        parameters_path.write_text(
            json.dumps(parameters, allow_nan=False, sort_keys=True),
            encoding="utf-8",
        )
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as errors:
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        _GENERATION_SCRIPT,
                        str(source_root),
                        str(parameters_path),
                        str(result_path),
                    ],
                    check=True,
                    stderr=errors,
                    timeout=_GENERATION_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    "GraphUniverse generation did not finish within "
                    f"{_GENERATION_TIMEOUT_SECONDS} seconds"
                ) from error
            except subprocess.CalledProcessError as error:
                errors.seek(0)
                details = errors.read().strip()
                message = (
                    "GraphUniverse generation failed in isolated process "
                    f"with exit code {error.returncode}"
                )
                if details:
                    message = f"{message}:\n{details}"
                raise RuntimeError(message) from error
        dataset = torch.load(
            result_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(dataset, GraphUniverseDataset):
            raise TypeError(
                "isolated GraphUniverse generation returned "
                f"{type(dataset).__name__}, expected GraphUniverseDataset"
            )
        if len(dataset) == 0:
            raise ValueError(
                "isolated GraphUniverse generation returned an empty dataset"
            )

    dataset.root = str(canonical_data_dir.parent)
    dataset.name = canonical_data_dir.name
    dataset.processed_root = str(canonical_data_dir)
    return dataset


class GraphUniverseDatasetLoader(AbstractLoader):
    """Load Graph Universe datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - data_type: Type of the dataset (e.g., "graph_classification")
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load Graph Universe dataset.

        Returns
        -------
        Dataset
            The loaded Graph Universe dataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """
        parameters = resolve_cache_config(
            self.parameters["generation_parameters"],
            context="GraphUniverse generation parameters",
        )
        if not isinstance(parameters, dict):
            raise TypeError("generation_parameters must resolve to a mapping")
        dataset = _materialize_in_isolated_process(
            Path(self.get_data_dir()),
            parameters,
        )
        dataset.feature_policy = "continuous"
        dataset.representation_version = (
            _GRAPH_UNIVERSE_REPRESENTATION_VERSION
        )
        dataset.parser_version = _GRAPH_UNIVERSE_PARSER_VERSION
        return dataset

    def load(self, **kwargs) -> tuple[Data, str]:
        """Load data.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        tuple[torch_geometric.data.Data, str]
            Tuple containing the loaded data and the data directory.
        """
        dataset, _ = super().load(**kwargs)
        data_dir = dataset.raw_dir

        return dataset, data_dir
