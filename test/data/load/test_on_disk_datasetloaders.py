"""Comprehensive test suite for on-disk dataset loaders."""

from pathlib import Path
from typing import Any

import hydra
import pytest
import torch_geometric
from torch_geometric.data import OnDiskDataset


class TestOnDiskLoaders:
    """
    Test suite for dataset loaders using on-disk storage.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """
        Set up Hydra and loader configuration before each test.
        """
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        base_dir = Path(__file__).resolve().parents[3]
        self.config_files = self._gather_config_files(base_dir)
        self.relative_config_dir = "../../../configs"
        self.test_splits = ["train", "val", "test"]

    def _gather_config_files(self, base_dir: Path) -> list[tuple[str, str]]:
        """
        Gather all dataset config files.

        Parameters
        ----------
        base_dir : Path
            Project base directory.

        Returns
        -------
        list of tuple of str
            List of (data_domain, config_file) pairs.
        """
        config_files: list[tuple[str, str]] = []
        config_base_dir = base_dir / "configs/dataset"

        exclude_datasets = {
            "karate_club.yaml",
            "REDDIT-BINARY.yaml",
            "IMDB-MULTI.yaml",
            "IMDB-BINARY.yaml",
            "ogbg-molpcba.yaml",
            "manual_dataset.yaml",
        }

        self.long_running_datasets = {
            "mantra_name.yaml",
            "mantra_orientation.yaml",
            "mantra_genus.yaml",
            "mantra_betti_numbers.yaml",
        }

        for dir_path in config_base_dir.iterdir():
            curr_dir = dir_path.name
            if dir_path.is_dir():
                for f in dir_path.glob("*.yaml"):
                    if f.name in exclude_datasets:
                        continue
                    config_files.append((curr_dir, f.name))

        return config_files

    def _load_dataset(
        self, data_domain: str, config_file: str
    ) -> tuple[OnDiskDataset | None, dict, Any]:
        """
        Load dataset and configuration for the given YAML file.

        Parameters
        ----------
        data_domain : str
            Dataset domain name.
        config_file : str
            Dataset config filename.

        Returns
        -------
        OnDiskDataset or None
            Loaded on-disk dataset or None if memory_type is not on_disk.
        dict
            Dataset directory metadata as returned by the loader.
        Any
            Full Hydra-composed configuration.
        """
        with hydra.initialize(
            version_base="1.3",
            config_path=self.relative_config_dir,
            job_name="run",
        ):
            parameters = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    f"dataset={data_domain}/{config_file}",
                    "model=graph/gat",
                ],
                return_hydra_config=True,
            )

            memory_type = parameters.dataset.loader.parameters.get(
                "memory_type", "in_memory"
            )

            if memory_type != "on_disk":
                return None, {}, parameters
                
            print(f"{config_file}")
            dataset_loader = hydra.utils.instantiate(parameters.dataset.loader)

            if config_file in self.long_running_datasets:
                dataset, data_dir = dataset_loader.load(slice=100)
            else:
                dataset, data_dir = dataset_loader.load()

        return dataset, data_dir, parameters

    def test_on_disk_dataset_loading_states(self):
        """
        Test loading and basic properties for on-disk datasets.

        For configs with memory_type == 'on_disk', this verifies that
        the loader returns a valid OnDiskDataset and that features and
        labels are non-empty.
        """
        for data_domain, config_file in self.config_files:
            dataset, _, parameters = self._load_dataset(
                data_domain, config_file
            )

            # Skip in-memory datasets.
            if dataset is None:
                continue

            # Check that the returned dataset is backed by OnDiskDataset.
            assert isinstance(dataset, (OnDiskDataset, torch_geometric.data.OnDiskDataset))

            # Dataset must contain at least one graph.
            assert len(dataset) > 0

            # Single-graph style (dataset.data) or multi-graph style (dataset[0]).
            data = dataset.data if hasattr(dataset, "data") else dataset[0]

            # Basic feature and label checks.
            assert hasattr(data, "x"), "Missing node features"
            assert hasattr(data, "y"), "Missing labels"
            assert data.x is not None and data.x.numel() > 0, "Empty node features"
            assert data.y is not None and data.y.numel() > 0, "Empty labels"

            # Node feature dimension consistency when available.
            if hasattr(dataset, "num_node_features"):
                assert data.x.size(1) == dataset.num_node_features

            # Basic repr should not crash.
            repr(dataset)
