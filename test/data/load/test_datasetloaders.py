"""Comprehensive test suite for all dataset loaders."""

import os
from pathlib import Path
from typing import Any

import hydra
import pytest
from torch_geometric.data import HeteroData

from topobench.utils.config_resolvers import register_all_resolvers

download_gated_datasets = frozenset({"DBLP.yaml", "OGB_MAG.yaml"})


class TestLoaders:
    """Comprehensive test suite for all dataset loaders."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment before each test method."""
        # Existing setup code remains the same
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        register_all_resolvers()
        base_dir = Path(__file__).resolve().parents[3]
        self.config_files = self._gather_config_files(base_dir)
        self.relative_config_dir = "../../../configs"
        self.test_splits = ["train", "val", "test"]

    # Existing helper methods remain the same
    def _gather_config_files(
        self,
        base_dir: Path,
    ) -> list[tuple[str, str]]:
        """Gather all relevant config files.

        Parameters
        ----------
        base_dir : Path
            Base directory to start searching for config files.

        Returns
        -------
        list[tuple[str, str]]
          Dataset-domain and configuration-filename pairs.
        """
        config_files = []
        config_base_dir = base_dir / "configs/dataset"
        # Below the datasets that have some default transforms manually overriten with no_transform,
        exclude_datasets = {
            "karate_club.yaml",
            # Below the datasets that have some default transforms with we manually overriten with no_transform,
            # due to lack of default transform for domain2domain
            "REDDIT-BINARY.yaml",
            "IMDB-MULTI.yaml",
            "IMDB-BINARY.yaml",  # "ZINC.yaml"
            "ogbg-molpcba.yaml",
            "manual_dataset.yaml",  # "ogbg-molhiv.yaml"
        }
        if os.environ.get("TOPOBENCH_ALLOW_DOWNLOADS") != "1":
            exclude_datasets.update(download_gated_datasets)

        # Below the datasets that takes quite some time to load and process
        self.long_running_datasets = {
            "mantra_name.yaml",
            "mantra_orientation.yaml",
            "mantra_genus.yaml",
            "mantra_betti_numbers.yaml",
        }

        for dir_path in config_base_dir.iterdir():
            curr_dir = dir_path.name
            if dir_path.is_dir():
                config_files.extend(
                    [
                        (curr_dir, f.name)
                        for f in dir_path.glob("*.yaml")
                        if f.name not in exclude_datasets
                    ]
                )
        return config_files

    def _load_dataset(
        self, data_domain: str, config_file: str
    ) -> tuple[Any, str]:
        """Load dataset with given config file.

        Parameters
        ----------
        data_domain : str
            Name of the data domain.
        config_file : str
          Name of the config file.

        Returns
        -------
        tuple[Any, str]
          Dataset and canonical dataset directory.
        """
        with hydra.initialize(
            version_base="1.3",
            config_path=self.relative_config_dir,
            job_name="run",
        ):
            print("Current config file: ", config_file)
            overrides = [
                f"dataset={data_domain}/{config_file}",
                "model=graph/gat",
            ]
            if data_domain == "heterogeneous":
                overrides.append("transforms=no_transform")
            parameters = hydra.compose(
                config_name="run.yaml",
                overrides=overrides,
                return_hydra_config=True,
            )
            dataset_loader = hydra.utils.instantiate(parameters.dataset.loader)
            print(repr(dataset_loader))

            if config_file in self.long_running_datasets:
                dataset, data_dir = dataset_loader.load(slice=100)
            else:
                dataset, data_dir = dataset_loader.load()
            return dataset, data_dir

    def test_dataset_loading_states(self):
        """Test different states and scenarios during dataset loading."""
        for config_data in self.config_files:
            data_domain, config_file = config_data
            dataset, _ = self._load_dataset(data_domain, config_file)

            # Test dataset size and dimensions
            data = dataset[0]
            if isinstance(data, HeteroData):
                assert data.node_types
                assert data.edge_types
                for node_type in data.node_types:
                    assert data[node_type].num_nodes > 0
            else:
                assert data.x.size(0) > 0, "Empty node features"
                assert data.y.size(0) > 0, "Empty labels"

            # Below brakes with manual dataset
            # else:
            #     assert dataset[0].x.size(0) > 0, "Empty node features"
            #     assert dataset[0].y.size(0) > 0, "Empty labels"

            # Test node feature dimensions
            if not isinstance(data, HeteroData) and hasattr(
                dataset, "num_node_features"
            ):
                assert dataset.data.x.size(1) == dataset.num_node_features

            # Below brakes with manual dataset
            # # Test label dimensions
            # if hasattr(dataset, 'num_classes'):
            #     assert torch.max(dataset.data.y) < dataset.num_classes

            repr(dataset)


def test_download_gated_datasets_are_excluded_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config enumeration cannot download DBLP or OGB-MAG by default."""
    monkeypatch.delenv("TOPOBENCH_ALLOW_DOWNLOADS", raising=False)
    gathered = TestLoaders()._gather_config_files(
        Path(__file__).resolve().parents[3]
    )
    filenames = {filename for _, filename in gathered}

    assert download_gated_datasets == {"DBLP.yaml", "OGB_MAG.yaml"}
    assert filenames.isdisjoint(download_gated_datasets)


def test_download_gated_datasets_are_enumerated_only_after_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The single environment gate enables configs that actually exist."""
    monkeypatch.setenv("TOPOBENCH_ALLOW_DOWNLOADS", "1")
    gathered = TestLoaders()._gather_config_files(
        Path(__file__).resolve().parents[3]
    )
    filenames = {filename for _, filename in gathered}

    assert download_gated_datasets.issubset(filenames)
