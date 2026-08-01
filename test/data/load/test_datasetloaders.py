"""Comprehensive test suite for all dataset loaders."""

import os
from pathlib import Path
from typing import Any

import hydra
import pytest
from torch_geometric.data import HeteroData

from topobench.utils.config_resolvers import register_all_resolvers

NETWORK_FREE_DATASET_SELECTORS = frozenset(
    {
        ("graph", "SyntheticGraph"),
        ("graph", "SyntheticGraphRegression"),
        ("graph", "SyntheticNodeGraph"),
        ("heterogeneous", "SyntheticHeterogeneous"),
        ("hypergraph", "SyntheticHypergraph"),
    }
)
MODEL_SELECTOR_BY_DOMAIN = {
    "graph": "graph/gat",
    "heterogeneous": "heterogeneous/hgt",
    "hypergraph": "hypergraph/hypergraph_conv",
}


class TestLoaders:
    """Comprehensive test suite for all dataset loaders."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment before each test method."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        register_all_resolvers()
        base_dir = Path(__file__).resolve().parents[3]
        self.config_selectors = self._gather_config_selectors(base_dir)
        self.relative_config_dir = "../../../configs"
        self.test_splits = ["train", "val", "test"]

    def _gather_config_selectors(
        self,
        base_dir: Path,
    ) -> list[tuple[str, str]]:
        """Gather every surviving dataset selector exercised by this test.

        Parameters
        ----------
        base_dir : Path
            Base directory to start searching for config files.

        Returns
        -------
        list[tuple[str, str]]
          Dataset-domain and configuration-selector pairs.
        """
        config_base_dir = base_dir / "configs/dataset"
        config_selectors = [
            (domain, config_file.stem)
            for domain in ("graph", "heterogeneous", "hypergraph")
            for config_file in (config_base_dir / domain).glob("*.yaml")
        ]
        if os.environ.get("TOPOBENCH_ALLOW_DOWNLOADS") == "1":
            return config_selectors
        return [
            selector
            for selector in config_selectors
            if selector in NETWORK_FREE_DATASET_SELECTORS
        ]

    def _load_dataset(
        self, data_domain: str, data_selector: str
    ) -> tuple[Any, str]:
        """Load the dataset selected by its domain-qualified Hydra config.

        Parameters
        ----------
        data_domain : str
            Name of the data domain.
        data_selector : str
          Name of the dataset config selector.

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
            print("Current dataset selector: ", data_selector)
            overrides = [
                f"dataset={data_domain}/{data_selector}",
                f"model={MODEL_SELECTOR_BY_DOMAIN[data_domain]}",
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

            dataset, data_dir = dataset_loader.load()
            return dataset, data_dir

    def test_dataset_loading_states(self):
        """Test different states and scenarios during dataset loading."""
        for data_domain, data_selector in self.config_selectors:
            dataset, _ = self._load_dataset(data_domain, data_selector)

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


            # Test node feature dimensions
            if not isinstance(data, HeteroData) and hasattr(
                dataset, "num_node_features"
            ):
                assert dataset.data.x.size(1) == dataset.num_node_features


            repr(dataset)


def test_only_packaged_datasets_are_enumerated_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ordinary suite loads only deterministic packaged datasets."""
    monkeypatch.delenv("TOPOBENCH_ALLOW_DOWNLOADS", raising=False)
    gathered = TestLoaders()._gather_config_selectors(
        Path(__file__).resolve().parents[3]
    )

    assert set(gathered) == NETWORK_FREE_DATASET_SELECTORS


def test_download_datasets_are_enumerated_only_after_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explicit environment gate enables every surviving selector."""
    monkeypatch.setenv("TOPOBENCH_ALLOW_DOWNLOADS", "1")
    gathered = TestLoaders()._gather_config_selectors(
        Path(__file__).resolve().parents[3]
    )
    selectors = set(gathered)

    assert NETWORK_FREE_DATASET_SELECTORS < selectors
    assert ("heterogeneous", "DBLP") in selectors
    assert ("heterogeneous", "OGB_MAG") in selectors
