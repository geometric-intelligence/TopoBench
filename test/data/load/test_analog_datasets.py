"""Tests for analog dataset loaders."""

import pytest
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from unittest.mock import patch
import pandas as pd
import torch

from topobench.data.datasets.aicircuit_datasets import AICircuitDataset
from topobench.data.datasets.analoggenie_datasets import AnalogGenieDataset

# Dummy data for testing (AICircuit)
SYNTHETIC_NUM_CIRCUITS_AICIRCUIT = 2
SYNTHETIC_NUM_NODES_AICIRCUIT = 5
SYNTHETIC_NUM_HYPEREDGES_AICIRCUIT = 3
SYNTHETIC_NUM_NODE_FEATURES_AICIRCUIT = 1
SYNTHETIC_NUM_CLASSES_AICIRCUIT = 3
SYNTHETIC_NUM_GRAPH_ATTR_AICIRCUIT = 4

# Dummy data for testing (AnalogGenie)
SYNTHETIC_NUM_CIRCUITS_ANALOGGENIE = 2
SYNTHETIC_NUM_NODES_ANALOGGENIE = 5
SYNTHETIC_NUM_HYPEREDGES_ANALOGGENIE = 3
SYNTHETIC_NUM_NODE_FEATURES_ANALOGGENIE = 1
SYNTHETIC_NUM_HYPEREDGE_ATTR_ANALOGGENIE = 5


def _write_dummy_aicircuit_raw_data(base_dir: Path, circuit_type: str):
    """Create synthetic raw data for AICircuit.

    Parameters
    ----------
    base_dir : Path
        Root directory for the dummy raw files.
    circuit_type : str
        Circuit subfolder name.
    """
    dataset_base_dir = base_dir / "Dataset"
    dataset_base_dir.mkdir(parents=True, exist_ok=True)
    dataset_type_dir = dataset_base_dir / circuit_type
    dataset_type_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dataset_type_dir / f"{circuit_type}.csv"
    
    df = pd.DataFrame({
        'Wbias': [4.5e-06, 5e-06],
        'Rd': [2000, 2500],
        'Wn1': [6e-06, 7e-06],
        'Wn2': [5e-06, 6e-06],
        'Bandwidth': [94400000.0, 95000000.0],
        'PowerConsumption': [0.000718, 0.000818],
        'VoltageGain': [15.18, 15.50]
    })
    df.to_csv(csv_path, index=False)

    simulation_base_dir = base_dir / "Simulation"
    simulation_base_dir.mkdir(parents=True, exist_ok=True)
    netlists_base_dir = simulation_base_dir / "Netlists"
    netlists_base_dir.mkdir(parents=True, exist_ok=True)
    netlist_type_dir = netlists_base_dir / circuit_type
    netlist_type_dir.mkdir(parents=True, exist_ok=True)
    netlist_path = netlist_type_dir / "netlist"

    netlist_content = """
M0 (IOUT1 net4 VSS VSS) nmos4
R0 (VDD net4) resistor
C0 (net4 VSS) capacitor
"""
    netlist_path.write_text(netlist_content)

def _write_dummy_analoggenie_raw_data(base_dir: Path, circuit_id: str):
    """Create synthetic raw data for AnalogGenie.

    Parameters
    ----------
    base_dir : Path
        Root directory for the dummy raw files.
    circuit_id : str
        Circuit ID subfolder name.
    """
    dataset_base_dir = base_dir / "Dataset"
    dataset_base_dir.mkdir(parents=True, exist_ok=True)
    circuit_dir = dataset_base_dir / circuit_id
    circuit_dir.mkdir(parents=True, exist_ok=True)
    cir_path = circuit_dir / f"{circuit_id}.cir"
    
    cir_content = """
M0 (IOUT1 net4 VSS VSS) nmos4
R0 (VDD net4) resistor
C0 (net4 VSS) capacitor
"""
    cir_path.write_text(cir_content)


@pytest.fixture(scope="function")
def analog_datasets_fixture(tmp_path):
    """Set up dummy raw data and mock download for analog datasets.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.
    """
    with patch('topobench.data.datasets.aicircuit_datasets.AICircuitDataset.download', autospec=True) as mock_aicircuit_download, \
         patch('topobench.data.datasets.analoggenie_datasets.AnalogGenieDataset.download', autospec=True) as mock_analoggenie_download:
        # Ensure download simply triggers processing on the local dummy data
        def mock_download_and_process(self, *args, **kwargs):
            return self.process()

        mock_aicircuit_download.side_effect = mock_download_and_process
        mock_analoggenie_download.side_effect = mock_download_and_process

        aicircuit_raw_root_dir = tmp_path / "AICircuit" / "raw"
        aicircuit_raw_root_dir.mkdir(parents=True, exist_ok=True)
        _write_dummy_aicircuit_raw_data(aicircuit_raw_root_dir, "CVA")

        analoggenie_raw_root_dir = tmp_path / "AnalogGenie" / "raw"
        analoggenie_raw_root_dir.mkdir(parents=True, exist_ok=True)
        _write_dummy_analoggenie_raw_data(analoggenie_raw_root_dir, "100")

        yield tmp_path # Yield tmp_path for other uses if needed


@pytest.fixture(scope="module")
def hydra_initialize():
    """Fixture to initialize Hydra."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    relative_config_dir = "../../../configs"
    with hydra.initialize(version_base="1.3", config_path=relative_config_dir, job_name="test_analog_datasets"):
        yield

def test_aicurcuit_dataset(hydra_initialize, analog_datasets_fixture):
    """Test AICircuit dataset loads synthetic data.

    Parameters
    ----------
    hydra_initialize : fixture
        Hydra init fixture.
    analog_datasets_fixture : Path
        Temporary raw data root.
    """
    tmp_path_str = str(analog_datasets_fixture)
    aicircuit_split_dir_str = str(Path(tmp_path_str) / 'data_splits' / 'AICircuit') # Construct split dir

    cfg = hydra.compose(config_name="dataset/hypergraph/aicircuit", overrides=[
        "++dataset.hypergraph.loader.parameters.data_dir=" + tmp_path_str, # Override loader's data_dir directly
        "++dataset.hypergraph.loader.parameters.data_name=AICircuit", # Ensure data_name is correctly set
        "++dataset.hypergraph.loader.parameters.data_domain=hypergraph",
        "++dataset.hypergraph.loader.parameters.data_type=analog_circuit",
        f"++dataset.hypergraph.split_params.data_split_dir={aicircuit_split_dir_str}"
    ])
    dataset_cfg = cfg.dataset.hypergraph
    dataset_loader = hydra.utils.instantiate(dataset_cfg.loader, cfg=dataset_cfg)
    dataset, _ = dataset_loader.load()

    assert dataset is not None
    assert len(dataset) > 0
    assert hasattr(dataset[0], 'graph_attr')
    assert dataset[0].x.shape[1] == 1
    assert dataset[0].y.shape[1] == 3

def test_analoggenie_dataset(hydra_initialize, analog_datasets_fixture):
    """Test AnalogGenie dataset loads synthetic data.

    Parameters
    ----------
    hydra_initialize : fixture
        Hydra init fixture.
    analog_datasets_fixture : Path
        Temporary raw data root.
    """
    tmp_path_str = str(analog_datasets_fixture)
    analoggenie_split_dir_str = str(Path(tmp_path_str) / 'data_splits' / 'AnalogGenie') # Construct split dir

    cfg = hydra.compose(config_name="dataset/hypergraph/analoggenie", overrides=[
        "++dataset.hypergraph.loader.parameters.data_dir=" + tmp_path_str, # Override loader's data_dir directly
        "++dataset.hypergraph.loader.parameters.data_name=AnalogGenie", # Ensure data_name is correctly set
        "++dataset.hypergraph.loader.parameters.data_domain=hypergraph",
        "++dataset.hypergraph.loader.parameters.data_type=analog_circuit",
        f"++dataset.hypergraph.split_params.data_split_dir={analoggenie_split_dir_str}"
    ])
    dataset_cfg = cfg.dataset.hypergraph
    dataset_loader = hydra.utils.instantiate(dataset_cfg.loader, cfg=dataset_cfg)
    dataset, _ = dataset_loader.load()

    assert dataset is not None
    assert len(dataset) > 0
    assert not hasattr(dataset[0], 'graph_attr')
    assert not hasattr(dataset[0], 'y')
    assert dataset[0].x.shape[1] == 1
