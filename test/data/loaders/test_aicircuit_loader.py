"""Tests for the AICircuit dataset and loader."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
import torch
import hydra
from omegaconf import OmegaConf
from unittest.mock import patch

from topobench.data.datasets.aicircuit_datasets import AICircuitDataset
from topobench.data.loaders.hypergraph.aicircuit_dataset_loader import AICircuitDatasetLoader

# Dummy data for testing
SYNTHETIC_NUM_CIRCUITS = 2
SYNTHETIC_NUM_NODES = 5
SYNTHETIC_NUM_HYPEREDGES = 3
SYNTHETIC_NUM_NODE_FEATURES = 1 # From aicurcuit_datasets.py -> x = torch.arange(num_nodes, dtype=torch.float).view(-1, 1)
SYNTHETIC_NUM_CLASSES = 3      # From aicurcuit_datasets.py -> y = torch.tensor(df.values[:, 4:], dtype=torch.float)
SYNTHETIC_NUM_GRAPH_ATTR = 4   # From aicurcuit_datasets.py -> graph_attr = torch.tensor(df.values[:, :4], dtype=torch.float)
SYNTHETIC_NUM_HYPEREDGE_ATTR = 16 # From _create_component_vocab (num_classes in one-hot encoding)

def mock_download_and_process(self):
    """Mock download that calls process directly after raw data is in place."""
    self.process() # Directly call process since raw data is pre-created by fixture

def _write_dummy_aicircuit_raw_data(base_dir: Path, circuit_type: str):
    """Create synthetic raw data for AICircuit.

    Parameters
    ----------
    base_dir : Path
        Root directory for the dummy raw files.
    circuit_type : str
        Circuit type subfolder.
    """
    
    # Create Dataset structure for CSV directly under base_dir
    dataset_base_dir = base_dir / "Dataset"
    dataset_base_dir.mkdir(parents=True, exist_ok=True)
    dataset_type_dir = dataset_base_dir / circuit_type
    dataset_type_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dataset_type_dir / f"{circuit_type}.csv"
    
    # Create dummy CSV content
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

    # Create Simulation/Netlists structure for netlist directly under base_dir
    simulation_base_dir = base_dir / "Simulation"
    simulation_base_dir.mkdir(parents=True, exist_ok=True)
    netlists_base_dir = simulation_base_dir / "Netlists"
    netlists_base_dir.mkdir(parents=True, exist_ok=True)
    netlist_type_dir = netlists_base_dir / circuit_type
    netlist_type_dir.mkdir(parents=True, exist_ok=True)
    netlist_path = netlist_type_dir / "netlist"

    # Create dummy netlist content
    netlist_content = """
M0 (IOUT1 net4 VSS VSS) nmos4
R0 (VDD net4) resistor
C0 (net4 VSS) capacitor
"""
    netlist_path.write_text(netlist_content)


@pytest.fixture
@patch('topobench.data.datasets.aicircuit_datasets.AICircuitDataset.download', new=mock_download_and_process)
def aicircuit_dataset_fixture(tmp_path):
    """Return a synthetic AICircuit dataset and its loader directory.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    tuple
        Tuple of dataset name, dataset instance, and loader config.
    """
    
    dataset_name = "AICircuit"
    circuit_type = "CVA" # We will test with CVA for simplicity

    # Simulate the raw_dir structure after download and extraction
    # The download method moves contents of 'AICircuit-main' directly into raw_dir
    raw_root_dir = tmp_path / dataset_name / "raw"
    raw_root_dir.mkdir(parents=True, exist_ok=True)

    _write_dummy_aicircuit_raw_data(raw_root_dir, circuit_type)

    loader_config = OmegaConf.create(
        {
            "_target_": "topobench.data.loaders.hypergraph.aicircuit_dataset_loader.AICircuitDatasetLoader",
            "parameters": {
                "data_domain": "hypergraph",
                "data_type": "analog_circuit",
                "data_name": dataset_name,
                "data_dir": str(tmp_path),
            }
        }
    )
    dataset_config = OmegaConf.create(
        {
            "parameters": {
                "num_features": SYNTHETIC_NUM_NODE_FEATURES
            },
            "split_params": {
                "learning_setting": "inductive",
                "data_seed": 0,
                "split_type": "random",
                "train_prop": 0.8,
                "standardize": True,
                "data_split_dir": str(tmp_path / "data_splits" / dataset_name),
            }
        }
    )
    loader = hydra.utils.instantiate(loader_config, cfg=dataset_config)
    dataset, _ = loader.load() # load() returns dataset, dataset_dir
    return dataset_name, dataset, raw_root_dir


def test_aicircuit_loader_instantiates_correctly():
    """Ensure the AICircuitDatasetLoader can be instantiated."""
    loader_config = OmegaConf.create(
        {
            "_target_": "topobench.data.loaders.hypergraph.aicircuit_dataset_loader.AICircuitDatasetLoader",
            "parameters": {
                "data_domain": "hypergraph",
                "data_type": "analog_circuit",
                "data_name": "AICircuit",
                "data_dir": "/tmp", # Dummy path for instantiation
            }
        }
    )
    dataset_config = OmegaConf.create(
        {
            "parameters": {
                "num_features": SYNTHETIC_NUM_NODE_FEATURES
            },
            "split_params": {
                "learning_setting": "inductive",
                "data_seed": 0,
                "split_type": "random",
                "train_prop": 0.8,
                "standardize": True,
                "data_split_dir": "/tmp/data_splits/AICircuit",
            }
        }
    )
    loader = hydra.utils.instantiate(loader_config, cfg=dataset_config)
    assert isinstance(loader, AICircuitDatasetLoader)

def test_aicircuit_loader_loads_dataset(aicircuit_dataset_fixture):
    """Ensure the loader loads an AICircuitDataset instance with correct length.

    Parameters
    ----------
    aicircuit_dataset_fixture : tuple
        Fixture providing dataset name, dataset instance, and loader config.
    """
    dataset_name, dataset, _ = aicircuit_dataset_fixture
    assert isinstance(dataset, AICircuitDataset)
    assert len(dataset) == 1 # Only one circuit type (CVA) in our dummy data


def test_aicircuit_dataset_properties(aicircuit_dataset_fixture):
    """Validate properties of a loaded AICircuitDataset sample.

    Parameters
    ----------
    aicircuit_dataset_fixture : tuple
        Fixture providing dataset name, dataset instance, and loader config.
    """
    dataset_name, dataset, _ = aicircuit_dataset_fixture
    
    data = dataset[0] # Get the first (and only) graph in our dummy dataset

    # Check node features
    assert data.x is not None
    assert data.x.shape[0] >= 1 # At least one node
    assert data.x.shape[1] == SYNTHETIC_NUM_NODE_FEATURES
    assert data.x.dtype == torch.float

    # Check hyperedge index
    assert data.hyperedge_index is not None
    assert data.hyperedge_index.shape[0] == 2 # (node_idx, hyperedge_idx)
    assert data.hyperedge_index.dtype == torch.long

    # Check hyperedge attributes (component types)
    assert data.hyperedge_attr is not None
    assert data.hyperedge_attr.shape[0] == SYNTHETIC_NUM_HYPEREDGES # Number of components in dummy netlist
    assert data.hyperedge_attr.dtype == torch.long

    # Check target labels (y)
    assert data.y is not None
    assert data.y.shape[0] == SYNTHETIC_NUM_CIRCUITS # Two rows in dummy CSV
    assert data.y.shape[1] == SYNTHETIC_NUM_CLASSES # Bandwidth, PowerConsumption, VoltageGain
    assert data.y.dtype == torch.float

    # Check graph attributes (design parameters)
    assert data.graph_attr is not None
    assert data.graph_attr.shape[0] == SYNTHETIC_NUM_CIRCUITS # Two rows in dummy CSV
    assert data.graph_attr.shape[1] == SYNTHETIC_NUM_GRAPH_ATTR # Wbias, Rd, Wn1, Wn2
    assert data.graph_attr.dtype == torch.float

    # Check data.name
    assert data.name == "CVA"

@patch('topobench.data.datasets.aicircuit_datasets.AICircuitDataset.download', new=mock_download_and_process)
def test_aicircuit_loader_handles_missing_files_gracefully(tmp_path):
    """Ensure the loader handles missing raw files gracefully.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.
    """
    dataset_name = "AICircuit"
    circuit_type = "CVA"

    raw_root_dir = tmp_path / dataset_name / "raw"
    raw_root_dir.mkdir(parents=True, exist_ok=True)

    # Create only netlist, no CSV
    netlist_dir = raw_root_dir / "Simulation" / "Netlists" / circuit_type
    netlist_dir.mkdir(parents=True, exist_ok=True)
    (netlist_dir / "netlist").write_text("M0 (IOUT1 net4 VSS VSS) nmos4") # Minimal content

    loader_config = OmegaConf.create(
        {
            "_target_": "topobench.data.loaders.hypergraph.aicircuit_dataset_loader.AICircuitDatasetLoader",
            "parameters": {
                "data_domain": "hypergraph",
                "data_type": "analog_circuit",
                "data_name": dataset_name,
                "data_dir": str(tmp_path),
            }
        }
    )
    dataset_config = OmegaConf.create(
        {
            "parameters": {
                "num_features": SYNTHETIC_NUM_NODE_FEATURES
            },
            "split_params": {
                "learning_setting": "inductive",
                "data_seed": 0,
                "split_type": "random",
                "train_prop": 0.8,
                "standardize": True,
                "data_split_dir": str(tmp_path / "data_splits" / dataset_name),
            }
        }
    )
    loader = hydra.utils.instantiate(loader_config, cfg=dataset_config)
    
    # Expect an empty dataset or error if no valid circuit types are found
    # The current process method skips if csv_path or netlist_path does not exist
    dataset, _ = loader.load()
    assert len(dataset) == 0 # No valid circuits were processed
