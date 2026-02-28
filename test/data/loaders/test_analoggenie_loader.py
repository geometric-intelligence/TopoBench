"""Tests for the AnalogGenie dataset and loader."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import hydra
from omegaconf import OmegaConf
from unittest.mock import patch

from topobench.data.datasets.analoggenie_datasets import AnalogGenieDataset
from topobench.data.loaders.hypergraph.analoggenie_dataset_loader import AnalogGenieDatasetLoader

# Dummy data for testing
SYNTHETIC_NUM_CIRCUITS = 2
SYNTHETIC_NUM_NODES = 5
SYNTHETIC_NUM_HYPEREDGES = 3
SYNTHETIC_NUM_NODE_FEATURES = 1 # From analoggenie_datasets.py -> x = torch.arange(num_nodes, dtype=torch.float).view(-1, 1)
SYNTHETIC_NUM_HYPEREDGE_ATTR = 5 # From _create_component_vocab (num_classes in one-hot encoding)

def mock_download_and_process(self):
    """Mock download that calls process directly after raw data is in place."""
    self.process() # Directly call process since raw data is pre-created by fixture

def _write_dummy_analoggenie_raw_data(base_dir: Path, circuit_id: str):
    """Create synthetic raw data for AnalogGenie.

    Parameters
    ----------
    base_dir : Path
        Root directory for the dummy raw files.
    circuit_id : str
        Circuit ID subfolder.
    """
    
    # Create Dataset structure for .cir file directly under base_dir
    dataset_base_dir = base_dir / "Dataset"
    dataset_base_dir.mkdir(parents=True, exist_ok=True)
    circuit_dir = dataset_base_dir / circuit_id
    circuit_dir.mkdir(parents=True, exist_ok=True)
    cir_path = circuit_dir / f"{circuit_id}.cir"
    
    # Create dummy .cir content
    cir_content = """
M0 (IOUT1 net4 VSS VSS) nmos4
R0 (VDD net4) resistor
C0 (net4 VSS) capacitor
"""
    cir_path.write_text(cir_content)


@pytest.fixture
@patch('topobench.data.datasets.analoggenie_datasets.AnalogGenieDataset.download', new=mock_download_and_process)
def analoggenie_dataset_fixture(tmp_path):
    """Return a synthetic AnalogGenie dataset and its loader directory.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    tuple
        Tuple of dataset name, dataset instance, and loader config.
    """
    
    dataset_name = "AnalogGenie"
    circuit_id = "100" # We will test with a specific ID for simplicity

    # Simulate the raw_dir structure after download and extraction
    raw_root_dir = tmp_path / dataset_name / "raw"
    raw_root_dir.mkdir(parents=True, exist_ok=True)

    _write_dummy_analoggenie_raw_data(raw_root_dir, circuit_id)

    loader_config = OmegaConf.create(
        {
            "_target_": "topobench.data.loaders.hypergraph.analoggenie_dataset_loader.AnalogGenieDatasetLoader",
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
                "standardize": False,
                "data_split_dir": str(tmp_path / "data_splits" / dataset_name),
            }
        }
    )
    loader = hydra.utils.instantiate(loader_config, cfg=dataset_config)
    dataset, _ = loader.load() # load() returns dataset, dataset_dir
    return dataset_name, dataset, raw_root_dir


def test_analoggenie_loader_instantiates_correctly():
    """Ensure the AnalogGenieDatasetLoader can be instantiated."""
    loader_config = OmegaConf.create(
        {
            "_target_": "topobench.data.loaders.hypergraph.analoggenie_dataset_loader.AnalogGenieDatasetLoader",
            "parameters": {
                "data_domain": "hypergraph",
                "data_type": "analog_circuit",
                "data_name": "AnalogGenie",
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
                "standardize": False,
                "data_split_dir": "/tmp/data_splits/AnalogGenie",
            }
        }
    )
    loader = hydra.utils.instantiate(loader_config, cfg=dataset_config)
    assert isinstance(loader, AnalogGenieDatasetLoader)

def test_analoggenie_loader_loads_dataset(analoggenie_dataset_fixture):
    """Ensure the loader loads an AnalogGenieDataset instance with correct length.

    Parameters
    ----------
    analoggenie_dataset_fixture : tuple
        Fixture providing dataset name, dataset instance, and loader config.
    """
    dataset_name, dataset, _ = analoggenie_dataset_fixture
    assert isinstance(dataset, AnalogGenieDataset)
    assert len(dataset) == 1 # Only one circuit type (CVA) in our dummy data


def test_analoggenie_dataset_properties(analoggenie_dataset_fixture):
    """Validate properties of a loaded AnalogGenieDataset sample.

    Parameters
    ----------
    analoggenie_dataset_fixture : tuple
        Fixture providing dataset name, dataset instance, and loader config.
    """
    dataset_name, dataset, _ = analoggenie_dataset_fixture
    
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

    # AnalogGenie has no y or graph_attr for unsupervised task
    assert not hasattr(data, 'y')
    assert not hasattr(data, 'graph_attr')

@patch('topobench.data.datasets.analoggenie_datasets.AnalogGenieDataset.download', new=mock_download_and_process)
def test_analoggenie_loader_handles_missing_files_gracefully(tmp_path):
    """Ensure the loader handles missing raw files gracefully.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.
    """
    dataset_name = "AnalogGenie"
    circuit_id = "100"

    raw_root_dir = tmp_path / dataset_name / "raw"
    raw_root_dir.mkdir(parents=True, exist_ok=True)

    # Do not create any .cir file
    
    loader_config = OmegaConf.create(
        {
            "_target_": "topobench.data.loaders.hypergraph.analoggenie_dataset_loader.AnalogGenieDatasetLoader",
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
                "standardize": False,
                "data_split_dir": str(tmp_path / "data_splits" / dataset_name),
            }
        }
    )
    loader = hydra.utils.instantiate(loader_config, cfg=dataset_config)
    
    # Expect an empty dataset if no valid circuits are found
    dataset, _ = loader.load()
    assert len(dataset) == 0 # No valid circuits were processed
