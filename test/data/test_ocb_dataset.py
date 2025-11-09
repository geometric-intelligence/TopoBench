"""Tests for the OCB dataset and its target statistics."""

import os
import pytest
import torch
import shutil
from omegaconf import OmegaConf
from topobench.data.datasets.ocb_dataset import OCB101Dataset
from topobench.data.loaders.graph.ocb_loader import OCB101DatasetLoader

# Define a base data directory for the test
TEST_ROOT_DIR = "./temp_test_data"

@pytest.fixture(scope="module")
def ocb101_dataset():
    """Fixture to load OCB101 dataset once for all tests in this module.

    Yields
    ------
    OCB101Dataset
        An instance of the OCB101 dataset.
    """
    # Clean up any previous test data
    if os.path.exists(TEST_ROOT_DIR):
        shutil.rmtree(TEST_ROOT_DIR)
    
    # Mock Hydra config for the loader
    config = OmegaConf.create({
        "data_domain": "graph",
        "data_type": "circuits",
        "data_name": "OCB101",
        "data_dir": os.path.join(TEST_ROOT_DIR, "graph", "circuits")
    })

    # Instantiate the loader
    loader = OCB101DatasetLoader(parameters=config)
    loader.root_data_dir = TEST_ROOT_DIR
    
    # Load the dataset
    dataset = loader.load_dataset()
    yield dataset
    
    # Teardown: Clean up the created test directory
    if os.path.exists(TEST_ROOT_DIR):
        shutil.rmtree(TEST_ROOT_DIR)

def test_ocb101_get_target_statistics(ocb101_dataset: OCB101Dataset):
    """Test the get_target_statistics method of OCB101Dataset.

    Parameters
    ----------
    ocb101_dataset : OCB101Dataset
        The OCB101 dataset fixture.
    """
    stats = ocb101_dataset.get_target_statistics()

    assert isinstance(stats, dict)
    assert "mean" in stats
    assert "std" in stats
    assert "min" in stats
    assert "max" in stats

    # Check types
    assert isinstance(stats["mean"], float)
    assert isinstance(stats["std"], float)
    assert isinstance(stats["min"], float)
    assert isinstance(stats["max"], float)

    # Check if values are reasonable (e.g., not zero or NaN for std)
    assert stats["mean"] > 0
    assert stats["std"] > 0
    assert stats["min"] < stats["max"]

    # Verify against manual calculation for a small subset (optional, but good for robustness)
    # For a full dataset, this might be too slow, but for a fixture-loaded dataset, it's fine.
    all_targets = torch.cat([ocb101_dataset[i].y for i in range(len(ocb101_dataset))])
    assert abs(stats["mean"] - float(all_targets.mean())) < 1e-6
    assert abs(stats["std"] - float(all_targets.std())) < 1e-6
    assert abs(stats["min"] - float(all_targets.min())) < 1e-6
    assert abs(stats["max"] - float(all_targets.max())) < 1e-6
