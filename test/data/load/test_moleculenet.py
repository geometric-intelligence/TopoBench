"""Unit tests for MoleculeNet datasets."""

import os
import pytest
import torch
import numpy as np
from pathlib import Path
from omegaconf import DictConfig

from topobench.data.loaders.graph import MoleculeNetDatasetLoader


class TestMoleculeNetDatasets:
    """Test suite for MoleculeNet datasets.

    Note: Some MoleculeNet datasets contain molecules with invalid SMILES strings
    that cannot be parsed by RDKit (see https://github.com/deepchem/deepchem/issues/2336). 
    These molecules are automatically skipped during dataset processing, resulting in slightly fewer graphs than the original dataset
    reports. For such datasets, we use 'min_graphs' to specify the expected minimum
    number of valid molecules after filtering.
    """

    # Expected statistics for each dataset
    # Note: #nodes and #edges are averages, so we test with tolerance
    # Note: Some datasets have invalid SMILES that get skipped, so we use min_graphs
    DATASET_STATS = {
        "ESOL": {
            "num_graphs": 1128,
            "avg_nodes": 13.3,
            "avg_edges": 27.4,
            "num_features": 9,
            "num_classes": 1,
            "task": "regression",
            "tolerance_nodes": 0.5,  # tolerance for average
            "tolerance_edges": 1.0,
        },
        "FreeSolv": {
            "num_graphs": 642,
            "avg_nodes": 8.7,
            "avg_edges": 16.8,
            "num_features": 9,
            "num_classes": 1,
            "task": "regression",
            "tolerance_nodes": 0.5,
            "tolerance_edges": 1.0,
        },
        "Lipo": {
            "num_graphs": 4200,
            "avg_nodes": 27.0,
            "avg_edges": 59.0,
            "num_features": 9,
            "num_classes": 1,
            "task": "regression",
            "tolerance_nodes": 0.5,
            "tolerance_edges": 1.0,
        },
        "PCBA": {
            "num_graphs": 437929,
            "min_graphs": 437927,  # 2 molecules skipped due to invalid SMILES
            "avg_nodes": 26.0,
            "avg_edges": 56.2,
            "num_features": 9,
            "num_classes": 128,
            "task": "classification",
            "tolerance_nodes": 0.5,
            "tolerance_edges": 1.0,
        },
        "MUV": {
            "num_graphs": 93087,
            "avg_nodes": 24.2,
            "avg_edges": 52.6,
            "num_features": 9,
            "num_classes": 17,
            "task": "classification",
            "tolerance_nodes": 0.5,
            "tolerance_edges": 1.0,
        },
        "HIV": {
            "num_graphs": 41127,
            "min_graphs": 41120,  # 7 molecules skipped due to invalid SMILES
            "avg_nodes": 25.5,
            "avg_edges": 54.9,
            "num_features": 9,
            "num_classes": 1,
            "task": "classification",
            "tolerance_nodes": 0.5,
            "tolerance_edges": 1.0,
        },
        "BACE": {
            "num_graphs": 1513,
            "avg_nodes": 34.1,
            "avg_edges": 73.7,
            "num_features": 9,
            "num_classes": 1,
            "task": "classification",
            "tolerance_nodes": 0.5,
            "tolerance_edges": 1.5,
        },
        "BBBP": {
            "num_graphs": 2050,
            "min_graphs": 2039,  # Some molecules skipped due to invalid SMILES
            "avg_nodes": 23.9,
            "avg_edges": 51.6,
            "num_features": 9,
            "num_classes": 1,
            "task": "classification",
            "tolerance_nodes": 0.5,
            "tolerance_edges": 1.0,
        },
        "Tox21": {
            "num_graphs": 7831,
            "min_graphs": 7823,  # 8 molecules skipped due to invalid SMILES
            "avg_nodes": 18.6,
            "avg_edges": 38.6,
            "num_features": 9,
            "num_classes": 12,
            "task": "classification",
            "tolerance_nodes": 0.5,
            "tolerance_edges": 1.0,
        },
        "ToxCast": {
            "num_graphs": 8597,
            "min_graphs": 8579,  # 18 molecules skipped due to invalid SMILES
            "avg_nodes": 18.7,
            "avg_edges": 38.4,
            "num_features": 9,
            "num_classes": 617,
            "task": "classification",
            "tolerance_nodes": 0.5,
            "tolerance_edges": 1.0,
        },
        "SIDER": {
            "num_graphs": 1427,
            "avg_nodes": 33.6,
            "avg_edges": 70.7,
            "num_features": 9,
            "num_classes": 27,
            "task": "classification",
            "tolerance_nodes": 0.5,
            "tolerance_edges": 1.5,
        },
        "ClinTox": {
            "num_graphs": 1484,
            "min_graphs": 1480,  # 4 molecules skipped due to invalid SMILES
            "avg_nodes": 26.1,
            "avg_edges": 55.5,
            "num_features": 9,
            "num_classes": 2,
            "task": "classification",
            "tolerance_nodes": 0.5,
            "tolerance_edges": 1.0,
        },
    }

    @pytest.fixture(scope="session")
    def test_data_dir(self):
        """Create persistent cache directory for test data.
        
        Uses session scope so datasets are downloaded once and reused
        across all test runs, which is important for large datasets
        like PCBA and MUV.
        
        Returns
        -------
        Path
            Persistent cache directory path.
        """
        cache_dir = Path(".test_tmp/moleculenet_datasets")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    @pytest.fixture
    def dataset_params(self, test_data_dir):
        """Create a function that generates parameters for a given dataset.
        
        Parameters
        ----------
        test_data_dir : Path
            Temporary directory for test data.
            
        Returns
        -------
        callable
            Function that takes dataset_name and returns DictConfig.
        """
        def _make_params(dataset_name: str) -> DictConfig:
            """Create dataset parameters.

            Parameters
            ----------
            dataset_name : str
                Name of the dataset.

            Returns
            -------
            DictConfig
                Dataset parameters.
            """
            return DictConfig({
                "data_dir": str(test_data_dir),
                "data_domain": "graph",
                "data_type": "MoleculeNet",
                "data_name": dataset_name
            })
        return _make_params

    SMALL_DATASETS = ["FreeSolv", "ESOL", "BACE", "BBBP", "ClinTox", "SIDER"]
    MEDIUM_DATASETS = ["Lipo", "Tox21", "ToxCast"]
    # Large datasets (may be slow, consider marking with pytest.mark.slow)
    LARGE_DATASETS = ["HIV", "PCBA", "MUV"]

    @pytest.mark.parametrize("dataset_name", SMALL_DATASETS + MEDIUM_DATASETS)
    def test_dataset_initialization(self, test_data_dir, dataset_params, dataset_name):
        """Test that dataset initializes correctly.
        
        Parameters
        ----------
        test_data_dir : Path
            Temporary directory for test data.
        dataset_params : callable
            Function to create dataset parameters.
        dataset_name : str
            Name of dataset to test.
        """
        params = dataset_params(dataset_name)
        loader = MoleculeNetDatasetLoader(params)
        dataset = loader.load_dataset()

        assert dataset is not None
        expected = self.DATASET_STATS[dataset_name]
        # Some datasets have invalid SMILES that get skipped during processing
        if "min_graphs" in expected:
            assert len(dataset) >= expected["min_graphs"], \
                f"Dataset has fewer graphs than minimum: {len(dataset)} < {expected['min_graphs']}"
            assert len(dataset) <= expected["num_graphs"], \
                f"Dataset has more graphs than expected: {len(dataset)} > {expected['num_graphs']}"
        else:
            assert len(dataset) == expected["num_graphs"], \
                f"Dataset has incorrect number of graphs: {len(dataset)} != {expected['num_graphs']}"

    @pytest.mark.parametrize("dataset_name", SMALL_DATASETS + MEDIUM_DATASETS)
    def test_dataset_structure(self, test_data_dir, dataset_params, dataset_name):
        """Test that loaded dataset has correct structure.
        
        Parameters
        ----------
        test_data_dir : Path
            Temporary directory for test data.
        dataset_params : callable
            Function to create dataset parameters.
        dataset_name : str
            Name of dataset to test.
        """
        params = dataset_params(dataset_name)
        loader = MoleculeNetDatasetLoader(params)
        dataset = loader.load_dataset()

        # Get first graph
        data = dataset[0]

        # Check required attributes exist
        assert hasattr(data, 'x'), "Missing node features"
        assert hasattr(data, 'edge_index'), "Missing edge_index"
        assert hasattr(data, 'y'), "Missing labels"

        # Check that x and edge_index are tensors
        assert isinstance(data.x, torch.Tensor), "Node features should be tensor"
        assert isinstance(data.edge_index, torch.Tensor), "Edge index should be tensor"
        assert isinstance(data.y, torch.Tensor), "Labels should be tensor"

    @pytest.mark.parametrize("dataset_name", SMALL_DATASETS)
    def test_dataset_statistics(self, test_data_dir, dataset_params, dataset_name):
        """Test that dataset has expected statistics.
        
        Tests exact number of graphs and approximate averages for nodes/edges.
        Only run on small datasets to keep test time reasonable.
        
        Parameters
        ----------
        test_data_dir : Path
            Temporary directory for test data.
        dataset_params : callable
            Function to create dataset parameters.
        dataset_name : str
            Name of dataset to test.
        """
        params = dataset_params(dataset_name)
        loader = MoleculeNetDatasetLoader(params)
        dataset = loader.load_dataset()

        expected = self.DATASET_STATS[dataset_name]

        # Check number of graphs (allow for skipped molecules)
        if "min_graphs" in expected:
            assert len(dataset) >= expected["min_graphs"], \
                f"Dataset has fewer graphs than minimum for {dataset_name}: {len(dataset)} < {expected['min_graphs']}"
        else:
            assert len(dataset) == expected["num_graphs"], \
                f"Incorrect number of graphs for {dataset_name}"

        # Calculate average nodes and edges
        total_nodes = 0
        total_edges = 0
        for i in range(len(dataset)):
            data = dataset[i]
            total_nodes += data.x.shape[0]
            total_edges += data.edge_index.shape[1]

        avg_nodes = total_nodes / len(dataset)
        avg_edges = total_edges / len(dataset)

        # Check averages with tolerance
        assert abs(avg_nodes - expected["avg_nodes"]) <= expected["tolerance_nodes"], \
            f"Average nodes mismatch for {dataset_name}: {avg_nodes} vs {expected['avg_nodes']}"
        assert abs(avg_edges - expected["avg_edges"]) <= expected["tolerance_edges"], \
            f"Average edges mismatch for {dataset_name}: {avg_edges} vs {expected['avg_edges']}"

    @pytest.mark.parametrize("dataset_name", SMALL_DATASETS + MEDIUM_DATASETS)
    def test_feature_dimensions(self, test_data_dir, dataset_params, dataset_name):
        """Test that node features have correct dimensions.
        
        Parameters
        ----------
        test_data_dir : Path
            Temporary directory for test data.
        dataset_params : callable
            Function to create dataset parameters.
        dataset_name : str
            Name of dataset to test.
        """
        params = dataset_params(dataset_name)
        loader = MoleculeNetDatasetLoader(params)
        dataset = loader.load_dataset()

        expected = self.DATASET_STATS[dataset_name]
        data = dataset[0]

        # Check feature dimensions
        assert data.x.shape[1] == expected["num_features"], \
            f"Incorrect number of features for {dataset_name}"

    @pytest.mark.parametrize("dataset_name", SMALL_DATASETS + MEDIUM_DATASETS)
    def test_edge_index_format(self, test_data_dir, dataset_params, dataset_name):
        """Test that edge_index has correct format.
        
        Parameters
        ----------
        test_data_dir : Path
            Temporary directory for test data.
        dataset_params : callable
            Function to create dataset parameters.
        dataset_name : str
            Name of dataset to test.
        """
        params = dataset_params(dataset_name)
        loader = MoleculeNetDatasetLoader(params)
        dataset = loader.load_dataset()

        data = dataset[0]

        # Edge index should be 2 x num_edges
        assert data.edge_index.shape[0] == 2, \
            f"Edge index should be 2xN for {dataset_name}"

        # Edge indices should be valid node indices
        num_nodes = data.x.shape[0]
        assert data.edge_index.min() >= 0, \
            f"Edge indices should be non-negative for {dataset_name}"
        assert data.edge_index.max() < num_nodes, \
            f"Edge indices should be < num_nodes for {dataset_name}"

    @pytest.mark.parametrize("dataset_name", SMALL_DATASETS)
    def test_label_format_regression(self, test_data_dir, dataset_params, dataset_name):
        """Test label format for regression tasks.
        
        Parameters
        ----------
        test_data_dir : Path
            Temporary directory for test data.
        dataset_params : callable
            Function to create dataset parameters.
        dataset_name : str
            Name of dataset to test.
        """
        expected = self.DATASET_STATS[dataset_name]

        # Skip if not a regression task
        if expected["task"] != "regression":
            pytest.skip(f"{dataset_name} is not a regression task")

        params = dataset_params(dataset_name)
        loader = MoleculeNetDatasetLoader(params)
        dataset = loader.load_dataset()

        data = dataset[0]

        # For regression, labels should be scalar values (shape: [1] or [])
        assert data.y.numel() == 1, \
            f"Regression labels should be scalar for {dataset_name}"

    @pytest.mark.parametrize("dataset_name", SMALL_DATASETS)
    def test_label_format_classification(self, test_data_dir, dataset_params, dataset_name):
        """Test label format for classification tasks.
        
        Parameters
        ----------
        test_data_dir : Path
            Temporary directory for test data.
        dataset_params : callable
            Function to create dataset parameters.
        dataset_name : str
            Name of dataset to test.
        """
        expected = self.DATASET_STATS[dataset_name]

        # Skip if not a classification task
        if expected["task"] != "classification":
            pytest.skip(f"{dataset_name} is not a classification task")

        params = dataset_params(dataset_name)
        loader = MoleculeNetDatasetLoader(params)
        dataset = loader.load_dataset()

        data = dataset[0]

        # For classification, check label dimensions
        if expected["num_classes"] == 1:
            # Binary classification (single label)
            assert data.y.numel() == 1, \
                f"Binary classification labels should be scalar for {dataset_name}"
        else:
            # Multi-label classification
            assert data.y.shape[-1] == expected["num_classes"] or data.y.numel() == expected["num_classes"], \
                f"Multi-label classification should have {expected['num_classes']} labels for {dataset_name}"

    @pytest.mark.parametrize("dataset_name", SMALL_DATASETS)
    def test_data_consistency(self, test_data_dir, dataset_params, dataset_name):
        """Test consistency across multiple graphs in dataset.
        
        Parameters
        ----------
        test_data_dir : Path
            Temporary directory for test data.
        dataset_params : callable
            Function to create dataset parameters.
        dataset_name : str
            Name of dataset to test.
        """
        params = dataset_params(dataset_name)
        loader = MoleculeNetDatasetLoader(params)
        dataset = loader.load_dataset()

        expected = self.DATASET_STATS[dataset_name]

        # Check first 10 graphs (or all if less than 10)
        num_to_check = min(10, len(dataset))

        for i in range(num_to_check):
            data = dataset[i]

            # All graphs should have same number of features
            assert data.x.shape[1] == expected["num_features"], \
                f"Feature dimension mismatch at graph {i} for {dataset_name}"

            # Edge index should be valid
            assert data.edge_index.shape[0] == 2, \
                f"Invalid edge_index shape at graph {i} for {dataset_name}"

            # Labels should exist
            assert data.y is not None, \
                f"Missing labels at graph {i} for {dataset_name}"

    @pytest.mark.parametrize("dataset_name", SMALL_DATASETS)
    def test_no_nan_values(self, test_data_dir, dataset_params, dataset_name):
        """Test that there are no NaN values in features.
        
        Parameters
        ----------
        test_data_dir : Path
            Temporary directory for test data.
        dataset_params : callable
            Function to create dataset parameters.
        dataset_name : str
            Name of dataset to test.
        """
        params = dataset_params(dataset_name)
        loader = MoleculeNetDatasetLoader(params)
        dataset = loader.load_dataset()

        # Check first 10 graphs for NaN values
        num_to_check = min(10, len(dataset))

        for i in range(num_to_check):
            data = dataset[i]

            # Check node features
            assert not torch.isnan(data.x).any(), \
                f"NaN values found in features at graph {i} for {dataset_name}"

            # Check labels (for non-missing labels)
            if data.y is not None and torch.numel(data.y) > 0:
                # Note: Some datasets may have missing labels (NaN) for certain tasks
                # This is expected for multi-task datasets
                pass

    @pytest.mark.parametrize("dataset_name", SMALL_DATASETS + MEDIUM_DATASETS)
    def test_loader_interface(self, test_data_dir, dataset_params, dataset_name):
        """Test that loader implements the required interface.
        
        Parameters
        ----------
        test_data_dir : Path
            Temporary directory for test data.
        dataset_params : callable
            Function to create dataset parameters.
        dataset_name : str
            Name of dataset to test.
        """
        params = dataset_params(dataset_name)
        loader = MoleculeNetDatasetLoader(params)

        # Test load_dataset method
        assert hasattr(loader, 'load_dataset'), \
            "Loader should have load_dataset method"

        dataset = loader.load_dataset()
        assert dataset is not None, \
            "load_dataset should return a dataset"

        # Test that dataset is iterable
        try:
            _ = dataset[0]
        except Exception as e:
            pytest.fail(f"Dataset should be indexable: {e}")

    def test_invalid_dataset_name(self, test_data_dir, dataset_params):
        """Test that invalid dataset name is handled properly.
        
        Parameters
        ----------
        test_data_dir : Path
            Temporary directory for test data.
        dataset_params : callable
            Function to create dataset parameters.
        """
        params = dataset_params("InvalidDataset")
        loader = MoleculeNetDatasetLoader(params)

        # Should raise an error when trying to load invalid dataset
        with pytest.raises(Exception):
            loader.load_dataset()

    @pytest.mark.slow
    @pytest.mark.parametrize("dataset_name", LARGE_DATASETS)
    def test_large_datasets(self, test_data_dir, dataset_params, dataset_name):
        """Test large datasets (marked as slow tests).
        
        These tests are marked with @pytest.mark.slow and should be run
        separately with: pytest -m slow
        
        Parameters
        ----------
        test_data_dir : Path
            Temporary directory for test data.
        dataset_params : callable
            Function to create dataset parameters.
        dataset_name : str
            Name of dataset to test.
        """
        params = dataset_params(dataset_name)
        loader = MoleculeNetDatasetLoader(params)
        dataset = loader.load_dataset()

        expected = self.DATASET_STATS[dataset_name]

        # Just check basic properties for large datasets
        # Some datasets have invalid SMILES that get skipped during processing
        if "min_graphs" in expected:
            assert len(dataset) >= expected["min_graphs"], \
                f"Dataset has fewer graphs than minimum for {dataset_name}: {len(dataset)} < {expected['min_graphs']}"
            assert len(dataset) <= expected["num_graphs"], \
                f"Dataset has more graphs than expected for {dataset_name}: {len(dataset)} > {expected['num_graphs']}"
        else:
            assert len(dataset) == expected["num_graphs"], \
                f"Incorrect number of graphs for {dataset_name}"

        # Check first graph only
        data = dataset[0]
        assert data.x.shape[1] == expected["num_features"], \
            f"Incorrect number of features for {dataset_name}"
