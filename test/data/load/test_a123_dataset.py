"""Unit tests for A123 mouse auditory cortex dataset.

Tests cover:
- Graph-level dataset loading and structure
- Triangle classification task functionality
- Triangle role classification logic
- Feature dimensions and data integrity
- Configuration parameter handling
"""

import pytest
import torch
import hydra
import networkx as nx
from pathlib import Path
from omegaconf import DictConfig
from torch_geometric.data import Data

# Import the dataset and loader
from topobench.data.datasets.a123 import A123CortexMDataset, TriangleClassifier
from topobench.data.loaders.graph.a123_loader import A123DatasetLoader


class TestA123GraphDataset:
    """Test suite for A123 graph-level dataset."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        self.config_path = "../../../configs"

    def test_dataset_loading(self):
        """Test basic dataset loading and instantiation."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.config_path,
            job_name="test"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "dataset=graph/a123",
                    "model=graph/gat",
                    "paths=test",
                ],
                return_hydra_config=True,
            )

            # Instantiate loader
            loader = hydra.utils.instantiate(cfg.dataset.loader)
            # Check loader type using class name to handle module reloading issues
            assert type(loader).__name__ == "A123DatasetLoader" or hasattr(loader, "load_dataset")

            # Load dataset
            dataset = loader.load_dataset(task_type="classification")
            assert dataset is not None
            assert hasattr(dataset, "data")
            assert isinstance(dataset.data, Data)

    def test_graph_dataset_properties(self):
        """Test graph-level dataset has correct properties."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.config_path,
            job_name="test"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "dataset=graph/a123",
                    "model=graph/gat",
                    "paths=test",
                ],
                return_hydra_config=True,
            )

            loader = hydra.utils.instantiate(cfg.dataset.loader)
            dataset = loader.load_dataset(task_type="classification")

            # Check dataset structure
            assert hasattr(dataset, "num_node_features")
            assert hasattr(dataset, "num_classes")
            # Note: InMemoryDataset doesn't expose num_graphs directly; it uses slices internally

            # Check feature dimensions
            assert dataset.num_node_features == 3  # mean_corr, std_corr, noise_diag
            assert dataset.num_classes == 9  # 9 frequency bins

            # Check data integrity
            assert dataset.data.x is not None
            assert dataset.data.edge_index is not None
            assert dataset.data.y is not None

            # Check labels are in valid range
            assert torch.all(dataset.data.y >= 0)
            assert torch.all(dataset.data.y < dataset.num_classes)

    def test_graph_node_features(self):
        """Test that node features are correctly structured."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.config_path,
            job_name="test"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "dataset=graph/a123",
                    "model=graph/gat",
                    "paths=test",
                ],
                return_hydra_config=True,
            )

            loader = hydra.utils.instantiate(cfg.dataset.loader)
            dataset = loader.load_dataset(task_type="classification")

            # Check node features
            x = dataset.data.x
            assert x.shape[1] == 3  # 3 features per node
            assert torch.isfinite(x).all()  # No NaN or Inf values


class TestTriangleClassifier:
    """Test suite for TriangleClassifier helper class."""

    def test_triangle_classifier_initialization(self):
        """Test TriangleClassifier can be instantiated."""
        classifier = TriangleClassifier(min_weight=0.2)
        assert classifier is not None
        assert classifier.min_weight == 0.2

    def test_role_to_label_mapping(self):
        """Test that triangle roles map to correct labels (0-6)."""
        classifier = TriangleClassifier(min_weight=0.2)

        # Test all 7 roles map to integers 0-6
        roles = [
            "core_strong",
            "core_medium",
            "bridge_strong",
            "bridge_medium",
            "isolated_strong",
            "isolated_medium",
            "isolated_weak",
        ]

        for i, role in enumerate(roles):
            label = classifier._role_to_label(role)
            assert isinstance(label, int)
            assert 0 <= label <= 6
            # Verify deterministic mapping (same role -> same label)
            assert label == classifier._role_to_label(role)

    def test_role_classification_logic(self):
        """Test that role classification works with synthetic triangle data."""
        classifier = TriangleClassifier(min_weight=0.2)

        # Create a simple networkx graph with a triangle
        G = nx.Graph()
        G.add_edge(0, 1, weight=0.8)
        G.add_edge(1, 2, weight=0.7)
        G.add_edge(0, 2, weight=0.6)
        # Add some other nodes connected to all three to test embedding class
        G.add_edge(3, 0, weight=0.5)
        G.add_edge(3, 1, weight=0.5)
        G.add_edge(3, 2, weight=0.5)

        # Test role classification with the graph
        nodes = (0, 1, 2)
        edge_weights = [0.8, 0.7, 0.6]
        
        role = classifier._classify_role(G, nodes, edge_weights)
        assert role is not None
        assert isinstance(role, str)
        assert role in [
            "core_strong",
            "core_medium",
            "bridge_strong",
            "bridge_medium",
            "isolated_strong",
            "isolated_medium",
            "isolated_weak",
        ]

    def test_triangle_extraction_simple(self):
        """Test triangle extraction on a simple graph."""
        classifier = TriangleClassifier(min_weight=0.2)

        # Create a simple graph: complete triangle (3-clique)
        # Nodes: 0, 1, 2
        # Edges: (0,1), (1,2), (0,2)
        edge_index = torch.tensor([[0, 1, 0, 1, 2, 0],
                                   [1, 0, 2, 2, 1, 2]])  # Undirected edges
        edge_weights = torch.tensor([0.9, 0.9, 0.8, 0.8, 0.9, 0.9])

        triangles = classifier.extract_triangles(edge_index, edge_weights, num_nodes=3)

        # Should find at least one triangle
        assert len(triangles) > 0

        # Each triangle should have required fields
        for tri in triangles:
            assert "nodes" in tri
            assert "edge_weights" in tri
            assert "label" in tri
            assert "role" in tri

            # Verify structure
            assert len(tri["nodes"]) == 3
            assert len(tri["edge_weights"]) == 3
            assert isinstance(tri["label"], int)
            assert 0 <= tri["label"] <= 6
            assert isinstance(tri["role"], str)


class TestTriangleTask:
    """Test suite for triangle classification task functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        self.config_path = "../../../configs"

    def test_triangle_task_configuration(self):
        """Test that triangle task is properly configured."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.config_path,
            job_name="test"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "dataset=graph/a123",
                    "model=graph/gat",
                    "paths=test",
                ],
                return_hydra_config=True,
            )

            # Check triangle task configuration
            assert hasattr(cfg.dataset.parameters, "triangle_task")
            tri_cfg = cfg.dataset.parameters.triangle_task
            assert hasattr(tri_cfg, "enabled")
            assert hasattr(tri_cfg, "num_triangle_classes")
            assert hasattr(tri_cfg, "num_triangle_features")

            # Check values
            assert tri_cfg.num_triangle_classes == 7
            assert tri_cfg.num_triangle_features == 3  # Optimized: edge weights only

    def test_minimal_features_in_config(self):
        """Test that triangle features are optimized to minimal set (3D edge weights)."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.config_path,
            job_name="test"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "dataset=graph/a123",
                    "model=graph/gat",
                    "paths=test",
                ],
                return_hydra_config=True,
            )

            # Verify features are minimal (3D)
            num_features = cfg.dataset.parameters.triangle_task.num_triangle_features
            assert num_features == 3, (
                f"Expected 3D features (edge weights only), got {num_features}D. "
                "Features should be: [weight_01, weight_02, weight_12]"
            )

    def test_triangle_loader_instantiation(self):
        """Test that dataset loader can instantiate with triangle task."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.config_path,
            job_name="test"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "dataset=graph/a123",
                    "model=graph/gat",
                    "paths=test",
                ],
                return_hydra_config=True,
            )

            loader = hydra.utils.instantiate(cfg.dataset.loader)
            # Check loader type using class name to handle module reloading issues
            assert type(loader).__name__ == "A123DatasetLoader" or hasattr(loader, "load_dataset")

            # Verify loader has load_dataset method with task_type parameter
            assert hasattr(loader, "load_dataset")
            assert callable(loader.load_dataset)

    def test_graph_vs_triangle_task_independent(self):
        """Test that graph and triangle tasks are independent.
        
        Graph task should always work. Triangle task may require
        prior processing, but shouldn't affect graph task.
        """
        with hydra.initialize(
            version_base="1.3",
            config_path=self.config_path,
            job_name="test"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "dataset=graph/a123",
                    "model=graph/gat",
                    "paths=test",
                ],
                return_hydra_config=True,
            )

            loader = hydra.utils.instantiate(cfg.dataset.loader)

            # Graph task should always work
            graph_dataset = loader.load_dataset(task_type="classification")
            assert graph_dataset is not None
            assert graph_dataset.num_classes == 9

            # Triangle task may fail if not processed yet, but shouldn't crash loader
            try:
                triangle_dataset = loader.load_dataset(task_type="triangle")
                if triangle_dataset is not None:
                    # If triangle dataset loaded, verify it has expected properties
                    assert hasattr(triangle_dataset, "data")
            except FileNotFoundError:
                # Expected if triangle processing hasn't been run
                pass


class TestA123Configuration:
    """Test suite for A123 configuration integrity."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        self.config_path = "../../../configs"

    def test_dataset_parameters(self):
        """Test that all required dataset parameters are configured."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.config_path,
            job_name="test"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "dataset=graph/a123",
                    "model=graph/gat",
                    "paths=test",
                ],
                return_hydra_config=True,
            )

            params = cfg.dataset.parameters

            # Check required parameters
            assert hasattr(params, "num_features")
            assert hasattr(params, "num_classes")
            assert hasattr(params, "task")
            assert hasattr(params, "data_name")
            assert hasattr(params, "min_neurons")
            assert hasattr(params, "corr_threshold")

            # Check values are sensible
            assert params.num_features == 3
            assert params.num_classes == 9
            assert params.task == "classification"
            assert params.min_neurons >= 3
            assert 0.0 <= params.corr_threshold <= 1.0

    def test_loader_parameters(self):
        """Test that loader configuration is correct."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.config_path,
            job_name="test"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "dataset=graph/a123",
                    "model=graph/gat",
                    "paths=test",
                ],
                return_hydra_config=True,
            )

            loader_cfg = cfg.dataset.loader

            # Check loader configuration
            assert hasattr(loader_cfg, "parameters")
            params = loader_cfg.parameters

            assert hasattr(params, "data_domain")
            assert hasattr(params, "data_type")
            assert hasattr(params, "is_undirected")

            # Verify values
            assert params.data_domain == "graph"
            assert params.data_type == "A123CortexM"
            assert params.is_undirected is True


class TestA123DataIntegrity:
    """Test suite for data integrity checks."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        self.config_path = "../../../configs"

    def test_feature_format(self):
        """Test that features are in correct format."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.config_path,
            job_name="test"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "dataset=graph/a123",
                    "model=graph/gat",
                    "paths=test",
                ],
                return_hydra_config=True,
            )

            loader = hydra.utils.instantiate(cfg.dataset.loader)
            dataset = loader.load_dataset(task_type="classification")

            # Check feature tensor
            x = dataset.data.x
            assert x.dtype in [torch.float32, torch.float64]
            assert torch.isfinite(x).all()

    def test_edge_index_format(self):
        """Test that edge indices are correctly formatted."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.config_path,
            job_name="test"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "dataset=graph/a123",
                    "model=graph/gat",
                    "paths=test",
                ],
                return_hydra_config=True,
            )

            loader = hydra.utils.instantiate(cfg.dataset.loader)
            dataset = loader.load_dataset(task_type="classification")

            # Check edge index
            edge_index = dataset.data.edge_index
            assert edge_index.dtype == torch.long
            assert edge_index.shape[0] == 2  # [2, num_edges]
            assert torch.all(edge_index >= 0)  # No negative indices

    def test_labels_format(self):
        """Test that labels are correctly formatted."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.config_path,
            job_name="test"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "dataset=graph/a123",
                    "model=graph/gat",
                    "paths=test",
                ],
                return_hydra_config=True,
            )

            loader = hydra.utils.instantiate(cfg.dataset.loader)
            dataset = loader.load_dataset(task_type="classification")

            # Check labels
            y = dataset.data.y
            assert y.dtype == torch.long
            assert torch.all(y >= 0)
            assert torch.all(y < dataset.num_classes)
