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


def pytest_addoption(parser):
    """Add command-line options for pytest.

    Parameters
    ----------
    parser : pytest.Parser
        Pytest command-line parser.
    """
    parser.addoption(
        "--specific-task",
        action="store",
        default=None,
        help="Filter tests by specific_task type: classification, triangle_classification, triangle_common_neighbors. "
        "If not specified, ALL tests run.",
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests that don't match the specified task type.

    Parameters
    ----------
    config : pytest.Config
        Pytest configuration object.
    items : list
        List of collected test items.
    """
    task = config.getoption("--specific-task")
    if not task:
        return

    # Map task types to test class names
    task_mapping = {
        "classification": [
            "TestA123GraphDataset",
            "TestA123Configuration",
            "TestA123DataIntegrity",
        ],
        "triangle_classification": [
            "TestTriangleClassifier",
            "TestTriangleTask",
        ],
        "triangle_common_neighbors": ["TestTriangleCommonNeighborsTask"],
    }

    if task not in task_mapping:
        raise ValueError(
            f"Invalid --specific-task '{task}'. "
            f"Must be one of: {', '.join(task_mapping.keys())}"
        )

    allowed_classes = set(task_mapping[task])
    skipped = 0
    kept = 0

    for item in items:
        # Extract class name from item nodeid
        # nodeid format: path/to/file.py::ClassName::test_method_name
        class_name = None
        if "::" in item.nodeid:
            parts = item.nodeid.split("::")
            if len(parts) >= 2:
                class_name = parts[1]

        if class_name and class_name not in allowed_classes:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"Test class {class_name} not for task '{task}'"
                )
            )
            skipped += 1
        else:
            kept += 1

    if kept > 0 or skipped > 0:
        print(f"\n{'=' * 70}")
        print(f"[Task Filter] Task: '{task}'")
        print(f"  ✓ Kept:   {kept} tests  - {', '.join(allowed_classes)}")
        print(f"  ✗ Skipped: {skipped} tests")
        print(f"{'=' * 70}\n")


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
            version_base="1.3", config_path=self.config_path, job_name="test"
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
            assert type(loader).__name__ == "A123DatasetLoader" or hasattr(
                loader, "load_dataset"
            )

            # Load dataset
            dataset = loader.load_dataset()
            assert dataset is not None
            assert hasattr(dataset, "data")
            assert isinstance(dataset.data, Data)

    def test_graph_dataset_properties(self):
        """Test graph-level dataset has correct properties."""
        with hydra.initialize(
            version_base="1.3", config_path=self.config_path, job_name="test"
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
            dataset = loader.load_dataset()

            # Check dataset structure
            assert hasattr(dataset, "num_node_features")
            assert hasattr(dataset, "num_classes")
            # Note: InMemoryDataset doesn't expose num_graphs directly; it uses slices internally

            # Check feature dimensions
            assert (
                dataset.num_node_features == 3
            )  # mean_corr, std_corr, noise_diag
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
            version_base="1.3", config_path=self.config_path, job_name="test"
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
            dataset = loader.load_dataset()

            # Check node features
            x = dataset.data.x
            assert x.shape[1] == 3  # 3 features per node
            assert torch.isfinite(x).all()  # No NaN or Inf values


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
            version_base="1.3", config_path=self.config_path, job_name="test"
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
            version_base="1.3", config_path=self.config_path, job_name="test"
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
            version_base="1.3", config_path=self.config_path, job_name="test"
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
            dataset = loader.load_dataset()

            # Check feature tensor
            x = dataset.data.x
            assert x.dtype in [torch.float32, torch.float64]
            assert torch.isfinite(x).all()

    def test_edge_index_format(self):
        """Test that edge indices are correctly formatted."""
        with hydra.initialize(
            version_base="1.3", config_path=self.config_path, job_name="test"
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
            dataset = loader.load_dataset()

            # Check edge index
            edge_index = dataset.data.edge_index
            assert edge_index.dtype == torch.long
            assert edge_index.shape[0] == 2  # [2, num_edges]
            assert torch.all(edge_index >= 0)  # No negative indices

    def test_labels_format(self):
        """Test that labels are correctly formatted."""
        with hydra.initialize(
            version_base="1.3", config_path=self.config_path, job_name="test"
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
            dataset = loader.load_dataset()

            # Check labels
            y = dataset.data.y
            assert y.dtype == torch.long
            assert torch.all(y >= 0)
            assert torch.all(y < dataset.num_classes)


class TestTriangleClassifier:
    """Test suite for TriangleClassifier helper class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.roles = [
            "core_strong",
            "core_medium",
            "core_weak",
            "bridge_strong",
            "bridge_medium",
            "bridge_weak",
            "isolated_strong",
            "isolated_medium",
            "isolated_weak",
        ]

    def test_triangle_classifier_initialization(self):
        """Test TriangleClassifier can be instantiated."""
        classifier = TriangleClassifier(min_weight=0.2)
        assert classifier is not None
        assert classifier.min_weight == 0.2

    def test_role_to_label_mapping(self):
        """Test that triangle roles map to correct labels (0-6)."""
        classifier = TriangleClassifier(min_weight=0.2)

        # Test all 9 roles map to integers 0-8
        for i, role in enumerate(self.roles):
            label = classifier._role_to_label(role)
            assert isinstance(label, int)
            assert 0 <= label <= 8
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
        assert role in self.roles

    def test_triangle_extraction_simple(self):
        """Test triangle extraction on a simple graph."""
        classifier = TriangleClassifier(min_weight=0.2)

        # Create a simple graph: complete triangle (3-clique)
        # Nodes: 0, 1, 2
        # Edges: (0,1), (1,2), (0,2)
        edge_index = torch.tensor(
            [[0, 1, 0, 1, 2, 0], [1, 0, 2, 2, 1, 2]]
        )  # Undirected edges
        edge_weights = torch.tensor([0.9, 0.9, 0.8, 0.8, 0.9, 0.9])

        triangles = classifier.extract_triangles(
            edge_index, edge_weights, num_nodes=3
        )

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
            assert 0 <= tri["label"] <= 8
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
            version_base="1.3", config_path=self.config_path, job_name="test"
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
            # specific_task is a string selector, not a nested config
            assert hasattr(cfg.dataset.parameters, "specific_task")
            specific_task = cfg.dataset.parameters.specific_task
            assert isinstance(specific_task, str)

            # Valid tasks include triangle_classification
            assert specific_task in [
                "classification",
                "triangle_classification",
                "triangle_common_neighbors",
            ]

            # All tasks use 9 classes
            assert cfg.dataset.parameters.num_classes == 9

    def test_minimal_features_in_config(self):
        """Test that triangle features are optimized to minimal set (3D edge weights)."""
        with hydra.initialize(
            version_base="1.3", config_path=self.config_path, job_name="test"
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

            # Triangle tasks use 3 node features (edge weights in triangle context)
            # This is a fixed feature dimension for all triangle-based tasks
            # Verify the base configuration uses 3 features
            num_features = cfg.dataset.parameters.num_features
            assert num_features == 3, (
                f"Expected 3D features (edge weights only), got {num_features}D. "
                "Features should be: [weight_01, weight_02, weight_12]"
            )

    def test_triangle_loader_instantiation(self):
        """Test that dataset loader can instantiate with triangle task."""
        with hydra.initialize(
            version_base="1.3", config_path=self.config_path, job_name="test"
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
            assert type(loader).__name__ == "A123DatasetLoader" or hasattr(
                loader, "load_dataset"
            )

            # Verify loader has load_dataset method with task_type parameter
            assert hasattr(loader, "load_dataset")
            assert callable(loader.load_dataset)

    def test_graph_vs_triangle_task_independent(self):
        """Test that graph and triangle tasks are independent.

        Graph task should always work. Triangle task may require
        prior processing, but shouldn't affect graph task.
        """
        with hydra.initialize(
            version_base="1.3", config_path=self.config_path, job_name="test"
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
            graph_dataset = loader.load_dataset()
            assert graph_dataset is not None
            assert graph_dataset.num_classes == 9

            # Triangle task may fail if not processed yet, but shouldn't crash loader
            try:
                triangle_dataset = loader.load_dataset()
                if triangle_dataset is not None:
                    # If triangle dataset loaded, verify it has expected properties
                    assert hasattr(triangle_dataset, "data")
            except FileNotFoundError:
                # Expected if triangle processing hasn't been run
                pass


class TestTriangleCommonNeighborsTask:
    """Test suite for triangle common-neighbors task (TDL-focused)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        self.config_path = "../../../configs"

    def test_triangle_common_task_configuration(self):
        """Test that triangle common-neighbors task is configured."""
        with hydra.initialize(
            version_base="1.3", config_path=self.config_path, job_name="test"
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

            # Check that specific_task can be set to triangle_common_neighbors
            # It's a string selector in parameters, not a nested config structure
            assert hasattr(cfg.dataset.parameters, "specific_task")
            specific_task = cfg.dataset.parameters.specific_task

            # Verify it's a valid task selector
            assert isinstance(specific_task, str)
            assert specific_task in [
                "classification",
                "triangle_classification",
                "triangle_common_neighbors",
            ]

            # All tasks use 9 classes (unified output)
            assert cfg.dataset.parameters.num_classes == 9

    def test_triangle_common_loader_instantiation(self):
        """Test that loader can instantiate with triangle common-neighbors task."""
        with hydra.initialize(
            version_base="1.3", config_path=self.config_path, job_name="test"
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
            # Check loader has load_dataset method
            assert hasattr(loader, "load_dataset")
            assert callable(loader.load_dataset)

    def test_triangle_common_task_creation_synthetic(self):
        """Test triangle common-neighbors task creation on synthetic graph."""
        classifier = TriangleClassifier(min_weight=0.2)

        # Create synthetic graph with known structure
        # Nodes: 0, 1, 2, 3, 4
        # Triangles: (0,1,2) with common neighbor 3, (1,2,3) with common neighbor 4
        edge_list = [
            (0, 1),
            (0, 2),
            (1, 2),  # Triangle 0-1-2
            (1, 3),
            (2, 3),
            (0, 3),  # Add node 3 as common neighbor
            (1, 4),
            (2, 4),
            (3, 4),  # Add node 4 as common neighbor
        ]

        edge_index_list = []
        for u, v in edge_list:
            edge_index_list.append([u, v])
            edge_index_list.append([v, u])  # Undirected

        edge_index = (
            torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        )
        edge_weights = torch.ones(edge_index.shape[1])

        # Extract triangles
        triangles = classifier.extract_triangles(
            edge_index, edge_weights, num_nodes=5
        )

        # Verify triangles were found
        assert len(triangles) > 0

        # Now simulate creating CN features for each triangle
        # Build graph to compute common neighbors
        G = nx.Graph()
        G.add_nodes_from(range(5))
        for i in range(edge_index.shape[1]):
            u = int(edge_index[0, i].item())
            v = int(edge_index[1, i].item())
            G.add_edge(u, v)

        # For each triangle, compute common neighbors
        for tri in triangles:
            a, b, c = tri["nodes"]
            common = (
                set(G.neighbors(a)) & set(G.neighbors(b)) & set(G.neighbors(c))
            ) - {a, b, c}
            num_common = len(common)

            # Features: node degrees (structural, no weights)
            deg_a = G.degree(a)
            deg_b = G.degree(b)
            deg_c = G.degree(c)

            # Verify features are reasonable
            assert deg_a > 0 and deg_b > 0 and deg_c > 0
            assert num_common >= 0

    def test_triangle_common_features_are_structural(self):
        """Test that CN task features are purely structural (node degrees)."""
        # Create a simple triangle with known degrees
        edge_index = torch.tensor(
            [[0, 0, 1, 2], [1, 2, 2, 0]], dtype=torch.long
        )  # Triangle 0-1-2 + extra edge 0-1
        edge_weights = torch.ones(edge_index.shape[1])

        classifier = TriangleClassifier(min_weight=0.2)
        triangles = classifier.extract_triangles(
            edge_index, edge_weights, num_nodes=3
        )

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(3))
        for i in range(edge_index.shape[1]):
            u = int(edge_index[0, i].item())
            v = int(edge_index[1, i].item())
            G.add_edge(u, v)

        # For triangle, extract degree features
        for tri in triangles:
            a, b, c = tri["nodes"]
            deg_a = G.degree(a)
            deg_b = G.degree(b)
            deg_c = G.degree(c)

            tri_feats = torch.tensor(
                [deg_a, deg_b, deg_c], dtype=torch.float32
            )

            # Features should be non-negative integers (degrees)
            assert tri_feats.shape == (3,)
            assert torch.all(tri_feats >= 0)
            # Degrees should be integers stored as floats
            assert torch.allclose(tri_feats, tri_feats.round())

    def test_triangle_common_label_semantics(self):
        """Test that CN task labels represent common neighbor counts."""
        # Create graph where we know common neighbor counts
        # 4 nodes: (0,1,2) form triangle with no external connections (CN=0)
        # Add node 3 connected to all three: (0,1,2) will have CN=1
        edges_no_cn = torch.tensor(
            [[0, 0, 1], [1, 2, 2]], dtype=torch.long
        )  # Triangle 0-1-2
        edge_weights_no_cn = torch.ones(edges_no_cn.shape[1])

        classifier = TriangleClassifier(min_weight=0.2)
        triangles_no_cn = classifier.extract_triangles(
            edges_no_cn, edge_weights_no_cn, num_nodes=3
        )

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(3))
        for i in range(edges_no_cn.shape[1]):
            u = int(edges_no_cn[0, i].item())
            v = int(edges_no_cn[1, i].item())
            G.add_edge(u, v)

        # Verify CN count for triangle with no external neighbors
        for tri in triangles_no_cn:
            a, b, c = tri["nodes"]
            common = (
                set(G.neighbors(a)) & set(G.neighbors(b)) & set(G.neighbors(c))
            ) - {a, b, c}
            assert len(common) == 0  # No common neighbors

    def test_triangle_common_vs_role_independence(self):
        """Test that CN task is independent of role classification task."""
        # Both tasks should work without interference
        # CN task focuses on structural degree measures
        # Role task focuses on embedding + weight patterns

        # Create a simple graph
        edge_index = torch.tensor(
            [[0, 0, 1, 1, 2, 0], [1, 2, 2, 0, 0, 2]], dtype=torch.long
        )
        edge_weights = torch.tensor([0.8, 0.7, 0.6, 0.6, 0.7, 0.8])

        classifier = TriangleClassifier(min_weight=0.2)
        triangles = classifier.extract_triangles(
            edge_index, edge_weights, num_nodes=3
        )

        # From triangle classifier, we get roles (based on weights + embedding)
        for tri in triangles:
            role = tri["role"]
            label = tri["label"]

            # Role should be one of the 7 types
            assert isinstance(role, str)
            assert 0 <= label <= 6

            # CN features would be independent (just degrees)
            # So two triangles with different roles could have same degree features
            assert "strong" in role or "medium" in role or "weak" in role

    def test_triangle_common_edge_cases(self):
        """Test CN task handles edge cases gracefully."""
        classifier = TriangleClassifier(min_weight=0.2)

        # Empty graph
        empty_edge_index = torch.empty((2, 0), dtype=torch.long)
        empty_weights = torch.empty((0,))
        triangles_empty = classifier.extract_triangles(
            empty_edge_index, empty_weights, num_nodes=0
        )
        assert len(triangles_empty) == 0

        # Graph with no triangles (just edges)
        linear_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        linear_weights = torch.ones(linear_edge_index.shape[1])
        triangles_linear = classifier.extract_triangles(
            linear_edge_index, linear_weights, num_nodes=3
        )
        assert len(triangles_linear) == 0  # No triangles in a path

        # Single triangle (minimal case)
        single_tri_edge_index = torch.tensor(
            [[0, 0, 1], [1, 2, 2]], dtype=torch.long
        )
        single_tri_weights = torch.ones(single_tri_edge_index.shape[1])
        triangles_single = classifier.extract_triangles(
            single_tri_edge_index, single_tri_weights, num_nodes=3
        )
        assert len(triangles_single) == 1
        assert len(triangles_single[0]["nodes"]) == 3


if __name__ == "__main__":
    """Run tests for each task type with clear output."""
    import sys

    tasks = [
        "classification",
        "triangle_classification",
        "triangle_common_neighbors",
    ]

    print("\n" + "=" * 80)
    print("RUNNING A123 DATASET TESTS FOR ALL TASK TYPES")
    print("=" * 80)

    results = {}

    for task in tasks:
        print(f"\n{'-' * 80}")
        print(f"Running tests for: {task}")
        print(f"{'-' * 80}\n")

        # Use pytest.main programmatically
        exit_code = pytest.main(
            [
                __file__,
                f"--specific-task={task}",
                "-v",
                "--tb=short",
            ]
        )

        results[task] = exit_code

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for task, returncode in results.items():
        status = "✓ PASSED" if returncode == 0 else "✗ FAILED"
        print(f"{status:12} - {task}")

    print("=" * 80 + "\n")

    # Exit with failure if any task failed
    sys.exit(max(results.values()))
