"""Tests for Transform DAG functionality."""

import pytest
from torch_geometric.data import Data

from topobench.data.preprocessor._ondisk.transform_dag import (
    TransformDAG,
    TransformNode,
)
from topobench.data.preprocessor._ondisk.transform_pipeline import (
    TransformPipeline,
)


class MockTransform:
    """Simple mock transform for testing."""

    def __init__(self, name: str):
        self.name = name
        self.parameters = {"name": name}

    def __call__(self, data: Data) -> Data:
        """Apply transform (identity operation for testing)."""
        return data


class TestTransformNode:
    """Test TransformNode dataclass."""

    def test_node_creation_and_serialization(self):
        """Test node creation with dependencies and serialization."""
        transform = MockTransform("test")

        # Test basic creation
        node = TransformNode(
            transform=transform,
            transform_id="MockTransform_0",
            tier="heavy",
            hash_value="abc123",
            dependencies=["OtherTransform_0"],
        )

        assert node.transform_id == "MockTransform_0"
        assert node.tier == "heavy"
        assert node.hash_value == "abc123"
        assert len(node.dependencies) == 1
        assert node.dependencies[0] == "OtherTransform_0"

        # Test serialization
        data = node.to_dict()
        assert data["transform_id"] == "MockTransform_0"
        assert data["transform_class"] == "MockTransform"
        assert data["tier"] == "heavy"
        assert data["hash_value"] == "abc123"
        assert data["dependencies"] == ["OtherTransform_0"]


class TestTransformDAG:
    """Test TransformDAG class."""

    def test_add_transforms_with_dependencies(self):
        """Test adding single/multiple transforms with dependencies."""
        dag = TransformDAG()

        # Empty initialization
        assert len(dag.nodes) == 0
        assert len(dag.execution_order) == 0

        # Add first transform
        t1 = MockTransform("transform1")
        id1 = dag.add_transform(t1, tier="heavy")

        assert id1 == "MockTransform_0"
        assert id1 in dag.nodes
        assert id1 in dag.execution_order
        assert dag.nodes[id1].tier == "heavy"

        # Add second transform with dependency
        t2 = MockTransform("transform2")
        id2 = dag.add_transform(t2, tier="light", dependencies=[id1])

        assert id2 == "MockTransform_1"
        assert len(dag.nodes) == 2
        assert len(dag.execution_order) == 2
        assert id1 in dag.nodes[id2].dependencies

    def test_hash_computation(self):
        """Test per-transform and combined hash computation."""
        dag = TransformDAG()
        t1 = MockTransform("test1")
        t2 = MockTransform("test2")

        id1 = dag.add_transform(t1, tier="heavy")
        id2 = dag.add_transform(t2, tier="light")

        # Per-transform hashes
        hash1 = dag.get_transform_hash(id1)
        hash2 = dag.get_transform_hash(id2)

        assert hash1 != hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0

        # Combined pipeline hash
        hash_combined = dag.compute_pipeline_hash([id1, id2])
        assert isinstance(hash_combined, str)
        assert hash_combined != hash1
        assert hash_combined != hash2

    def test_affected_transforms_linear_chain(self):
        """Test affected transforms in linear dependency chain (core DAG functionality)."""
        dag = TransformDAG()
        t1 = MockTransform("t1")
        t2 = MockTransform("t2")
        t3 = MockTransform("t3")

        id1 = dag.add_transform(t1, tier="heavy")
        id2 = dag.add_transform(t2, tier="heavy", dependencies=[id1])
        id3 = dag.add_transform(t3, tier="light", dependencies=[id2])

        # Changing first transform affects all downstream
        affected = dag.get_affected_transforms(id1)
        assert set(affected) == {id1, id2, id3}

        # Changing middle transform affects itself and downstream
        affected = dag.get_affected_transforms(id2)
        assert set(affected) == {id2, id3}

        # Changing last transform affects only itself
        affected = dag.get_affected_transforms(id3)
        assert set(affected) == {id3}

    def test_error_handling(self):
        """Test error handling for invalid transform IDs."""
        dag = TransformDAG()

        # Invalid hash query
        with pytest.raises(KeyError, match="Transform invalid_id not in DAG"):
            dag.get_transform_hash("invalid_id")

        # Invalid affected transforms query
        with pytest.raises(KeyError, match="Transform invalid_id not in DAG"):
            dag.get_affected_transforms("invalid_id")

    def test_serialization(self):
        """Test DAG serialization and deserialization."""
        dag = TransformDAG()
        t1 = MockTransform("t1")
        t2 = MockTransform("t2")

        id1 = dag.add_transform(t1, tier="heavy")
        _id2 = dag.add_transform(t2, tier="light", dependencies=[id1])

        # Serialize
        data = dag.to_dict()
        assert "nodes" in data
        assert "execution_order" in data
        assert len(data["nodes"]) == 2
        assert len(data["execution_order"]) == 2

        # Deserialize
        dag2 = TransformDAG.from_dict(data)
        assert dag2.execution_order == dag.execution_order

        # Test repr
        repr_str = repr(dag)
        assert "TransformDAG" in repr_str
        assert "nodes=2" in repr_str
        assert "heavy=1" in repr_str
        assert "light=1" in repr_str


class TestTransformPipelineDAGIntegration:
    """Test DAG integration with TransformPipeline."""

    def test_pipeline_builds_dag_automatically(self):
        """Test pipeline automatically builds DAG with correct structure."""
        transforms = [MockTransform(f"t{i}") for i in range(3)]
        pipeline = TransformPipeline(transforms, transform_tier="all_heavy")

        dag = pipeline.get_dag()

        # DAG created with correct size
        assert len(dag.nodes) == 3
        assert len(dag.execution_order) == 3

        # All transforms are heavy
        heavy_nodes = [n for n in dag.nodes.values() if n.tier == "heavy"]
        assert len(heavy_nodes) == 3

        # Summary includes DAG info
        summary = pipeline.get_summary()
        assert "dag_nodes" in summary
        assert summary["dag_nodes"] == 3

    def test_dag_sequential_dependencies(self):
        """Test DAG has correct sequential dependencies (N depends on N-1)."""
        transforms = [MockTransform(f"t{i}") for i in range(3)]
        pipeline = TransformPipeline(transforms, transform_tier="all_heavy")

        dag = pipeline.get_dag()
        nodes_list = [dag.nodes[tid] for tid in dag.execution_order]

        # First node has no dependencies
        assert len(nodes_list[0].dependencies) == 0

        # Subsequent nodes depend on previous
        assert len(nodes_list[1].dependencies) == 1
        assert nodes_list[1].dependencies[0] == nodes_list[0].transform_id

        assert len(nodes_list[2].dependencies) == 1
        assert nodes_list[2].dependencies[0] == nodes_list[1].transform_id

    def test_affected_transforms_through_pipeline(self):
        """Test accessing affected transforms through pipeline (core use case)."""
        transforms = [MockTransform(f"t{i}") for i in range(3)]
        pipeline = TransformPipeline(transforms, transform_tier="all_heavy")

        dag = pipeline.get_dag()
        first_id = dag.execution_order[0]

        # Changing first transform affects all downstream
        affected = dag.get_affected_transforms(first_id)
        assert len(affected) == 3
        assert set(affected) == set(dag.execution_order)
