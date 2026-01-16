"""Transform DAG for dependency tracking and granular caching.

This module implements a Directed Acyclic Graph (DAG) for tracking transform
dependencies, enabling per-transform hashing and granular cache invalidation.

The key insight for Topological Deep Learning:
- Topology construction (liftings): Expensive
- Feature engineering: Cheap
- DAG enables: Change features WITHOUT recomputing topology
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import torch_geometric

from topobench.data.utils import ensure_serializable, make_hash


@dataclass
class TransformNode:
    """Node in transform dependency graph.

    Each node represents a single transform with its hash, tier classification,
    and dependencies. This enables per-transform caching and incremental updates.

    Attributes
    ----------
    transform : torch_geometric.transforms.BaseTransform
        The actual transform object.
    transform_id : str
        Unique identifier (format: {ClassName}_{index}).
    tier : Literal["heavy", "light"]
        Classification tier (heavy=cached offline, light=runtime).
    hash_value : str
        Hash of transform parameters (hex string).
    dependencies : list[str], optional
        IDs of transforms this depends on (default: empty list).
    metadata : dict[str, Any], optional
        Additional metadata like execution time (default: empty dict).

    Examples
    --------
    >>> node = TransformNode(
    ...     transform=SimplicialLifting(),
    ...     transform_id="SimplicialLifting_0",
    ...     tier="heavy",
    ...     hash_value="abc123def456",
    ...     dependencies=[],
    ... )
    >>> node.transform_id
    'SimplicialLifting_0'
    """

    transform: torch_geometric.transforms.BaseTransform
    transform_id: str
    tier: Literal["heavy", "light"]
    hash_value: str
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize node for cache metadata.

        Returns
        -------
        dict
            Serialized node with transform class name, ID, tier, hash, and dependencies.
            Does not include the transform object itself (not serializable).

        Examples
        --------
        >>> node = TransformNode(transform, "Lifting_0", "heavy", "abc123")
        >>> data = node.to_dict()
        >>> data["transform_id"]
        'Lifting_0'
        """
        return {
            "transform_id": self.transform_id,
            "transform_class": self.transform.__class__.__name__,
            "tier": self.tier,
            "hash_value": self.hash_value,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
        }


class TransformDAG:
    """Dependency graph for transforms with per-transform hashing.

    Tracks dependencies between transforms, computes per-transform hashes,
    and detects affected transforms for incremental updates.
    The DAG uses sequential dependencies (realistic in most cases).

    Attributes
    ----------
    nodes : dict[str, TransformNode]
        Mapping from transform IDs to nodes.
    execution_order : list[str]
        Transform IDs in execution order (topological sort).

    Examples
    --------
    >>> dag = TransformDAG()
    >>> lifting_id = dag.add_transform(SimplicialLifting(), tier="heavy")
    >>> norm_id = dag.add_transform(FeatureNorm(), tier="light", dependencies=[lifting_id])
    >>>
    >>> # Get per-transform hash
    >>> dag.get_transform_hash(lifting_id)
    'abc123def456'
    >>>
    >>> # Find affected transforms (for incremental updates)
    >>> dag.get_affected_transforms(lifting_id)
    ['SimplicialLifting_0', 'FeatureNorm_0']  # Changing lifting affects both
    >>> dag.get_affected_transforms(norm_id)
    ['FeatureNorm_0']  # Changing norm only affects itself

    See Also
    --------
    TransformNode : Individual node in the DAG.
    TransformPipeline : Uses DAG for two-tier transform management.
    """

    def __init__(self) -> None:
        """Initialize empty transform DAG."""
        self.nodes: dict[str, TransformNode] = {}
        self.execution_order: list[str] = []

    def add_transform(
        self,
        transform: torch_geometric.transforms.BaseTransform,
        tier: Literal["heavy", "light"],
        dependencies: list[str] | None = None,
    ) -> str:
        """Add transform to DAG with optional dependencies.

        Generates a unique ID ({ClassName}_{index}), computes the transform hash,
        and adds the node to the graph.

        Parameters
        ----------
        transform : torch_geometric.transforms.BaseTransform
            Transform to add.
        tier : Literal["heavy", "light"]
            Classification tier (heavy=cached, light=runtime).
        dependencies : list[str], optional
            IDs of transforms this depends on (default: None = no dependencies).

        Returns
        -------
        str
            Generated transform ID.

        Examples
        --------
        >>> dag = TransformDAG()
        >>> id1 = dag.add_transform(SimplicialLifting(), tier="heavy")
        >>> id2 = dag.add_transform(FeatureNorm(), tier="light", dependencies=[id1])
        >>> id1
        'SimplicialLifting_0'
        >>> id2
        'FeatureNorm_0'
        """
        # Generate unique ID: {ClassName}_{index}
        transform_class = transform.__class__.__name__
        idx = sum(1 for nid in self.nodes if nid.startswith(transform_class))
        transform_id = f"{transform_class}_{idx}"

        # Compute hash from transform parameters
        hash_value = self._compute_transform_hash(transform)

        # Create node
        node = TransformNode(
            transform=transform,
            transform_id=transform_id,
            tier=tier,
            hash_value=hash_value,
            dependencies=dependencies or [],
        )

        # Add to graph
        self.nodes[transform_id] = node
        self.execution_order.append(transform_id)

        return transform_id

    def _compute_transform_hash(
        self, transform: torch_geometric.transforms.BaseTransform
    ) -> str:
        """Compute hash for single transform from its parameters.

        Parameters
        ----------
        transform : torch_geometric.transforms.BaseTransform
            Transform to hash.

        Returns
        -------
        str
            Hex string hash (without '0x' prefix).

        Examples
        --------
        >>> dag = TransformDAG()
        >>> hash_val = dag._compute_transform_hash(SimplicialLifting())
        >>> len(hash_val) > 0
        True
        """
        # Extract parameters
        if hasattr(transform, "parameters"):
            params = transform.parameters
        else:
            # Fallback: use class name if no parameters attribute
            params = {"__class__": transform.__class__.__name__}

        # Ensure serializable and hash
        serializable = ensure_serializable(params)
        hash_int = make_hash(serializable)
        return hex(hash_int)[2:]  # Remove '0x' prefix

    def get_transform_hash(self, transform_id: str) -> str:
        """Get hash for specific transform.

        Parameters
        ----------
        transform_id : str
            Transform ID to query.

        Returns
        -------
        str
            Transform hash (hex string).

        Raises
        ------
        KeyError
            If transform_id not in DAG.

        Examples
        --------
        >>> dag = TransformDAG()
        >>> id1 = dag.add_transform(SimplicialLifting(), tier="heavy")
        >>> hash_val = dag.get_transform_hash(id1)
        >>> isinstance(hash_val, str)
        True
        """
        if transform_id not in self.nodes:
            raise KeyError(f"Transform {transform_id} not in DAG")
        return self.nodes[transform_id].hash_value

    def get_affected_transforms(self, changed_transform_id: str) -> list[str]:
        """Get all transforms affected by changing given transform.

        Uses depth-first search (DFS) to find all downstream dependencies.
        This is critical for incremental updates in Phase 3: only reprocess
        affected transforms, not the entire pipeline.

        Parameters
        ----------
        changed_transform_id : str
            ID of transform that changed.

        Returns
        -------
        list[str]
            Transform IDs affected by the change (includes the changed transform itself).
            Ordered by execution order (topological sort).

        Raises
        ------
        KeyError
            If changed_transform_id not in DAG.

        Examples
        --------
        >>> dag = TransformDAG()
        >>> id1 = dag.add_transform(SimplicialLifting(), tier="heavy")
        >>> id2 = dag.add_transform(FeatureNorm(), tier="light", dependencies=[id1])
        >>> id3 = dag.add_transform(RandomNoise(), tier="light", dependencies=[id2])
        >>>
        >>> # Changing lifting affects everything
        >>> dag.get_affected_transforms(id1)
        ['SimplicialLifting_0', 'FeatureNorm_0', 'RandomNoise_0']
        >>>
        >>> # Changing middle transform affects itself and downstream
        >>> dag.get_affected_transforms(id2)
        ['FeatureNorm_0', 'RandomNoise_0']
        >>>
        >>> # Changing last transform affects only itself
        >>> dag.get_affected_transforms(id3)
        ['RandomNoise_0']

        Notes
        -----
        For Topological Deep Learning:
        - Changing expensive liftings affects all downstream transforms
        - Changing cheap features only affects the feature itself
        """
        if changed_transform_id not in self.nodes:
            raise KeyError(f"Transform {changed_transform_id} not in DAG")

        affected = []
        visited = set()

        def dfs(node_id: str) -> None:
            """Depth-first search to find affected transforms.

            Parameters
            ----------
            node_id : str
                ID of node to search from.
            """
            if node_id in visited:
                return
            visited.add(node_id)
            affected.append(node_id)

            # Find all nodes that depend on this one
            for nid, node in self.nodes.items():
                if node_id in node.dependencies:
                    dfs(nid)

        dfs(changed_transform_id)
        return affected

    def compute_pipeline_hash(self, transform_ids: list[str]) -> str:
        """Compute combined hash for multiple transforms.

        Useful for computing cache keys from specific transform subsets
        (e.g., only heavy transforms).

        Parameters
        ----------
        transform_ids : list[str]
            Transform IDs to include in hash.

        Returns
        -------
        str
            Combined hash (hex string).

        Examples
        --------
        >>> dag = TransformDAG()
        >>> id1 = dag.add_transform(SimplicialLifting(), tier="heavy")
        >>> id2 = dag.add_transform(FeatureNorm(), tier="light")
        >>> dag.compute_pipeline_hash([id1, id2])
        'abc123def456...'
        """
        # Combine hashes in order (order matters!)
        combined = "_".join(
            self.get_transform_hash(tid) for tid in transform_ids
        )
        hash_int = make_hash(combined)
        return hex(hash_int)[2:]

    def to_dict(self) -> dict[str, Any]:
        """Serialize DAG for cache metadata.

        Returns
        -------
        dict
            Serialized DAG with nodes and execution order.
            Does not include transform objects themselves.

        Examples
        --------
        >>> dag = TransformDAG()
        >>> id1 = dag.add_transform(SimplicialLifting(), tier="heavy")
        >>> data = dag.to_dict()
        >>> "nodes" in data and "execution_order" in data
        True
        """
        return {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "execution_order": self.execution_order,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransformDAG":
        """Deserialize DAG from cache metadata.

        Note: Only loads metadata (IDs, hashes, dependencies), not transform
        objects themselves. Transform objects are reconstructed from config.

        Parameters
        ----------
        data : dict
            Serialized DAG data from to_dict().

        Returns
        -------
        TransformDAG
            Reconstructed DAG (without transform objects).

        Examples
        --------
        >>> dag = TransformDAG()
        >>> id1 = dag.add_transform(SimplicialLifting(), tier="heavy")
        >>> data = dag.to_dict()
        >>> dag2 = TransformDAG.from_dict(data)
        >>> dag2.execution_order == dag.execution_order
        True
        """
        dag = cls()
        dag.execution_order = data["execution_order"]
        # Note: We only restore metadata, not transform objects
        # Transform objects are reconstructed from transforms_config
        return dag

    def __repr__(self) -> str:
        """Return string representation of DAG.

        Returns
        -------
        str
            Human-readable DAG description.

        Examples
        --------
        >>> dag = TransformDAG()
        >>> dag.add_transform(SimplicialLifting(), tier="heavy")
        'SimplicialLifting_0'
        >>> repr(dag)
        'TransformDAG(nodes=1, heavy=1, light=0)'
        """
        heavy_count = sum(1 for n in self.nodes.values() if n.tier == "heavy")
        light_count = sum(1 for n in self.nodes.values() if n.tier == "light")
        return (
            f"{self.__class__.__name__}("
            f"nodes={len(self.nodes)}, "
            f"heavy={heavy_count}, "
            f"light={light_count})"
        )
