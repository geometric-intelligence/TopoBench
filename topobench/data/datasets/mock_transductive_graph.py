"""Mock transductive graph dataset for testing on-disk transductive pipeline.

This module provides a synthetic single-graph dataset for testing transductive
learning pipelines without needing to download large datasets like OGBN-products.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs


class MockTransductiveGraph(InMemoryDataset):
    """Synthetic single transductive graph for testing.

    Generates a random graph with realistic properties:
    - Community structure (for better structure detection)
    - Node features (random)
    - Multi-class labels
    - Train/val/test masks

    This is useful for:
    - Testing on-disk transductive pipeline without downloads
    - Quick iteration during development
    - CI/CD testing
    - Benchmarking with controlled sizes

    Parameters
    ----------
    root : str or Path
        Root directory for dataset storage.
    num_nodes : int, optional
        Number of nodes in the graph (default: 100).
    avg_degree : int, optional
        Average node degree (default: 10).
    num_features : int, optional
        Dimensionality of node features (default: 100).
    num_classes : int, optional
        Number of output classes (default: 47 for OGBN-products-like).
    train_ratio : float, optional
        Fraction of nodes for training (default: 0.08 like OGBN-products).
    val_ratio : float, optional
        Fraction of nodes for validation (default: 0.02).
    seed : int, optional
        Random seed for reproducibility (default: 42).

    Examples
    --------
    >>> from topobench.data.datasets import MockTransductiveGraph
    >>>
    >>> # Small graph for quick testing (default)
    >>> dataset = MockTransductiveGraph(root='./data/mock')
    >>> graph = dataset[0]
    >>> print(f"Nodes: {graph.num_nodes}, Edges: {graph.edge_index.shape[1]}")
    >>>
    >>> # Larger graph (but still reasonable for structure indexing)
    >>> dataset = MockTransductiveGraph(
    ...     root='./data/mock',
    ...     num_nodes=500,
    ...     avg_degree=20,
    ...     num_classes=47
    ... )

    Notes
    -----
    - Graph is generated using Watts-Strogatz model for realistic structure
    - Community structure creates triangles (good for clique detection)
    - Much faster than downloading real datasets
    - Consistent results with same seed

    See Also
    --------
    topobench.data.loaders.ogbn_products_loader.OGBNProductsLoader :
        Loader that can use this mock dataset.
    """

    def __init__(
        self,
        root: str | Path,
        num_nodes: int = 100,
        avg_degree: int = 10,
        num_features: int = 100,
        num_classes: int = 47,
        train_ratio: float = 0.08,
        val_ratio: float = 0.02,
        seed: int = 42,
    ):
        """Initialize mock transductive graph dataset.

        Parameters
        ----------
        root : str or Path
            Root directory for dataset.
        num_nodes : int
            Number of nodes.
        avg_degree : int
            Average node degree.
        num_features : int
            Node feature dimensionality.
        num_classes : int
            Number of classes.
        train_ratio : float
            Training set fraction.
        val_ratio : float
            Validation set fraction.
        seed : int
            Random seed.
        """
        self._num_nodes = num_nodes
        self._avg_degree = avg_degree
        self._num_features = num_features
        self._num_classes = num_classes
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._seed = seed

        super().__init__(root)
        out = fs.torch_load(self.processed_paths[0])
        if len(out) == 4:
            data, self.slices, self.sizes, data_cls = out
            self.data = (
                data_cls.from_dict(data) if isinstance(data, dict) else data
            )
        else:
            data, self.slices, self.sizes = out
            self.data = data

    @property
    def raw_file_names(self):
        """Raw file names (none needed for synthetic data)."""
        return []

    @property
    def processed_file_names(self):
        """Processed file name."""
        return f"mock_transductive_{self._num_nodes}n_{self._seed}s.pt"

    def download(self):
        """Download (not needed for synthetic data)."""

    def process(self):
        """Generate synthetic transductive graph."""
        # Set seed for reproducibility
        torch.manual_seed(self._seed)
        import random

        random.seed(self._seed)

        # Generate graph with community structure (Watts-Strogatz)
        # This creates realistic triangles for structure detection
        k = min(self._avg_degree, self._num_nodes - 1)
        G = nx.watts_strogatz_graph(
            n=self._num_nodes,
            k=k,
            p=0.1,  # Small-world rewiring probability
            seed=self._seed,
        )

        # Convert to PyG format
        edges = list(G.edges())
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        # Make undirected
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

        # Generate node features (random)
        x = torch.randn(self._num_nodes, self._num_features)

        # Generate labels (random)
        y = torch.randint(0, self._num_classes, (self._num_nodes,))

        # Create train/val/test masks
        num_train = int(self._num_nodes * self._train_ratio)
        num_val = int(self._num_nodes * self._val_ratio)

        train_mask = torch.zeros(self._num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(self._num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(self._num_nodes, dtype=torch.bool)

        # Random split
        indices = torch.randperm(self._num_nodes)
        train_mask[indices[:num_train]] = True
        val_mask[indices[num_train : num_train + num_val]] = True
        test_mask[indices[num_train + num_val :]] = True

        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            num_nodes=self._num_nodes,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

        # Save
        self.data, self.slices = self.collate([data])
        fs.torch_save(
            (self.data.to_dict(), self.slices, {}, self.data.__class__),
            self.processed_paths[0],
        )

    def __repr__(self):
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"num_nodes={self._num_nodes}, "
            f"avg_degree={self._avg_degree}, "
            f"num_classes={self._num_classes})"
        )
