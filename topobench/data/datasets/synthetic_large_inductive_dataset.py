"""Synthetic large inductive dataset for validation testing."""

import os.path as osp

import networkx as nx
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs


class SyntheticLargeInductiveDataset(InMemoryDataset):
    """Synthetic dataset with many graphs designed to test memory limits.

    This dataset generates synthetic graphs with controlled parameters
    to demonstrate the memory limitations of in-memory preprocessing.

    Parameters
    ----------
    root : str
        Root directory where the dataset will be saved.
    name : str
        Name of the dataset.
    parameters : DictConfig
        Configuration parameters:
        - num_graphs : int
            Number of graphs to generate
        - nodes_per_graph : int
            Average nodes per graph
        - degree : int
            Average degree per node
        - num_features : int
            Number of node features
        - num_classes : int
            Number of classes for classification
    """

    def __init__(
        self,
        root: str,
        name: str,
        parameters: DictConfig,
    ) -> None:
        self.name = name
        self.parameters = parameters

        # Dataset parameters
        self._num_graphs = parameters.get("num_graphs", 5000)
        self._nodes_per_graph = parameters.get("nodes_per_graph", 80)
        self._degree = parameters.get("degree", 15)
        self._num_node_features = parameters.get("num_features", 16)
        self._num_classes = parameters.get("num_classes", 5)

        super().__init__(root)

        out = fs.torch_load(self.processed_paths[0])
        assert len(out) == 3 or len(out) == 4

        if len(out) == 3:  # Backward compatibility
            data, self.slices, self.sizes = out
            data_cls = Data
        else:
            data, self.slices, self.sizes, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

        assert isinstance(self._data, Data)

    def __repr__(self) -> str:
        return (
            f"{self.name}(root={self.root}, num_graphs={self._num_graphs}, "
            f"nodes_per_graph={self._nodes_per_graph})"
        )

    @property
    def raw_dir(self) -> str:
        """Return the path to the raw directory.

        Returns
        -------
        str
            Path to the raw directory.
        """
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory.

        Returns
        -------
        str
            Path to the processed directory.
        """
        config_str = (
            f"n{self._num_graphs}_v{self._nodes_per_graph}_d{self._degree}"
        )
        self.processed_root = osp.join(self.root, self.name, config_str)
        return osp.join(self.processed_root, "processed")

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw file names (none needed for synthetic).

        Returns
        -------
        list[str]
            Empty list (no raw files needed).
        """
        return []

    @property
    def processed_file_names(self) -> str:
        """Return the processed file name.

        Returns
        -------
        str
            Processed file name.
        """
        return "data.pt"

    def download(self) -> None:
        """Download is not needed for synthetic data."""

    def process(self) -> None:
        """Generate synthetic graphs and save them.

        This method creates a large number of synthetic graphs using
        Watts-Strogatz model, which creates graphs with high clustering
        (many triangles) for testing memory limits.
        """
        print(f"Generating {self._num_graphs} synthetic graphs...")
        print(f"  Nodes per graph: ~{self._nodes_per_graph}")
        print(f"  Degree: {self._degree}")
        print(f"  Features: {self._num_node_features}")
        print(f"  Classes: {self._num_classes}")

        data_list = []

        for i in range(self._num_graphs):
            if i % 500 == 0 and i > 0:
                print(f"  Generated {i}/{self._num_graphs} graphs...")

            # Vary graph size slightly
            n = self._nodes_per_graph + torch.randint(-5, 5, (1,)).item()
            n = max(20, n)

            # Generate Watts-Strogatz graph (creates many triangles)
            G = nx.watts_strogatz_graph(
                n=n,
                k=self._degree,
                p=0.3,  # Rewiring probability
                seed=42 + i,
            )

            # Convert to PyG Data
            edges = list(G.edges())
            if not edges:
                # Skip empty graphs
                continue

            edge_index = torch.tensor(edges, dtype=torch.long).t()
            # Add reverse edges for undirected graph
            edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

            # Random features
            x = torch.randn(n, self._num_node_features)

            # Random graph-level label
            y = torch.randint(0, self._num_classes, (1,))

            data = Data(x=x, edge_index=edge_index, y=y, num_nodes=n)

            data_list.append(data)

        print(f"✓ Generated {len(data_list)} graphs")
        print()

        # Collate the graphs
        self.data, self.slices = self.collate(data_list)
        self._data_list = None  # Reset cache

        # Save processed data
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )

        print(f"✓ Dataset saved to {self.processed_paths[0]}")
