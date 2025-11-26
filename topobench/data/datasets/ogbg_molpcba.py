"""OGB molecular dataset adapters for TopoBench."""

from pathlib import Path

import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import Data

from topobench.data.datasets.base_inductive import BaseOnDiskInductiveDataset


class OGBGMolPCBADataset(BaseOnDiskInductiveDataset):
    """OGB molecular property prediction dataset (437K graphs, 128 tasks).
    
    Parameters
    ----------
    root : str or Path
        Root directory for dataset storage.
    split : str, optional
        Dataset split: "train", "valid", or "test" (default: "train").
    subset_size : int, optional
        Use only first N samples for testing (default: None = all).
    cache_samples : bool, optional
        Cache samples to disk (default: True).
    """
    
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        subset_size: int | None = None,
        cache_samples: bool = True,
    ):
        self.split = split
        self.subset_size = subset_size
        self._root = Path(root)
        
        # Download dataset if needed (downloads to disk, not kept in memory)
        ogb_root = self._root / "ogb_raw"
        ogb_root.mkdir(parents=True, exist_ok=True)
        
        # Patch torch.load for PyTorch 2.6+ compatibility
        # OGB uses pickle which requires weights_only=False
        original_load = torch.load
        
        def patched_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        torch.load = patched_load
        
        try:
            # This downloads if needed, but we immediately get indices and discard
            temp_dataset = PygGraphPropPredDataset(name="ogbg-molpcba", root=str(ogb_root))
            split_idx = temp_dataset.get_idx_split()
            self._indices = split_idx[split].tolist()
            
            if subset_size is not None:
                self._indices = self._indices[:subset_size]
            
            # Store paths for on-demand loading (not the dataset itself!)
            self._ogb_root = ogb_root
            del temp_dataset  # Free memory immediately
        finally:
            # Restore original torch.load
            torch.load = original_load
        
        super().__init__(root=root, cache_samples=cache_samples)
    
    def _get_num_samples(self) -> int:
        return len(self._indices)
    
    def _generate_or_load_sample(self, idx: int) -> Data:
        from ogb.graphproppred import PygGraphPropPredDataset
        
        actual_idx = self._indices[idx]
        
        # Patch torch.load for PyTorch 2.6+ compatibility
        original_load = torch.load
        
        def patched_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        torch.load = patched_load
        
        try:
            # Load dataset on-demand (OGB caches to disk, this is lightweight)
            temp_dataset = PygGraphPropPredDataset(name="ogbg-molpcba", root=str(self._ogb_root))
            data = temp_dataset[actual_idx]
            del temp_dataset  # Free memory
        finally:
            # Restore original torch.load
            torch.load = original_load
        
        # Convert features to float (OGB uses integer encodings)
        if hasattr(data, 'x') and data.x is not None:
            data.x = data.x.float()
        
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = data.edge_attr.float()
        
        if data.y is not None:
            # OGB data is [1, num_tasks], keep it that way for proper batching
            data.y = data.y.float()
        
        # OGB provides bidirectional edges, but SimplicalCliqueLifting
        # needs single-direction edges to avoid "duplicate nodes" error
        # Filter to only keep edges where src < dst and mark as undirected
        if hasattr(data, "edge_index") and data.edge_index is not None:
            from torch_geometric.utils import is_undirected, to_undirected
            
            # If already undirected, nothing to do
            if not is_undirected(data.edge_index, data.edge_attr if hasattr(data, "edge_attr") else None):
                # Convert to undirected (this ensures proper structure)
                if hasattr(data, "edge_attr") and data.edge_attr is not None:
                    data.edge_index, data.edge_attr = to_undirected(data.edge_index, data.edge_attr)
                else:
                    data.edge_index = to_undirected(data.edge_index)
        
        return data
    
    def _get_pickle_args(self) -> tuple:
        return (str(self.root), self.split, self.subset_size, self.cache_samples)
    
    @property
    def num_classes(self) -> int:
        return 128
    
    @property
    def num_features(self) -> int:
        return 9
    


class MockMolecularDataset(BaseOnDiskInductiveDataset):
    """Safe mock molecular dataset for testing (100 samples, ~20 nodes each).
    
    Mimics ogbg-molpcba structure without downloading data. Perfect for
    verifying pipelines on memory-constrained machines.
    
    Parameters
    ----------
    root : str or Path
        Root directory for caching.
    num_samples : int, optional
        Number of molecules to generate (default: 100).
    num_tasks : int, optional
        Number of classification tasks (default: 128).
    seed : int, optional
        Random seed for reproducibility (default: 42).
    """
    
    def __init__(
        self,
        root: str | Path,
        num_samples: int = 100,
        num_tasks: int = 128,
        seed: int = 42,
        cache_samples: bool = True,
    ):
        self.num_samples = num_samples
        self.num_tasks = num_tasks
        self.seed = seed
        super().__init__(root=root, cache_samples=cache_samples)
    
    def _get_num_samples(self) -> int:
        return self.num_samples
    
    def _generate_or_load_sample(self, idx: int) -> Data:
        import networkx as nx
        
        torch.manual_seed(self.seed + idx)
        
        num_nodes = torch.randint(15, 25, (1,)).item()
        
        # Generate a clean molecular-like graph using networkx
        # This ensures no self-loops, no multi-edges, connected structure
        G = nx.connected_watts_strogatz_graph(num_nodes, k=min(4, num_nodes-1), p=0.3, seed=self.seed + idx)
        
        # Convert to PyG format
        edges = list(G.edges())
        if len(edges) == 0:  # Failsafe
            edges = [(i, i+1) for i in range(num_nodes-1)]
        
        # Normalize to have src < dst (only ONE direction per edge)
        edges_normalized = [(min(u, v), max(u, v)) for u, v in edges]
        edges_unique = list(set(edges_normalized))
        
        # Store ONLY one direction - the lifting will handle undirected properly
        edge_index = torch.tensor(edges_unique, dtype=torch.long).t().contiguous()
        
        x = torch.randn(num_nodes, 9)
        edge_attr = torch.randn(edge_index.shape[1], 3)
        
        y = torch.rand(self.num_tasks)
        y = (y > 0.5).float()
        mask = torch.rand(self.num_tasks) > 0.3
        y[~mask] = float("nan")
        
        # Reshape to [1, num_tasks] for proper batching in graph-level tasks
        y = y.unsqueeze(0)  # Shape: [1, 128]
        
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=num_nodes
        )
    
    def _get_pickle_args(self) -> tuple:
        return (str(self.root), self.num_samples, self.num_tasks, self.seed, self.cache_samples)
    
    @property
    def num_classes(self) -> int:
        return self.num_tasks
    
    @property
    def num_features(self) -> int:
        return 9
