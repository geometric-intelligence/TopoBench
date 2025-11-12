"""
DAWN temporal hypergraph dataset loader (TopoBench-compatible).

This module provides a PyTorch Geometric InMemoryDataset implementation that
parses raw node, simplex, and label files and produces a processed dataset
ready for graph learning tasks.
"""

import gzip
import os
import os.path as osp
import shutil
import warnings

import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs
from torch_sparse import coalesce


class DawnDataset(InMemoryDataset):
    """
    TopoBench-compatible loader for the DAWN temporal hypergraph dataset from Cornell.

    Handles raw files (gz or txt) located in `raw_dir`:

    - `simplices.txt.gz` / `simplices.txt` : timestamped simplices
    - `nodes.txt.gz` / `nodes.txt`         : optional node features
    - `labels.txt.gz` / `labels.txt`       : optional node labels

    Produces a single processed file: `data.pt`.

    Parameters
    ----------
    root : str
        Root directory where the dataset should be saved.
    transform : callable, optional
        A function/transform that takes in a `Data` object and returns a
        transformed version.
    pre_transform : callable, optional
        A function/transform applied before saving the processed data.
    pre_filter : callable, optional
        A function that decides whether a `Data` object should be included.
    """

    def __init__(
        self,
        root: str,
        name: str = "DAWN",
        parameters: DictConfig = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """
        Initialize the DAWN dataset.

        Parameters
        ----------
        root : str
            Root directory where the dataset should be saved.
        name : str, optional
            Name of the dataset. Defaults to "DAWN".
        parameters : DictConfig, optional
            Configuration parameters for the dataset.
        transform : callable, optional
            A function/transform that takes in a `Data` object and returns a
            transformed version.
        pre_transform : callable, optional
            A function/transform applied before saving the processed data.
        pre_filter : callable, optional
            A function that decides whether a `Data` object should be included.
        """
        self.name = name
        self.parameters = parameters
        # Call the parent InMemoryDataset constructor
        super().__init__(root, transform, pre_transform, pre_filter)
        # Load the processed dataset from disk
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

    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        return (
            f"{self.name}(root={self.root}, name={self.name}, "
            f"parameters={self.parameters})"
        )

    @property
    def raw_dir(self) -> str:
        """Return the path to the raw directory of the dataset.

        Returns
        -------
        str
            Path to the raw directory.
        """
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory of the dataset.

        Returns
        -------
        str
            Path to the processed directory.
        """
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> list:
        """
        List of raw files expected in `raw_dir`.

        Returns
        -------
        list
            Names of raw files to check for extraction.
        """
        return [
            f"{self.name}-nverts.txt",
            f"{self.name}-simplices.txt",
            f"{self.name}-times.txt",
            f"{self.name}-node-labels.txt",
        ]

    @property
    def processed_file_names(self) -> str:
        """Return the processed file name for the dataset.

        Returns
        -------
        str
            Processed file name.
        """
        return "data.pt"

    def download(self) -> None:
        """Copy files from configs directory if they exist.

        For DAWN, the files are already in configs/dataset/hypergraph/DAWN/.
        This method copies them to the raw directory if needed.
        Also handles gzipped files if they exist.
        """
        # Check if files already exist in raw_dir
        all_exist = all(
            osp.exists(osp.join(self.raw_dir, fname))
            for fname in self.raw_file_names
        )

        if all_exist:
            return

        # Try to copy from configs directory (relative to project root)
        # Get project root by going up from topobench/data/datasets/
        project_root = osp.dirname(osp.dirname(osp.dirname(osp.dirname(__file__))))
        configs_dir = osp.join(
            project_root,
            "configs",
            "dataset",
            "hypergraph",
            self.name,
        )

        if osp.exists(configs_dir):
            os.makedirs(self.raw_dir, exist_ok=True)
            for fname in self.raw_file_names:
                src = osp.join(configs_dir, fname)
                dst = osp.join(self.raw_dir, fname)
                if osp.exists(src) and not osp.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"Copied {fname} from configs to raw directory")

        # Also handle gzipped files if they exist
        for fname in self.raw_file_names:
            gz_path = os.path.join(self.raw_dir, fname + ".gz")
            txt_path = os.path.join(self.raw_dir, fname)
            # Extract .gz only if the corresponding .txt does not already exist
            if os.path.exists(gz_path) and not os.path.exists(txt_path):
                print(f"Extracting {gz_path} → {txt_path}")
                with (
                    gzip.open(gz_path, "rb") as f_in,
                    open(txt_path, "wb") as f_out,
                ):
                    # Copy contents of gzipped file to txt
                    shutil.copyfileobj(f_in, f_out)

    def validate_and_normalize(
        self,
        num_nodes: int,
        x: torch.Tensor | None,
        y: torch.Tensor | None,
        edge_index: torch.LongTensor,
    ) -> tuple[int, torch.Tensor | None, torch.Tensor | None]:
        """
        Validate and normalize inputs.

        Ensures edge_index is non-empty, infers num_nodes from edges,
        validates x and y shapes, and converts 1-indexed labels to 0-indexed.

        Parameters
        ----------
        num_nodes : int
            Initial estimate of number of nodes.
        x : torch.Tensor, optional
            Node feature tensor of shape (num_nodes, num_features).
        y : torch.Tensor, optional
            Node label tensor of shape (num_nodes,).
        edge_index : torch.LongTensor
            Edge index tensor in COO format.

        Returns
        -------
        tuple[int, Optional[torch.Tensor], Optional[torch.Tensor]]
            Validated and normalized (num_nodes, x, y).

        Raises
        ------
        ValueError
            If edge_index is empty, if x/y dimensions are invalid, if x/y have
            fewer entries than required, or if labels contain negative values.
        """
        # Ensure that there is at least one hyperedge
        if edge_index.numel() == 0:
            raise ValueError(
                "Parsed edge_index is empty — no hyperedges found in simplices.txt."
            )

        # Infer number of nodes from edges
        max_node_id = int(edge_index[0].max().item())
        inferred_num_nodes = max_node_id + 1
        num_nodes = max(num_nodes, inferred_num_nodes)

        # Validate node features tensor
        if x is not None:
            if x.dim() != 2:
                raise ValueError(
                    f"Node features must be 2D (num_nodes x feat_dim), got {tuple(x.shape)}"
                )
            if x.size(0) < num_nodes:
                raise ValueError(
                    f"Node feature rows ({x.size(0)}) < inferred num_nodes ({num_nodes}) from simplices."
                )
            if x.size(0) > num_nodes:
                warnings.warn(
                    f"Node features ({x.size(0)}) contain more rows than inferred num_nodes ({num_nodes}). "
                    "Extra rows will be kept but may be ignored downstream.",
                    stacklevel=2,
                )

        # Validate and normalize label tensor
        if y is not None:
            if y.dim() != 1:
                raise ValueError(
                    f"Labels tensor must be 1D, got shape {tuple(y.shape)}"
                )
            if y.size(0) < num_nodes:
                raise ValueError(
                    f"Labels ({y.size(0)}) contain fewer entries than inferred num_nodes ({num_nodes})."
                )
            if y.size(0) > num_nodes:
                warnings.warn(
                    f"Labels ({y.size(0)}) contain more entries than inferred num_nodes ({num_nodes}). "
                    "Extra entries will be ignored.",
                    stacklevel=2,
                )
                y = y[:num_nodes]

            # Convert 1-indexed labels to 0-indexed
            y_min = int(y.min().item())
            if y_min == 1:
                warnings.warn(
                    "Detected labels starting at 1 — converting to 0-indexed labels.",
                    stacklevel=2,
                )
                y = y - 1

            # Ensure no negative labels remain
            if int(y.min().item()) < 0:
                raise ValueError(
                    "Labels contain negative values after normalization."
                )

        return num_nodes, x, y

    def process(self) -> None:
        """Process the DAWN dataset files into a PyG Data object.

        Reads the nverts, simplices, and times files and constructs
        a hypergraph with timestamps.
        """
        # Ensure files are available
        self.download()

        # File paths
        nverts_path = osp.join(self.raw_dir, f"{self.name}-nverts.txt")
        simplices_path = osp.join(self.raw_dir, f"{self.name}-simplices.txt")
        times_path = osp.join(self.raw_dir, f"{self.name}-times.txt")

        # Read nverts (number of vertices per simplex)
        with open(nverts_path, "r") as f:
            nverts = [int(line.strip()) for line in f if line.strip()]

        num_simplices = len(nverts)

        # Read simplices (contiguous list of node IDs)
        with open(simplices_path, "r") as f:
            all_node_ids = [int(line.strip()) for line in f if line.strip()]

        # Verify the total number of nodes matches sum of nverts
        total_nodes_in_simplices = sum(nverts)
        assert len(all_node_ids) == total_nodes_in_simplices, (
            f"Mismatch: sum of nverts ({total_nodes_in_simplices}) != "
            f"length of simplices ({len(all_node_ids)})"
        )

        # Read timestamps
        with open(times_path, "r") as f:
            timestamps = [int(line.strip()) for line in f if line.strip()]

        assert len(timestamps) == num_simplices, (
            f"Mismatch: number of timestamps ({len(timestamps)}) != "
            f"number of simplices ({num_simplices})"
        )

        # Build edge_index (incidence matrix: [node_id, hyperedge_id])
        node_list = []
        edge_list = []
        node_idx = 0

        for e_idx, nv in enumerate(nverts):
            # Get nodes for this simplex
            simplex_nodes = all_node_ids[node_idx : node_idx + nv]
            # Remove duplicates within simplex (if any)
            simplex_nodes = list(set(simplex_nodes))
            # Add to edge_index
            for node_id in simplex_nodes:
                node_list.append(node_id)
                edge_list.append(e_idx)
            node_idx += nv

        # Convert to 0-indexed (DAWN uses 1-indexed)
        max_node_id = max(node_list) if node_list else 0
        num_nodes = max_node_id  # Will be adjusted after 0-indexing

        # Convert to 0-indexed
        node_list = [n - 1 for n in node_list]  # 1-indexed -> 0-indexed
        num_nodes = max(node_list) + 1 if node_list else 0

        # Build edge_index tensor
        edge_index = torch.tensor([node_list, edge_list], dtype=torch.long)

        # Coalesce to remove duplicates and sort
        edge_index, _ = coalesce(
            edge_index, None, num_nodes, num_simplices
        )

        # Create node features (default: ones)
        x = torch.ones((num_nodes, 1), dtype=torch.float)

        # Create timestamps tensor
        edge_timestamps = torch.tensor(timestamps, dtype=torch.float)

        # Create labels (None for DAWN - no node labels)
        y = None

        # Create Data object
        data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            incidence_hyperedges=edge_index,
            edge_timestamps=edge_timestamps,
            num_nodes=num_nodes,
        )

        # Create sparse incidence matrix
        data.incidence_hyperedges = torch.sparse_coo_tensor(
            edge_index,
            values=torch.ones(edge_index.shape[1]),
            size=(num_nodes, num_simplices),
        )

        # Apply pre-filter and pre-transform if provided
        if self.pre_filter is not None and not self.pre_filter(data):
            return
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # Collate and save
        data_list = [data]
        self.data, self.slices = self.collate(data_list)
        self._data_list = None  # Reset cache

        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )
