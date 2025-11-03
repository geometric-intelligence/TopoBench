import os.path as osp
import random
from collections.abc import Callable

import torch
from torch import Tensor
from torch_geometric.data import Data, OnDiskDataset
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected


# Helpers taken from their torch_gometric.FakeDataset, unchanged
def get_num_nodes(avg_num_nodes: int, avg_degree: float) -> int:
    min_num_nodes = max(3 * avg_num_nodes // 4, int(avg_degree))
    max_num_nodes = 5 * avg_num_nodes // 4
    return random.randint(min_num_nodes, max_num_nodes)


def get_num_channels(num_channels: int) -> int:
    min_num_channels = 3 * num_channels // 4
    max_num_channels = 5 * num_channels // 4
    return random.randint(min_num_channels, max_num_channels)


def get_edge_index(
    num_src_nodes: int,
    num_dst_nodes: int,
    avg_degree: float,
    is_undirected: bool = False,
    remove_loops: bool = False,
) -> Tensor:
    num_edges = int(num_src_nodes * avg_degree)
    row = torch.randint(num_src_nodes, (num_edges,), dtype=torch.int64)
    col = torch.randint(num_dst_nodes, (num_edges,), dtype=torch.int64)
    edge_index = torch.stack([row, col], dim=0)

    if remove_loops:
        edge_index, _ = remove_self_loops(edge_index)

    num_nodes = max(num_src_nodes, num_dst_nodes)
    if is_undirected:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    else:
        edge_index = coalesce(edge_index, num_nodes=num_nodes)

    return edge_index


# FakeOnDiskDataset: API & generation logic taken from torch_geometric FakeDataset,
# but persisted via OnDiskDataset.
class FakeOnDiskDataset(OnDiskDataset):
    r"""A fake dataset that returns randomly generated Data objects,
    persisted to disk via :class:`~torch_geometric.data.OnDiskDataset`.

    Args:
        root (str): Root directory where the processed DB will live.
        num_graphs (int, optional): Number of graphs. (default: 1)
        avg_num_nodes (int, optional): Average nodes per graph. (default: 1000)
        avg_degree (float, optional): Average degree per node. (default: 10.0)
        num_channels (int, optional): Node feature dimension. (default: 64)
        edge_dim (int, optional): Edge feature dimension. (default: 0)
        num_classes (int, optional): Number of classes. (default: 10)
        task (str, optional): 'node', 'graph', or 'auto'. (default: 'auto')
        is_undirected (bool, optional): Make edges undirected. (default: True)
        transform (callable, optional): Applied on read. (default: None)
        pre_transform (callable, optional): (unused, kept for parity)
        backend (str, optional): 'sqlite' or 'rocksdb'. (default: 'sqlite')
        force_reload (bool, optional): Rebuild DB even if present. (default: False)
        **kwargs (optional): Extra attributes & shapes, e.g. global_features=5.
    """

    def __init__(
        self,
        root: str,
        name: str,
        num_graphs: int = 1,
        avg_num_nodes: int = 1000,
        avg_degree: float = 10.0,
        num_channels: int = 64,
        edge_dim: int = 0,
        num_classes: int = 10,
        task: str = "auto",
        is_undirected: bool = True,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        backend: str = "sqlite",
        **kwargs,
    ):
        if task == "auto":
            task = "graph" if num_graphs > 1 else "node"
        assert task in ["node", "graph"]
        self.name = name
        self.num_graphs = max(int(num_graphs), 1)
        self.avg_num_nodes = max(int(avg_num_nodes), int(avg_degree))
        self.avg_degree = max(float(avg_degree), 1.0)
        self.num_channels = int(num_channels)
        self.edge_dim = int(edge_dim)
        self._num_classes = int(num_classes)
        self.task = task
        self.is_undirected = bool(is_undirected)
        self.root = root
        self.kwargs = kwargs

        # Static schema consistent with configuration:
        # Always store num_nodes (typed) to avoid KeyErrors.
        schema: dict[str, object] = {
            "edge_index": dict(dtype=torch.int64, size=(2, -1)),
            "num_nodes": dict(dtype=torch.int64, size=(1,)),
            "x": (dict(dtype=torch.float32, size=(-1, self.num_channels))
                  if self.num_channels > 0 else object),
            "edge_attr": (dict(dtype=torch.float32, size=(-1, self.edge_dim))
                          if self.edge_dim > 1 else object),
            "edge_weight": (dict(dtype=torch.float32, size=(-1,))
                            if self.edge_dim == 1 else object),
            "y": (
                dict(dtype=torch.int64, size=(-1,)) if (self._num_classes > 0 and self.task == "node")
                else dict(dtype=torch.int64, size=(1,)) if (self._num_classes > 0 and self.task == "graph")
                else object
            ),
        }

        super().__init__(self.root, transform, backend=backend, schema=schema)

    @property
    def raw_file_names(self) -> list[str]:
        return []
        
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


    def download(self) -> None:
        pass

    def process(self) -> None:
        """Generate synthetic graphs and write them to disk in batches.

        This mirrors the batch-writing logic from pcqm4m: we never keep
        all graphs in memory at once, only up to `batch_size` at a time.
        """
        batch_size = 1000  # same idea as pcqm4m

        data_list: list[Data] = []

        for i in range(self.num_graphs):
            data = self.generate_data()
            data_list.append(data)

            # flush either every `batch_size` items, or at the very end
            if (i + 1) == self.num_graphs or ((i + 1) % batch_size == 0):
                # this writes the batch to disk / processed dir
                self.extend(data_list)
                # free the batch from RAM
                data_list = []

    # Mirrors torch_geometric.FakeDataset.generate_data():
    def generate_data(self) -> Data:
        num_nodes = get_num_nodes(self.avg_num_nodes, self.avg_degree)

        data = Data()
        data.num_nodes = num_nodes  # always store for schema consistency

        if self._num_classes > 0 and self.task == "node":
            data.y = torch.randint(self._num_classes, (num_nodes,))
        elif self._num_classes > 0 and self.task == "graph":
            data.y = torch.tensor([random.randint(0, self._num_classes - 1)])

        data.edge_index = get_edge_index(
            num_nodes, num_nodes, self.avg_degree, self.is_undirected, remove_loops=True
        )

        if self.num_channels > 0:
            x = torch.randn(num_nodes, self.num_channels)
            if self._num_classes > 0 and self.task == "node":
                x = x + data.y.unsqueeze(1)  # weak correlation like theirs
            elif self._num_classes > 0 and self.task == "graph":
                x = x + data.y
            data.x = x

        if self.edge_dim > 1:
            data.edge_attr = torch.rand(data.num_edges, self.edge_dim)
        elif self.edge_dim == 1:
            data.edge_weight = torch.rand(data.num_edges)

        for feature_name, feature_shape in self.kwargs.items():
            setattr(data, feature_name, torch.randn(feature_shape))
        return data

    # Map Data -> dict following the schema (always include num_nodes).
    def serialize(self, data: Data) -> dict[str, object]:
        row: dict[str, object] = {
            "edge_index": data.edge_index,
            "num_nodes": torch.tensor([data.num_nodes], dtype=torch.int64),
            "x": getattr(data, "x", None) if self.num_channels > 0 else None,
            "edge_attr": getattr(data, "edge_attr", None) if self.edge_dim > 1 else None,
            "edge_weight": getattr(data, "edge_weight", None) if self.edge_dim == 1 else None,
            "y": getattr(data, "y", None) if self._num_classes > 0 else None,
        }
        return row

    # Map dict -> Data; only set optional fields if present/non-None.
    def deserialize(self, row: dict[str, object]) -> Data:
        d = Data()
        d.edge_index = row["edge_index"]  # type: ignore[assignment]
        # num_nodes is guaranteed present & typed
        n = row["num_nodes"]
        d.num_nodes = int(n[0]) if isinstance(n, torch.Tensor) else int(n[0])  # type: ignore[index]

        x = row.get("x")
        if x is not None:
            d.x = x  # type: ignore[assignment]

        ea = row.get("edge_attr")
        if ea is not None:
            d.edge_attr = ea  # type: ignore[assignment]

        ew = row.get("edge_weight")
        if ew is not None:
            d.edge_weight = ew  # type: ignore[assignment]

        y = row.get("y")
        if y is not None:
            d.y = y  # type: ignore[assignment]

        return d
