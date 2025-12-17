"""Neighbor Feature Aggregation (multi-agg, multi-hop)."""

from typing import List, Optional, Dict, Union
import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

def get_neighbor_ids(
    edge_index: Tensor,
    k: int,
    num_nodes: Optional[int] = None,
    directed: bool = True,
    include_self: bool = False,
) -> List[List[int]]:
    """
    Return a list-of-lists where out[i] contains the node ids that are
    at EXACTLY k hops from node i (shortest-path distance == k).

    Parameters
    ----------
    edge_index : Tensor
        Shape [2, E], COO format (src -> dst).
    k : int
        Hop distance (>= 1).
    num_nodes : Optional[int]
        If None, inferred from edge_index.max()+1.
    directed : bool
        If False, treats edges as undirected (adds reverse edges).
    include_self : bool
        If True, allows returning self if reachable at exactly k hops
        (rare unless there are cycles). Usually keep False.

    Returns
    -------
    List[List[int]]
        neighbors_k[i] = list of node ids at exactly k hops from i.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    # Handle empty edge_index (no edges)
    if edge_index.numel() == 0:
        if num_nodes is None:
            return []
        return [[] for _ in range(num_nodes)]

    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    # Build adjacency list
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    for s, d in zip(src, dst):
        adj[s].append(d)

    if not directed:
        for s, d in zip(src, dst):
            adj[d].append(s)

    neighbors_k: List[List[int]] = []

    for start in range(num_nodes):
        visited = {start}  # nodes within <= current hop
        frontier = {start}  # nodes at exactly current hop (starts at 0-hop)

        for _ in range(k):
            next_frontier = set()
            for u in frontier:
                next_frontier.update(adj[u])

            # Remove nodes already seen at smaller distance to keep EXACT k-hop
            next_frontier.difference_update(visited)

            visited.update(next_frontier)
            frontier = next_frontier

            if not frontier:
                break  # no more nodes reachable

        if not include_self:
            frontier.discard(start)

        neighbors_k.append(sorted(frontier))

    return neighbors_k

def aggregate_neighbor_features(
    x: Tensor,
    neighbor_ids: List[List[int]],
    agg: str,
) -> Tensor:
    """
    Aggregate features of neighbors (given as list-of-lists per node).

    Parameters
    ----------
    x : Tensor
        Node features, shape [N, F].
    neighbor_ids : List[List[int]]
        neighbor_ids[i] = list of neighbor node indices for node i.
    agg : str
        One of {"sum","mean","min","max"}.

    Returns
    -------
    Tensor
        Aggregated features, shape [N, F].
        For empty neighbor sets -> zeros.
    """
    if agg not in {"sum", "mean", "min", "max"}:
        raise ValueError("agg must be one of {'sum','mean','min','max'}")

    N, F = x.size(0), x.size(1)
    out = torch.zeros((N, F), device=x.device, dtype=x.dtype)

    for i, nbrs in enumerate(neighbor_ids):
        if len(nbrs) == 0:
            continue

        feats = x[nbrs]  # [num_nbrs, F]

        if agg == "sum":
            out[i] = feats.sum(dim=0)
        elif agg == "mean":
            out[i] = feats.mean(dim=0)
        elif agg == "min":
            out[i] = feats.min(dim=0).values
        elif agg == "max":
            out[i] = feats.max(dim=0).values

    return out

class NeighborFeatureAggregation(BaseTransform):
    r"""Compute neighbor-aggregated features for multiple hops and multiple reducers.

    For each hop k in {1..K} and each agg in aggs, computes:
        x_{k,agg} = AGG( x_{k-1} over k-hop neighbors )
    where x_0 is the original x_field.

    Naming:
        out_field = f"{prefix}{k}_hop_{agg}"  (e.g., "x_1_hop_mean")

    Parameters
    ----------
    selected_fields : List[str]
        Fields to process (e.g., ["edge_index"] or ["adj_t"]).
        If none match and data.edge_index exists -> uses "edge_index".
        NOTE: this implementation uses ONE field (the first match) for clarity.
    aggs : List[str], default=("mean",)
        Aggregations to compute. Supported: {"sum","mean","min","max"}.
    num_hops : int, default=1
        Number of hops K.
    prefix : Optional[str], default=None
        Prefix for "store" mode fields. If None, uses f"{x_field}_".
    x_field : str, default="x"
        Node feature field name.
    """

    SUPPORTED_AGGS = {"sum", "mean", "min", "max"}

    def __init__(
        self,
        aggs: List[str] = ("mean",),
        num_hops: int = 1,
        prefix: Optional[str] = None,
        x_field: str = "x",
        edge_field: str = "edge_index",
        **kwargs,
    ):
        super().__init__()
        if isinstance(aggs, (tuple, set)):
            aggs = list(aggs)
        assert all(a in self.SUPPORTED_AGGS for a in aggs), f"aggs must be in {self.SUPPORTED_AGGS}"
        assert num_hops >= 1

        self.type = "neighbor_feature_aggregation"
        self.parameters = {
            "aggs": aggs,
            "num_hops": num_hops,
            "prefix": prefix,
            "x_field": x_field,
            "edge_field":edge_field,
            **kwargs,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r}, parameters={self.parameters!r})"

    # ---- public API ----
    def forward(self, data: Data) -> Data:
        x_field = self.parameters["x_field"]
        if not hasattr(data, x_field):
            raise AttributeError(f"`data` has no '{x_field}' features to aggregate.")

        # Pick connectivity field
        connectivity_field = self.parameters["edge_field"]

        x0: Tensor = getattr(data, x_field)
        edge_index: Tensor = getattr(data, connectivity_field)

        for k in range(1, self.parameters["num_hops"] + 1):
            # getting the k-hop neighbors indexes
            # it is a list of lists where each sublist contains the neighbor ids for each node
            neighbor_ids = get_neighbor_ids(edge_index, k)

            for agg in self.parameters["aggs"]:
                out_field = (
                    f"{self.parameters['prefix']}{k}_hop_{agg}"
                    if self.parameters["prefix"] is not None
                    else f"{x_field}_{k}_hop_{agg}"
                )
                aggregated_features = aggregate_neighbor_features(
                    x0, neighbor_ids, agg
                )
                setattr(data, out_field, aggregated_features)

        return data
