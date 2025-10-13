"""Neighbor Feature Aggregation."""

from typing import List, Optional
import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import scatter


class NeighborFeatureAggregation(BaseTransform):
    r"""Aggregate neighbor node features and (optionally) combine with `data.x`.

    Parameters
    ----------
    selected_fields : List[str]
        Fields to process (e.g., ["edge_index"]
    agg : {"mean","sum","max"}, default="mean"
        Aggregation function over neighbor features.
    self_loops : bool, default=True
        Whether to include each node's own features as part of aggregation.
    combine : {"store","replace"}, default="store"
        - "store": put result in `out_field`.
        - "replace": `data.x = agg_x`.
    out_field : Optional[str], default=None
        Field to store aggregated features when `combine="store"`. If None, uses
        `"x_neighbor_<agg>"`.
    x_field : str, default="x"
        Name of the node feature field to aggregate.
    """

    def __init__(
        self,
        selected_fields: List[str],
        agg: str = "mean",
        self_loops: bool = True,
        combine: str = "store",
        out_field: Optional[str] = None,
        x_field: str = "x",
        **kwargs,
    ):
        super().__init__()
        assert agg in {"mean", "sum", "max"}
        assert combine in {"store", "replace"}
        self.type = "neighbor_feature_aggregation"
        self.parameters = {
            "selected_fields": selected_fields,
            "agg": agg,
            "self_loops": self_loops,
            "combine": combine,
            "out_field": out_field,
            "x_field": x_field,
            **kwargs,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r}, parameters={self.parameters!r})"

    # ---- public API ----
    def forward(self, data: Data) -> Data:
        selected_fields = self.parameters["selected_fields"]
        x_field = self.parameters["x_field"]

        if not hasattr(data, x_field):
            raise AttributeError(f"`data` has no '{x_field}' features to aggregate.")

        # Pick fields to process, mirroring your selection pattern.
        field_to_process = [
            key
            for key in data.to_dict()
            for sub in selected_fields
            if (sub in key) and key != "incidence_0"
        ]
        # If none matched, try default to 'edge_index' when present.
        if not field_to_process and hasattr(data, "edge_index"):
            field_to_process = ["edge_index"]

        # Start from the given features
        x: Tensor = getattr(data, x_field)

        # Apply aggregation for each selected connectivity field; if multiple
        # fields are present, aggregate sequentially (last one wins unless combined).
        agg_x = None
        for field in field_to_process:
            agg_x = self._aggregate_from_field(data, field, x)

            # If multiple fields, feed the newly aggregated features into next step
            x = agg_x

        if agg_x is None:
            # Nothing to do
            return data

        # Combine strategy
        combine = self.parameters["combine"]
        if combine == "store":
            out_field = self.parameters["out_field"] or f"{x_field}_neighbor_{self.parameters['agg']}"
            data[out_field] = agg_x
        elif combine == "replace":
            data[x_field] = agg_x

        return data

    # ---- helpers ----
    def _aggregate_from_field(self, data: Data, field: str, x: Tensor) -> Tensor:
        agg = self.parameters["agg"]
        include_self = self.parameters["self_loops"]

        # Number of nodes
        num_nodes = data.num_nodes if data.num_nodes is not None else x.size(0)

        # Case 1: sparse adjacency-like tensor available -> use matmul for sum/mean
        if hasattr(data, field) and isinstance(data[field], Tensor) and data[field].is_sparse:
            A = data[field]  # shape [N, N]
            if include_self:
                I = torch.eye(num_nodes, device=A.device).to_sparse_coo()
                A = (A + I).coalesce()

            # Sum aggregation via sparse matmul
            agg_sum = torch.sparse.mm(A, x)

            if agg == "sum":
                return agg_sum

            # Degree for mean
            deg = torch.sparse.sum(A.abs(), dim=1).to_dense().clamp_min(1).unsqueeze(-1)
            if agg == "mean":
                return agg_sum / deg

            # Max isn't supported by matmul; fall back to edge-wise scatter
            # Convert to edge_index
            edge_index = A.coalesce().indices()
            return self._scatter_from_edges(edge_index, x, num_nodes, include_self, reduce="max")

        # Case 2: standard edge_index
        if field == "edge_index":
            edge_index = data.edge_index
            reduce = {"sum": "sum", "mean": "mean", "max": "max"}[agg]
            return self._scatter_from_edges(edge_index, x, num_nodes, include_self, reduce=reduce)

        raise NotImplementedError(
            f"Aggregation from field '{field}' is only implemented for sparse adjacency or 'edge_index'."
        )

    @staticmethod
    def _scatter_from_edges(
        edge_index: Tensor,
        x: Tensor,
        num_nodes: int,
        include_self: bool,
        reduce: str,
    ) -> Tensor:
        # Optionally add self loops to include own features
        if include_self:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        src, dst = edge_index  # messages from src -> dst
        # Gather source features and scatter to destination with reduction.
        out = scatter(x[src], dst, dim=0, dim_size=num_nodes, reduce=reduce)

        # Ensure no NaNs for empty nodes (possible for mean), fill with zeros.
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out
