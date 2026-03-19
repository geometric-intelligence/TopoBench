"""K-hop feature Encoding (KFE) for Hasse graphs Transform."""

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj


class KHopFE(BaseTransform):
    r"""
    K-hop Feature Encodings (KHopFE) transform.

    Parameters
    ----------
    max_hop: int
        The maximum hop neighbourhood.
    concat_to_x : bool, optional
        If True, concatenates the encodings with existing node features in
        ``data.x``. If ``data.x`` is None, creates it. Default is True.
    aggregation : str, optional
        Aggregation function to reduce over the feature dimension.
        Options: "mean", "sum", "max", "min". Default is "mean".
    **kwargs : dict
        Additional arguments (not used).
    """

    _AGG_FN_MAP = {"mean": "mean", "sum": "sum", "max": "amax", "min": "amin"}

    def __init__(
        self,
        max_hop: int,
        concat_to_x: bool = True,
        aggregation: str = "mean",
        **kwargs,
    ):
        self.concat_to_x = concat_to_x
        self.max_hop = (
            max_hop - 1
        )  # The 0-th hop is always the features themselves
        if aggregation not in self._AGG_FN_MAP:
            raise ValueError(
                f"Unknown aggregation '{aggregation}'. "
                f"Choose from: {list(self._AGG_FN_MAP.keys())}"
            )
        self.aggregation = aggregation

    def forward(self, data: Data) -> Data:
        """Compute the K-hop feature encodings for the input graph.

        Parameters
        ----------
        data : Data
            Input graph data object.

        Returns
        -------
        Data
            Graph data object with K-hop feature encodings added.
        """
        if data.x is None:
            raise ValueError(
                "KHopFE requires node features (data.x cannot be None)"
            )

        fe = self._compute_khopfe(data.x, data.edge_index, data.num_nodes)

        if self.concat_to_x:
            if data.x is None:
                data.x = fe
            else:
                data.x = torch.cat([data.x, fe], dim=-1)
        else:
            data.KHopFE = fe

        return data

    def _compute_khopfe(
        self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Internal method to compute K-hop feature encodings.

        Propagates features through K hops and aggregates over input features
        to produce a fixed-dimension output (matching PSE pattern).

        Parameters
        ----------
        x : torch.Tensor
            Node features of the graph.
        edge_index : torch.Tensor
            Edge indices of the graph.
        num_nodes : int
            Number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            K-hop feature encodings of shape [N, max_hop].
        """
        device = edge_index.device
        x = x.to(device)
        khop_fe = []
        if edge_index.size(1) == 0 or num_nodes <= 1:
            return torch.zeros(num_nodes, self.max_hop, device=device)

        A = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
        # Symmetric norm adjacency matrix
        deg = A.sum(dim=1)
        deg_inv_sqrt = torch.diagflat(torch.pow(deg + 1e-8, -0.5))
        A_norm = deg_inv_sqrt @ A @ deg_inv_sqrt

        for hop in range(self.max_hop):
            x = A_norm @ x
            khop_fe.append(x)
        khop_fe = torch.stack(khop_fe, dim=1)  # [N, max_hop, F]
        # Aggregate over features to produce fixed-dimension output (like PSEs)
        agg_fn = getattr(khop_fe, self._AGG_FN_MAP[self.aggregation])
        khop_fe = agg_fn(dim=-1)  # [N, max_hop]
        return khop_fe.float()
