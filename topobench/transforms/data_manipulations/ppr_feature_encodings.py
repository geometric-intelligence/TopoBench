"""Personalized Page Rank Feature Encoding (PPRFE) Transform."""

import numpy as np
import omegaconf
import torch
from scipy.linalg import inv
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj


class PPRFE(BaseTransform):
    r"""
    Personalized Page Rank Feature Encodings (PPRFE) transform.

    Computes PPR diffusion of node features using the formula:
    PPR = α(I - (1-α)Ã)^{-1}
    where Ã = D^{-1/2} A^ D^{-1/2} is the normalized adjacency matrix
    and A^ = A + I (adjacency with self-loops).

    Parameters
    ----------
    alpha_param_PPRFE : tuple of float
        Tuple specifying the start and end teleport probabilities (alpha values).
        Values should be in (0, 1]. Higher alpha = more local, lower = more global.
    concat_to_x : bool, optional
        If True, concatenates the encodings with existing node features in
        ``data.x``. If ``data.x`` is None, creates it. Default is True.
    aggregation : str, optional
        Aggregation function to reduce over the feature dimension.
        Options: "mean", "sum", "max", "min". Default is "mean".
    self_loop : bool, optional
        If True, adds self-loops to the adjacency matrix. Default is True.
    **kwargs : dict
        Additional arguments (not used).
    """

    _AGG_FN_MAP = {"mean": "mean", "sum": "sum", "max": "amax", "min": "amin"}

    def __init__(
        self,
        alpha_param_PPRFE: tuple,
        concat_to_x: bool = True,
        aggregation: str = "mean",
        self_loop: bool = True,
        **kwargs,
    ):
        self.alpha_param_PPRFE = alpha_param_PPRFE
        self.concat_to_x = concat_to_x
        self.self_loop = self_loop
        if aggregation not in self._AGG_FN_MAP:
            raise ValueError(
                f"Unknown aggregation '{aggregation}'. "
                f"Choose from: {list(self._AGG_FN_MAP.keys())}"
            )
        self.aggregation = aggregation
        # Compute fe_dim from tuple/list
        if (
            isinstance(alpha_param_PPRFE, (list, tuple))
            or type(alpha_param_PPRFE) is omegaconf.listconfig.ListConfig
        ):
            # Number of alpha values to use
            self.fe_dim = alpha_param_PPRFE[1]
        else:
            self.fe_dim = alpha_param_PPRFE

    def forward(self, data: Data) -> Data:
        """Compute the PPR feature encodings for the input graph.

        Parameters
        ----------
        data : Data
            Input graph data object.

        Returns
        -------
        Data
            Graph data object with PPR feature encodings added.
        """
        if data.x is None:
            raise ValueError(
                "PPRFE requires node features (data.x cannot be None)"
            )

        fe = self._compute_pprfe(data.x, data.edge_index, data.num_nodes)

        if self.concat_to_x:
            data.x = torch.cat([data.x, fe], dim=-1)
        else:
            data.PPRFE = fe

        return data

    def _compute_pprfe(
        self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Internal method to compute PPR feature encodings.

        Computes PPR diffusion at multiple alpha values and aggregates
        over input features to produce a fixed-dimension output.

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
            PPR feature encodings of shape [N, fe_dim].
        """
        device = edge_index.device

        if edge_index.size(1) == 0 or num_nodes <= 1:
            return torch.zeros(num_nodes, self.fe_dim, device=device)

        # Convert to dense adjacency matrix
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        adj_np = adj.cpu().numpy().astype(np.float64)

        # Add self-loops: A^ = A + I
        if self.self_loop:
            adj_np = adj_np + np.eye(num_nodes)

        # Compute degree matrix D
        deg = np.sum(adj_np, axis=1)
        # Handle isolated nodes
        deg_safe = np.where(deg > 0, deg, 1.0)
        deg_inv_sqrt = np.diag(1.0 / np.sqrt(deg_safe))

        # Normalized adjacency: Ã = D^{-1/2} A^ D^{-1/2}
        adj_norm = deg_inv_sqrt @ adj_np @ deg_inv_sqrt

        # Generate alpha values (teleport probabilities)
        start, num_alphas = (
            self.alpha_param_PPRFE[0],
            self.alpha_param_PPRFE[1],
        )
        # Use linear spacing for alpha in (0, 1]
        # Start from a small alpha (more global) to larger alpha (more local)
        alpha_values = np.linspace(start, 0.9, num_alphas)

        x_np = x.detach().cpu().numpy().astype(np.float64)
        ppr_fe = []

        identity = np.eye(num_nodes)
        for alpha in alpha_values:
            # PPR = α(I - (1-α)Ã)^{-1}
            ppr_matrix = alpha * inv(identity - (1 - alpha) * adj_norm)
            # Diffuse features
            x_ppr = ppr_matrix @ x_np
            ppr_fe.append(torch.from_numpy(x_ppr).float().to(device))

        ppr_fe = torch.stack(ppr_fe, dim=1)  # [N, fe_dim, F]

        # Aggregate over features to produce fixed-dimension output
        agg_fn = getattr(ppr_fe, self._AGG_FN_MAP[self.aggregation])
        ppr_fe = agg_fn(dim=-1)  # [N, fe_dim]

        if torch.any(torch.isnan(ppr_fe)):
            raise ValueError("PPRFE contains NaNs")

        return ppr_fe.float()
