"""Random Walk Structural Encodings (RWSE) Transform."""
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree


class RWSE(BaseTransform):
    r"""
    Random Walk Structural Encoding (RWSE) transform.

    Computes return probabilities of random walks of length 1..K
    for each node in the graph, and appends them as structural
    encodings to node features.

    Parameters
    ----------
    max_pe_dim : int
        Maximum walk length (number of RWSE dimensions).
    concat_to_x : bool, optional
        If True, concatenates the encodings with existing node
        features in ``data.x``. If ``data.x`` is None, creates it.
        Default is True.
    **kwargs : dict
        Additional arguments (not used).
    """

    def __init__(self, max_pe_dim: int, concat_to_x: bool = True, **kwargs):
        self.max_pe_dim = max_pe_dim
        self.concat_to_x = concat_to_x

    def forward(self, data: Data) -> Data:
        pe = self._compute_rwse(data.edge_index, data.num_nodes)

        if self.concat_to_x:
            if data.x is None:
                data.x = pe
            else:
                data.x = torch.cat([data.x, pe], dim=-1)
        else:
            setattr(data, "RWSE", pe)

        return data

    def _compute_rwse(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Internal method to compute RWSE return probabilities."""
        device = edge_index.device

        if edge_index.numel() == 0 or num_nodes <= 1:
            return torch.zeros(num_nodes, self.max_pe_dim, device=device)

        # Degree and adjacency
        deg = degree(edge_index[0], num_nodes=num_nodes).float().to(device)
        deg = torch.where(deg == 0, torch.ones_like(deg), deg)

        adj = torch.zeros(num_nodes, num_nodes, device=device)
        adj[edge_index[0], edge_index[1]] = 1.0

        # Transition matrix
        P = adj / deg.unsqueeze(-1)

        # RWSE features
        rwse = torch.zeros(num_nodes, self.max_pe_dim, device=device)
        P_power = torch.eye(num_nodes, device=device)

        for k in range(1, self.max_pe_dim + 1):
            P_power = P_power @ P
            rwse[:, k - 1] = P_power.diag()  # return probs

        return rwse
