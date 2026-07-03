"""Dir-GNN: Directed Graph Neural Network.

Reference
---------
Rossi et al., "Edge Directionality Improves Learning on Heterophilic Graphs",
Learning on Graphs Conference, 2024. https://arxiv.org/abs/2305.10498

Equations (from paper):
    h_v^{(l+1)} = MLP( h_v^{(l)},
                        AGG_{u:(u,v) in E}( h_u^{(l)} ),   # in-neighbours
                        AGG_{u:(v,u) in E}( h_u^{(l)} ) )  # out-neighbours
    
    where alpha controls the mixing of in/out aggregations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class DirSageConv(MessagePassing):
    """Single Dir-GNN convolutional layer using mean aggregation (GraphSAGE-style).

    Performs separate aggregations over in-neighbours and out-neighbours,
    then combines with the node's own features via a learned MLP.

    Parameters
    ----------
    in_channels : int
        Dimensionality of input node features.
    out_channels : int
        Dimensionality of output node features.
    alpha : float
        Mixing coefficient in [0,1] for in vs out aggregation.
        alpha=0.5 gives equal weight. (Eq. 4 in paper.)
    """

    def __init__(self, in_channels: int, out_channels: int, alpha: float = 0.5):
        # aggr=None: we handle aggregation manually for in/out separation
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha

        # Separate linear transforms for in-agg, out-agg, and self
        self.lin_in = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_out = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using Glorot uniform initialization."""
        nn.init.xavier_uniform_(self.lin_in.weight)
        nn.init.xavier_uniform_(self.lin_out.weight)
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape [N, in_channels]
            Node feature matrix.
        edge_index : Tensor, shape [2, E]
            Edge indices in COO format. edge_index[0] = source, edge_index[1] = target.

        Returns
        -------
        Tensor, shape [N, out_channels]
        """
        num_nodes = x.size(0)

        # --- In-neighbour aggregation: edges point TO v ---
        # Standard: propagate from src (edge_index[0]) to dst (edge_index[1])
        deg_in = degree(edge_index[1], num_nodes, dtype=x.dtype).clamp(min=1)
        agg_in = self.propagate(edge_index, x=x, norm=1.0 / deg_in[edge_index[1]])

        # --- Out-neighbour aggregation: edges point FROM v ---
        # Reverse edge_index to treat out-neighbours as in-neighbours
        edge_index_t = edge_index.flip(0)  # [2, E], now [dst, src]
        deg_out = degree(edge_index_t[1], num_nodes, dtype=x.dtype).clamp(min=1)
        agg_out = self.propagate(edge_index_t, x=x, norm=1.0 / deg_out[edge_index_t[1]])

        # --- Combine (Eq. 4 in paper) ---
        # h = W_self * x + alpha * W_in * agg_in + (1-alpha) * W_out * agg_out
        out = (
            self.lin_self(x)
            + self.alpha * self.lin_in(agg_in)
            + (1.0 - self.alpha) * self.lin_out(agg_out)
            + self.bias
        )
        return out

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        """Compute normalized messages from neighbors.

        Parameters
        ----------
        x_j : Tensor
            Neighbor features.
        norm : Tensor
            Per-edge normalization factor (1 / degree).
        """
        return norm.view(-1, 1) * x_j


class DirGNN(nn.Module):
    """Directed Graph Neural Network (Dir-GNN).

    A stack of DirSageConv layers with ReLU activations and optional dropout,
    implementing the framework from Rossi et al. (2024).

    Each layer separately aggregates in-neighbours and out-neighbours,
    giving the model awareness of edge directionality — particularly
    beneficial on heterophilic graphs.

    Parameters
    ----------
    in_channels : int
        Dimensionality of input node features.
    hidden_channels : int
        Dimensionality of hidden representations.
    out_channels : int
        Dimensionality of output representations (number of classes).
    num_layers : int
        Number of Dir-GNN convolutional layers.
    dropout : float
        Dropout probability applied after each hidden layer.
    alpha : float
        Mixing coefficient for in/out aggregation (see DirSageConv).

    Examples
    --------
    >>> model = DirGNN(in_channels=16, hidden_channels=64, out_channels=7,
    ...                num_layers=2, dropout=0.5, alpha=0.5)
    >>> x = torch.randn(100, 16)
    >>> edge_index = torch.randint(0, 100, (2, 300))
    >>> out = model(x, edge_index)
    >>> out.shape
    torch.Size([100, 7])
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        # Input layer
        self.convs.append(DirSageConv(in_channels, hidden_channels, alpha=alpha))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(DirSageConv(hidden_channels, hidden_channels, alpha=alpha))
        # Output layer
        if num_layers > 1:
            self.convs.append(DirSageConv(hidden_channels, out_channels, alpha=alpha))

        self.num_layers = num_layers

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass through all Dir-GNN layers.

        Parameters
        ----------
        x : Tensor, shape [N, in_channels]
            Node feature matrix.
        edge_index : Tensor, shape [2, E]
            Edge connectivity in COO format.

        Returns
        -------
        Tensor, shape [N, out_channels]
            Node-level output representations.
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x