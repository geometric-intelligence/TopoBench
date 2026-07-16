"""FAGCN backbone for TopoBench graph tasks.

This module implements "Beyond Low-frequency Information in Graph Convolutional
Networks" (Bo et al., AAAI 2021) for supervised TopoBench graph tasks.

FAGCN replaces the fixed low-pass filter of GCN with a self-gating mechanism
that learns a signed edge coefficient alpha_ij in (-1, 1) via a tanh attention.
Positive alpha_ij aggregates low-frequency (smoothing) information; negative
alpha_ij aggregates high-frequency (sharpening) information. This allows the
model to adapt to both homophilic and heterophilic graphs.

The update rule for node i at layer l is (Eq. 5 in the paper):

    h_i^(l) = eps * h_i^(l-1)
               + sum_{j in N(i)} alpha_ij / sqrt(d_i * d_j) * h_j^(l-1)

where alpha_ij = tanh(a^T [h_i^(l-1) || h_j^(l-1)]) and eps is a small
residual weight that retains self-information.

The forward pass returns node embeddings of shape [num_nodes, out_channels]
and is compatible with :class:`topobench.nn.wrappers.GNNWrapper`.

Reference
---------
Bo, D., Wang, X., Shi, C., & Shen, H. (2021).
Beyond Low-frequency Information in Graph Convolutional Networks.
AAAI 2021. https://arxiv.org/abs/2101.00797
Official code: https://github.com/bdy9527/FAGCN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FAConv
from torch_geometric.utils import add_self_loops, degree

"""
class FAConv(MessagePassing):
    # Single frequency-adaptive graph convolution layer.

    # Implements Eq. 5 from Bo et al. (AAAI 2021):

    #     h_i' = eps * h_i + sum_{j in N(i)} alpha_ij / sqrt(d_i * d_j) * h_j

    # where alpha_ij = tanh(a^T [h_i || h_j]).

    # Parameters
    # ----------
    # channels : int
    #     Node feature dimension (input and output are the same size).
    # eps : float, optional
    #     Residual self-loop weight. Defaults to 0.1.
    # dropout : float, optional
    #     Dropout probability applied to attention coefficients. Defaults to 0.0.
    

    def __init__(
        self,
        channels: int,
        eps: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(aggr="add")
        self.channels = channels
        self.eps = eps
        self.dropout = dropout

        # Attention vector: a^T [h_i || h_j] -> scalar
        self.att = nn.Linear(2 * channels, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        # Compute one FA-GCN layer forward pass.

        # Parameters
        # ----------
        # x : torch.Tensor
        #     Node features of shape ``[num_nodes, channels]``.
        # edge_index : torch.Tensor
        #     Graph connectivity in COO format of shape ``[2, num_edges]``.

        # Returns
        # -------
        # torch.Tensor
        #     Updated node embeddings of shape ``[num_nodes, channels]``.
        
        num_nodes = x.size(0)

        # Compute degree-based normalisation: 1 / sqrt(d_i)
        row, col = edge_index
        deg = degree(col, num_nodes=num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0

        # Propagate: aggregates alpha_ij / sqrt(d_i*d_j) * h_j
        agg = self.propagate(
            edge_index,
            x=x,
            deg_inv_sqrt=deg_inv_sqrt,
        )

        # Residual self term: eps * h_i
        return self.eps * x + agg

    def message(
        self,
        x_i,
        x_j,
        deg_inv_sqrt_i,
        deg_inv_sqrt_j,
    ):
        # Compute signed, degree-normalised messages.

        # Parameters
        # ----------
        # x_i : torch.Tensor
        #     Target node features of shape ``[num_edges, channels]``.
        # x_j : torch.Tensor
        #     Source node features of shape ``[num_edges, channels]``.
        # deg_inv_sqrt_i : torch.Tensor
        #     Inverse sqrt degree of target nodes, shape ``[num_edges]``.
        # deg_inv_sqrt_j : torch.Tensor
        #     Inverse sqrt degree of source nodes, shape ``[num_edges]``.

        # Returns
        # -------
        # torch.Tensor
        #     Messages of shape ``[num_edges, channels]``.
        
        # alpha_ij = tanh(a^T [h_i || h_j])  in (-1, 1)
        alpha = torch.tanh(self.att(torch.cat([x_i, x_j], dim=-1)))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Degree normalisation: 1 / sqrt(d_i * d_j)
        norm = deg_inv_sqrt_i * deg_inv_sqrt_j

        return alpha * norm.unsqueeze(-1) * x_j
"""

class FAGCN(nn.Module):
    """Frequency Adaptive Graph Convolutional Network (FAGCN).

    Implements the full FAGCN model from Bo et al. (AAAI 2021).

    A linear feature projection maps inputs to a fixed ``hidden_channels``
    dimension, then ``num_layers`` FAConv layers are stacked. The final
    node representations are returned without a classification head so they
    can be handled by the TopoBench readout.

    Parameters
    ----------
    in_channels : int
        Input node feature dimension.
    hidden_channels : int
        Hidden and output embedding dimension.
    out_channels : int, optional
        Output node embedding dimension. Defaults to ``hidden_channels``.
    num_layers : int, optional
        Number of FA-GCN message-passing layers. Defaults to 2.
    eps : float, optional
        Residual self-loop weight in FAConv. Defaults to 0.1.
    dropout : float, optional
        Dropout on attention coefficients and between layers. Defaults to 0.5.

    Examples
    --------
    >>> import torch
    >>> from topobench.nn.backbones.graph.fagcn import FAGCN
    >>> model = FAGCN(in_channels=16, hidden_channels=32)
    >>> x = torch.randn(10, 16)
    >>> edge_index = torch.tensor([[0,1,2],[1,2,0]])
    >>> out = model(x=x, edge_index=edge_index)
    >>> out.shape
    torch.Size([10, 32])
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int | None = None,
        num_layers: int = 2,
        eps: float = 0.1,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be a positive integer.")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be a positive integer.")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels if out_channels is not None else hidden_channels
        self.num_layers = num_layers
        self.eps = eps
        self.dropout = dropout

        # Input projection (paper uses a single linear layer + dropout)
        self.lin = nn.Linear(in_channels, hidden_channels)

        # Stack of FAConv layers (all operate at hidden_channels)
        self.convs = nn.ModuleList(
            [FAConv(hidden_channels, eps=eps, dropout=dropout) for _ in range(num_layers)]
        )

        # Optional output projection if out_channels != hidden_channels
        if self.out_channels != hidden_channels:
            self.out_proj = nn.Linear(hidden_channels, self.out_channels)
        else:
            self.out_proj = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute FAGCN node embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``[num_nodes, in_channels]``.
        edge_index : torch.Tensor
            Graph connectivity in COO format, shape ``[2, num_edges]``.
        edge_weight : torch.Tensor, optional
            Ignored. Accepted for API compatibility with GNNWrapper.
        edge_attr : torch.Tensor, optional
            Ignored. Accepted for API compatibility with GNNWrapper.
        batch : torch.Tensor, optional
            Batch vector of shape ``[num_nodes]``. Ignored by the backbone
            but accepted for API compatibility.

        Returns
        -------
        torch.Tensor
            Node embeddings of shape ``[num_nodes, out_channels]``.
        """
        x0 = x
        # Input projection + activation + dropout  (Sec. 4 in paper)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin(x))

        # Stack FA-GCN layers
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, x0, edge_index)

        return self.out_proj(x)
