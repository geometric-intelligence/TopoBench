"""This module implements a GPS-based model[1] that can be used with the training framework.

GPS combines local message passing with global attention mechanisms.
Uses the official PyTorch Geometric GPSConv implementation.

[1] Rampášek, Ladislav, et al. "Recipe for a general, powerful, scalable graph transformer."
Advances in Neural Information Processing Systems 35 (2022): 14501-14515.
"""

from typing import Any

import torch
import torch.nn as nn
from torch_geometric.nn import (
    GINConv,
    GPSConv,
    PNAConv,
)
from torch_geometric.nn.attention import PerformerAttention


class RedrawProjection:
    """
    Helper class to handle redrawing of random projections in Performer attention.

    This is crucial for maintaining the quality of the random feature approximation.

    Parameters
    ----------
    model : torch.nn.Module
        The model containing PerformerAttention modules.
    redraw_interval : int or None, optional
        Interval for redrawing random projections. If None, projections are not redrawn. Default is None.
    """

    def __init__(
        self, model: torch.nn.Module, redraw_interval: int | None = None
    ):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        """Redraw random projections in PerformerAttention modules if needed.

        Returns
        -------
        None
            None.
        """
        if not self.model.training or self.redraw_interval is None:
            return

        if self.num_last_redraw >= self.redraw_interval:
            # Find all PerformerAttention modules in the model
            fast_attentions = [
                module
                for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]

            # Redraw projections for each PerformerAttention module
            for fast_attention in fast_attentions:
                if hasattr(fast_attention, "redraw_projection_matrix"):
                    fast_attention.redraw_projection_matrix()

            self.num_last_redraw = 0
            return

        self.num_last_redraw += 1


class GPSEncoder(torch.nn.Module):
    """
    GPS Encoder that can be used with the training framework.

    Uses the official PyTorch Geometric GPSConv implementation.
    This encoder combines local message passing with global attention mechanisms
    for powerful graph representation learning.

    Parameters
    ----------
    input_dim : int
        Dimension of input node features.
    hidden_dim : int
        Dimension of hidden layers.
    num_layers : int, optional
        Number of GPS layers. Default is 4.
    heads : int, optional
        Number of attention heads in GPSConv layers. Default is 4.
    dropout : float, optional
        Dropout rate for GPSConv layers. Default is 0.1.
    attn_type : str, optional
        Type of attention mechanism to use. Options are 'multihead', 'performer', etc.
        Default is 'multihead'.
    local_conv_type : str, optional
        Type of local message passing layer. Options are 'gin', 'pna', etc.
        Default is 'gin'.
    use_edge_attr : bool, optional
        Whether to use edge attributes in GPSConv layers. Default is False.
    redraw_interval : int or None, optional
        Interval for redrawing random projections in Performer attention.
        If None, projections are not redrawn. Default is None.
    attn_kwargs : dict, optional
        Additional keyword arguments for the attention mechanism.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        attn_type: str = "multihead",
        local_conv_type: str = "gin",
        use_edge_attr: bool = False,
        redraw_interval: int | None = None,
        attn_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.attn_type = attn_type
        self.use_edge_attr = use_edge_attr

        # GPS layers using official PyG GPSConv
        self.convs = nn.ModuleList()
        attn_kwargs = attn_kwargs or {}

        for _ in range(num_layers):
            # Create local MPNN
            if local_conv_type == "gin":
                nn_module = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                # Always use GINConv (no edge attributes) for simplicity
                local_conv = GINConv(nn_module)
            elif local_conv_type == "pna":
                # PNA aggregators and scalers
                aggregators = ["mean", "min", "max", "std"]
                scalers = ["identity", "amplification", "attenuation"]
                # Assume degree statistics for PNA (these would normally be computed from data)
                # For now, use reasonable defaults
                deg = torch.tensor([1, 2, 3, 4, 5, 10, 20], dtype=torch.long)
                local_conv = PNAConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    aggregators=aggregators,
                    scalers=scalers,
                    deg=deg,
                    towers=1,
                    pre_layers=1,
                    post_layers=1,
                    divide_input=False,
                )
            else:
                raise ValueError(
                    f"Unsupported local conv type: {local_conv_type}. Supported: 'gin', 'pna'"
                )

            # Create GPS layer using PyG's implementation
            conv = GPSConv(
                channels=hidden_dim,
                conv=local_conv,
                heads=heads,
                dropout=dropout,
                attn_type=attn_type,
                attn_kwargs=attn_kwargs,
            )
            self.convs.append(conv)

        # Setup redraw projection for Performer attention
        if attn_type == "performer":
            redraw_interval = redraw_interval or 1000
        else:
            redraw_interval = None

        self.redraw_projection = RedrawProjection(
            self.convs, redraw_interval=redraw_interval
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of GPS encoder.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, input_dim].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].
        batch : torch.Tensor, optional
            Batch vector assigning each node to a specific graph. Shape [num_nodes]. Default is None.
        edge_attr : torch.Tensor, optional
            Edge feature matrix of shape [num_edges, edge_dim]. Default is None.
        **kwargs : dict
            Additional arguments (not used).

        Returns
        -------
        torch.Tensor
            Output node feature matrix of shape [num_nodes, hidden_dim].
        """
        # Redraw projections if using Performer attention
        if self.training:
            self.redraw_projection.redraw_projections()

        # Apply GPS layers (no edge attributes)
        for conv in self.convs:
            x = conv(x, edge_index, batch=batch)

        return x
