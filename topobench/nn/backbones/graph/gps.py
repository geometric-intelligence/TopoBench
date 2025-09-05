"""
This module implements a GPS encoder that can be used with the training framework.
GPS combines local message passing with global attention mechanisms.

Uses the official PyTorch Geometric GPSConv implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GPSConv, GINEConv, GINConv, PNAConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.attention import PerformerAttention
from typing import Dict, List, Optional, Tuple, Any, Union

class RedrawProjection:
    """
    Helper class to handle redrawing of random projections in Performer attention.
    This is crucial for maintaining the quality of the random feature approximation.
    """
    def __init__(self, model: torch.nn.Module, redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        """Redraw random projections in PerformerAttention modules if needed."""
        if not self.model.training or self.redraw_interval is None:
            return
        
        if self.num_last_redraw >= self.redraw_interval:
            # Find all PerformerAttention modules in the model
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            
            # Redraw projections for each PerformerAttention module
            for fast_attention in fast_attentions:
                if hasattr(fast_attention, 'redraw_projection_matrix'):
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
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        pe_dim: int = 10,
        pe_type: str = 'laplacian',
        pe_norm: bool = False,
        heads: int = 4,
        dropout: float = 0.1,
        attn_type: str = 'multihead',
        local_conv_type: str = 'gin',
        use_edge_attr: bool = False,
        redraw_interval: Optional[int] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pe_type = pe_type
        self.heads = heads 
        self.pe_dim = pe_dim
        self.pe_norm = pe_norm
        self.dropout = dropout
        self.attn_type = attn_type
        self.use_edge_attr = use_edge_attr
        
        # Input projection
        if pe_type in ['laplacian', 'degree', 'rwse']:
            # If using PE, split channels between node features and PE
            self.node_proj = nn.Linear(input_dim, hidden_dim - pe_dim)
            self.pe_proj = nn.Linear(pe_dim, pe_dim)
            if self.pe_norm:
                self.pe_norm = nn.BatchNorm1d(pe_dim)
            else:
                self.pe_norm = None
        elif pe_type is None or pe_type == "None":
            # No PE, use full hidden_dim for node features
            self.node_proj = nn.Linear(input_dim, hidden_dim)
            self.pe_proj = None
            self.pe_norm = None
        else:
            raise ValueError(f"Invalid PE type: {pe_type}. Supported: 'laplacian', 'degree', 'rwse', None")
        
        # GPS layers using official PyG GPSConv
        self.convs = nn.ModuleList()
        attn_kwargs = attn_kwargs or {}
        
        for _ in range(num_layers):
            # Create local MPNN
            if local_conv_type == 'gin':
                nn_module = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                # Always use GINConv (no edge attributes) for simplicity
                local_conv = GINConv(nn_module)
            elif local_conv_type == 'pna':
                # PNA aggregators and scalers
                aggregators = ['mean', 'min', 'max', 'std']
                scalers = ['identity', 'amplification', 'attenuation']
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
                    divide_input=False
                )
            else:
                raise ValueError(f"Unsupported local conv type: {local_conv_type}. Supported: 'gin', 'pna'")
            
            # Create GPS layer using PyG's implementation
            conv = GPSConv(
                channels=hidden_dim,
                conv=local_conv,
                heads=heads,
                dropout=dropout,
                attn_type=attn_type,
                attn_kwargs=attn_kwargs
            )
            self.convs.append(conv)
        
        # Setup redraw projection for Performer attention
        if attn_type == 'performer':
            redraw_interval = redraw_interval or 1000
        else:
            redraw_interval = None
            
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=redraw_interval
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None, 
                edge_attr: Optional[torch.Tensor] = None,
                pe: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        Forward pass of GPS encoder.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices [num_nodes] (optional)
            edge_attr: Edge attributes [num_edges, edge_attr_dim] (optional)
            pe: Positional encodings [num_nodes, pe_dim] (optional)
            
        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        # Redraw projections if using Performer attention
        if self.training:
            self.redraw_projection.redraw_projections()
        
        # Node feature projection
        h = self.node_proj(x)
        
        # Add positional encoding if available
        if self.pe_proj is not None:
            if pe is not None:
                # Normalize PE
                if self.pe_norm is not None and pe.size(0) > 1:
                    pe_normalized = self.pe_norm(pe)
                else:
                    pe_normalized = pe
                
                # Project PE
                pe_proj = self.pe_proj(pe_normalized)
                
                # Concatenate node features with PE
                h = torch.cat([h, pe_proj], dim=-1)
            else:
                raise ValueError(f"If PE type is not None, PE must be provided in forward pass")
        
        
        # Apply GPS layers (no edge attributes)
        for conv in self.convs:
            h = conv(h, edge_index, batch=batch)
        
        return h