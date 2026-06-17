"""Directed Simplicial Neural Network Backbone.

This module implements the DirSNN architecture for processing directed
simplicial complexes within the TopoBench framework.
"""

import torch
import torch.nn as nn


class DirSNNLayer(nn.Module):
    """Directed Simplicial Neural Network Layer.

    Computes message passing over directed simplices using oriented boundary matrices.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    dropout : float, optional
        Base dropout probability. Default is 0.0.
    """

    def __init__(self, in_features, out_features, dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable weight matrices for lower, upper, and self neighborhoods
        self.W_lower = nn.Linear(in_features, out_features, bias=False)
        self.W_upper = nn.Linear(in_features, out_features, bias=False)
        self.W_self = nn.Linear(in_features, out_features, bias=True)

        # --- Target 4: Independent Asymmetric Regularization ---
        # Upper pathway receives the full dropout to combat dense-clique noise.
        # Lower and self pathways receive half-dropout to preserve stable signals.
        self.drop_down = nn.Dropout(dropout * 0.5)
        self.drop_up = nn.Dropout(dropout)
        self.drop_self = nn.Dropout(dropout * 0.5)

        # --- Target 2: Harmonic Space & Pathway Equalization Weights ---
        # Initialized to 1/3 to allow dynamic isolation of cyclic flows
        self.alpha = nn.Parameter(torch.tensor([0.333]))
        self.beta = nn.Parameter(torch.tensor([0.333]))
        self.gamma = nn.Parameter(torch.tensor([0.333]))

        # --- Target 5: Source/Sink Gradient Masking Bypass ---
        # Learnable epsilon residuals to prevent dead-weights on topological boundaries
        self.eps_down = nn.Parameter(torch.tensor([1e-4]))
        self.eps_up = nn.Parameter(torch.tensor([1e-4]))

        self.activation = nn.ReLU()

    def forward(self, x, boundary_lower, boundary_upper):
        """Forward pass for the directed layer."""

        # Compute message from lower adjacent simplices with epsilon bypass
        msg_lower = torch.sparse.mm(boundary_lower, x) + (self.eps_down * x)
        out_lower = self.drop_down(self.W_lower(msg_lower))

        # Compute message from upper adjacent simplices with epsilon bypass
        msg_upper = torch.sparse.mm(boundary_upper, x) + (self.eps_up * x)
        out_upper = self.drop_up(self.W_upper(msg_upper))

        # Self-loop update
        out_self = self.drop_self(self.W_self(x))

        # Aggregate directed signals with dynamic weighting
        out = self.activation(
            (self.alpha * out_lower)
            + (self.beta * out_upper)
            + (self.gamma * out_self)
        )
        return out


class DirSNN(nn.Module):
    """The main DirSNN Backbone that integrates into TopoBench.

    Parameters
    ----------
    in_channels : int
        Input feature dimensions.
    hidden_channels : int
        Hidden feature dimensions.
    out_channels : int
        Output feature dimensions.
    num_layers : int, optional
        Number of message passing layers. Default is 2.
    dropout : float, optional
        Dropout probability. Default is 0.5.
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = num_layers

        # --- Target 6 Config: Density Threshold ---
        # Maximum allowed average triangles per edge before active pruning engages
        self.max_upper_degree = 32

        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(DirSNNLayer(in_channels, hidden_channels, dropout))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                DirSNNLayer(hidden_channels, hidden_channels, dropout)
            )

        # Output layer
        self.layers.append(DirSNNLayer(hidden_channels, out_channels, dropout))

    def _sparsify_boundary(self, B, max_density):
        """
        Target 6: Active Simplicial Sparsification.
        Dynamically drops 2-simplices (triangles) if the local clique density
        exceeds the memory-safe threshold during training.
        """
        if B._nnz() == 0:
            return B

        # Calculate the average number of triangles connected to each edge
        avg_degree = B._nnz() / B.shape[0]

        if self.training and avg_degree > max_density:
            # Calculate the probability needed to enforce the strict density limit
            keep_prob = max_density / avg_degree

            # Generate a boolean mask to randomly drop excess topological connections
            mask = torch.rand(B._nnz(), device=B.device) < keep_prob

            # Reconstruct the sparse tensor with the pruned connections
            B_pruned = torch.sparse_coo_tensor(
                B.indices()[:, mask],
                B.values()[mask],
                B.size()
            )
            return B_pruned.coalesce()

        return B

    def forward(self, batch):
        """Forward pass through the backbone."""

        # Extract edge features (1-simplices) as the primary representation
        x = batch.edge_x
        B1 = batch.B1
        B2 = batch.B2

        # --- Target 7: Sparse Autograd Engine Stabilization ---
        # Quarantine the structural math so PyTorch does not materialize dense gradients
        with torch.no_grad():

            # --- Target 1: Boundary Guard ---
            if self.training:
                boundary_product = torch.sparse.mm(B1, B2).coalesce()
                if (
                    boundary_product._nnz() > 0
                    and torch.norm(boundary_product.values()) > 1e-5
                ):
                    raise ValueError(
                        "Topological condition violated: B1 @ B2 != 0. Check dataset orientation."
                    )

            # --- Target 6: Simplicial Sparsification Guard ---
            B2 = self._sparsify_boundary(B2, max_density=self.max_upper_degree)

            # Compute the directed Hodge Laplacians safely
            L_down = torch.sparse.mm(B1.t(), B1).coalesce()
            L_up = torch.sparse.mm(B2, B2.t()).coalesce()

        # Pass messages through layers (Autograd now safely only tracks `x` and weights)
        for layer in self.layers:
            x = layer(x, L_down, L_up)

        return x
