"""Sheaf diffusion model implementations."""

from abc import abstractmethod

import numpy as np

# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F
from torch import nn


class SheafLearner(nn.Module):
    """
    Base model that learns a sheaf from the features and the graph structure.

    This abstract class provides the interface for learning sheaf structures,
    including storing the learned Laplacian.
    """

    def __init__(self):
        super().__init__()
        self.L = None

    @abstractmethod
    def forward(self, x, edge_index):
        """
        Learn sheaf structure from node features and graph structure.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix.
        edge_index : torch.Tensor
            Edge indices of the graph.

        Returns
        -------
        torch.Tensor
            Learned sheaf parameters.

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError()

    def set_L(self, weights):
        """
        Store the learned Laplacian weights.

        Parameters
        ----------
        weights : torch.Tensor
            Laplacian weights to store.

        Returns
        -------
        None
            None.
        """
        self.L = weights.clone().detach()


class LocalConcatSheafLearner(SheafLearner):
    """
    Sheaf learner that concatenates source and target node features.

    This learner computes sheaf parameters by concatenating the features of
    connected nodes and passing them through a linear layer with activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels per node.
    out_shape : tuple of int
        Shape of output sheaf parameters. Should be (d,) for diagonal sheaf
        or (d, d) for general sheaf.
    sheaf_act : str, optional
        Activation function to apply. Options are 'id', 'tanh', or 'elu'.
        Default is 'tanh'.

    Raises
    ------
    ValueError
        If sheaf_act is not one of the supported activation functions.
    """

    def __init__(
        self, in_channels: int, out_shape: tuple[int, ...], sheaf_act="tanh"
    ):
        super().__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape

        self.linear1 = torch.nn.Linear(
            in_channels * 2, int(np.prod(out_shape)), bias=False
        )

        if sheaf_act == "id":
            self.act = lambda x: x
        elif sheaf_act == "tanh":
            self.act = torch.tanh
        elif sheaf_act == "elu":
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(self, x, edge_index):
        """
        Compute sheaf parameters from concatenated node features.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, in_channels].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].

        Returns
        -------
        torch.Tensor
            Sheaf parameters of shape [num_edges, *out_shape].
        """
        # Use PyG's efficient indexing (same as original but potentially more optimized)
        row, col = edge_index
        x_row = x[row]  # Source node features
        x_col = x[col]  # Target node features

        # Concatenate source and target features
        x_concat = torch.cat([x_row, x_col], dim=1)

        # Apply linear transformation
        maps = self.linear1(x_concat)

        # Apply activation
        maps = self.act(maps)

        # Reshape to output shape
        if len(self.out_shape) == 2:
            result = maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            result = maps.view(-1, self.out_shape[0])

        return result


class IdentitySheafLearner(SheafLearner):
    """
    Sheaf learner that returns fixed identity restriction maps.

    Unlike :class:`LocalConcatSheafLearner`, this learner has no learnable
    parameters: every restriction map ``F_{v<e}`` is the identity. The
    resulting sheaf Laplacian therefore collapses to the standard graph
    Laplacian, yielding the Identity Sheaf Network (ISN) baseline of
    Hernandez Caralt et al. used to ablate sheaf learning [1].

    Parameters
    ----------
    in_channels : int
        Number of input channels per node. Unused (kept for interface
        compatibility with :class:`LocalConcatSheafLearner`).
    out_shape : tuple of int
        Shape of the restriction maps per edge. ``(d,)`` for a diagonal
        sheaf (identity diagonal is all ones) or ``(d, d)`` for a general
        sheaf (the d x d identity matrix).

    References
    ----------
    [1] Hernandez Caralt et al. "On the Necessity of Learnable Sheaf
        Laplacians." GRaM Workshop, ICLR 2026. https://arxiv.org/abs/2603.05395
    """

    def __init__(self, in_channels: int, out_shape: tuple[int, ...]):
        super().__init__()
        assert len(out_shape) in [1, 2]
        self.in_channels = in_channels
        self.out_shape = out_shape

    def forward(self, x, edge_index):
        """
        Return identity restriction maps for every edge.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, in_channels]. Used only
            for dtype/device; its values do not affect the output.
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].

        Returns
        -------
        torch.Tensor
            Identity restriction maps of shape [num_edges, *out_shape].
        """
        num_edges = edge_index.size(1)
        if len(self.out_shape) == 1:
            # Diagonal sheaf: the identity diagonal is all ones.
            return torch.ones(
                num_edges,
                self.out_shape[0],
                dtype=x.dtype,
                device=x.device,
            )
        # General sheaf: broadcast the d x d identity matrix to every edge.
        d = self.out_shape[0]
        eye = torch.eye(d, dtype=x.dtype, device=x.device)
        return eye.unsqueeze(0).expand(num_edges, d, d)
