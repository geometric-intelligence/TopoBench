# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Inductive Neural Sheaf Diffusion models.

This module implements three variants of inductive sheaf diffusion:
- Diagonal: Diagonal restriction maps
- Bundle: Orthogonal restriction maps with normalization
- General: Full matrix restriction maps
"""

import torch
import torch.nn.functional as F
import torch_sparse
from torch import nn

from .laplacian_builders import (
    DiagLaplacianBuilder,
    GeneralLaplacianBuilder,
    NormConnectionLaplacianBuilder,
)
from .sheaf_base import SheafDiffusion
from .sheaf_models import LocalConcatSheafLearner


class InductiveDiscreteDiagSheafDiffusion(SheafDiffusion):
    """
    Inductive sheaf diffusion with diagonal restriction maps.

    This model learns diagonal d x d restriction maps for each edge,
    parameterized by d scalar values. Suitable for problems where
    feature channels can be processed independently.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
        - d (int): Dimension of stalk space (must be > 0).
        - layers (int): Number of diffusion layers.
        - hidden_channels (int): Hidden channels per stalk dimension.
        - input_dim (int): Input feature dimension.
        - output_dim (int): Output feature dimension.
        - device (str): Device to run on.
        - input_dropout (float): Input layer dropout rate.
        - dropout (float): Hidden layer dropout rate.
        - sheaf_act (str): Activation for sheaf learning.
    """

    def __init__(self, config):
        super().__init__(None, config)
        assert config["d"] > 0

        self.config = config
        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        for _i in range(self.layers):
            self.lin_right_weights.append(
                nn.Linear(
                    self.hidden_channels, self.hidden_channels, bias=False
                )
            )
            nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        for _i in range(self.layers):
            self.lin_left_weights.append(nn.Linear(self.d, self.d, bias=False))
            nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers)
        for _i in range(num_sheaf_learners):
            self.sheaf_learners.append(
                LocalConcatSheafLearner(
                    self.hidden_dim,
                    out_shape=(self.d,),
                    sheaf_act=self.sheaf_act,
                )
            )

        self.epsilons = nn.ParameterList()
        for _i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, edge_index):
        """
        Forward pass of diagonal sheaf diffusion.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, input_dim].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].

        Returns
        -------
        torch.Tensor
            Output node features of shape [num_nodes, output_dim].
        """
        # Get actual number of nodes dynamically
        actual_num_nodes = x.size(0)

        # Create laplacian builder for this specific graph
        laplacian_builder = DiagLaplacianBuilder(
            actual_num_nodes,
            edge_index,
            d=self.d,
        )

        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Use actual number of nodes
        x = x.view(actual_num_nodes * self.d, -1)

        x0 = x
        for layer in range(self.layers):
            x_maps = F.dropout(
                x,
                p=self.dropout if layer > 0 else 0.0,
                training=self.training,
            )
            # Reshape using actual number of nodes
            maps = self.sheaf_learners[layer](
                x_maps.reshape(actual_num_nodes, -1), edge_index
            )
            L, trans_maps = laplacian_builder(maps)
            self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            x = x.t().reshape(-1, self.d)
            x = self.lin_left_weights[layer](x)
            x = x.reshape(-1, actual_num_nodes * self.d).t()
            x = self.lin_right_weights[layer](x)

            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)
            x = F.elu(x)

            # Use actual number of nodes for epsilon tiling
            coeff = 1 + torch.tanh(self.epsilons[layer]).tile(
                actual_num_nodes, 1
            )
            x0 = coeff * x0 - x
            x = x0

        # Reshape using actual number of nodes
        x = x.reshape(actual_num_nodes, -1)
        x = self.lin2(x)
        return x


class InductiveDiscreteBundleSheafDiffusion(SheafDiffusion):
    """
    Inductive sheaf diffusion with orthogonal bundle restriction maps.

    This model learns orthogonal d x d restriction maps for each edge,
    ensuring isometric transport between stalks. Uses normalized Laplacian
    and Cayley/matrix exponential parameterization for orthogonality.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
        - d (int): Dimension of stalk space (must be > 1).
        - layers (int): Number of diffusion layers.
        - hidden_channels (int): Hidden channels per stalk dimension.
        - input_dim (int): Input feature dimension.
        - output_dim (int): Output feature dimension.
        - device (str): Device to run on.
        - input_dropout (float): Input layer dropout rate.
        - dropout (float): Hidden layer dropout rate.
        - sheaf_act (str): Activation for sheaf learning.
        - orth (str): Orthogonalization method ('cayley' or 'matrix_exp').

    Raises
    ------
    AssertionError
        If d is not greater than 1 or hidden_dim is not divisible by d.
    """

    def __init__(self, config):
        super().__init__(None, config)
        assert config["d"] > 1
        assert self.hidden_dim % self.d == 0

        self.config = config
        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        for _i in range(self.layers):
            self.lin_right_weights.append(
                nn.Linear(
                    self.hidden_channels, self.hidden_channels, bias=False
                )
            )
            nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        for _i in range(self.layers):
            self.lin_left_weights.append(nn.Linear(self.d, self.d, bias=False))
            nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers)
        for _i in range(num_sheaf_learners):
            self.sheaf_learners.append(
                LocalConcatSheafLearner(
                    self.hidden_dim,
                    out_shape=(self.get_param_size(),),
                    sheaf_act=self.sheaf_act,
                )
            )

        self.epsilons = nn.ParameterList()
        for _i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def get_param_size(self):
        """
        Get the number of parameters needed for orthogonal maps.

        Returns
        -------
        int
            Number of parameters (d*(d+1)/2 for lower triangular parameterization).
        """
        return self.d * (self.d + 1) // 2

    def left_right_linear(self, x, left, right, actual_num_nodes):
        """
        Apply left and right linear transformations to stalk vectors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [num_nodes * d, hidden_channels].
        left : nn.Linear
            Left linear transformation (acts on stalk dimension).
        right : nn.Linear
            Right linear transformation (acts on hidden channels).
        actual_num_nodes : int
            Number of nodes in the current graph.

        Returns
        -------
        torch.Tensor
            Transformed tensor of shape [num_nodes * d, hidden_channels].
        """
        x = x.t().reshape(-1, self.d)
        x = left(x)
        x = x.reshape(-1, actual_num_nodes * self.d).t()
        x = right(x)

        return x

    def forward(self, x, edge_index):
        """
        Forward pass of bundle sheaf diffusion.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, input_dim].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].

        Returns
        -------
        torch.Tensor
            Output node features of shape [num_nodes, output_dim].
        """
        # Get actual number of nodes dynamically
        actual_num_nodes = x.size(0)

        # Create laplacian builder for this specific graph
        laplacian_builder = NormConnectionLaplacianBuilder(
            actual_num_nodes,
            edge_index,
            d=self.d,
            orth_map=self.orth_trans,
        )

        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Use actual number of nodes
        x = x.view(
            actual_num_nodes * self.d, -1
        )  # So for each node, we put reshape the output of the lin1 to a tensor of size (final_d, hidden_dim // final_d)
        # This means that if we set "hidden_dim" to 64 and "final_d" to 2, then we have that for each node, we have a tensor of size (2, 32)

        x0, L = x, None
        for layer in range(self.layers):
            # Time each component of the forward pass
            x_maps = F.dropout(
                x,
                p=self.dropout if layer > 0 else 0.0,
                training=self.training,
            )
            x_maps = x_maps.reshape(
                actual_num_nodes, -1
            )  # Reshape using actual number of nodes (so back to the original shape)
            maps = self.sheaf_learners[layer](x_maps, edge_index)
            L, trans_maps = laplacian_builder(maps)
            self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            # Pass actual_num_nodes to left_right_linear
            x = self.left_right_linear(
                x,
                self.lin_left_weights[layer],
                self.lin_right_weights[layer],
                actual_num_nodes,
            )

            # Use the adjacency matrix rather than the diagonal
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            x = F.elu(x)

            # Use actual number of nodes for epsilon tiling
            x0 = (
                1 + torch.tanh(self.epsilons[layer]).tile(actual_num_nodes, 1)
            ) * x0 - x
            x = x0

        # Reshape using actual number of nodes
        x = x.reshape(actual_num_nodes, -1)
        x = self.lin2(x)
        return x


class InductiveDiscreteGeneralSheafDiffusion(SheafDiffusion):
    """
    Inductive sheaf diffusion with general (unrestricted) restriction maps.

    This model learns arbitrary d x d restriction maps for each edge,
    providing maximum expressiveness but requiring more parameters.
    Each restriction map is a full d x d matrix.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
        - d (int): Dimension of stalk space (must be > 1).
        - layers (int): Number of diffusion layers.
        - hidden_channels (int): Hidden channels per stalk dimension.
        - input_dim (int): Input feature dimension.
        - output_dim (int): Output feature dimension.
        - device (str): Device to run on.
        - input_dropout (float): Input layer dropout rate.
        - dropout (float): Hidden layer dropout rate.
        - sheaf_act (str): Activation for sheaf learning.

    Raises
    ------
    AssertionError
        If d is not greater than 1.
    """

    def __init__(self, config):
        super().__init__(None, config)
        assert config["d"] > 1

        self.config = config
        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        for _i in range(self.layers):
            self.lin_right_weights.append(
                nn.Linear(
                    self.hidden_channels, self.hidden_channels, bias=False
                )
            )
            nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        for _i in range(self.layers):
            self.lin_left_weights.append(nn.Linear(self.d, self.d, bias=False))
            nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers)
        for _i in range(num_sheaf_learners):
            self.sheaf_learners.append(
                LocalConcatSheafLearner(
                    self.hidden_dim,
                    out_shape=(self.d, self.d),
                    sheaf_act=self.sheaf_act,
                )
            )

        self.epsilons = nn.ParameterList()
        for _i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def left_right_linear(self, x, left, right, actual_num_nodes):
        """
        Apply left and right linear transformations to stalk vectors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [num_nodes * d, hidden_channels].
        left : nn.Linear
            Left linear transformation (acts on stalk dimension).
        right : nn.Linear
            Right linear transformation (acts on hidden channels).
        actual_num_nodes : int
            Number of nodes in the current graph.

        Returns
        -------
        torch.Tensor
            Transformed tensor of shape [num_nodes * d, hidden_channels].
        """
        x = x.t().reshape(-1, self.d)
        x = left(x)
        x = x.reshape(-1, actual_num_nodes * self.d).t()
        x = right(x)
        return x

    def forward(self, x, edge_index):
        """
        Forward pass of general sheaf diffusion.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, input_dim].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].

        Returns
        -------
        torch.Tensor
            Output node features of shape [num_nodes, output_dim].
        """
        # Get actual number of nodes dynamically
        actual_num_nodes = x.size(0)

        # Create laplacian builder for this specific graph
        laplacian_builder = GeneralLaplacianBuilder(
            actual_num_nodes,
            edge_index,
            d=self.d,
        )

        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Use actual number of nodes
        x = x.view(actual_num_nodes * self.d, -1)

        x0, L = x, None
        for layer in range(self.layers):
            x_maps = F.dropout(
                x,
                p=self.dropout if layer > 0 else 0.0,
                training=self.training,
            )
            # Reshape using actual number of nodes
            maps = self.sheaf_learners[layer](
                x_maps.reshape(actual_num_nodes, -1), edge_index
            )
            L, trans_maps = laplacian_builder(maps)
            self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            # Pass actual_num_nodes to left_right_linear
            x = self.left_right_linear(
                x,
                self.lin_left_weights[layer],
                self.lin_right_weights[layer],
                actual_num_nodes,
            )

            # Use the adjacency matrix rather than the diagonal
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            x = F.elu(x)

            # Use actual number of nodes for epsilon tiling
            x0 = (
                1 + torch.tanh(self.epsilons[layer]).tile(actual_num_nodes, 1)
            ) * x0 - x
            x = x0

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))

        # Reshape using actual number of nodes
        x = x.reshape(actual_num_nodes, -1)
        x = self.lin2(x)
        return x


class InductiveDiscreteDiagSheafPropagation(
    InductiveDiscreteDiagSheafDiffusion
):
    """
    Neural Sheaf Propagation (NSP) with diagonal restriction maps.

    This is the sheaf **wave** model of Suk et al. [1] (vs the diffusion/heat
    dynamics of NSD). It discretises the sheaf wave equation
    ``X''(t) = -Delta_F X(t)`` with the leapfrog method:

        X_{t+1} = 2 X_t - X_{t-1} - h^2 * sigma(Delta_F (I (x) W1) X_t W2)

    where ``h = step_size`` is the leapfrog step. The central second-difference
    discretisation of ``X''(t) = -Delta_F X`` scales the force term by ``h^2``;
    this is the leapfrog stability (CFL) control: without it the wave update
    diverges on dense/high-degree graphs. The original paper sweeps
    ``Step Size in [0.1, 1.0]`` (Table 2); we expose it as ``step_size``
    (default 0.5) and, unlike a naive implementation, actually use it. We do NOT carry a diffusion ``epsilons``
    parameter (the diffusion residual coefficient is meaningless for a wave
    update), so it is removed after construction.

    Two optional appendix knobs from the paper (Table 2) are supported:

    * ``second_linear`` (bool, default False) — an extra input projection applied
      after ``lin1`` and before propagation begins (NOT the sheaf, NOT ``W2``).
    * ``new_laplacian_each_step`` (bool, default True) — if True the sheaf
      ``Delta_F(t)`` is recomputed from ``X_t`` at every layer (dynamic geometry);
      if False it is built once from the encoded ``X_0`` and reused for all layers
      (fixed geometry). In the fixed case only a single sheaf learner is kept, so
      there are no unused parameters.

    Parameters
    ----------
    config : dict
        Same configuration as :class:`InductiveDiscreteDiagSheafDiffusion`, plus
        optional ``step_size`` (float, default 0.5), ``second_linear``
        (bool, default False) and ``new_laplacian_each_step`` (bool, default True).

    References
    ----------
    [1] Suk et al. "Surfing on the Neural Sheaf." NeurIPS 2022 Workshop on
        Symmetry and Geometry in Neural Representations. OpenReview:xOXFkyRzTlu.
    """

    def __init__(self, config):
        super().__init__(config)
        # Wave propagation is second-order: it uses a leapfrog step size, not
        # the first-order diffusion residual epsilons. Remove the unused param.
        del self.epsilons
        self.step_size = float(config.get("step_size", 0.5))

        # Optional extra input projection (appendix "Use Second Linear Transform").
        self.second_linear = bool(config.get("second_linear", False))
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Dynamic vs fixed sheaf geometry (appendix "New Delta each step").
        self.new_laplacian_each_step = bool(
            config.get("new_laplacian_each_step", True)
        )
        # Fixed geometry only needs one sheaf learner; drop the rest so there
        # are no dead parameters. (Per-layer W1/W2 are still used either way.)
        if not self.new_laplacian_each_step:
            self.sheaf_learners = self.sheaf_learners[:1]

    def forward(self, x, edge_index):
        """
        Forward pass of diagonal sheaf wave propagation (leapfrog).

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, input_dim].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].

        Returns
        -------
        torch.Tensor
            Output node features of shape [num_nodes, output_dim].
        """
        actual_num_nodes = x.size(0)

        laplacian_builder = DiagLaplacianBuilder(
            actual_num_nodes,
            edge_index,
            d=self.d,
        )

        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Optional extra input projection before propagation begins.
        if self.second_linear:
            x = self.lin12(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.view(actual_num_nodes * self.d, -1)

        # Fixed geometry: build Delta_F once from the encoded X_0 and reuse it.
        fixed_L = None
        if not self.new_laplacian_each_step:
            maps = self.sheaf_learners[0](
                x.reshape(actual_num_nodes, -1), edge_index
            )
            fixed_L, trans_maps = laplacian_builder(maps)
            self.sheaf_learners[0].set_L(trans_maps)

        # Leapfrog needs the previous and current states.
        x_prev = x
        x_curr = x
        for layer in range(self.layers):
            if self.new_laplacian_each_step:
                # Dynamic geometry: recompute Delta_F(t) from X_t each layer.
                x_maps = F.dropout(
                    x_curr,
                    p=self.dropout if layer > 0 else 0.0,
                    training=self.training,
                )
                maps = self.sheaf_learners[layer](
                    x_maps.reshape(actual_num_nodes, -1), edge_index
                )
                L, trans_maps = laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)
            else:
                L = fixed_L

            # Force term: sigma(Delta_F (I (x) W1) x_curr W2).
            x_layer = F.dropout(x_curr, p=self.dropout, training=self.training)
            x_layer = x_layer.t().reshape(-1, self.d)
            x_layer = self.lin_left_weights[layer](x_layer)
            x_layer = x_layer.reshape(-1, actual_num_nodes * self.d).t()
            x_layer = self.lin_right_weights[layer](x_layer)
            x_layer = torch_sparse.spmm(
                L[0], L[1], x_layer.size(0), x_layer.size(0), x_layer
            )
            x_layer = F.elu(x_layer)

            # Stabilised leapfrog wave update. step_size is the leapfrog step h;
            # the central second-difference of X''(t) = -Delta_F X scales the
            # force/acceleration term by h^2 (see paper derivation). The original
            # PR omitted this entirely (implicit h=1 + an unused epsilons param).
            x_new = (
                2 * x_curr - x_prev - (self.step_size**2) * x_layer
            )
            x_prev = x_curr
            x_curr = x_new

        x = x_curr.reshape(actual_num_nodes, -1)
        x = self.lin2(x)
        return x
