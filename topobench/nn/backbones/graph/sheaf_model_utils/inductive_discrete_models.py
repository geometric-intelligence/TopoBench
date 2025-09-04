# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import torch_sparse
import sys
import os
import time
import copy
from torch import nn
from .sheaf_base import SheafDiffusion
from .laplacian_builders import DiagLaplacianBuilder, NormConnectionLaplacianBuilder, GeneralLaplacianBuilder
from .sheaf_models import LocalConcatSheafLearner, EdgeWeightLearner, LocalConcatSheafLearnerVariant, LocalConcatSheafLearnerPyg


class InductiveDiscreteDiagSheafDiffusion(SheafDiffusion):

    def __init__(self, config):
        super(InductiveDiscreteDiagSheafDiffusion, self).__init__(None, config)
        assert config['d'] > 0

        self.config = config
        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d,), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d,), sheaf_act=self.sheaf_act))

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, edge_index):
        # Get actual number of nodes dynamically
        actual_num_nodes = x.size(0)
        
        # Create laplacian builder for this specific graph
        laplacian_builder = DiagLaplacianBuilder(actual_num_nodes, edge_index, d=self.d,
                                                normalised=self.normalised,
                                                deg_normalised=self.deg_normalised,
                                                add_hp=self.add_hp, add_lp=self.add_lp,
                                                )
        
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        
        # Use actual number of nodes
        x = x.view(actual_num_nodes * self.final_d, -1)

        x0 = x
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                # Reshape using actual number of nodes
                maps = self.sheaf_learners[layer](x_maps.reshape(actual_num_nodes, -1), edge_index)
                L, trans_maps = laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, actual_num_nodes * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            # Use actual number of nodes for epsilon tiling
            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(actual_num_nodes, 1))
            x0 = coeff * x0 - x
            x = x0

        # Reshape using actual number of nodes
        x = x.reshape(actual_num_nodes, -1)
        x = self.lin2(x)
        return x


class InductiveDiscreteBundleSheafDiffusion(SheafDiffusion):

    def __init__(self, config):
        super(InductiveDiscreteBundleSheafDiffusion, self).__init__(None, config)
        assert config['d'] > 1
        assert not self.deg_normalised
        assert self.hidden_dim % self.final_d == 0 # So that we can reshape the output of the lin1 to a tensor of size (final_d, hidden_dim // final_d)

        self.config = config
        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearnerPyg(
                    self.hidden_dim, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
            
            if self.use_edge_weights:
                # Initialize with dummy edge_index, will be updated in forward pass
                dummy_edge_index = torch.zeros((2, 1), dtype=torch.long)
                self.weight_learners.append(EdgeWeightLearner(self.hidden_dim, dummy_edge_index))

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left, right, actual_num_nodes):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, actual_num_nodes * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def forward(self, x, edge_index):
        # Get actual number of nodes dynamically
        actual_num_nodes = x.size(0)
        
        # Create laplacian builder for this specific graph
        laplacian_builder = NormConnectionLaplacianBuilder(
            actual_num_nodes, edge_index, d=self.d, add_hp=self.add_hp,
            add_lp=self.add_lp, orth_map=self.orth_trans)
        
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        
        # Use actual number of nodes
        x = x.view(actual_num_nodes * self.final_d, -1) # So for each node, we put reshape the output of the lin1 to a tensor of size (final_d, hidden_dim // final_d)
        # This means that if we set "hidden_dim" to 64 and "final_d" to 2, then we have that for each node, we have a tensor of size (2, 32)

        x0, L = x, None
        for layer in range(self.layers):
            # Time each component of the forward pass
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                x_maps = x_maps.reshape(actual_num_nodes, -1) # Reshape using actual number of nodes (so back to the original shape)
                maps = self.sheaf_learners[layer](x_maps, edge_index)
                
                if self.use_edge_weights:
                    # Update edge_index for weight learner
                    self.weight_learners[layer].update_edge_index(edge_index)
                    edge_weights = self.weight_learners[layer](x_maps, edge_index)
                else:
                    edge_weights = None
                
                L, trans_maps = laplacian_builder(maps, edge_weights)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            # Pass actual_num_nodes to left_right_linear
            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer], actual_num_nodes)

            # Use the adjacency matrix rather than the diagonal
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            # Use actual number of nodes for epsilon tiling
            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(actual_num_nodes, 1)) * x0 - x
            x = x0

        # Reshape using actual number of nodes
        x = x.reshape(actual_num_nodes, -1)
        x = self.lin2(x)
        return x


class InductiveDiscreteGeneralSheafDiffusion(SheafDiffusion):

    def __init__(self, config):
        super(InductiveDiscreteGeneralSheafDiffusion, self).__init__(None, config)
        assert config['d'] > 1

        self.config = config
        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def left_right_linear(self, x, left, right, actual_num_nodes):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, actual_num_nodes * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def forward(self, x, edge_index):
        # Get actual number of nodes dynamically
        actual_num_nodes = x.size(0)
        
        # Create laplacian builder for this specific graph
        laplacian_builder = GeneralLaplacianBuilder(
            actual_num_nodes, edge_index, d=self.d, add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)
        
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.second_linear:
            x = self.lin12(x)
        
        # Use actual number of nodes
        x = x.view(actual_num_nodes * self.final_d, -1)

        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                # Reshape using actual number of nodes
                maps = self.sheaf_learners[layer](x_maps.reshape(actual_num_nodes, -1), edge_index)
                L, trans_maps = laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            # Pass actual_num_nodes to left_right_linear
            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer], actual_num_nodes)

            # Use the adjacency matrix rather than the diagonal
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            # Use actual number of nodes for epsilon tiling
            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(actual_num_nodes, 1)) * x0 - x
            x = x0

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))

        # Reshape using actual number of nodes
        x = x.reshape(actual_num_nodes, -1)
        x = self.lin2(x)
        return x