import os
import sys
from abc import abstractmethod

import numpy as np

# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F
from torch import nn

# PyTorch Geometric imports

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .lib import laplace as lap

# from experiments.neural_sheaf_diffusion.timing_utils import global_tracker


class SheafLearner(nn.Module):
    """Base model that learns a sheaf from the features and the graph structure."""

    def __init__(self):
        super(SheafLearner, self).__init__()
        self.L = None

    @abstractmethod
    def forward(self, x, edge_index):
        raise NotImplementedError()

    def set_L(self, weights):
        # global_tracker.start_timer("SheafLearner.set_L")
        self.L = weights.clone().detach()
        # global_tracker.end_timer("SheafLearner.set_L")


class LocalConcatSheafLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(
        self, in_channels: int, out_shape: tuple[int, ...], sheaf_act="tanh"
    ):
        super(LocalConcatSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.linear1 = torch.nn.Linear(
            in_channels * 2, int(np.prod(out_shape)), bias=False
        )  # 2 because we concatenate the local node features of two neighboring nodes

        if sheaf_act == "id":
            self.act = lambda x: x
        elif sheaf_act == "tanh":
            self.act = torch.tanh
        elif sheaf_act == "elu":
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(self, x, edge_index):
        # global_tracker.start_timer("LocalConcatSheafLearner.forward")

        # global_tracker.start_timer("LocalConcatSheafLearner.forward.index_select")
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        # global_tracker.end_timer("LocalConcatSheafLearner.forward.index_select")

        # global_tracker.start_timer("LocalConcatSheafLearner.forward.concatenate")
        x_concat = torch.cat([x_row, x_col], dim=1)
        # global_tracker.end_timer("LocalConcatSheafLearner.forward.concatenate")

        # global_tracker.start_timer("LocalConcatSheafLearner.forward.linear")
        maps = self.linear1(x_concat)
        # global_tracker.end_timer("LocalConcatSheafLearner.forward.linear")

        # global_tracker.start_timer("LocalConcatSheafLearner.forward.activation")
        maps = self.act(maps)
        # global_tracker.end_timer("LocalConcatSheafLearner.forward.activation")

        # sign = maps.sign()
        # maps = maps.abs().clamp(0.05, 1.0) * sign

        # global_tracker.start_timer("LocalConcatSheafLearner.forward.reshape")
        if len(self.out_shape) == 2:
            result = maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            result = maps.view(-1, self.out_shape[0])
        # global_tracker.end_timer("LocalConcatSheafLearner.forward.reshape")

        # global_tracker.end_timer("LocalConcatSheafLearner.forward")
        return result


class LocalConcatSheafLearnerVariant(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(
        self,
        d: int,
        hidden_channels: int,
        out_shape: tuple[int, ...],
        sheaf_act="tanh",
    ):
        super(LocalConcatSheafLearnerVariant, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.d = d
        self.hidden_channels = hidden_channels
        self.linear1 = torch.nn.Linear(
            hidden_channels * 2, int(np.prod(out_shape)), bias=False
        )
        # self.linear2 = torch.nn.Linear(self.d, 1, bias=False)

        # std1 = 1.414 * math.sqrt(2. / (hidden_channels * 2 + 1))
        # std2 = 1.414 * math.sqrt(2. / (d + 1))
        #
        # nn.init.normal_(self.linear1.weight, 0.0, std1)
        # nn.init.normal_(self.linear2.weight, 0.0, std2)

        if sheaf_act == "id":
            self.act = lambda x: x
        elif sheaf_act == "tanh":
            self.act = torch.tanh
        elif sheaf_act == "elu":
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(self, x, edge_index):
        # global_tracker.start_timer("LocalConcatSheafLearnerVariant.forward")

        # global_tracker.start_timer("LocalConcatSheafLearnerVariant.forward.index_select")
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        # global_tracker.end_timer("LocalConcatSheafLearnerVariant.forward.index_select")

        # global_tracker.start_timer("LocalConcatSheafLearnerVariant.forward.concatenate")
        x_cat = torch.cat([x_row, x_col], dim=-1)
        # global_tracker.end_timer("LocalConcatSheafLearnerVariant.forward.concatenate")

        # global_tracker.start_timer("LocalConcatSheafLearnerVariant.forward.reshape_sum")
        x_cat = x_cat.reshape(-1, self.d, self.hidden_channels * 2).sum(dim=1)
        # global_tracker.end_timer("LocalConcatSheafLearnerVariant.forward.reshape_sum")

        # global_tracker.start_timer("LocalConcatSheafLearnerVariant.forward.linear")
        x_cat = self.linear1(x_cat)
        # global_tracker.end_timer("LocalConcatSheafLearnerVariant.forward.linear")

        # x_cat = x_cat.t().reshape(-1, self.d)
        # x_cat = self.linear2(x_cat)
        # x_cat = x_cat.reshape(-1, edge_index.size(1)).t()

        # global_tracker.start_timer("LocalConcatSheafLearnerVariant.forward.activation")
        maps = self.act(x_cat)
        # global_tracker.end_timer("LocalConcatSheafLearnerVariant.forward.activation")

        # global_tracker.start_timer("LocalConcatSheafLearnerVariant.forward.reshape")
        if len(self.out_shape) == 2:
            result = maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            result = maps.view(-1, self.out_shape[0])
        # global_tracker.end_timer("LocalConcatSheafLearnerVariant.forward.reshape")

        # global_tracker.end_timer("LocalConcatSheafLearnerVariant.forward")
        return result


class AttentionSheafLearner(SheafLearner):
    def __init__(self, in_channels, d):
        super(AttentionSheafLearner, self).__init__()
        self.d = d
        self.linear1 = torch.nn.Linear(in_channels * 2, d**2, bias=False)

    def forward(self, x, edge_index):
        # global_tracker.start_timer("AttentionSheafLearner.forward")

        # global_tracker.start_timer("AttentionSheafLearner.forward.index_select")
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        # global_tracker.end_timer("AttentionSheafLearner.forward.index_select")

        # global_tracker.start_timer("AttentionSheafLearner.forward.concatenate_linear")
        maps = self.linear1(torch.cat([x_row, x_col], dim=1)).view(
            -1, self.d, self.d
        )
        # global_tracker.end_timer("AttentionSheafLearner.forward.concatenate_linear")

        # global_tracker.start_timer("AttentionSheafLearner.forward.softmax")
        id = torch.eye(
            self.d, device=edge_index.device, dtype=maps.dtype
        ).unsqueeze(0)
        result = id - torch.softmax(maps, dim=-1)
        # global_tracker.end_timer("AttentionSheafLearner.forward.softmax")

        # global_tracker.end_timer("AttentionSheafLearner.forward")
        return result


class EdgeWeightLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, in_channels: int, edge_index):
        super(EdgeWeightLearner, self).__init__()
        self.in_channels = in_channels
        self.linear1 = torch.nn.Linear(in_channels * 2, 1, bias=False)
        # global_tracker.start_timer("EdgeWeightLearner.__init__.compute_indices")
        self.full_left_right_idx, _ = lap.compute_left_right_map_index(
            edge_index, full_matrix=True
        )
        # global_tracker.end_timer("EdgeWeightLearner.__init__.compute_indices")

    def forward(self, x, edge_index):
        # global_tracker.start_timer("EdgeWeightLearner.forward")

        # global_tracker.start_timer("EdgeWeightLearner.forward.index_select")
        _, full_right_idx = self.full_left_right_idx
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        # global_tracker.end_timer("EdgeWeightLearner.forward.index_select")

        # global_tracker.start_timer("EdgeWeightLearner.forward.linear_sigmoid")
        weights = self.linear1(torch.cat([x_row, x_col], dim=1))
        weights = torch.sigmoid(weights)
        # global_tracker.end_timer("EdgeWeightLearner.forward.linear_sigmoid")

        # global_tracker.start_timer("EdgeWeightLearner.forward.compute_edge_weights")
        edge_weights = weights * torch.index_select(
            weights, index=full_right_idx, dim=0
        )
        # global_tracker.end_timer("EdgeWeightLearner.forward.compute_edge_weights")

        # global_tracker.end_timer("EdgeWeightLearner.forward")
        return edge_weights

    def update_edge_index(self, edge_index):
        # global_tracker.start_timer("EdgeWeightLearner.update_edge_index")
        self.full_left_right_idx, _ = lap.compute_left_right_map_index(
            edge_index, full_matrix=True
        )
        # global_tracker.end_timer("EdgeWeightLearner.update_edge_index")


class QuadraticFormSheafLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, in_channels: int, out_shape: tuple[int]):
        super(QuadraticFormSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape

        tensor = (
            torch.eye(in_channels)
            .unsqueeze(0)
            .tile(int(np.prod(out_shape)), 1, 1)
        )
        self.tensor = nn.Parameter(tensor)

    def forward(self, x, edge_index):
        # global_tracker.start_timer("QuadraticFormSheafLearner.forward")

        # global_tracker.start_timer("QuadraticFormSheafLearner.forward.index_select")
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        # global_tracker.end_timer("QuadraticFormSheafLearner.forward.index_select")

        # global_tracker.start_timer("QuadraticFormSheafLearner.forward.map_builder")
        maps = self.map_builder(torch.cat([x_row, x_col], dim=1))
        # global_tracker.end_timer("QuadraticFormSheafLearner.forward.map_builder")

        # global_tracker.start_timer("QuadraticFormSheafLearner.forward.tanh_reshape")
        maps = torch.tanh(maps)
        if len(self.out_shape) == 2:
            result = maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            result = maps.view(-1, self.out_shape[0])
        # global_tracker.end_timer("QuadraticFormSheafLearner.forward.tanh_reshape")

        # global_tracker.end_timer("QuadraticFormSheafLearner.forward")
        return result


class LocalConcatSheafLearnerPyg(SheafLearner):
    """PyG version of LocalConcatSheafLearner using PyG utilities for better efficiency."""

    def __init__(
        self, in_channels: int, out_shape: tuple[int, ...], sheaf_act="tanh"
    ):
        super(LocalConcatSheafLearnerPyg, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape

        # Initialize the linear layer
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
