# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time

import torch
from torch import nn
from torch_geometric.utils import degree
from torch_scatter import scatter_add

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .lib import laplace as lap
from .orthogonal import Orthogonal


class LaplacianBuilder(nn.Module):
    def __init__(
        self,
        size,
        edge_index,
        d,
        normalised=False,
        deg_normalised=False,
        add_hp=False,
        add_lp=False,
        augmented=True,
    ):
        super(LaplacianBuilder, self).__init__()
        assert not (normalised and deg_normalised)

        self.d = d
        self.final_d = d
        if add_hp:
            self.final_d += 1
        if add_lp:
            self.final_d += 1
        self.size = size
        self.edges = edge_index.size(1) // 2
        self.edge_index = edge_index
        self.normalised = normalised
        self.deg_normalised = deg_normalised
        self.device = edge_index.device
        self.add_hp = add_hp
        self.add_lp = add_lp
        self.augmented = augmented

        # Preprocess the sparse indices required to compute the Sheaf Laplacian.
        self.full_left_right_idx, _ = lap.compute_left_right_map_index(
            edge_index, full_matrix=True
        )
        self.left_right_idx, self.vertex_tril_idx = (
            lap.compute_left_right_map_index(edge_index)
        )
        if self.add_lp or self.add_hp:
            self.fixed_diag_indices, self.fixed_tril_indices = (
                lap.compute_fixed_diag_laplacian_indices(
                    size, self.vertex_tril_idx, self.d, self.final_d
                )
            )
        self.deg = degree(self.edge_index[0], num_nodes=self.size)

        # Always compute fixed indices if needed (these depend on hyperparameters)
        if self.add_lp or self.add_hp:
            self.fixed_diag_indices, self.fixed_tril_indices = (
                lap.compute_fixed_diag_laplacian_indices(
                    size, self.vertex_tril_idx, self.d, self.final_d
                )
            )

    def get_fixed_maps(self, size, dtype):
        assert self.add_lp or self.add_hp

        fixed_diag, fixed_non_diag = [], []
        if self.add_lp:
            fixed_diag.append(self.deg.view(-1, 1))
            fixed_non_diag.append(
                torch.ones(size=(size, 1), device=self.device, dtype=dtype)
            )
        if self.add_hp:
            fixed_diag.append(self.deg.view(-1, 1))
            fixed_non_diag.append(
                -torch.ones(size=(size, 1), device=self.device, dtype=dtype)
            )

        fixed_diag = torch.cat(fixed_diag, dim=1)
        fixed_non_diag = torch.cat(fixed_non_diag, dim=1)

        assert self.fixed_tril_indices.size(1) == fixed_non_diag.numel()
        assert self.fixed_diag_indices.size(1) == fixed_diag.numel()

        return fixed_diag, fixed_non_diag

    def scalar_normalise(self, diag, tril, row, col):
        if tril.dim() > 2:
            assert tril.size(-1) == tril.size(-2)
            assert diag.dim() == 2
        d = diag.size(-1)

        if self.augmented:
            diag_sqrt_inv = (diag + 1).pow(-0.5)
        else:
            diag_sqrt_inv = diag.pow(-0.5)
            diag_sqrt_inv.masked_fill_(diag_sqrt_inv == float("inf"), 0)
        diag_sqrt_inv = (
            diag_sqrt_inv.view(-1, 1, 1)
            if tril.dim() > 2
            else diag_sqrt_inv.view(-1, d)
        )
        left_norm = diag_sqrt_inv[row]
        right_norm = diag_sqrt_inv[col]
        non_diag_maps = left_norm * tril * right_norm

        diag_sqrt_inv = (
            diag_sqrt_inv.view(-1, 1, 1)
            if diag.dim() > 2
            else diag_sqrt_inv.view(-1, d)
        )
        diag_maps = diag_sqrt_inv**2 * diag

        return diag_maps, non_diag_maps

    def append_fixed_maps(
        self, size, diag_indices, diag_maps, tril_indices, tril_maps
    ):
        if not self.add_lp and not self.add_hp:
            return (diag_indices, diag_maps), (tril_indices, tril_maps)

        fixed_diag, fixed_non_diag = self.get_fixed_maps(size, tril_maps.dtype)
        tril_row, tril_col = self.vertex_tril_idx

        # Normalise the fixed parts.
        if self.normalised:
            fixed_diag, fixed_non_diag = self.scalar_normalise(
                fixed_diag, fixed_non_diag, tril_row, tril_col
            )
        fixed_diag, fixed_non_diag = (
            fixed_diag.view(-1),
            fixed_non_diag.view(-1),
        )
        # Combine the learnable and fixed parts.
        tril_indices, tril_maps = lap.mergesp(
            self.fixed_tril_indices, fixed_non_diag, tril_indices, tril_maps
        )
        diag_indices, diag_maps = lap.mergesp(
            self.fixed_diag_indices, fixed_diag, diag_indices, diag_maps
        )

        return (diag_indices, diag_maps), (tril_indices, tril_maps)

    def create_with_new_edge_index(self, edge_index):
        assert edge_index.max() <= self.size
        new_builder = self.__class__(
            self.size,
            edge_index,
            self.d,
            normalised=self.normalised,
            deg_normalised=self.deg_normalised,
            add_hp=self.add_hp,
            add_lp=self.add_lp,
            augmented=self.augmented,
        )  # No precomputed indices for new edge index
        new_builder.train(self.training)
        return new_builder


class DiagLaplacianBuilder(LaplacianBuilder):
    """Learns a a Sheaf Laplacian with diagonal restriction maps"""

    def __init__(
        self,
        size,
        edge_index,
        d,
        normalised=False,
        deg_normalised=False,
        add_hp=False,
        add_lp=False,
        augmented=True,
    ):
        super(DiagLaplacianBuilder, self).__init__(
            size,
            edge_index,
            d,
            normalised,
            deg_normalised,
            add_hp,
            add_lp,
            augmented,
        )

        self.diag_indices, self.tril_indices = (
            lap.compute_learnable_diag_laplacian_indices(
                size, self.vertex_tril_idx, self.d, self.final_d
            )
        )

    def normalise(self, diag, tril, row, col):
        if self.normalised:
            d_sqrt_inv = (
                (diag + 1).pow(-0.5) if self.augmented else diag.pow(-0.5)
            )
            left_norm, right_norm = d_sqrt_inv[row], d_sqrt_inv[col]
            tril = left_norm * tril * right_norm
            diag = d_sqrt_inv * diag * d_sqrt_inv
        elif self.deg_normalised:
            deg_sqrt_inv = (
                (self.deg + 1).pow(-0.5)
                if self.augmented
                else self.deg.pow(-0.5)
            )
            deg_sqrt_inv = deg_sqrt_inv.unsqueeze(-1)
            deg_sqrt_inv.masked_fill_(deg_sqrt_inv == float("inf"), 0)
            left_norm, right_norm = deg_sqrt_inv[row], deg_sqrt_inv[col]
            tril = left_norm * tril * right_norm
            diag = deg_sqrt_inv * diag * deg_sqrt_inv
        return diag, tril

    def forward(self, maps):
        assert len(maps.size()) == 2
        assert maps.size(1) == self.d
        left_idx, right_idx = self.left_right_idx
        tril_row, tril_col = self.vertex_tril_idx
        row, _ = self.edge_index

        # Compute the un-normalised Laplacian entries.
        left_maps = torch.index_select(maps, index=left_idx, dim=0)
        right_maps = torch.index_select(maps, index=right_idx, dim=0)
        tril_maps = -left_maps * right_maps
        saved_tril_maps = tril_maps.detach().clone()
        diag_maps = scatter_add(maps**2, row, dim=0, dim_size=self.size)

        # Normalise the entries if the normalised Laplacian is used.
        diag_maps, tril_maps = self.normalise(
            diag_maps, tril_maps, tril_row, tril_col
        )
        tril_indices, diag_indices = self.tril_indices, self.diag_indices
        tril_maps, diag_maps = tril_maps.view(-1), diag_maps.view(-1)

        # Append fixed diagonal values in the non-learnable dimensions.
        (diag_indices, diag_maps), (tril_indices, tril_maps) = (
            self.append_fixed_maps(
                len(left_maps),
                diag_indices,
                diag_maps,
                tril_indices,
                tril_maps,
            )
        )

        # Add the upper triangular part
        triu_indices = torch.empty_like(tril_indices)
        triu_indices[0], triu_indices[1] = tril_indices[1], tril_indices[0]
        non_diag_indices, non_diag_values = lap.mergesp(
            tril_indices, tril_maps, triu_indices, tril_maps
        )

        # Merge diagonal and non-diagonal
        edge_index, weights = lap.mergesp(
            non_diag_indices, non_diag_values, diag_indices, diag_maps
        )

        return (edge_index, weights), saved_tril_maps


class NormConnectionLaplacianBuilder(LaplacianBuilder):
    """Learns a a Sheaf Laplacian with diagonal restriction maps"""

    def __init__(
        self,
        size,
        edge_index,
        d,
        add_hp=False,
        add_lp=False,
        orth_map=None,
        augmented=True,
    ):
        start_time = time.time()
        super(NormConnectionLaplacianBuilder, self).__init__(
            size,
            edge_index,
            d,
            add_hp=add_hp,
            add_lp=add_lp,
            normalised=True,
            augmented=augmented,
        )
        # print("TIME DEBUG: Time taken for NormConnectionLaplacianBuilder init: ", time.time() - start_time)
        start_time = time.time()
        self.orth_transform = Orthogonal(d=self.d, orthogonal_map=orth_map)
        self.orth_map = orth_map
        # print("TIME DEBUG: Time taken for Orthogonal init: ", time.time() - start_time)

        start_time = time.time()
        _, self.tril_indices = lap.compute_learnable_laplacian_indices(
            size, self.vertex_tril_idx, self.d, self.final_d
        )
        # print("TIME DEBUG: Time taken for compute_learnable_laplacian_indices: ", time.time() - start_time)
        start_time = time.time()
        self.diag_indices, _ = lap.compute_learnable_diag_laplacian_indices(
            size, self.vertex_tril_idx, self.d, self.final_d
        )
        # print("TIME DEBUG: Time taken for compute_learnable_diag_laplacian_indices: ", time.time() - start_time)

    def create_with_new_edge_index(self, edge_index):
        assert edge_index.max() <= self.size
        new_builder = self.__class__(
            self.size,
            edge_index,
            self.d,
            add_hp=self.add_hp,
            add_lp=self.add_lp,
            augmented=self.augmented,
            orth_map=self.orth_map,
        )  # No precomputed indices for new edge index
        new_builder.train(self.training)
        return new_builder

    def normalise(self, diag, tril, row, col):
        if tril.dim() > 2:
            assert tril.size(-1) == tril.size(-2)
            assert diag.dim() == 2
        d = diag.size(-1)

        if self.augmented:
            diag_sqrt_inv = (diag + 1).pow(-0.5)
        else:
            diag_sqrt_inv = diag.pow(-0.5)
            diag_sqrt_inv.masked_fill_(diag_sqrt_inv == float("inf"), 0)
        diag_sqrt_inv = (
            diag_sqrt_inv.view(-1, 1, 1)
            if tril.dim() > 2
            else diag_sqrt_inv.view(-1, d)
        )
        left_norm = diag_sqrt_inv[row]
        right_norm = diag_sqrt_inv[col]
        non_diag_maps = left_norm * tril * right_norm

        diag_sqrt_inv = (
            diag_sqrt_inv.view(-1, 1, 1)
            if diag.dim() > 2
            else diag_sqrt_inv.view(-1, d)
        )
        diag_maps = diag_sqrt_inv**2 * diag

        return diag_maps, non_diag_maps

    def forward(self, map_params, edge_weights=None):
        if edge_weights is not None:
            assert edge_weights.size(1) == 1
        assert len(map_params.size()) == 2
        if self.orth_map in ["matrix_exp", "cayley"]:
            assert map_params.size(1) == self.d * (self.d + 1) // 2
        else:
            assert map_params.size(1) == self.d * (self.d - 1) // 2

        _, full_right_idx = self.full_left_right_idx
        left_idx, right_idx = self.left_right_idx
        tril_row, tril_col = self.vertex_tril_idx
        tril_indices, diag_indices = self.tril_indices, self.diag_indices
        row, _ = self.edge_index

        # Convert the parameters to orthogonal matrices.
        maps = self.orth_transform(map_params)
        if edge_weights is None:
            diag_maps = self.deg.unsqueeze(-1)
        else:
            diag_maps = scatter_add(
                edge_weights**2, row, dim=0, dim_size=self.size
            )
            maps = maps * edge_weights.unsqueeze(-1)

        # Compute the transport maps.
        left_maps = torch.index_select(maps, index=left_idx, dim=0)
        right_maps = torch.index_select(maps, index=right_idx, dim=0)
        tril_maps = -torch.bmm(torch.transpose(left_maps, -1, -2), right_maps)
        saved_tril_maps = tril_maps.detach().clone()

        # Normalise the entries if the normalised Laplacian is used.
        diag_maps, tril_maps = self.scalar_normalise(
            diag_maps, tril_maps, tril_row, tril_col
        )
        tril_maps, diag_maps = (
            tril_maps.view(-1),
            diag_maps.expand(-1, self.d).reshape(-1),
        )

        # Append fixed diagonal values in the non-learnable dimensions.
        (diag_indices, diag_maps), (tril_indices, tril_maps) = (
            self.append_fixed_maps(
                len(left_maps),
                diag_indices,
                diag_maps,
                tril_indices,
                tril_maps,
            )
        )

        # Add the upper triangular part
        triu_indices = torch.empty_like(tril_indices)
        triu_indices[0], triu_indices[1] = tril_indices[1], tril_indices[0]
        non_diag_indices, non_diag_values = lap.mergesp(
            tril_indices, tril_maps, triu_indices, tril_maps
        )

        # Merge diagonal and non-diagonal
        edge_index, weights = lap.mergesp(
            non_diag_indices, non_diag_values, diag_indices, diag_maps
        )

        return (edge_index, weights), saved_tril_maps


class GeneralLaplacianBuilder(LaplacianBuilder):
    """Learns a multi-dimensional Sheaf Laplacian from data."""

    def __init__(
        self,
        size,
        edge_index,
        d,
        normalised=False,
        deg_normalised=False,
        add_hp=False,
        add_lp=False,
        augmented=True,
    ):
        super(GeneralLaplacianBuilder, self).__init__(
            size,
            edge_index,
            d,
            normalised=normalised,
            deg_normalised=deg_normalised,
            add_hp=add_hp,
            add_lp=add_lp,
            augmented=augmented,
        )

        # Preprocess the sparse indices required to compute the Sheaf Laplacian.
        self.diag_indices, self.tril_indices = (
            lap.compute_learnable_laplacian_indices(
                size, self.vertex_tril_idx, self.d, self.final_d
            )
        )

    def normalise(self, diag_maps, non_diag_maps, tril_row, tril_col):
        if self.normalised:
            # Normalise the entries if the normalised Laplacian is used.
            if self.training:
                # During training, we perturb the matrices to ensure they have different singular values.
                # Without this, the gradients of batched_sym_matrix_pow, which uses SVD are non-finite.
                eps = (
                    torch.FloatTensor(self.d)
                    .uniform_(-0.001, 0.001)
                    .to(device=self.device)
                )
            else:
                eps = torch.zeros(self.d, device=self.device)

            to_be_inv_diag_maps = (
                diag_maps + torch.diag(1.0 + eps).unsqueeze(0)
                if self.augmented
                else diag_maps
            )
            d_sqrt_inv = lap.batched_sym_matrix_pow(to_be_inv_diag_maps, -0.5)
            assert torch.all(torch.isfinite(d_sqrt_inv))
            left_norm = d_sqrt_inv[tril_row]
            right_norm = d_sqrt_inv[tril_col]
            non_diag_maps = (left_norm @ non_diag_maps @ right_norm).clamp(
                min=-1, max=1
            )
            diag_maps = (d_sqrt_inv @ diag_maps @ d_sqrt_inv).clamp(
                min=-1, max=1
            )
            assert torch.all(torch.isfinite(non_diag_maps))
            assert torch.all(torch.isfinite(diag_maps))
        elif self.deg_normalised:
            # These are general d x d maps so we need to divide by 1 / sqrt(deg * d), their maximum possible norm.
            deg_sqrt_inv = (
                (self.deg * self.d + 1).pow(-1 / 2)
                if self.augmented
                else (self.deg * self.d + 1).pow(-1 / 2)
            )
            deg_sqrt_inv = deg_sqrt_inv.view(-1, 1, 1)
            left_norm = deg_sqrt_inv[tril_row]
            right_norm = deg_sqrt_inv[tril_col]
            non_diag_maps = left_norm * non_diag_maps * right_norm
            diag_maps = deg_sqrt_inv * diag_maps * deg_sqrt_inv
        return diag_maps, non_diag_maps

    def forward(self, maps):
        left_idx, right_idx = self.left_right_idx
        tril_row, tril_col = self.vertex_tril_idx
        tril_indices, diag_indices = self.tril_indices, self.diag_indices
        row, _ = self.edge_index

        # Compute transport maps.
        assert torch.all(torch.isfinite(maps))
        left_maps = torch.index_select(maps, index=left_idx, dim=0)
        right_maps = torch.index_select(maps, index=right_idx, dim=0)
        tril_maps = -torch.bmm(
            torch.transpose(left_maps, dim0=-1, dim1=-2), right_maps
        )
        saved_tril_maps = tril_maps.detach().clone()
        diag_maps = torch.bmm(torch.transpose(maps, dim0=-1, dim1=-2), maps)
        diag_maps = scatter_add(diag_maps, row, dim=0, dim_size=self.size)

        # Normalise the transport maps.
        diag_maps, tril_maps = self.normalise(
            diag_maps, tril_maps, tril_row, tril_col
        )
        diag_maps, tril_maps = diag_maps.view(-1), tril_maps.view(-1)

        # Append fixed diagonal values in the non-learnable dimensions.
        (diag_indices, diag_maps), (tril_indices, tril_maps) = (
            self.append_fixed_maps(
                len(left_maps),
                diag_indices,
                diag_maps,
                tril_indices,
                tril_maps,
            )
        )

        # Add the upper triangular part.
        triu_indices = torch.empty_like(tril_indices)
        triu_indices[0], triu_indices[1] = tril_indices[1], tril_indices[0]
        non_diag_indices, non_diag_values = lap.mergesp(
            tril_indices, tril_maps, triu_indices, tril_maps
        )

        # Merge diagonal and non-diagonal
        edge_index, weights = lap.mergesp(
            non_diag_indices, non_diag_values, diag_indices, diag_maps
        )

        return (edge_index, weights), saved_tril_maps
