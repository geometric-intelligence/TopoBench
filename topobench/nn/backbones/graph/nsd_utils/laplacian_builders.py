# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Laplacian builders for Neural Sheaf Diffusion.

This module provides builders for constructing different types of sheaf Laplacians:
diagonal, bundle (with orthogonal maps), and general (full matrices).
"""

import os
import sys

import torch
from torch import nn
from torch_geometric.utils import degree
from torch_scatter import scatter_add

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .laplace import (
    compute_fixed_diag_laplacian_indices,
    compute_learnable_diag_laplacian_indices,
    compute_learnable_laplacian_indices,
    compute_left_right_map_index,
    mergesp,
)
from .orthogonal import Orthogonal


class LaplacianBuilder(nn.Module):
    """
    Base class for building sheaf Laplacians.

    This class provides common functionality for all Laplacian builders,
    including preprocessing edge indices and computing normalization.

    Parameters
    ----------
    size : int
        Number of nodes in the graph.
    edge_index : torch.Tensor
        Edge indices of shape [2, num_edges].
    d : int
        Dimension of the stalk space.
    normalised : bool, optional
        Use the augmented normalized sheaf Laplacian ((D+1)^-1/2 L (D+1)^-1/2). Default is False.
    deg_normalised : bool, optional
        Use degree normalization instead. Mutually exclusive with
        ``normalised``. Default is False.
    add_hp : bool, optional
        Append a fixed high-pass channel, growing each stalk by one dimension.
        Default is False.
    add_lp : bool, optional
        Append a fixed low-pass channel, growing each stalk by one dimension.
        Default is False.

    Notes
    -----
    Normalization is always augmented with a self-loop, i.e. it uses
    ``(deg + 1)`` on the diagonal, matching the reference implementation's
    default.
    """

    def __init__(
        self,
        size,
        edge_index,
        d,
        normalised=False,
        deg_normalised=False,
        add_hp=False,
        add_lp=False,
    ):
        super().__init__()
        assert not (normalised and deg_normalised), (
            "normalised and deg_normalised are mutually exclusive"
        )

        self.d = d
        # Fixed high-/low-pass channels each add one dimension to every stalk.
        self.final_d = d + int(add_hp) + int(add_lp)
        self.size = size
        self.edges = edge_index.size(1) // 2
        self.edge_index = edge_index
        self.normalised = normalised
        self.deg_normalised = deg_normalised
        self.add_hp = add_hp
        self.add_lp = add_lp
        self.device = edge_index.device

        # Preprocess the sparse indices required to compute the Sheaf Laplacian.
        self.full_left_right_idx, _ = compute_left_right_map_index(
            edge_index, full_matrix=True
        )
        self.left_right_idx, self.vertex_tril_idx = (
            compute_left_right_map_index(edge_index)
        )
        # Positions for the fixed high-/low-pass dimensions (only if enabled).
        if self.add_hp or self.add_lp:
            self.fixed_diag_indices, self.fixed_tril_indices = (
                compute_fixed_diag_laplacian_indices(
                    size, self.vertex_tril_idx, self.d, self.final_d
                )
            )
        self.deg = degree(self.edge_index[0], num_nodes=self.size)

    def scalar_normalise(self, diag, tril, row, col):
        """
        Apply scalar normalization to Laplacian entries.

        Normalizes diagonal and off-diagonal entries by node degrees,
        similar to symmetric normalization in standard graph Laplacians.

        Parameters
        ----------
        diag : torch.Tensor
            Diagonal block values.
        tril : torch.Tensor
            Lower triangular block values.
        row : torch.Tensor
            Row indices of edges.
        col : torch.Tensor
            Column indices of edges.

        Returns
        -------
        diag_maps : torch.Tensor
            Normalized diagonal block values.
        non_diag_maps : torch.Tensor
            Normalized off-diagonal block values.
        """
        if tril.dim() > 2:
            assert tril.size(-1) == tril.size(-2)
            assert diag.dim() == 2
        d = diag.size(-1)
        # Augmented (self-loop) symmetric normalization: (deg + 1)^-1/2.
        diag_sqrt_inv = (diag + 1).pow(-0.5)

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

    def get_fixed_maps(self, num_edges, dtype):
        """
        Build the fixed (non-learnable) high-/low-pass Laplacian values.

        Both channels put the node degree on the diagonal. Off-diagonal:
        the low-pass channel uses ``+1`` (the signless graph Laplacian, D + A),
        the high-pass channel uses ``-1`` (the standard graph Laplacian, D - A).

        Parameters
        ----------
        num_edges : int
            Number of lower-triangular edges to fill.
        dtype : torch.dtype
            Dtype of the returned tensors.

        Returns
        -------
        fixed_diag : torch.Tensor
            Fixed diagonal values, one column per enabled channel.
        fixed_non_diag : torch.Tensor
            Fixed off-diagonal values, one column per enabled channel.
        """
        assert self.add_lp or self.add_hp

        # Column order MUST match the ascending fixed-dimension indices from
        # compute_fixed_diag_laplacian_indices: low-pass occupies the first
        # fixed dim, high-pass the second. Do not reorder these branches.
        fixed_diag, fixed_non_diag = [], []
        if self.add_lp:
            fixed_diag.append(self.deg.view(-1, 1))
            fixed_non_diag.append(
                torch.ones(
                    size=(num_edges, 1), device=self.device, dtype=dtype
                )
            )
        if self.add_hp:
            fixed_diag.append(self.deg.view(-1, 1))
            fixed_non_diag.append(
                -torch.ones(
                    size=(num_edges, 1), device=self.device, dtype=dtype
                )
            )

        fixed_diag = torch.cat(fixed_diag, dim=1)
        fixed_non_diag = torch.cat(fixed_non_diag, dim=1)
        # Guard against index/value misalignment before the sparse merge.
        assert self.fixed_tril_indices.size(1) == fixed_non_diag.numel()
        assert self.fixed_diag_indices.size(1) == fixed_diag.numel()
        return fixed_diag, fixed_non_diag

    def append_fixed_maps(
        self, num_edges, diag_indices, diag_maps, tril_indices, tril_maps
    ):
        """
        Merge the fixed high-/low-pass entries into the learnable Laplacian.

        No-op when neither ``add_hp`` nor ``add_lp`` is enabled.

        Parameters
        ----------
        num_edges : int
            Number of lower-triangular edges.
        diag_indices, diag_maps : torch.Tensor
            Learnable diagonal indices/values.
        tril_indices, tril_maps : torch.Tensor
            Learnable off-diagonal indices/values.

        Returns
        -------
        tuple
            ``((diag_indices, diag_maps), (tril_indices, tril_maps))`` with the
            fixed entries merged in.
        """
        if not self.add_lp and not self.add_hp:
            return (diag_indices, diag_maps), (tril_indices, tril_maps)

        fixed_diag, fixed_non_diag = self.get_fixed_maps(
            num_edges, tril_maps.dtype
        )
        tril_row, tril_col = self.vertex_tril_idx

        # The fixed channels only support symmetric normalisation (not
        # deg-normalisation), so normalise them only when ``normalised`` is set.
        if self.normalised:
            fixed_diag, fixed_non_diag = self.scalar_normalise(
                fixed_diag, fixed_non_diag, tril_row, tril_col
            )
        fixed_diag, fixed_non_diag = (
            fixed_diag.view(-1),
            fixed_non_diag.view(-1),
        )

        tril_indices, tril_maps = mergesp(
            self.fixed_tril_indices, fixed_non_diag, tril_indices, tril_maps
        )
        diag_indices, diag_maps = mergesp(
            self.fixed_diag_indices, fixed_diag, diag_indices, diag_maps
        )
        return (diag_indices, diag_maps), (tril_indices, tril_maps)


class DiagLaplacianBuilder(LaplacianBuilder):
    """
    Builder for sheaf Laplacian with diagonal restriction maps.

    This builder constructs a sheaf Laplacian where the restriction maps
    are diagonal matrices, parameterized by d values per edge.

    Parameters
    ----------
    size : int
        Number of nodes in the graph.
    edge_index : torch.Tensor
        Edge indices of shape [2, num_edges].
    d : int
        Dimension of the diagonal stalk space.
    normalised, deg_normalised, add_hp, add_lp : optional
        See :class:`LaplacianBuilder`.
    """

    def __init__(
        self,
        size,
        edge_index,
        d,
        normalised=False,
        deg_normalised=False,
        add_hp=False,
        add_lp=False,
    ):
        super().__init__(
            size,
            edge_index,
            d,
            normalised=normalised,
            deg_normalised=deg_normalised,
            add_hp=add_hp,
            add_lp=add_lp,
        )

        # Learnable block indices span the full stalk (final_d) so the fixed
        # high-/low-pass dimensions line up in the same sparse matrix.
        self.diag_indices, self.tril_indices = (
            compute_learnable_diag_laplacian_indices(
                size, self.vertex_tril_idx, self.d, self.final_d
            )
        )

    def normalise(self, diag, tril, row, col):
        """
        Return the normalised learnable diagonal Laplacian entries.

        Applies augmented symmetric normalization ``(D+1)^-1/2 L (D+1)^-1/2``
        when ``normalised`` (delegating to :meth:`scalar_normalise`), or
        degree normalization when ``deg_normalised``; otherwise returns the
        entries unchanged.

        Parameters
        ----------
        diag : torch.Tensor
            Diagonal block values of shape [size, d].
        tril : torch.Tensor
            Lower-triangular block values of shape [num_edges, d].
        row, col : torch.Tensor
            Endpoint node indices of the lower-triangular edges.

        Returns
        -------
        diag : torch.Tensor
            The (possibly) normalized diagonal block values.
        tril : torch.Tensor
            The (possibly) normalized lower-triangular block values.
        """
        if self.normalised:
            # Share the symmetric-normalization math with the fixed maps.
            diag, tril = self.scalar_normalise(diag, tril, row, col)
        elif self.deg_normalised:
            deg_sqrt_inv = (self.deg + 1).pow(-0.5).unsqueeze(-1)
            deg_sqrt_inv.masked_fill_(deg_sqrt_inv == float("inf"), 0)
            tril = deg_sqrt_inv[row] * tril * deg_sqrt_inv[col]
            diag = deg_sqrt_inv * diag * deg_sqrt_inv
        return diag, tril

    def forward(self, maps):
        """
        Build the sheaf Laplacian from diagonal restriction maps.

        Parameters
        ----------
        maps : torch.Tensor
            Diagonal restriction map parameters of shape [num_edges, d].

        Returns
        -------
        L : tuple of torch.Tensor
            Sparse Laplacian representation as (indices, values).
        saved_tril_maps : torch.Tensor
            Saved lower triangular restriction maps for analysis.
        """
        assert len(maps.size()) == 2
        assert maps.size(1) == self.d
        left_idx, right_idx = self.left_right_idx
        tril_row, tril_col = self.vertex_tril_idx
        row, _ = self.edge_index

        # Compute the (un-normalised) learnable Laplacian entries.
        left_maps = torch.index_select(maps, index=left_idx, dim=0)
        right_maps = torch.index_select(maps, index=right_idx, dim=0)
        tril_maps = -left_maps * right_maps
        saved_tril_maps = tril_maps.detach().clone()
        diag_maps = scatter_add(maps**2, row, dim=0, dim_size=self.size)

        # Optionally normalise the learnable part.
        diag_maps, tril_maps = self.normalise(
            diag_maps, tril_maps, tril_row, tril_col
        )
        tril_indices, diag_indices = self.tril_indices, self.diag_indices
        tril_maps, diag_maps = tril_maps.view(-1), diag_maps.view(-1)

        # Append the fixed high-/low-pass entries (no-op unless enabled).
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
        non_diag_indices, non_diag_values = mergesp(
            tril_indices, tril_maps, triu_indices, tril_maps
        )

        # Merge diagonal and non-diagonal
        edge_index, weights = mergesp(
            non_diag_indices, non_diag_values, diag_indices, diag_maps
        )

        return (edge_index, weights), saved_tril_maps


class NormConnectionLaplacianBuilder(LaplacianBuilder):
    """
    Builder for normalized bundle sheaf Laplacian with orthogonal restriction maps.

    This builder constructs a normalized sheaf Laplacian where the restriction maps
    are orthogonal matrices parameterized via Cayley transform or matrix exponential.
    Used for bundle sheaf models.

    Parameters
    ----------
    size : int
        Number of nodes in the graph.
    edge_index : torch.Tensor
        Edge indices of shape [2, num_edges].
    d : int
        Dimension of the stalk space.
    orth_map : str or None, optional
        Method for orthogonalization ('cayley' or 'matrix_exp'). Default is None.
    """

    def __init__(self, size, edge_index, d, orth_map=None):
        super().__init__(
            size,
            edge_index,
            d,
            normalised=True,
        )
        self.orth_transform = Orthogonal(d=self.d, orthogonal_map=orth_map)
        self.orth_map = orth_map

        _, self.tril_indices = compute_learnable_laplacian_indices(
            size, self.vertex_tril_idx, self.d, self.d
        )
        self.diag_indices, _ = compute_learnable_diag_laplacian_indices(
            size, self.vertex_tril_idx, self.d, self.d
        )

    def forward(self, map_params):
        """
        Build the normalized sheaf Laplacian from orthogonal restriction maps.

        Parameters
        ----------
        map_params : torch.Tensor
            Orthogonal map parameters of shape [num_edges, d*(d+1)/2].

        Returns
        -------
        L : tuple of torch.Tensor
            Sparse normalized Laplacian representation as (indices, values).
        saved_tril_maps : torch.Tensor
            Saved lower triangular transport maps for analysis.
        """
        assert len(map_params.size()) == 2
        assert map_params.size(1) == self.d * (self.d + 1) // 2

        _, full_right_idx = self.full_left_right_idx
        left_idx, right_idx = self.left_right_idx
        tril_row, tril_col = self.vertex_tril_idx
        tril_indices, diag_indices = self.tril_indices, self.diag_indices
        row, _ = self.edge_index

        # Convert the parameters to orthogonal matrices.
        maps = self.orth_transform(map_params)
        diag_maps = self.deg.unsqueeze(-1)

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

        # Add the upper triangular part
        triu_indices = torch.empty_like(tril_indices)
        triu_indices[0], triu_indices[1] = tril_indices[1], tril_indices[0]
        non_diag_indices, non_diag_values = mergesp(
            tril_indices, tril_maps, triu_indices, tril_maps
        )

        # Merge diagonal and non-diagonal
        edge_index, weights = mergesp(
            non_diag_indices, non_diag_values, diag_indices, diag_maps
        )

        return (edge_index, weights), saved_tril_maps


class GeneralLaplacianBuilder(LaplacianBuilder):
    """
    Builder for general sheaf Laplacian with full matrix restriction maps.

    This builder constructs a sheaf Laplacian where the restriction maps
    are arbitrary d x d matrices learned from data.

    Parameters
    ----------
    size : int
        Number of nodes in the graph.
    edge_index : torch.Tensor
        Edge indices of shape [2, num_edges].
    d : int
        Dimension of the stalk space.
    augmented : bool, optional
        Whether to use augmented representation (not currently used). Default is True.
    """

    def __init__(
        self,
        size,
        edge_index,
        d,
        augmented=True,
    ):
        super().__init__(
            size,
            edge_index,
            d,
        )

        # Preprocess the sparse indices required to compute the Sheaf Laplacian.
        self.diag_indices, self.tril_indices = (
            compute_learnable_laplacian_indices(
                size, self.vertex_tril_idx, self.d, self.d
            )
        )

    def forward(self, maps):
        """
        Build the sheaf Laplacian from general restriction maps.

        Parameters
        ----------
        maps : torch.Tensor
            General restriction map matrices of shape [num_edges, d, d].

        Returns
        -------
        L : tuple of torch.Tensor
            Sparse Laplacian representation as (indices, values).
        saved_tril_maps : torch.Tensor
            Saved lower triangular transport maps for analysis.
        """
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
        diag_maps, tril_maps = diag_maps.view(-1), tril_maps.view(-1)

        # Add the upper triangular part.
        triu_indices = torch.empty_like(tril_indices)
        triu_indices[0], triu_indices[1] = tril_indices[1], tril_indices[0]
        non_diag_indices, non_diag_values = mergesp(
            tril_indices, tril_maps, triu_indices, tril_maps
        )

        # Merge diagonal and non-diagonal
        edge_index, weights = mergesp(
            non_diag_indices, non_diag_values, diag_indices, diag_maps
        )

        return (edge_index, weights), saved_tril_maps
