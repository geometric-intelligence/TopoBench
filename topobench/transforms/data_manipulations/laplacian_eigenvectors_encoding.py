"""Laplacian Positional Encoding (LapPE) Transform."""

import numpy as np
import torch
from scipy.sparse.linalg import eigsh
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix


class LapPE(BaseTransform):
    r"""
    Laplacian Positional Encoding (LapPE) transform.

    This computes the smallest eigenvectors of the normalized Laplacian
    matrix and appends them as node features (structural encodings).
    Optionally pads to a fixed dimension.

    Parameters
    ----------
    max_pe_dim : int
        Maximum number of eigenvectors to use (dimensionality of the encoding).
    include_eigenvalues : bool, optional
        If True, concatenates eigenvalues alongside eigenvectors.
        Shape then becomes ``[num_nodes, 2 * max_pe_dim]``.
        Default is False.
    include_first : bool, optional
        If False, removes eigenvectors corresponding to (near-)zero eigenvalues
        (constant eigenvector in connected graphs). Default is False.
    concat_to_x : bool, optional
        If True, concatenates the encodings with existing node features in
        ``data.x``. If ``data.x`` is None, creates it. Default is True.
    **kwargs : dict
        Additional arguments (not used).
    """

    def __init__(
        self,
        max_pe_dim: int,
        include_eigenvalues: bool = False,
        include_first: bool = False,
        concat_to_x: bool = True,
        **kwargs,
    ):
        self.max_pe_dim = max_pe_dim
        self.include_eigenvalues = include_eigenvalues
        self.include_first = include_first
        self.concat_to_x = concat_to_x

    def forward(self, data: Data) -> Data:
        pe = self._compute_lap_pe(data.edge_index, data.num_nodes)

        if self.concat_to_x:
            if data.x is None:
                data.x = pe
            else:
                data.x = torch.cat([data.x, pe], dim=-1)
        else:
            data.LapPE = pe

        return data

    def _compute_lap_pe(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Internal method to compute Laplacian eigenvector encodings."""
        device = edge_index.device

        if edge_index.size(1) == 0 or num_nodes <= 1:
            return torch.zeros(num_nodes, self.max_pe_dim, device=device)

        # Normalized Laplacian
        edge_index_lap, edge_weight = get_laplacian(
            edge_index, normalization="sym", num_nodes=num_nodes
        )
        L = to_scipy_sparse_matrix(
            edge_index_lap, edge_weight, num_nodes
        ).astype(np.float64)

        k = min(self.max_pe_dim, max(1, num_nodes - 1))

        try:
            evals, evecs = eigsh(L, k=k, which="SM", tol=1e-6)
            idx = np.argsort(evals)
            evals, evecs = evals[idx], evecs[:, idx]
        except Exception:
            evals, evecs = np.linalg.eigh(L.toarray())
            idx = np.argsort(evals)
            evals, evecs = evals[idx], evecs[:, idx]

        # Drop trivial eigenvectors if requested
        eps = 1e-6
        if not self.include_first:
            mask = evals > eps
            evals, evecs = evals[mask], evecs[:, mask]

        # Take up to k
        evals, evecs = evals[: self.max_pe_dim], evecs[:, : self.max_pe_dim]

        # Fix sign ambiguity
        for i in range(evecs.shape[1]):
            max_idx = np.argmax(np.abs(evecs[:, i]))
            if evecs[max_idx, i] < 0:
                evecs[:, i] *= -1

        # Pad if fewer than max_pe_dim
        if evecs.shape[1] < self.max_pe_dim:
            pad_width = self.max_pe_dim - evecs.shape[1]
            evecs = np.pad(evecs, ((0, 0), (0, pad_width)), mode="constant")
            evals = np.pad(evals, (0, pad_width), mode="constant")

        pe = torch.from_numpy(evecs).to(dtype=torch.float32, device=device)

        if self.include_eigenvalues:
            eigvals_broadcast = torch.from_numpy(evals).to(
                dtype=torch.float32, device=device
            )
            eigvals_broadcast = eigvals_broadcast.unsqueeze(0).repeat(
                num_nodes, 1
            )
            pe = torch.cat([pe, eigvals_broadcast], dim=-1)

        return pe
