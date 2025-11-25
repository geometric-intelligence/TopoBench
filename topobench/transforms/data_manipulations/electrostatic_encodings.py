"""Laplacian Positional Encoding (LapPE) Transform."""

from copy import deepcopy

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    get_laplacian,
    remove_self_loops,
    to_scipy_sparse_matrix,
)


class ElectrostaticPE(BaseTransform):
    r"""
    Electrostatic Positional Encoding (ElectrostaticPE) transform.

    Kernel based on the electrostatic interaction between nodes.

    Parameters
    ----------
    max_pe_dim : int
        Maximum number of eigenvectors to use (dimensionality of the encoding).
    include_eigenvalues : bool, optional
        If True, concatenates eigenvalues alongside eigenvectors.
        Shape then becomes ``[num_nodes, 2 * max_pe_dim]``. Default is False.
    include_first : bool, optional
        If False, removes eigenvectors corresponding to (near-)zero eigenvalues
        (constant eigenvector in connected graphs). Default is False.
    concat_to_x : bool, optional
        If True, concatenates the encodings with existing node features in
        ``data.x``. If ``data.x`` is None, creates it. Default is True.
    eps : float, optional
        Small value to avoid division by zero. Default is 1e-6.
    **kwargs : dict
        Additional arguments (not used).
    """

    def __init__(
        self,
        max_pe_dim: int,
        include_eigenvalues: bool = False,
        include_first: bool = False,
        concat_to_x: bool = True,
        eps: float = 1e-6,
        **kwargs,
    ):
        self.max_pe_dim = max_pe_dim
        self.include_eigenvalues = include_eigenvalues
        self.include_first = include_first
        self.concat_to_x = concat_to_x
        self.eps = eps

    def forward(self, data: Data) -> Data:
        """Compute the Laplacian positional encodings for the input graph.

        Parameters
        ----------
        data : Data
            Input graph data object.

        Returns
        -------
        Data
            Graph data object with Laplacian positional encodings added.
        """
        pe = self._compute_electrostatic_pe(data.edge_index, data.num_nodes)

        if self.concat_to_x:
            if data.x is None:
                data.x = pe
            else:
                data.x = torch.cat([data.x, pe], dim=-1)
        else:
            data.ElectrostaticPE = pe

        return data

    def _compute_electrostatic_pe(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Internal method to compute electrostatic positional encodings.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of the graph.
        num_nodes : int
            Number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            Electrostatic positional encodings.
        """
        device = edge_index.device

        if edge_index.size(1) == 0 or num_nodes <= 1:
            return torch.zeros(num_nodes, self.max_pe_dim, device=device)

        # Normalized Laplacian
        edge_index_lap, edge_weight = get_laplacian(
            edge_index, normalization="sym", num_nodes=num_nodes
        )
        L = (
            to_scipy_sparse_matrix(edge_index_lap, edge_weight, num_nodes)
            .astype(np.float64)
            .todense()
        )

        L = torch.as_tensor(L)
        Dinv = torch.eye(L.shape[0]) * ((L.diag() + 1e-6) ** -1)
        A = deepcopy(L).abs()
        A.fill_diagonal_(0)
        DinvA = Dinv.matmul(A)

        # evals, evecs = torch.linalg.eigh(L)
        # try:
        #     #evals, evecs = torch.linalg.eigh(L)
        # except:
        # IMDB-BINARY has some issue with scipy.sparse.linalg.eigsh deep in scipy library.
        evals, evecs = np.linalg.eigh(L.numpy())
        # back to torch
        evals = torch.from_numpy(evals)
        evecs = torch.from_numpy(evecs)

        offset = (evals < self.eps).sum().item()
        if offset == num_nodes:
            return torch.zeros(num_nodes, 7, dtype=torch.float32)

        electrostatic = (
            evecs[:, offset:] / evals[offset:] @ evecs[:, offset:].T
        )
        electrostatic = electrostatic - electrostatic.diag()
        electrostatic_encoding = torch.stack(
            [
                electrostatic.min(dim=0)[0],  # Min of Vi -> j
                electrostatic.mean(dim=0),  # Mean of Vi -> j
                electrostatic.std(dim=0),  # Std of Vi -> j
                electrostatic.min(dim=1)[0],  # Min of Vj -> i
                electrostatic.std(dim=1),  # Std of Vj -> i
                (DinvA * electrostatic).sum(
                    dim=0
                ),  # Mean of interaction on direct neighbour
                (DinvA * electrostatic).sum(
                    dim=1
                ),  # Mean of interaction from direct neighbour
            ],
            dim=1,
        )

        # TODO: some corner case when N=2 on MUTAG
        if torch.all(electrostatic_encoding == 0) and N > 2:
            if list(remove_self_loops(edge_index)[0].cpu().shape) == [2, 0]:
                # Case when there is no connectivity
                pass
            else:
                raise ValueError("ElectrostaticPE is all zeros")

        if torch.any(torch.isnan(electrostatic_encoding)):
            raise ValueError("ElectrostaticPE contains NaNs")
        return electrostatic_encoding
