"""Laplacian Positional Encoding (LapPE) Transform."""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    get_laplacian,
    remove_self_loops,
    to_scipy_sparse_matrix,
)


class HKdiagSE(BaseTransform):
    r"""
    Heat Kernel Diagonal Structural Encoding (HKdiagSE) transform.

    Diagonals of heat kernel diffusion.

    Parameters
    ----------
    max_pe_dim : int
        Maximum number of eigenvectors to use (dimensionality of the encoding).
    kernel_param_HKdiagSE : tuple of int
        Tuple specifying the start and end diffusion times for the heat kernel.
    space_dim : int, optional
        Estimated dimensionality of the space. Used to
        correct the diffusion diagonal by a factor `t^(space_dim/2)`. In
        euclidean space, this correction means that the height of the
        gaussian stays constant across time, if `space_dim` is the dimension
        of the euclidean space. Default is 0 (no correction).
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
        kernel_param_HKdiagSE: tuple,
        space_dim: int = 0,
        include_eigenvalues: bool = False,
        include_first: bool = False,
        concat_to_x: bool = True,
        eps: float = 1e-6,
        **kwargs,
    ):
        self.max_pe_dim = max_pe_dim
        self.kernel_param_HKdiagSE = kernel_param_HKdiagSE
        self.space_dim = space_dim
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
        pe = self._compute_hkdiag_se(data.edge_index, data.num_nodes)

        if self.concat_to_x:
            if data.x is None:
                data.x = pe
            else:
                data.x = torch.cat([data.x, pe], dim=-1)
        else:
            data.HKdiagSE = pe

        return data

    def _compute_hkdiag_se(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Internal method to compute heat kernel diagonal structural encodings.

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
        L = to_scipy_sparse_matrix(
            edge_index_lap, edge_weight, num_nodes
        ).astype(np.float64)

        evals, evects = np.linalg.eigh(L.toarray())
        evals = torch.from_numpy(evals)
        evects = torch.from_numpy(evects)

        start, end = (
            self.kernel_param_HKdiagSE[0],
            self.kernel_param_HKdiagSE[1],
        )
        kernel_times = range(start, end)

        if len(kernel_times) == 0:
            raise ValueError("Diffusion times are required for heat kernel")

        hk_diag = []
        evects = F.normalize(evects, p=2.0, dim=0)

        # Remove eigenvalues == 0 from the computation of the heat kernel
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]

        # Change the shapes for the computations
        evals = evals.unsqueeze(-1)  # lambda_{i, ..., ...}
        evects = evects.transpose(0, 1)  # phi_{i,j}: i-th eigvec X j-th node

        # Compute the heat kernels diagonal only for each time
        eigvec_mul = evects**2
        for t in kernel_times:
            # sum_{i>0}(exp(-2 t lambda_i) * phi_{i, j} * phi_{i, j})
            this_kernel = torch.sum(
                torch.exp(-t * evals) * eigvec_mul, dim=0, keepdim=False
            )

            # Multiply by `t` to stabilize the values, since the gaussian height
            # is proportional to `1/t`
            hk_diag.append(this_kernel * (t ** (self.space_dim / 2)))
        hk_diag = torch.stack(hk_diag, dim=0).transpose(0, 1)

        # TODO: some corner case when N=2 on MUTAG
        if (torch.all(hk_diag == 0)) and (num_nodes > 2):
            # Case when there is no connectivity
            if list(remove_self_loops(edge_index)[0].cpu().shape) == [2, 0]:
                pass
            else:
                raise ValueError("HKdiagSE is all zeros")

        if torch.any(torch.isnan(hk_diag)):
            raise ValueError("HKdiagSE contains NaNs")
        return hk_diag
