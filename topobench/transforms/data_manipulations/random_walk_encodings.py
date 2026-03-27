"""Random Walk Structural Encodings (RWSE) Transform."""

import time

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree


class RWSE(BaseTransform):
    r"""Random Walk Structural Encoding (RWSE) transform.

    Parameters
    ----------
    max_pe_dim : int
        Maximum walk length (number of RWSE dimensions).
    concat_to_x : bool, optional
        If True, concatenates the encodings with existing node features.
        Default is True.
    method : str, optional
        Computation method: "dense" (Dense MatMul) or "sparse" (Sparse MatMul).
        Default is "sparse".
    debug : bool, optional
        If True, runs both methods and prints error/timing metrics.
        Default is False.
    **kwargs : dict
        Additional arguments (not used).
    """

    def __init__(
        self,
        max_pe_dim: int,
        concat_to_x: bool = True,
        method: str = "sparse",
        debug: bool = True,
        **kwargs,
    ):
        self.max_pe_dim = max_pe_dim
        self.concat_to_x = concat_to_x
        self.debug = debug

        if method not in ["dense", "sparse"]:
            raise ValueError("Method must be 'dense' or 'sparse'.")
        self.method = method

    def forward(self, data: Data) -> Data:
        """Compute the RWSE for the input graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Graph data object with RWSE added to ``data.x`` or ``data.RWSE``.
        """
        if self.debug:
            print("\n--- RWSE Debug Report ---")

            # Dense Method
            t0 = time.time()
            pe_dense = self._compute_dense(data.edge_index, data.num_nodes)
            t_dense = time.time() - t0
            print(f"Dense compute time:  {t_dense:.4f}s")

            # Sparse Method
            t0 = time.time()
            pe_sparse = self._compute_sparse(data.edge_index, data.num_nodes)
            t_sparse = time.time() - t0
            print(f"Sparse compute time: {t_sparse:.4f}s")

            # Compare
            diff = torch.abs(pe_dense - pe_sparse)
            speedup = (t_dense / t_sparse) if t_sparse > 0 else float("inf")
            print(f"Speedup Factor:      {speedup:.2f}x")
            print(f"Mean Abs Error:      {diff.mean().item():.6e}")
            print(f"Max Abs Error:       {diff.max().item():.6e}")
            print("---------------------------\n")

            pe = pe_dense if self.method == "dense" else pe_sparse
        else:
            if self.method == "dense":
                pe = self._compute_dense(data.edge_index, data.num_nodes)
            else:
                pe = self._compute_sparse(data.edge_index, data.num_nodes)

        if self.concat_to_x:
            if data.x is None:
                data.x = pe
            else:
                data.x = torch.cat([data.x, pe], dim=-1)
        else:
            data.RWSE = pe

        return data

    def _compute_dense(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Compute RWSE using original dense matrix multiplication.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of the graph of shape ``[2, num_edges]``.
        num_nodes : int
            Number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            RWSE return probabilities of shape ``[num_nodes, max_pe_dim]``.
        """
        device = edge_index.device
        if edge_index.numel() == 0 or num_nodes <= 1:
            return torch.zeros(num_nodes, self.max_pe_dim, device=device)

        deg = degree(edge_index[0], num_nodes=num_nodes).float().to(device)
        deg = torch.where(deg == 0, torch.ones_like(deg), deg)

        adj = torch.zeros(num_nodes, num_nodes, device=device)
        adj[edge_index[0], edge_index[1]] = 1.0

        P = adj / deg.unsqueeze(-1)
        rwse = torch.zeros(num_nodes, self.max_pe_dim, device=device)
        P_power = torch.eye(num_nodes, device=device)

        for k in range(1, self.max_pe_dim + 1):
            P_power = P_power @ P
            rwse[:, k - 1] = P_power.diag()

        return rwse.float()

    def _compute_sparse(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Compute RWSE using optimized PyTorch sparse matrix multiplication.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of the graph of shape ``[2, num_edges]``.
        num_nodes : int
            Number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            RWSE return probabilities of shape ``[num_nodes, max_pe_dim]``.
        """
        device = edge_index.device
        if edge_index.numel() == 0 or num_nodes <= 1:
            return torch.zeros(num_nodes, self.max_pe_dim, device=device)

        row, col = edge_index

        # 1. Compute Out-Degree
        deg = degree(row, num_nodes=num_nodes, dtype=torch.float32)
        deg_inv = 1.0 / deg.clamp_(min=1.0)

        # 2. Transition probabilities: P_{i,j} = 1 / deg(i)
        edge_weight = deg_inv[row]

        # 3. Create Sparse Transition Matrix P
        P = torch.sparse_coo_tensor(
            edge_index, edge_weight, (num_nodes, num_nodes), device=device
        ).coalesce()

        rwse = []
        Pk = P

        # Pre-allocate a zero tensor to avoid re-allocating memory inside the loop
        pe_k = torch.zeros(num_nodes, device=device)

        for _ in range(self.max_pe_dim):
            # 1. Grab coordinates and values
            row, col = Pk.indices()
            val = Pk.values()

            # 2. Find the diagonal elements (where row index == col index)
            mask = row == col

            # 3. Drop them into the pre-allocated zero tensor and save
            pe_k.zero_()  # Reset the tensor inplace
            pe_k.scatter_(0, row[mask], val[mask])
            rwse.append(pe_k.clone())  # Clone to save this step's state

            # 4. Advance the random walk
            Pk = torch.sparse.mm(Pk, P)

        return torch.stack(rwse, dim=1)
