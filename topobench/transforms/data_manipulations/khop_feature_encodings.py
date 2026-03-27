"""K-hop feature Encoding (KFE) for Hasse graphs Transform."""
import time
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj, degree


class KHopFE(BaseTransform):
    r"""
    K-hop Feature Encodings (KHopFE) transform.

    Parameters
    ----------
    max_hop: int
        The maximum hop neighbourhood.
    concat_to_x : bool, optional
        If True, concatenates the encodings with existing node features in
        ``data.x``. If ``data.x`` is None, creates it. Default is True.
    aggregation : str, optional
        Aggregation function to reduce over the feature dimension.
        Options: "mean", "sum", "max", "min". Default is "mean".
    method : str, optional
        Computation method: "dense" or "sparse". Default is "sparse".
    debug : bool, optional
        If True, runs both methods and prints error/timing metrics. Default is False.
    **kwargs : dict
        Additional arguments (not used).
    """

    _AGG_FN_MAP = {"mean": "mean", "sum": "sum", "max": "amax", "min": "amin"}

    def __init__(
        self,
        max_hop: int,
        concat_to_x: bool = True,
        aggregation: str = "mean",
        method: str = "sparse",
        debug: bool = False,
        **kwargs,
    ):
        self.concat_to_x = concat_to_x
        self.max_hop = max_hop - 1  # The 0-th hop is always the features themselves
        self.method = method
        self.debug = debug
        
        if aggregation not in self._AGG_FN_MAP:
            raise ValueError(
                f"Unknown aggregation '{aggregation}'. "
                f"Choose from: {list(self._AGG_FN_MAP.keys())}"
            )
        self.aggregation = aggregation
        
        if method not in ["dense", "sparse"]:
            raise ValueError("Method must be 'dense' or 'sparse'.")

    def forward(self, data: Data) -> Data:
        if data.x is None:
            raise ValueError("KHopFE requires node features (data.x cannot be None)")

        fe = self._compute_khopfe(data.x, data.edge_index, data.num_nodes)

        if self.concat_to_x:
            if data.x is None:
                data.x = fe
            else:
                data.x = torch.cat([data.x, fe], dim=-1)
        else:
            data.KHopFE = fe

        return data

    def _compute_khopfe(
        self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        device = edge_index.device
        x = x.to(device)

        if edge_index.size(1) == 0 or num_nodes <= 1:
            return torch.zeros(num_nodes, self.max_hop, device=device)

        if self.debug:
            print("\n--- KHopFE Debug Report ---")
            
            # Exact (Dense)
            t0 = time.time()
            fe_dense = self._compute_dense(x, edge_index, num_nodes, device)
            t_dense = time.time() - t0
            print(f"Dense compute time:  {t_dense:.4f}s")

            # Approx (Sparse)
            t0 = time.time()
            fe_sparse = self._compute_sparse(x, edge_index, num_nodes, device)
            t_sparse = time.time() - t0
            print(f"Sparse compute time: {t_sparse:.4f}s")

            # Compare
            diff = torch.abs(fe_dense - fe_sparse)
            speedup = (t_dense / t_sparse) if t_sparse > 0 else float('inf')
            print(f"Speedup Factor:      {speedup:.2f}x")
            print(f"Mean Abs Error:      {diff.mean().item():.6e}")
            print(f"Max Abs Error:       {diff.max().item():.6e}")
            print("---------------------------\n")

            fe_raw = fe_dense if self.method == "dense" else fe_sparse
        else:
            if self.method == "dense":
                fe_raw = self._compute_dense(x, edge_index, num_nodes, device)
            else:
                fe_raw = self._compute_sparse(x, edge_index, num_nodes, device)

        # Aggregate over features
        agg_fn = getattr(fe_raw, self._AGG_FN_MAP[self.aggregation])
        khop_fe = agg_fn(dim=-1)

        if torch.any(torch.isnan(khop_fe)):
            raise ValueError("KHopFE contains NaNs")

        return khop_fe.float()

    def _compute_dense(self, x, edge_index, num_nodes, device):
        """Original implementation using dense adjacency matrices."""
        khop_fe = []
        A = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
        
        # Symmetric norm adjacency matrix
        deg = A.sum(dim=1)
        deg_inv_sqrt = torch.diagflat(torch.pow(deg + 1e-8, -0.5))
        A_norm = deg_inv_sqrt @ A @ deg_inv_sqrt

        curr_x = x
        for _ in range(self.max_hop):
            curr_x = A_norm @ curr_x
            khop_fe.append(curr_x)
            
        return torch.stack(khop_fe, dim=1)

    def _compute_sparse(self, x, edge_index, num_nodes, device):
        """Optimized implementation using pure PyTorch sparse tensors."""
        khop_fe = []
        row, col = edge_index

        # 1. Compute node degrees using the row indices (out-degree)
        deg = degree(row, num_nodes, dtype=torch.float32)
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)

        # 2. Compute symmetric normalized edge weights: (D^-0.5)[i] * (D^-0.5)[j]
        # Since A[i,j] is 1 for existing edges, the weight is just the product of the inverse sqrts
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 3. Create the sparse symmetric normalized adjacency matrix
        A_sparse = torch.sparse_coo_tensor(
            edge_index, edge_weight, (num_nodes, num_nodes), device=device
        ).coalesce()

        # 4. Iteratively propagate features via Sparse Matrix-Matrix multiplication
        curr_x = x
        for _ in range(self.max_hop):
            curr_x = torch.sparse.mm(A_sparse, curr_x)
            khop_fe.append(curr_x)

        return torch.stack(khop_fe, dim=1)