"""Heat Kernel feature Encoding (HKFE) Transform."""

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
)
from scipy.sparse.linalg import expm_multiply
import omegaconf


class HKFE(BaseTransform):
    r"""
    Heat Kernel Feature Encodings (HKFE) transform.

    Parameters
    ----------
    kernel_param_HKFE : tuple of int
        Tuple specifying the start and end diffusion times for the heat kernel.
    concat_to_x : bool, optional
        If True, concatenates the encodings with existing node features in
        ``data.x``. If ``data.x`` is None, creates it. Default is True.
    aggregation : str, optional
        Aggregation function to reduce over the feature dimension.
        Options: "mean", "sum", "max", "min". Default is "mean".
    **kwargs : dict
        Additional arguments (not used).
    """

    _AGG_FN_MAP = {"mean": "mean", "sum": "sum", "max": "amax", "min": "amin"}

    def __init__(
        self,
        kernel_param_HKFE: tuple,
        concat_to_x: bool = True,
        aggregation: str = "mean",
        **kwargs,
    ):
        self.kernel_param_HKFE = kernel_param_HKFE
        self.concat_to_x = concat_to_x
        if aggregation not in self._AGG_FN_MAP:
            raise ValueError(
                f"Unknown aggregation '{aggregation}'. "
                f"Choose from: {list(self._AGG_FN_MAP.keys())}"
            )
        self.aggregation = aggregation
        # Compute fe_dim from tuple/list or use directly if int
        if (
            isinstance(kernel_param_HKFE, (list, tuple))
            or type(kernel_param_HKFE) is omegaconf.listconfig.ListConfig
        ):
            self.fe_dim = kernel_param_HKFE[1] - kernel_param_HKFE[0]
        else:
            self.fe_dim = kernel_param_HKFE

    def forward(self, data: Data) -> Data:
        """Compute the Heat Kernel feature encodings for the input graph.

        Parameters
        ----------
        data : Data
            Input graph data object.

        Returns
        -------
        Data
            Graph data object with heat kernel feature encodings added.
        """
        if data.x is None:
            raise ValueError(
                "HKFE requires node features (data.x cannot be None)"
            )

        fe = self._compute_hkfe(data.x, data.edge_index, data.num_nodes)

        if self.concat_to_x:
            if data.x is None:
                data.x = fe
            else:
                data.x = torch.cat([data.x, fe], dim=-1)
        else:
            data.HKFE = fe

        return data

    def _compute_hkfe(
        self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Internal method to compute heat kernel feature encodings.

        Computes heat kernel diffusion at multiple time scales and aggregates
        over input features to produce a fixed-dimension output (matching PSE pattern).

        Parameters
        ----------
        x : torch.Tensor
            Node features of the graph.
        edge_index : torch.Tensor
            Edge indices of the graph.
        num_nodes : int
            Number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            Heat Kernel feature encodings of shape [N, fe_dim].
        """
        device = edge_index.device
        hk_fe = []
        if edge_index.size(1) == 0 or num_nodes <= 1:
            return torch.zeros(num_nodes, self.fe_dim, device=device)

        # Normalized Laplacian
        edge_index_lap, edge_weight = get_laplacian(
            edge_index, normalization="sym", num_nodes=num_nodes
        )
        L = to_scipy_sparse_matrix(
            edge_index_lap, edge_weight, num_nodes
        ).astype(np.float64)

        start, end = (
            self.kernel_param_HKFE[0],
            self.kernel_param_HKFE[1],
        )
        kernel_times = np.geomspace(start, end, self.fe_dim)
        if len(kernel_times) == 0:
            raise ValueError("Diffusion times are required for heat kernel")

        x = x.detach().cpu().numpy().astype(np.float64)
        for t in kernel_times:
            x_t = expm_multiply((-float(t)) * L, x)
            hk_fe.append(torch.from_numpy(x_t).float().to(device))
        hk_fe = torch.stack(hk_fe, dim=1)  # [N, fe_dim, F]
        # Aggregate over features to produce fixed-dimension output (like PSEs)
        agg_fn = getattr(hk_fe, self._AGG_FN_MAP[self.aggregation])
        hk_fe = agg_fn(dim=-1)  # [N, fe_dim]

        if torch.any(torch.isnan(hk_fe)):
            raise ValueError("HKFE contains NaNs")
        return hk_fe.float()
