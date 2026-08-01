"""Native homogeneous graph encoder for distance graph matching."""

from __future__ import annotations

from numbers import Integral

from torch import Tensor, nn
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn.norm import GraphNorm

from topobench.nn.encoders.base import AbstractFeatureEncoder

from .kdgm import DGM_d


class _DGMProjection(nn.Module):
    """Normalize and project native graph node features for DGM."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = GraphNorm(in_channels)
        self.linear = nn.Linear(in_channels, out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        """Project one native homogeneous node-feature matrix."""
        x = self.norm(x, batch=batch) if batch.numel() else self.norm(x)
        return self.dropout(self.activation(self.linear(x)))


class DGMStructureFeatureEncoder(AbstractFeatureEncoder):
    """Apply one DGM encoder to native homogeneous graph features.

    Parameters
    ----------
    in_channels : int
        Width of ``data.x``.
    out_channels : int
        Width assigned to encoded ``data.x``.
    proj_dropout : float, default=0
        Dropout used by the DGM base and embedding encoders.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        proj_dropout: float = 0,
    ) -> None:
        super().__init__()
        if isinstance(in_channels, bool) or not isinstance(
            in_channels, Integral
        ):
            raise TypeError("in_channels must be an integer")
        self.in_channels = int(in_channels)
        self.out_channels = out_channels
        self.encoder = DGM_d(
            base_enc=_DGMProjection(
                self.in_channels,
                self.out_channels,
                dropout=proj_dropout,
            ),
            embed_f=_DGMProjection(
                self.in_channels,
                self.out_channels,
                dropout=proj_dropout,
            ),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels})"
        )

    def forward(self, data: Data) -> Data:
        """Replace native graph features and retain DGM loss outputs."""
        if not isinstance(data, Data) or isinstance(data, HeteroData):
            raise TypeError(
                "DGMStructureFeatureEncoder requires homogeneous Data"
            )
        x = data.get("x")
        batch = data.get("batch")
        if not isinstance(x, Tensor):
            raise ValueError("data.x is required")
        if not isinstance(batch, Tensor):
            raise ValueError("data.batch is required")

        x, x_aux, edges_dgm, logprobs = self.encoder(x, batch)
        data.x = x
        data.dgm_aux = x_aux
        data.dgm_edge_index = edges_dgm
        data.dgm_logprobs = logprobs
        return data
