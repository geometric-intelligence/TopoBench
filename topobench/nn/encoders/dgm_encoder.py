"""Native homogeneous graph encoder for distance graph matching."""

from __future__ import annotations

from numbers import Integral

from torch import Tensor
from torch_geometric.data import Data, HeteroData

from topobench.nn.encoders.all_cell_encoder import BaseEncoder
from topobench.nn.encoders.base import AbstractFeatureEncoder

from .kdgm import DGM_d


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
            base_enc=BaseEncoder(
                self.in_channels,
                self.out_channels,
                dropout=proj_dropout,
            ),
            embed_f=BaseEncoder(
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
        data.x_aux_0 = x_aux
        data.edges_index = edges_dgm
        data.logprobs_0 = logprobs
        return data
