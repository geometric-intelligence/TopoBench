"""Encoder class to apply BaseEncoder."""

import torch_geometric

from topobench.nn.encoders.all_cell_encoder import BaseEncoder
from topobench.nn.encoders.base import AbstractFeatureEncoder

from .kdgm import DGM_d


class DGMStructureFeatureEncoder(AbstractFeatureEncoder):
    r"""Encoder class to apply BaseEncoder.

    The BaseEncoder is applied to the features of higher order
    structures. The class creates a BaseEncoder for each dimension specified in
    selected_dimensions. Then during the forward pass, the BaseEncoders are
    applied to the features of the corresponding dimensions.

    Parameters
    ----------
    in_channels : list[int]
        Input dimensions for the features.
    out_channels : list[int]
        Output dimensions for the features.
    proj_dropout : float, optional
        Dropout for the BaseEncoders (default: 0).
    selected_dimensions : list[int], optional
        List of indexes to apply the BaseEncoders to (default: None).
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        proj_dropout=0,
        selected_dimensions=None,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dimensions = (
            selected_dimensions
            if (
                selected_dimensions is not None
            )  # and len(selected_dimensions) <= len(self.in_channels))
            else range(len(self.in_channels))
        )
        for i in self.dimensions:
            base_enc = BaseEncoder(
                self.in_channels[i],
                self.out_channels,
                dropout=proj_dropout,
            )
            embed_f = BaseEncoder(
                self.in_channels[i],
                self.out_channels,
                dropout=proj_dropout,
            )

            setattr(
                self,
                f"encoder_{i}",
                DGM_d(base_enc=base_enc, embed_f=embed_f),
            )

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, dimensions={self.dimensions})"

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Forward pass.

        The method applies the BaseEncoders to the features of the selected_dimensions.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data object which should contain x_{i} features for each i in the selected_dimensions.

        Returns
        -------
        torch_geometric.data.Data
            Output data object with updated x_{i} features.
        """
        if not hasattr(data, "x_0"):
            data.x_0 = data.x

        for i in self.dimensions:
            if hasattr(data, f"x_{i}") and hasattr(self, f"encoder_{i}"):
                batch = getattr(data, f"batch_{i}")
                x_, x_aux, edges_dgm, logprobs = getattr(self, f"encoder_{i}")(
                    data[f"x_{i}"], batch
                )
                data[f"x_{i}"] = x_
                data[f"x_aux_{i}"] = x_aux
                data["edges_index"] = edges_dgm
                data[f"logprobs_{i}"] = logprobs
        return data
