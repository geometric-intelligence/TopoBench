"""Flat graph encoder."""

import torch_geometric

from topobench.nn.encoders.base import AbstractFeatureEncoder


class FlatEncoder(AbstractFeatureEncoder):
    r"""Abstract class to define a custom feature encoder.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    **kwargs
        Additional keyword arguments.
    """

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Forward pass of the flat encoder.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data object which should contain x features.

        Returns
        -------
        torch_geometric.data.Data
            Output data object with flattened features.
        """
        if not hasattr(data, "x_0"):
            data.x_0 = data.x
        data.labels = data.y
        data.x_0 = data.x_0.view(data.batch_size, -1)
        return data
