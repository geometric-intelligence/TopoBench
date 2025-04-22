"""Encoder class to apply SimpleEncoder."""

import torch
import torch_geometric

from topobench.nn.encoders.base import AbstractFeatureEncoder
from topobench.transforms.data_manipulations import (
    AddGPSEInformation,
)


class SANNOnlineFeatureEncoder(AbstractFeatureEncoder):
    r"""Encoder class to apply SimpleEncoder.

    The SimpleEncoder is applied to the features of each cell
    according to a simp

    Parameters
    ----------
    in_channels : list[list[int]]
        Input dimensions for the features.
    out_channels : list[int]
        Output dimensions for the features.
    proj_dropout : float, optional
        Dropout for the BaseEncoders (default: 0).
    selected_dimensions : list[int], optional
        List of indexes to apply the BaseEncoders to (default: None).
    max_hop : list[int], optional
        List of indexes to apply the BaseEncoders to in terms of hops (default: None).
    batch_norm : bool, optional
        Wether to apply batch normalizaiton when encoding (default: False).
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        proj_dropout=0,
        selected_dimensions=None,
        max_hop=3,
        batch_norm=False,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dimensions = (
            selected_dimensions
            if (selected_dimensions is not None)
            else range(len(self.in_channels))
        )
        self.hops = max_hop
        for i in self.dimensions:
            for j in range(self.hops):
                setattr(
                    self,
                    f"encoder_{i}_{j}",
                    SimpleEncoder(
                        self.in_channels[i][j],
                        self.out_channels,
                        dropout=proj_dropout,
                        batch_norm=batch_norm,
                    ),
                )

        config_dict = {
            "max_rank": kwargs["max_rank"],
            "copy_initial": kwargs["copy_initial"],
            "neighborhoods": kwargs["neighborhoods"],
            "dim_out": kwargs["dim_out"],
            "max_hop": f"{max_hop}",
            "dim_target_node": kwargs["dim_target_node"],
            "dim_target_graph": kwargs["dim_target_graph"],
            "pretrain_model": kwargs["pretrain_model"],
            "in_channels": in_channels,
            "device": kwargs["device"],
            "cuda": kwargs["cuda"],
        }
        self.gpse = AddGPSEInformation(**config_dict)

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
        data = self.gpse(data).clone()  # Apply the GPSE transformation first
        for i in self.dimensions:
            batch = getattr(data, f"batch_{i}")
            for j in range(self.hops):
                data[f"x{i}_{j}"] = getattr(self, f"encoder_{i}_{j}")(
                    data[f"x{i}_{j}"], batch
                )
        return data


class SimpleEncoder(torch.nn.Module):
    r"""SimpleEncoder used by SANN.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimensions of output features.
    dropout : float, optional
        Percentage of channels to discard between the two linear layers (default: 0).
    batch_norm : bool, optional
        Wether to perform batch normalization (default: False).
    """

    def __init__(self, in_channels, out_channels, dropout=0, batch_norm=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.linear1 = torch.nn.Linear(in_channels, out_channels)
        # self.linear2 = torch.nn.Linear(out_channels, out_channels)
        self.relu = torch.nn.ReLU()
        self.BN = (
            torch.nn.BatchNorm1d(out_channels)
            if batch_norm
            else torch.nn.Identity()
        )
        self.dropout = (
            torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.linear1.in_features}, out_channels={self.linear1.out_features})"

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the encoder.

        Applies a linear layer and a ReLu activation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of dimensions [N, in_channels].
        batch : torch.Tensor
            The batch vector which assigns each element to a specific example.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [N, out_channels].
        """

        x = self.linear1(x)
        if self.batch_norm:
            x = self.BN(x, batch=batch) if batch.shape[0] > 0 else self.BN(x)
        x = self.dropout(x)
        return x
