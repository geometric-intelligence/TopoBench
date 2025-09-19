"""Encoder class to apply SimpleEncoder."""

import torch
import torch_geometric
from torch_geometric.nn.models import MLP
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from topobench.nn.encoders.base import AbstractFeatureEncoder


class SANNFeatureEncoder(AbstractFeatureEncoder):
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
        use_atom_encoder=False,
        use_bond_encoder=False,
        fuse_pse2cell=True,
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
        self.fuse_pse2cell = fuse_pse2cell
        for i in self.dimensions:
            for j in range(self.hops):
                if use_atom_encoder and i == 0 and j == 0:
                    setattr(
                        self,
                        f"encoder_{i}_{j}",
                        SimpleAtomEncoder(self.out_channels),
                    )
                elif use_bond_encoder and i == 1 and j == 0:
                    setattr(
                        self,
                        f"encoder_{i}_{j}",
                        SimpleBondEncoder(self.out_channels),
                    )
                else:
                    true_in_channels = self.in_channels[i][j]

                    if self.fuse_pse2cell and j == 0:
                        true_in_channels = self.out_channels

                    setattr(
                        self,
                        f"encoder_{i}_{j}",
                        MLP(
                            in_channels=true_in_channels,
                            hidden_channels=true_in_channels,
                            out_channels=self.out_channels,
                            dropout=proj_dropout,
                            batch_norm=batch_norm,
                            num_layers=1,
                            act="relu",
                        ),
                    )

        # Rebuttal update
        if self.fuse_pse2cell:
            for i in self.dimensions:
                in_channel_total = 0
                for j in range(self.hops):
                    in_channel_total += self.in_channels[i][j]
                setattr(
                    self,
                    f"feature_mixing_{i}",
                    MLP(
                        in_channels=in_channel_total,
                        hidden_channels=in_channel_total,
                        out_channels=self.out_channels,
                        dropout=proj_dropout,
                        batch_norm=None,
                        num_layers=1,
                        act=None
                    )
                )
            # Instantiate self.hops layer normalization
            # self.LN_pse2cell = torch.nn.ModuleList(
            #     torch.nn.LayerNorm(self.out_channels) for _ in range(self.hops)
            # )


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
        if self.fuse_pse2cell:
            for i in self.dimensions:
                # node_and_pse_encodings = [
                #     self.LN_pse2cell[j](data[f"x{i}_{j}"])
                #     for j in range(self.hops)
                # ]
                # Concatenate the encodings along the last dimension
                concatenated = torch.cat([data[f"x{i}_{j}"] for j in range(self.hops)], dim=-1)
                data[f"x{i}_0"] = getattr(self, f"feature_mixing_{i}")(concatenated)

                # data[f"x_{i}"] = sum(node_and_pse_encodings)

        for i in self.dimensions:
            batch = getattr(data, f"batch_{i}")
            for j in range(self.hops):
                data[f"x{i}_{j}"] = getattr(self, f"encoder_{i}_{j}")(
                    data[f"x{i}_{j}"], batch
                )

        return data


class SimpleAtomEncoder(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.atom_encoder = AtomEncoder(in_channels)

    def forward(self, x, batch):
        x = self.atom_encoder(x.long())
        return x


class SimpleBondEncoder(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.bond_encoder = BondEncoder(in_channels)

    def forward(self, x, batch):
        x = self.bond_encoder(x.long())
        return x
