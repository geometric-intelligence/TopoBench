"""Base classes for sheaf neural network layers."""
# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

from torch import nn


class SheafDiffusion(nn.Module):
    """
    Base class for sheaf diffusion models.

    This class provides the foundational structure for all sheaf diffusion variants,
    storing common parameters and configurations.

    Parameters
    ----------
    edge_index : torch.Tensor or None
        Edge indices of shape [2, num_edges]. Can be None for inductive models.
    args : dict
        Configuration dictionary containing:
        - d (int): Dimension of the stalk space (must be > 0).
        - hidden_channels (int): Number of hidden channels per stalk dimension.
        - device (str): Device to run the model on.
        - layers (int): Number of diffusion layers.
        - input_dropout (float): Dropout rate for input layer.
        - dropout (float): Dropout rate for hidden layers.
        - input_dim (int): Dimension of input features.
        - output_dim (int): Dimension of output features.
        - sheaf_act (str): Activation function for sheaf learning.
        - orth (str): Orthogonalization method.
        - add_hp (bool, optional): Add a fixed high-pass channel (default False).
        - add_lp (bool, optional): Add a fixed low-pass channel (default False).
        - normalised (bool, optional): Use the normalized sheaf Laplacian
          (default False).
        - deg_normalised (bool, optional): Use degree normalization instead;
          mutually exclusive with ``normalised`` (default False).
        - second_linear (bool, optional): Add an extra input projection before
          propagation (default False).
    """

    def __init__(self, edge_index, args):
        super().__init__()

        assert args["d"] > 0
        self.d = args["d"]
        # Optional fixed high-/low-pass channels grow each stalk by one dim each.
        self.add_hp = args.get("add_hp", False)
        self.add_lp = args.get("add_lp", False)
        self.final_d = self.d + int(self.add_hp) + int(self.add_lp)
        # Laplacian normalization options (mutually exclusive).
        self.normalised = args.get("normalised", False)
        self.deg_normalised = args.get("deg_normalised", False)
        # Optional extra input projection before propagation.
        self.second_linear = args.get("second_linear", False)

        self.edge_index = edge_index
        self.hidden_channels = args["hidden_channels"]
        self.hidden_dim = self.hidden_channels * self.final_d
        self.device = args["device"]
        self.layers = args["layers"]
        self.input_dropout = args["input_dropout"]
        self.dropout = args["dropout"]
        self.input_dim = args["input_dim"]
        self.output_dim = args["output_dim"]
        self.sheaf_act = args["sheaf_act"]
        self.orth_trans = args["orth"]
        self.laplacian_builder = None
