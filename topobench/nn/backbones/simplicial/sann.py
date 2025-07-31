"""SANN network."""

import torch
import torch.nn.functional


class SANN(torch.nn.Module):
    r"""SANN network.

    Parameters
    ----------
    in_channels : tuple of int or int
        Dimension of the hidden layers.
    hidden_channels : int
        Dimension of the output layer.
    update_func : str
        Update function.
    complex_dim : int
        Dimension of the complex.
    max_hop : int
        Number of hops.
    n_layers : int
        Number of layers.
    layer_norm : bool, optional
        Wether to perform layer normalization.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        update_func=None,
        complex_dim=2,
        max_hop=3,
        n_layers=2,
        layer_norm=True,
    ):
        super().__init__()
        self.complex_dim = complex_dim
        self.max_hop = max_hop
        self.layer_norm = layer_norm

        assert n_layers >= 1

        if isinstance(in_channels, int):  # If only one value is passed
            in_channels = [in_channels] * self.max_hop


        self.f_x_MLPS = torch.nn.ModuleList(
                SANNLayer(
                    [hidden_channels] * max_hop,
                    [hidden_channels] * max_hop,
                    update_func=update_func,
                    max_hop=max_hop,
                )
                for i in range(complex_dim+1)
                )

    def forward(self, x, batch):
        r"""Forward pass of the model.

        Parameters
        ----------
        x : tuple(tuple(torch.Tensor))
            Tuple of tuple containing the input tensors for each simplex.

        Returns
        -------
        tuple(tuple(torch.Tensor))
            Tuple of tuples of final hidden state tensors.
        """

        # The follwing line will mean the same as:
        # # For each k: 0 to k (k=0,1,2)
        # x_0_tup = tuple(self.in_linear_0[i](x[0][i]) for i in range(3))
        # # For each k: 1 to k (k=0,1,2)
        # x_1_tup = tuple(self.in_linear_1[i](x[1][i]) for i in range(3))
        # # For each k: 2 to k (k=0,1,2)
        # x_2_tup = tuple(self.in_linear_2[i](x[2][i]) for i in range(3))

        # For each layer in the network
        # For each simplex dimension (0, 1, 2)
        x = tuple(self.f_x_MLPS[i](x[i], batch[i]) for i in range(self.complex_dim+1))
        return x


class SANNLayer(torch.nn.Module):
    r"""One layer in the SANN architecture.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    max_hop : int
        Number of hop representations to consider.
    aggr_norm : bool
        Whether to perform aggregation normalization.
    update_func : str
        Update function.
    initialization : str
        Initialization method.

    Returns
    -------
    torch.Tensor
        Output
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        max_hop,
        aggr_norm: bool = True,
        update_func=None,
        initialization: str = "xavier_uniform",
        layer_norm: bool = True,
        block_depth = 4
    ) -> None:
        super().__init__()

        assert max_hop == len(in_channels), (
            "Number of hops must be equal to the number of input channels."
        )
        assert max_hop == len(out_channels), (
            "Number of hops must be equal to the number of output channels."
        )

        self.max_hop = max_hop
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.aggr_norm = aggr_norm
        self.update_func = update_func
        self.initialization = initialization

        self.layer_norm = layer_norm

        assert initialization in ["xavier_uniform", "xavier_normal"]
        assert self.in_channels[0] == self.out_channels[0]

        # One element for each hop (1, 2, 3, ... , k)
        self.list_hops = torch.nn.ModuleList()
        self.list_ln_hops = torch.nn.ModuleList()

        # Iterate over each of the hops 
        for i in range(max_hop):
            # Create a list of linear layers for each hop
            list_block = torch.nn.ModuleList(
                torch.nn.Linear(
                    in_features=self.in_channels[i],
                    out_features=self.out_channels[i],
                )
                for _ in range(block_depth)
            )
            self.list_hops.append(list_block)

            list_ln = torch.nn.ModuleList(
                torch.nn.BatchNorm1d(
                    num_features=self.out_channels[i],
                )
                for _ in range(block_depth-1)
            )
            self.list_ln_hops.append(list_ln)



        # if self.layer_norm:
        #     self.LN = torch.nn.ModuleList(
        #         torch.nn.BatchNorm1d(self.out_channels[i])
        #         for i in range(max_hop)
        #     )
        #     # self.LN = torch.nn.ModuleList(
        #     #     torch.nn.LayerNorm(self.out_channels[i])
        #     #     for i in range(max_hop)
        #     # )
        # else:
        #     self.LN = torch.nn.ModuleList(
        #         torch.nn.Identity() for i in range(max_hop)
        #     )

    def update(self, x: torch.Tensor):
        """Update embeddings on each cell (step 4).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Updated tensor.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(x)
        if self.update_func == "relu":
            return torch.nn.functional.relu(x)
        if self.update_func == "leaky_relu":
            return torch.nn.functional.leaky_relu(x)
        if self.update_func == "gelu":
            return torch.nn.functional.gelu(x)
        if self.update_func == "silu":
            return torch.nn.functional.silu(x)
        return None

    def forward(self, x_all: dict[int, torch.Tensor], batch):
        r"""Forward computation.

        Parameters
        ----------
        x_all : Dict[Int, torch.Tensor]
            Dictionary of tensors where each simplex dimension (node, edge, face) represents a key, 0-indexed.

        Returns
        -------
        torch.Tensor
            Output tensors for each 0-cell.
        torch.Tensor
            Output tensors for each 1-cell.
        torch.Tensor
            Output tensors for each 2-cell.
        """

        x_out = []
        for x, linear_layer, ln in zip(x_all, self.list_hops, self.list_ln_hops):
            prev_x = x
            for i, layer in enumerate(linear_layer):
                x = layer(x)
                if i < len(linear_layer) - 1:
                    x = ln[i](x)
                x = self.update(x)
            x = self.update(x + prev_x)
            x_out.append(x)


        # Maybe add skip-connections here: x = LN(x + x_0)
        # x_all
        # x_all = tuple([self.update(y_t) for y_t in y_k_t.values()])

        return tuple(x_out)
