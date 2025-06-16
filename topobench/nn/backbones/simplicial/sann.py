"""SANN network."""

import torch
import torch.nn.functional
from torch.nn import ParameterList
from torch.nn.parameter import Parameter


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
        complex_dim=3,
        max_hop=3,
        n_layers=2,
        layer_norm=None,
    ):
        super().__init__()
        self.complex_dim = complex_dim
        self.max_hop = max_hop
        self.layer_norm = layer_norm

        assert n_layers >= 1

        if self.layer_norm:
            self.layernorm = torch.nn.LayerNorm(hidden_channels, eps=1e-6)

        if isinstance(in_channels, int):  # If only one value is passed
            in_channels = [in_channels] * self.max_hop

        self.layers = torch.nn.ModuleList()

        # Set of simplices layers
        self.layers_0 = torch.nn.ModuleList(
            SANNLayer(
                [in_channels[i] for i in range(max_hop)],
                [hidden_channels] * max_hop,
                update_func=update_func,
                max_hop=max_hop,
            )
            for i in range(complex_dim)
        )
        self.layers.append(self.layers_0)

        # From layer 1 to n_layers
        for i in range(1, n_layers):
            self.layers.append(
                torch.nn.ModuleList(
                    SANNLayer(
                        [hidden_channels] * max_hop,
                        [hidden_channels] * max_hop,
                        update_func=update_func,
                        max_hop=max_hop,
                    )
                    for i in range(complex_dim)
                )
            )

    def forward(self, x):
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
        for layer in self.layers:
            # Temporary list
            x_i = list()

            # For each i-simplex (i=0,1,2) to all other k-simplices
            for i in range(self.complex_dim):
                # Goes from i-simplex to all other simplices k<=i
                x_i_to_t = (
                    [self.layernorm(x_j) for x_j in x[i]]
                    if self.layer_norm
                    else x[i]
                )
                x_i_to_t = layer[i](x_i_to_t)
                # Update the i-th simplex to all other simplices embeddings
                x_i.append(tuple(x_i_to_t))
            x = tuple(x_i)
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
        initialization: str = "xavier_normal",
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

        assert initialization in ["xavier_uniform", "xavier_normal"]

        self.weights = ParameterList(
            [
                Parameter(
                    torch.Tensor(
                        self.in_channels[i],
                        self.out_channels[i],
                    )
                )
                for i in range(max_hop)
            ]
        )
        self.biases = ParameterList(
            [
                Parameter(
                    torch.Tensor(
                        self.out_channels[i],
                    )
                )
                for i in range(max_hop)
            ]
        )

        self.LN = torch.nn.ModuleList(
            torch.nn.LayerNorm(self.out_channels[i]) for i in range(max_hop)
        )

        self.reset_parameters()

    def reset_parameters(self, gain: float = 1.414):
        r"""Reset learnable parameters.

        Parameters
        ----------
        gain : float
            Gain for the weight initialization.
        """
        if self.initialization == "xavier_uniform":
            for i in range(len(self.weights)):
                torch.nn.init.xavier_uniform_(self.weights[i], gain=gain)
                torch.nn.init.zeros_(self.biases[i])
        elif self.initialization == "xavier_normal":
            for i in range(len(self.weights)):
                torch.nn.init.xavier_normal_(self.weights[i], gain=gain)
                torch.nn.init.zeros_(self.biases[i])
        else:
            raise RuntimeError(
                "Initialization method not recognized. "
                "Should be either xavier_uniform or xavier_normal."
            )

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
        if self.update_func == "lrelu":
            return torch.nn.functional.leaky_relu(x)
        if self.update_func == 'gelu':
            return torch.nn.functional.gelu(x)
        if self.update_func == 'silu':
            return torch.nn.functional.silu(x)
        return None

    def forward(self, x_all: dict[int, torch.Tensor]):
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
        x_all_0 = [x.clone() for x in x_all]
        # Extract all cells to all cells
        x_k_t = {i: x_all[i] for i in range(self.max_hop)}

        y_k_t = {
            i: torch.mm(x_k_t[i], self.weights[i]) + self.biases[i]
            for i in range(self.max_hop)
        }

        if self.update_func is None:
            return tuple(y_k_t.values())

        # Maybe add skip-connections here: x = LN(x + x_0)
        # x_all
        x_all = tuple([self.update(y_t) for y_t in y_k_t.values()])

        x_out = []
        for i, xs in enumerate(zip(x_all_0, x_all, strict=False)):
            x_0, x = xs
            x_out.append(self.LN[i](x + x_0))
            # x_out.append(x)

        return tuple(x_out)
