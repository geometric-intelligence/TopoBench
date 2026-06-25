"""
GSN-GIN message-passing layers with concatenated structural encodings.

Implements node- and edge-mode variants of Graph Substructure Network (GSN)
combined with GIN-style sum aggregation, following Bouritsas et al. (2022).
Structural encodings are concatenated to node features prior to aggregation
rather than being processed in a separate encoder stream.
"""

from abc import ABC, abstractmethod
from typing import LiteralString

import torch
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops


class GSNGINconcatBaseLayer(ABC, MessagePassing):
    """
    Abstract base class for GSN-GIN layers with structural encoding concatenation.

    Implements GIN-style sum aggregation extended with Graph Substructure Network
    (GSN) structural encodings via concatenation. Subclasses define node- or
    edge-level encoding strategies.

    Parameters
    ----------
    in_channels : int
        Dimensionality of input node features.
    in_gsn_channels : int
        Dimensionality of GSN structural encodings.
    out_channels : int
        Dimensionality of output node representations.
    train_eps : bool, optional
        If True, the epsilon scaling parameter is learnable. Default is False.
    extra_mlp_in_channel : int, optional
        Additional input channels prepended to the MLP input (used by
        edge-mode subclasses). Default is 0.
    mlp_architecture : list of int or None, optional
        Hidden layer widths for the update MLP. If None, a single linear
        layer followed by ReLU is used.

    Attributes
    ----------
    UP : torch.nn.Sequential
        Update MLP applied after neighbourhood aggregation.
    eps : torch.nn.Parameter or torch.Tensor
        Epsilon scaling factor for the self term; learnable or fixed at 0.
    """

    def __init__(
        self,
        in_channels: int,
        in_gsn_channels: int,
        out_channels: int,
        train_eps: bool = False,
        extra_mlp_in_channel: int = 0,
        mlp_architecture: list[int] | None = None,
    ):
        super().__init__(aggr="sum")

        self.in_channels: int = in_channels
        self.in_gsn_channels: int = in_gsn_channels
        self.out_channels: int = out_channels
        self.train_eps: bool = train_eps
        self.mlp_architecture: list[int] = mlp_architecture
        self.extra_mlp_in_channel: int = extra_mlp_in_channel

        if self.mlp_architecture is None:
            self.UP = torch.nn.Sequential(
                torch.nn.Linear(
                    self.in_channels
                    + self.in_gsn_channels
                    + self.extra_mlp_in_channel,
                    self.out_channels,
                ),
                torch.nn.ReLU(),
            )
        else:
            if len(self.mlp_architecture) < 1:
                raise ValueError()

            self.UP = torch.nn.Sequential()

            for lidx, lsize in enumerate(mlp_architecture):
                ic: int = -1
                oc: int = -1

                if lidx == 0:
                    ic = (
                        self.in_channels
                        + self.in_gsn_channels
                        + self.extra_mlp_in_channel
                    )
                    oc = lsize
                else:
                    if lidx == len(self.mlp_architecture) - 1:
                        ic = self.mlp_architecture[lidx - 1]
                        oc = self.out_channels
                    else:
                        ic = self.mlp_architecture[lidx - 1]
                        oc = lsize

                self.UP.append(torch.nn.Linear(ic, oc))
                self.UP.append(torch.nn.ReLU())

        if self.train_eps:
            self.eps = torch.nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("eps", torch.tensor(0.0))

    @abstractmethod
    def forward(
        self,
        edge_index: torch.Tensor,
        x: torch.Tensor,
        gsn_encodings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute updated node representations.

        Parameters
        ----------
        edge_index : torch.Tensor of shape (2, E)
            Graph connectivity in COO format.
        x : torch.Tensor of shape (N, in_channels)
            Input node features.
        gsn_encodings : torch.Tensor
            GSN structural encodings; shape and semantics depend on subclass.

        Returns
        -------
        torch.Tensor of shape (N, out_channels)
            Updated node representations.
        """


class GSNGINconcatVLayer(GSNGINconcatBaseLayer):
    """
    GSN-GIN layer with node-level structural encoding concatenation.

    Concatenates GSN node encodings with node features prior to GIN-style sum
    aggregation. Self-loops are excluded from the neighbourhood sum and handled
    via the ``(1 + eps) * x'`` self term.
    """

    def forward(
        self,
        edge_index: torch.Tensor,
        x: torch.Tensor,
        gsn_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute updated node representations using node-level GSN encodings.

        Parameters
        ----------
        edge_index : torch.Tensor of shape (2, E)
            Graph connectivity in COO format; self-loops are removed internally.
        x : torch.Tensor of shape (N, in_channels)
            Input node features.
        gsn_embeddings : torch.Tensor of shape (N, in_gsn_channels)
            Node-level GSN structural encodings.

        Returns
        -------
        torch.Tensor of shape (N, out_channels)
            Updated node representations.
        """
        edge_index, _ = remove_self_loops(edge_index=edge_index)

        xprime = torch.cat([x, gsn_embeddings], dim=-1)

        out = self.propagate(edge_index=edge_index, x=xprime)

        out += (1 + self.eps) * xprime

        return self.UP(out)


class GSNGINconcatELayer(GSNGINconcatBaseLayer):
    """
    GSN-GIN layer with edge-level structural encoding concatenation.

    Passes GSN edge encodings as edge attributes during message passing.
    Self-loop edges receive a dummy attribute block with a leading binary
    indicator; the ``(1 + eps)`` self-loop scaling is applied via per-edge
    weights rather than an additive self term.

    Parameters
    ----------
    in_channels : int
        Dimensionality of input node features.
    in_gsn_channels : int
        Dimensionality of GSN structural encodings per edge.
    out_channels : int
        Dimensionality of output node representations.
    train_eps : bool, optional
        If True, epsilon is learnable. Default is False.
    mlp_architecture : list of int or None, optional
        Hidden layer widths for the update MLP. Default is None.

    Notes
    -----
    The MLP input dimension is automatically extended by 1 to accommodate
    the self-loop indicator prepended to each edge attribute vector.
    """

    def __init__(
        self,
        in_channels,
        in_gsn_channels,
        out_channels,
        train_eps=False,
        mlp_architecture=None,
    ):
        super().__init__(
            in_channels,
            in_gsn_channels,
            out_channels,
            train_eps,
            extra_mlp_in_channel=1,
            mlp_architecture=mlp_architecture,
        )

    def forward(
        self,
        edge_index: torch.Tensor,
        x: torch.Tensor,
        gsn_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute updated node representations using edge-level GSN encodings.

        Parameters
        ----------
        edge_index : torch.Tensor of shape (2, E)
            Graph connectivity in COO format; self-loops are added internally.
        x : torch.Tensor of shape (N, in_channels)
            Input node features.
        gsn_embeddings : torch.Tensor of shape (E, in_gsn_channels)
            Edge-level GSN structural encodings (excluding self-loops).

        Returns
        -------
        torch.Tensor of shape (N, out_channels)
            Updated node representations.
        """
        device = x.device

        edge_index, _ = remove_self_loops(edge_index=edge_index)
        num_nodes = x.size(0)

        self_loop_dummy_block = torch.zeros(
            (num_nodes, self.in_gsn_channels + 1), device=device
        )
        self_loop_dummy_block[:, 0] = 1.0

        edge_weights = torch.zeros(
            (num_nodes + gsn_embeddings.size(0), 1), device=device
        )
        edge_weights[gsn_embeddings.size(0) :] = 1.0

        print(edge_weights.requires_grad)

        edge_weights = edge_weights * (1 + self.eps)

        print(edge_weights.requires_grad)

        gsn_embeddings = torch.cat(
            [
                torch.zeros(gsn_embeddings.size(0), 1, device=device),
                gsn_embeddings,
            ],
            dim=-1,
        )
        edge_attr = torch.cat([gsn_embeddings, self_loop_dummy_block], dim=0)

        edge_index, _ = add_self_loops(
            edge_index=edge_index, num_nodes=num_nodes
        )

        out = self.propagate(
            edge_index=edge_index,
            x=x,
            edge_attr=edge_attr,
            edge_weights=edge_weights,
        )

        return self.UP(out)

    def message(self, x_j, edge_attr, edge_weights):
        """
        Construct messages from source nodes.

        Concatenates source node features with edge attributes and scales by
        the per-edge weight, which encodes the ``(1 + eps)`` self-loop factor.

        Parameters
        ----------
        x_j : torch.Tensor of shape (E', in_channels)
            Source node features for each edge in the expanded edge set.
        edge_attr : torch.Tensor of shape (E', in_gsn_channels + 1)
            Edge attributes with a leading binary self-loop indicator column.
        edge_weights : torch.Tensor of shape (E', 1)
            Per-edge scaling weights.

        Returns
        -------
        torch.Tensor of shape (E', in_channels + in_gsn_channels + 1)
            Scaled concatenated messages.
        """
        return torch.cat([x_j, edge_attr], dim=-1) * edge_weights


class GSNGINconcatModel(torch.nn.Module):
    """
    Multi-layer GSN-GIN model with concatenated structural encodings.

    Stacks ``GSNGINconcatVLayer`` or ``GSNGINconcatELayer`` layers depending on
    ``mode``. Structural encodings are read once from the input Data object and
    passed unchanged to every layer.

    Parameters
    ----------
    mode : {'node', 'edge'}
        Whether to use node- or edge-level GSN encodings.
    num_layers : int
        Number of message-passing layers; must be >= 1.
    in_feature_channels : int
        Dimensionality of input node features.
    in_gsn_channels : int
        Dimensionality of GSN structural encodings.
    out_channels : int
        Dimensionality of the final per-node output.
    width : int
        Hidden layer width for intermediate layers.
    mlp_architecture : list of int or None, optional
        Hidden layer widths for each layer's update MLP. Default is None.
    train_eps : bool, optional
        If True, epsilon is learnable in every layer. Default is False.
    gsn_keyword : str or None, optional
        Attribute name on the Data object holding GSN encodings. Defaults to
        ``'node_gsn_encodings'`` or ``'edge_gsn_encodings'`` based on ``mode``.

    Attributes
    ----------
    _layers : torch.nn.ModuleList
        Sequence of GSN-GIN message-passing layers.

    Raises
    ------
    ValueError
        If ``num_layers < 1`` or ``mode`` is not ``'node'`` or ``'edge'``.
    """

    def __init__(
        self,
        mode: LiteralString,
        num_layers: int,
        in_feature_channels: int,
        in_gsn_channels: int,
        out_channels: int,
        width: int,
        mlp_architecture: list[int] | None = None,
        train_eps: bool = False,
        gsn_keyword: str | None = None,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError(
                f"`num_layers` must be at least 1, got {num_layers}."
            )

        if mode not in ["node", "edge"]:
            raise ValueError(
                f"argument `mode` should be either 'node' or 'edge', got '{mode}'."
            )

        self._node_mode = mode == "node"
        self._layer_cls = (
            GSNGINconcatVLayer if self._node_mode else GSNGINconcatELayer
        )
        self.num_layers: int = num_layers
        self.width: int = width
        self.gsn_kword: str = (
            gsn_keyword if gsn_keyword is not None else f"{mode}_gsn_encodings"
        )
        self.in_feature_channels: int = in_feature_channels
        self.in_gsn_channels: int = in_gsn_channels
        self.out_channels: int = out_channels
        self.train_eps: bool = train_eps
        self.mlp_architecture = mlp_architecture

        _layers = []

        for layer_index in range(self.num_layers):
            ic: int = -1
            oc: int = -1

            if layer_index == 0:
                ic = self.in_feature_channels
                oc = self.width if self.num_layers > 1 else self.out_channels
            else:
                if layer_index == self.num_layers - 1:
                    ic = self.width
                    oc = self.out_channels
                else:
                    ic = self.width
                    oc = self.width

            _layers.append(
                self._layer_cls(
                    in_channels=ic,
                    in_gsn_channels=self.in_gsn_channels,
                    out_channels=oc,
                    train_eps=self.train_eps,
                    mlp_architecture=mlp_architecture,
                )
            )

        self._layers = torch.nn.ModuleList(_layers)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Run a full forward pass through all message-passing layers.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph containing node features ``data.x``, edge connectivity
            ``data.edge_index``, and GSN encodings at ``data[gsn_keyword]``.

        Returns
        -------
        torch.Tensor of shape (N, out_channels)
            Node representations after all message-passing layers.
        """
        x, edge_index, gsn_embeddings = (
            data.x,
            data.edge_index,
            data[self.gsn_kword],
        )

        for layer in self._layers:
            x = layer(edge_index, x, gsn_embeddings)

        return x
