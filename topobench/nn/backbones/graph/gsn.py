"""
Graph Substructure Network (GSN) message-passing layers and models.

Implements the GSN family of structurally-aware message-passing layers from
Bouritsas et al. (2022), in which precomputed substructure (orbit) counts are
injected into the aggregation as node- or edge-level structural encodings.

Two flavours are provided:

- GIN-concat layers (`GSNGINconcatVLayer`, `GSNGINconcatELayer`) and the
  `GSNGINconcatModel` that stacks them. These use GIN-style sum aggregation
  with a learnable ``(1 + eps)`` self term, and inject the structural
  encodings by concatenation: node encodings are concatenated to node
  features (node mode), or edge encodings are passed as edge attributes
  (edge mode).
- A general MPNN baseline layer (`GSNMPNNBaselineVLayer`) implementing the
  full GSN-v scheme with separate inner (message) and outer (update) MLPs,
  without the GIN simplification.

The module-level helpers `mlp_dimension_builder` and `mlp_builder` assemble
the per-layer MLPs shared by these layers.
"""

from abc import ABC, abstractmethod
from typing import Literal

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
    edge_dim : int, optional
        Dimensionality of the original edge features carried in the message
        (used by edge-mode subclasses). Added to the MLP input width. Default
        is 0.
    train_eps : bool, optional
        If True, the epsilon scaling parameter is learnable. Default is False.
    extra_mlp_in_channel : int, optional
        Additional input channels prepended to the MLP input (used by
        edge-mode subclasses). Default is 0.
    mlp_architecture : list of int or None, optional
        Hidden layer widths for the update MLP. If None, a single linear
        layer is used.
    initial_eps : float, optional
        Initial value of the epsilon scaling parameter. Default is 0.0.

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
        edge_dim: int = 0,
        train_eps: bool = False,
        extra_mlp_in_channel: int = 0,
        mlp_architecture: list[int] | None = None,
        initial_eps: float = 0.0,
    ):
        super().__init__(aggr="sum")

        self.in_channels: int = in_channels
        self.in_gsn_channels: int = in_gsn_channels
        self.out_channels: int = out_channels
        self.train_eps: bool = train_eps
        self.extra_mlp_in_channel: int = extra_mlp_in_channel
        self._initial_eps = initial_eps
        self.edge_dim = edge_dim

        tmp_architecture = [] if mlp_architecture is None else mlp_architecture
        self.mlp_architecture = (
            [
                self.in_channels
                + self.in_gsn_channels
                + self.extra_mlp_in_channel
                + self.edge_dim
            ]
            + tmp_architecture
            + [self.out_channels]
        )

        self.UP = mlp_builder(mlp_dimension_builder(self.mlp_architecture))

        if self.train_eps:
            self.eps = torch.nn.Parameter(torch.tensor(self._initial_eps))
        else:
            self.register_buffer("eps", torch.tensor(self._initial_eps))

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

    Parameters
    ----------
    in_channels : int
        Dimensionality of input node features.
    in_gsn_channels : int
        Dimensionality of GSN structural encodings.
    out_channels : int
        Dimensionality of output node representations.
    edge_dim : int, optional
        Must be 0; node-mode GSN does not consume edge features. A value
        greater than 0 raises ``ValueError``. Default is 0.
    train_eps : bool, optional
        If True, the epsilon scaling parameter is learnable. Default is False.
    extra_mlp_in_channel : int, optional
        Additional input channels prepended to the MLP input. Default is 0.
    mlp_architecture : list of int or None, optional
        Hidden layer widths for the update MLP. If None, a single linear
        layer is used. Default is None.
    initial_eps : float, optional
        Initial value of the epsilon scaling parameter. Default is 0.0.
    """

    def __init__(
        self,
        in_channels,
        in_gsn_channels,
        out_channels,
        edge_dim=0,
        train_eps=False,
        extra_mlp_in_channel=0,
        mlp_architecture=None,
        initial_eps=0.0,
    ):
        """
        Initialize the node-mode GSN-GIN layer.

        Accepts the same arguments as `GSNGINconcatBaseLayer`. Node-mode GSN
        does not consume edge features, so ``edge_dim`` must be 0; it is
        forwarded as 0 to the base class regardless.

        Raises
        ------
        ValueError
            If ``edge_dim > 0``, since this layer does not support edge
            features.
        """
        if edge_dim > 0:
            raise ValueError(
                "Layer `GSNGINconcatVLayer` is incompatible with `edge_dim`>0"
            )

        super().__init__(
            in_channels=in_channels,
            in_gsn_channels=in_gsn_channels,
            out_channels=out_channels,
            edge_dim=0,  # explicitly set this to 0
            train_eps=train_eps,
            extra_mlp_in_channel=extra_mlp_in_channel,
            mlp_architecture=mlp_architecture,
            initial_eps=initial_eps,
        )

    def forward(
        self,
        edge_index: torch.Tensor,
        x: torch.Tensor,
        gsn_embeddings: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
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
        edge_attr : torch.Tensor or None, optional
            Accepted for interface compatibility with the edge-mode layer, but
            ignored: node-mode GSN does not use edge features. Default is None.

        Returns
        -------
        torch.Tensor of shape (N, out_channels)
            Updated node representations.
        """
        edge_index, _ = remove_self_loops(edge_index=edge_index)

        xprime = torch.cat([x, gsn_embeddings], dim=-1)

        out = self.propagate(edge_index=edge_index, x=xprime)

        out = out + (1 + self.eps) * xprime

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
    edge_dim : int, optional
        Dimensionality of the original edge features. If greater than 0, an
        ``edge_attr`` tensor of this width must be supplied to ``forward`` and
        is concatenated into each message. Default is 0 (no edge features).

    Notes
    -----
    The MLP input dimension is automatically extended by 1 to accommodate
    the self-loop indicator prepended to each edge attribute vector, plus
    ``edge_dim`` additional channels when original edge features are used.
    """

    def __init__(
        self,
        in_channels,
        in_gsn_channels,
        out_channels,
        train_eps=False,
        mlp_architecture=None,
        edge_dim: int = 0,
    ):
        super().__init__(
            in_channels=in_channels,
            in_gsn_channels=in_gsn_channels,
            out_channels=out_channels,
            train_eps=train_eps,
            extra_mlp_in_channel=1,
            mlp_architecture=mlp_architecture,
            edge_dim=edge_dim,
        )

    def forward(
        self,
        edge_index: torch.Tensor,
        x: torch.Tensor,
        gsn_embeddings: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
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
        edge_attr : torch.Tensor of shape (E, edge_dim) or None, optional
            Original edge features (excluding self-loops). Must be provided
            with width ``edge_dim`` if and only if ``edge_dim > 0``. Default
            is None.

        Returns
        -------
        torch.Tensor of shape (N, out_channels)
            Updated node representations.

        Raises
        ------
        ValueError
            If ``gsn_embeddings`` does not align with ``edge_index``, or if the
            presence/width of ``edge_attr`` is inconsistent with ``edge_dim``.
        """
        device = x.device

        # remove potentially existing self-loops and remove them from edge_attr
        # we dont have to remove them from GSN encodings, as the encoder
        # itself removes self-loops prior to computing GSN encodings
        edge_index, edge_attr = remove_self_loops(
            edge_index=edge_index, edge_attr=edge_attr
        )

        if not (gsn_embeddings.size(0) == edge_index.size(1)):
            raise ValueError(
                f"Size of GSN Embeddings ({gsn_embeddings.size(0)}) doesn't match edge_index ({edge_index.size(1)})"
            )

        if edge_attr is not None and edge_attr.size(1) != self.edge_dim:
            raise ValueError(
                f"edge_attr width {edge_attr.size(1)} != edge_dim {self.edge_dim}"
            )

        if edge_attr is None and self.edge_dim != 0:
            raise ValueError("edge_dim > 0 but no edge_attr was provided")

        num_nodes = x.size(0)

        self_loop_dummy_block = torch.zeros(
            (num_nodes, self.in_gsn_channels + 1), device=device
        )
        self_loop_dummy_block[:, 0] = 1.0

        # here we build the edge weights: 1+eps for self-loops, 1 for non-self-loops
        self_weights = torch.ones((num_nodes, 1), device=device) * (
            1 + self.eps
        )
        normal_weights = torch.ones((edge_index.size(1), 1), device=device)
        edge_weights = torch.cat([normal_weights, self_weights], dim=0)

        gsn_embeddings = torch.cat(
            [
                torch.zeros(gsn_embeddings.size(0), 1, device=device),
                gsn_embeddings,
            ],
            dim=-1,
        )

        full_edge_attr = torch.cat(
            [gsn_embeddings, self_loop_dummy_block], dim=0
        )

        if edge_attr is not None:
            # we need to add a zero block to the original edge features
            edge_attr = torch.cat(
                [
                    edge_attr,
                    torch.zeros(num_nodes, edge_attr.size(1), device=device),
                ]
            )
            full_edge_attr = torch.cat([full_edge_attr, edge_attr], dim=-1)

        edge_index, _ = add_self_loops(
            edge_index=edge_index, num_nodes=num_nodes
        )

        out = self.propagate(
            edge_index=edge_index,
            x=x,
            edge_attr=full_edge_attr,
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
        edge_attr : torch.Tensor of shape (E', in_gsn_channels + 1 + edge_dim)
            Edge attributes with a leading binary self-loop indicator column,
            the GSN structural encodings, and (when ``edge_dim > 0``) the
            original edge features.
        edge_weights : torch.Tensor of shape (E', 1)
            Per-edge scaling weights.

        Returns
        -------
        torch.Tensor of shape (E', in_channels + in_gsn_channels + 1 + edge_dim)
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
    edge_dim : int, optional
        Dimensionality of the original edge features (``data.edge_attr``) fed
        to every layer. Only supported in edge mode; must be 0 in node mode.
        Default is 0.

    Attributes
    ----------
    _layers : torch.nn.ModuleList
        Sequence of GSN-GIN message-passing layers.

    Raises
    ------
    ValueError
        If ``num_layers < 1``, ``mode`` is not ``'node'`` or ``'edge'``, or
        ``mode`` is ``'node'`` while ``edge_dim > 0``.
    """

    def __init__(
        self,
        mode: Literal["node", "edge"],
        num_layers: int,
        in_feature_channels: int,
        in_gsn_channels: int,
        out_channels: int,
        width: int,
        mlp_architecture: list[int] | None = None,
        train_eps: bool = False,
        gsn_keyword: str | None = None,
        edge_dim: int = 0,
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

        if (mode == "node") and (edge_dim > 0):
            raise ValueError(
                f"mode `node` is incompatible with `edge_dim` > 0 (got {edge_dim})"
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
        self.edge_dim = edge_dim

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
                    edge_dim=edge_dim,
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
            When present, ``data.edge_attr`` is forwarded to every layer as the
            original edge features (relevant only in edge mode with
            ``edge_dim > 0``).

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

        edge_attr: torch.Tensor | None = None
        if hasattr(data, "edge_attr"):
            edge_attr = data.edge_attr

        for layer in self._layers:
            x = layer(
                edge_index=edge_index,
                x=x,
                gsn_embeddings=gsn_embeddings,
                edge_attr=edge_attr,
            )

        return x


def mlp_dimension_builder(layer_dims: list[int]) -> list[tuple[int, int]]:
    """
    Turn a flat MLP architecture into per-layer (in, out) channel pairs.

    Converts a list of layer dimensions (input dimension, hidden widths,
    output dimension) into the list of ``(in_channels, out_channels)``
    tuples describing each consecutive linear layer.

    Parameters
    ----------
    layer_dims : list of int
        Layer dimensions, ordered from input to output. Must contain at
        least two entries (the input and output dimension).

    Returns
    -------
    list of (int, int)
        One ``(in_channels, out_channels)`` tuple per linear layer, i.e.
        ``len(layer_dims) - 1`` tuples.
    """
    return [
        (layer_dims[j], layer_dims[j + 1]) for j in range(len(layer_dims) - 1)
    ]


def mlp_builder(
    layer_dims: list[tuple[int, int]], activation_fn=torch.nn.ReLU
) -> torch.nn.Module:
    """
    Build a sequential MLP from per-layer (in, out) channel pairs.

    Stacks one ``torch.nn.Linear`` per entry in ``layer_dims``, inserting
    an activation function between consecutive linear layers. No activation
    is appended after the final linear layer, so the MLP output is
    unbounded.

    Parameters
    ----------
    layer_dims : list of (int, int)
        One ``(in_channels, out_channels)`` tuple per linear layer, as
        produced by `mlp_dimension_builder`.
    activation_fn : type of torch.nn.Module, optional
        Activation module class instantiated between consecutive linear
        layers. Default is ``torch.nn.ReLU``.

    Returns
    -------
    torch.nn.Sequential
        The assembled MLP, with activations between (but not after) the
        linear layers.
    """
    model = torch.nn.Sequential()
    for lidx, (ic, oc) in enumerate(layer_dims):
        model.append(torch.nn.Linear(ic, oc))

        # check if we are in the last layer already, then dont append activation fn
        if lidx < len(layer_dims) - 1:
            model.append(activation_fn())

    return model


class GSNMPNNBaselineVLayer(MessagePassing):
    """
    General MPNN baseline GSN layer with node-level structural encodings.

    Implements the full Graph Substructure Network vertex variant (GSN-v)
    message-passing scheme of Bouritsas et al. (2022), without the GIN
    simplification used by `GSNGINconcatVLayer`. Messages are produced by an
    inner MLP from the source and target node features, their GSN node
    encodings, and (optionally) edge attributes; aggregated messages are then
    combined with the target node features by an outer MLP:

    ``m_uv = INNER([h_v, h_u, x_v, x_u, e_uv])``
    ``h_v' = OUTER([h_v, sum_u m_uv])``

    where ``x_*`` are the GSN structural encodings and ``e_uv`` the edge
    attributes.

    Parameters
    ----------
    in_channels : int
        Dimensionality of input node features ``h``.
    out_channels : int
        Dimensionality of output node representations.
    gsn_channels : int
        Dimensionality of the per-node GSN structural encodings.
    edge_dim : int, optional
        Dimensionality of edge attributes. If 0 (default), edge attributes
        are not used and must not be passed to ``forward``.
    message_dim : int or None, optional
        Dimensionality of the messages produced by the inner MLP. If None
        (default), ``in_channels`` is used.
    inner_mlp_architecture : list of int or None, optional
        Hidden layer widths of the inner (message) MLP. If None (default),
        the inner MLP is a single linear layer.
    outer_mlp_architecture : list of int or None, optional
        Hidden layer widths of the outer (update) MLP. If None (default),
        the outer MLP is a single linear layer.

    Attributes
    ----------
    inner_mlp : torch.nn.Sequential
        MLP mapping ``[h_v, h_u, x_v, x_u, e_uv]`` to a message of size
        ``message_dim``.
    outer_mlp : torch.nn.Sequential
        MLP mapping ``[h_v, aggregated_message]`` to the ``out_channels``
        output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gsn_channels: int,
        edge_dim: int = 0,
        message_dim: int | None = None,
        inner_mlp_architecture: list[int] | None = None,
        outer_mlp_architecture: list[int] | None = None,
    ):
        super().__init__(aggr="sum")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gsn_channels = gsn_channels
        self.edge_dim = edge_dim
        self.message_dim = in_channels if message_dim is None else message_dim

        # here we create the inner MLP for creating the messages that then summed

        # inner MLP: [h^t_v; h^t_u; x_v; x_u, e_uv;] -> message_dim
        # inner MLP: 2xin_channels+edge_channels+2xGSN_channels -> message_dim
        ima = [] if inner_mlp_architecture is None else inner_mlp_architecture
        ima = (
            [2 * self.in_channels + 2 * self.gsn_channels + edge_dim]
            + ima
            + [self.message_dim]
        )
        ima = mlp_dimension_builder(ima)

        self.inner_mlp = mlp_builder(ima)

        # outer MLP: [h^t_v; m^(t+1)_v] -> out
        # outer MLP: [in_features; VARIABLE_WIDTH] -> out_channels
        oma = [] if outer_mlp_architecture is None else outer_mlp_architecture
        oma = [self.in_channels + self.message_dim] + oma + [self.out_channels]
        oma = mlp_dimension_builder(oma)

        self.outer_mlp = mlp_builder(oma)

    def forward(
        self,
        edge_index: torch.Tensor,
        x: torch.Tensor,
        gsn_embeddings: torch.Tensor,
        edge_attr: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute updated node representations via inner/outer MLP message passing.

        Parameters
        ----------
        edge_index : torch.Tensor of shape (2, E)
            Graph connectivity in COO format. Self-loops, if desired, must
            already be present in ``edge_index``.
        x : torch.Tensor of shape (N, in_channels)
            Input node features.
        gsn_embeddings : torch.Tensor of shape (N, gsn_channels)
            Per-node GSN structural encodings.
        edge_attr : torch.Tensor of shape (E, edge_dim) or None, optional
            Edge attributes. Must be provided if and only if ``edge_dim > 0``.

        Returns
        -------
        torch.Tensor of shape (N, out_channels)
            Updated node representations.

        Raises
        ------
        ValueError
            If the presence of ``edge_attr`` does not match ``edge_dim > 0``.
        """
        if (edge_attr is not None) != (self.edge_dim > 0):
            raise ValueError("edge_attr presence must match edge_dim > 0")

        out = self.propagate(
            edge_index=edge_index, x=x, gsn=gsn_embeddings, edge_attr=edge_attr
        )
        out = torch.cat([x, out], dim=-1)

        return self.outer_mlp(out)

    def message(self, x_i, x_j, gsn_i, gsn_j, edge_attr=None):
        """
        Construct messages from source-target node and edge information.

        Concatenates the target node features ``x_i``, source node features
        ``x_j``, their respective GSN encodings ``gsn_i`` / ``gsn_j``, and
        (optionally) the edge attributes, then maps the result through the
        inner MLP.

        Parameters
        ----------
        x_i : torch.Tensor of shape (E, in_channels)
            Target node features for each edge.
        x_j : torch.Tensor of shape (E, in_channels)
            Source node features for each edge.
        gsn_i : torch.Tensor of shape (E, gsn_channels)
            GSN structural encodings of the target nodes.
        gsn_j : torch.Tensor of shape (E, gsn_channels)
            GSN structural encodings of the source nodes.
        edge_attr : torch.Tensor of shape (E, edge_dim) or None, optional
            Edge attributes. Concatenated to the message input when provided.

        Returns
        -------
        torch.Tensor of shape (E, message_dim)
            Per-edge messages produced by the inner MLP.
        """
        if edge_attr is None:
            mlp_input = torch.cat([x_i, x_j, gsn_i, gsn_j], dim=-1)

            return self.inner_mlp(mlp_input)

        return self.inner_mlp(
            torch.cat([x_i, x_j, gsn_i, gsn_j, edge_attr], dim=-1)
        )
