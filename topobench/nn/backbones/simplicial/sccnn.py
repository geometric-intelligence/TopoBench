"""Implementation of the Simplicial Complex Convolutional Neural Network (SCCNN) for complex classification."""

import torch
from torch.nn.parameter import Parameter


class SCCNNCustom(torch.nn.Module):
    """SCCNN implementation for complex classification.

    Note: In this task, we can consider the output on any order of simplices for the
    classification task, which of course can be amended by a readout layer.

    Parameters
    ----------
    in_channels_all : tuple of int
        Dimension of input features on each rank (nodes, edges, faces, ...).
    hidden_channels_all : tuple of int
        Dimension of features of hidden layers on each rank.
    conv_order : int
        Order of convolutions, we consider the same order for all convolutions.
    sc_order : int
        Order of simplicial complex (max_rank + 1).
    aggr_norm : bool, optional
        Whether to normalize the aggregation (default: False).
    update_func : str, optional
        Update function for the simplicial complex convolution (default: None).
    n_layers : int, optional
        Number of layers (default: 2).
    """

    def __init__(
        self,
        in_channels_all,
        hidden_channels_all,
        conv_order,
        sc_order,
        aggr_norm=False,
        update_func=None,
        n_layers=2,
    ):
        super().__init__()

        self.max_rank = len(in_channels_all) - 1

        # Create input linear layers for each rank dynamically
        self.in_linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_channels_all[i], hidden_channels_all[i])
                for i in range(len(in_channels_all))
            ]
        )

        self.layers = torch.nn.ModuleList(
            SCCNNLayer(
                in_channels=hidden_channels_all,
                out_channels=hidden_channels_all,
                conv_order=conv_order,
                sc_order=sc_order,
                aggr_norm=aggr_norm,
                update_func=update_func,
            )
            for _ in range(n_layers)
        )

    def forward(self, x_all, laplacian_all, incidence_all):
        """Forward computation.

        Parameters
        ----------
        x_all : tuple(tensors)
            Tuple of feature tensors for each rank (x_0, x_1, ..., x_k).
        laplacian_all : tuple(tensors)
            Tuple of Laplacian tensors.
        incidence_all : tuple(tensors)
            Tuple of incidence matrices.

        Returns
        -------
        tuple(tensors)
            Tuple of final hidden state tensors for each rank.
        """
        # Apply input linear transformations to each rank
        x_all_transformed = tuple(
            self.in_linears[i](x_all[i]) for i in range(len(x_all))
        )

        # Forward through SCCNN layers
        for layer in self.layers:
            x_all_transformed = layer(
                x_all_transformed, laplacian_all, incidence_all
            )

        return x_all_transformed


class SCCNNLayer(torch.nn.Module):
    r"""Layer of a Simplicial Complex Convolutional Neural Network.

    Parameters
    ----------
    in_channels : tuple of int
        Dimensions of input features for each rank.
    out_channels : tuple of int
        Dimensions of output features for each rank.
    conv_order : int
        Convolution order of the simplicial filters.
    sc_order : int
        SC order (max_rank + 1).
    aggr_norm : bool, optional
        Whether to normalize the aggregated message by the neighborhood size (default: False).
    update_func : str, optional
        Activation function used in aggregation layers (default: None).
    initialization : str, optional
        Initialization method for the weights (default: "xavier_normal").
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_order,
        sc_order,
        aggr_norm: bool = False,
        update_func=None,
        initialization: str = "xavier_normal",
    ) -> None:
        super().__init__()

        self.in_channels = tuple(in_channels)
        self.out_channels = tuple(out_channels)
        self.max_rank = len(in_channels) - 1

        self.conv_order = conv_order
        self.sc_order = sc_order

        self.aggr_norm = aggr_norm
        self.update_func = update_func
        self.initialization = initialization

        assert initialization in ["xavier_uniform", "xavier_normal"]
        assert self.conv_order > 0

        # Create weight parameters for each rank
        self.weights = torch.nn.ParameterList()

        for rank in range(self.max_rank + 1):
            # Calculate weight tensor dimensions based on message types
            # For rank k, we have:
            # - Identity: 1
            # - Self convolutions: conv_order (down) + conv_order (up) for k>0, or just conv_order for k=0
            # - Lower messages (from k-1): 1 + conv_order (identity + convolutions)
            # - Upper messages (from k+1): 1 + conv_order (identity + convolutions)

            num_message_types = self._compute_message_types(rank)

            weight = Parameter(
                torch.Tensor(
                    self.in_channels[rank],
                    self.out_channels[rank],
                    num_message_types,
                )
            )
            self.weights.append(weight)

        self.reset_parameters()

    def _compute_message_types(self, rank):
        """Compute the maximum number of message types for a given rank.

        Parameters
        ----------
        rank : int
            Rank to consider.

        Returns
        -------
        int
            Number of message types for the given rank.
        """
        count = 0

        # Self messages
        if rank == 0:
            # Rank 0: identity + up Laplacian convolutions
            count += 1 + self.conv_order
        else:
            # Rank k>0: identity + down Laplacian + up Laplacian convolutions
            count += 1 + self.conv_order + self.conv_order

        # Lower messages (from rank-1 projected to rank)
        if rank > 0:
            # Identity + convolutions with down and up Laplacians at current rank
            count += 1 + self.conv_order + self.conv_order

        # Upper messages (from rank+1 projected to rank)
        if rank < self.max_rank:
            # Identity + convolutions with down and up Laplacians at current rank
            # Special case: rank 0 only has up Laplacian
            if rank == 0:
                count += 1 + self.conv_order
            else:
                count += 1 + self.conv_order + self.conv_order

        return count

    def reset_parameters(self, gain: float = 1.414):
        r"""Reset learnable parameters.

        Parameters
        ----------
        gain : float
            Gain for the weight initialization.
        """
        if self.initialization == "xavier_uniform":
            for weight in self.weights:
                torch.nn.init.xavier_uniform_(weight, gain=gain)
        elif self.initialization == "xavier_normal":
            for weight in self.weights:
                torch.nn.init.xavier_normal_(weight, gain=gain)
        else:
            raise RuntimeError(
                "Initialization method not recognized. "
                "Should be either xavier_uniform or xavier_normal."
            )

    def aggr_norm_func(self, conv_operator, x):
        r"""Perform aggregation normalization.

        Parameters
        ----------
        conv_operator : torch.sparse
            Convolution operator.
        x : torch.Tensor
            Feature tensor.

        Returns
        -------
        torch.Tensor
            Normalized feature tensor.
        """
        neighborhood_size = torch.sum(conv_operator.to_dense(), dim=1)
        neighborhood_size_inv = 1 / neighborhood_size
        neighborhood_size_inv[~(torch.isfinite(neighborhood_size_inv))] = 0

        x = torch.einsum("i,ij->ij ", neighborhood_size_inv, x)
        x[~torch.isfinite(x)] = 0
        return x

    def update(self, x):
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
        return None

    def chebyshev_conv(self, conv_operator, conv_order, x):
        r"""Perform Chebyshev convolution.

        Parameters
        ----------
        conv_operator : torch.sparse
            Convolution operator.
        conv_order : int
            Order of the convolution.
        x : torch.Tensor
            Feature tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        num_simplices, num_channels = x.shape
        X = torch.empty(size=(num_simplices, num_channels, conv_order)).to(
            x.device
        )

        if self.aggr_norm:
            X[:, :, 0] = torch.mm(conv_operator, x)
            X[:, :, 0] = self.aggr_norm_func(conv_operator, X[:, :, 0])
            for k in range(1, conv_order):
                X[:, :, k] = torch.mm(conv_operator, X[:, :, k - 1])
                X[:, :, k] = self.aggr_norm_func(conv_operator, X[:, :, k])
        else:
            X[:, :, 0] = torch.mm(conv_operator, x)
            for k in range(1, conv_order):
                X[:, :, k] = torch.mm(conv_operator, X[:, :, k - 1])
        return X

    def forward(self, x_all, laplacian_all, incidence_all):
        r"""Forward computation for arbitrary ranks.

        Parameters
        ----------
        x_all : tuple of tensors
            Tuple of input feature tensors for each rank.
        laplacian_all : tuple of tensors
            Tuple of Laplacian tensors organized as:
            (L_0, L_down_1, L_up_1, L_down_2, L_up_2, ...).
        incidence_all : tuple of tensors
            Tuple of incidence matrices (B_1, B_2, ..., B_k).

        Returns
        -------
        tuple of tensors
            Output tensors for each rank after message passing.
        """
        outputs = []

        for rank in range(self.max_rank + 1):
            x_rank = x_all[rank]

            # Skip empty ranks (no cells at this dimension)
            if x_rank.shape[0] == 0:
                # Create empty output tensor for this rank
                outputs.append(
                    torch.zeros(
                        0, self.out_channels[rank], device=x_rank.device
                    )
                )
                continue

            # Get Laplacians for this rank
            laplacians = self._get_laplacians_for_rank(rank, laplacian_all)

            # Get incidence matrices
            incidence_lower = (
                incidence_all[rank - 1]
                if rank > 0 and rank - 1 < len(incidence_all)
                else None
            )
            incidence_upper = (
                incidence_all[rank] if rank < len(incidence_all) else None
            )

            # Compute all messages for this rank
            messages = self._compute_messages_for_rank(
                rank,
                x_rank,
                x_all,
                laplacians,
                incidence_lower,
                incidence_upper,
            )

            # Apply weight and aggregate
            # Use only the first k dimensions of weights that match the number of messages
            num_messages = messages.shape[2]
            weight_slice = self.weights[rank][:, :, :num_messages]
            y_rank = torch.einsum("nik,iok->no", messages, weight_slice)

            # Apply activation if specified
            if self.update_func is not None:
                y_rank = self.update(y_rank)

            outputs.append(y_rank)

        return tuple(outputs)

    def _get_laplacians_for_rank(self, rank, laplacian_all):
        """Extract Laplacians for a given rank from laplacian_all.

        Parameters
        ----------
        rank : int
            The rank to extract Laplacians for.
        laplacian_all : tuple
            All Laplacians organized as (L_0, L_down_1, L_up_1, L_down_2, L_up_2, ...).

        Returns
        -------
        dict
            Dictionary with keys 'hodge', 'down', 'up' containing the relevant Laplacians.
        """
        laplacians = {}

        if rank == 0:
            # Rank 0 only has Hodge Laplacian
            laplacians["hodge"] = (
                laplacian_all[0] if len(laplacian_all) > 0 else None
            )
            laplacians["down"] = None
            laplacians["up"] = None
        else:
            # For rank k > 0: index is 1 + 2*(k-1) for down, 1 + 2*(k-1) + 1 for up
            idx_down = 1 + 2 * (rank - 1)
            idx_up = idx_down + 1

            laplacians["hodge"] = None
            laplacians["down"] = (
                laplacian_all[idx_down]
                if idx_down < len(laplacian_all)
                else None
            )
            laplacians["up"] = (
                laplacian_all[idx_up] if idx_up < len(laplacian_all) else None
            )

        return laplacians

    def _compute_messages_for_rank(
        self, rank, x_rank, x_all, laplacians, incidence_lower, incidence_upper
    ):
        """Compute all messages for a given rank.

        Parameters
        ----------
        rank : int
            The rank to compute messages for.
        x_rank : tensor
            Features of cells at this rank.
        x_all : tuple
            Features of all ranks.
        laplacians : dict
            Dictionary of Laplacians for this rank.
        incidence_lower : tensor or None
            Incidence matrix from rank-1 to rank.
        incidence_upper : tensor or None
            Incidence matrix from rank to rank+1.

        Returns
        -------
        tensor
            Concatenated messages of shape (num_cells, num_channels, num_message_types).
        """
        message_list = []

        # 1. Lower messages (from rank-1)
        if rank > 0 and incidence_lower is not None and rank - 1 < len(x_all):
            x_lower = x_all[rank - 1]
            # Only process if lower rank is not empty
            if x_lower.shape[0] > 0:
                # Project features from rank-1 to rank
                x_lower_proj = torch.mm(incidence_lower.T, x_lower)

                message_list.append(x_lower_proj.unsqueeze(2))

                # Apply down and up Laplacians
                if laplacians["down"] is not None:
                    x_lower_down = self.chebyshev_conv(
                        laplacians["down"], self.conv_order, x_lower_proj
                    )
                    message_list.append(x_lower_down)

                if laplacians["up"] is not None:
                    x_lower_up = self.chebyshev_conv(
                        laplacians["up"], self.conv_order, x_lower_proj
                    )
                    message_list.append(x_lower_up)

        # 2. Self messages (identity + convolutions)
        if rank == 0:
            message_list.append(x_rank.unsqueeze(2))
            if laplacians["hodge"] is not None:
                x_conv = self.chebyshev_conv(
                    laplacians["hodge"], self.conv_order, x_rank
                )
                message_list.append(x_conv)
        else:
            message_list.append(x_rank.unsqueeze(2))
            if laplacians["down"] is not None:
                x_down = self.chebyshev_conv(
                    laplacians["down"], self.conv_order, x_rank
                )
                message_list.append(x_down)
            if laplacians["up"] is not None:
                x_up = self.chebyshev_conv(
                    laplacians["up"], self.conv_order, x_rank
                )
                message_list.append(x_up)

        # 3. Upper messages (from rank+1)
        if (
            rank < self.max_rank
            and incidence_upper is not None
            and rank + 1 < len(x_all)
        ):
            x_upper = x_all[rank + 1]
            if x_upper.shape[0] > 0:
                x_upper_proj = torch.mm(incidence_upper, x_upper)
                message_list.append(x_upper_proj.unsqueeze(2))

                # Apply Laplacians (Hodge for rank 0, both down/up for rank > 0)
                if rank == 0:
                    if laplacians["hodge"] is not None:
                        x_upper_hodge = self.chebyshev_conv(
                            laplacians["hodge"], self.conv_order, x_upper_proj
                        )
                        message_list.append(x_upper_hodge)
                else:
                    if laplacians["down"] is not None:
                        x_upper_down = self.chebyshev_conv(
                            laplacians["down"], self.conv_order, x_upper_proj
                        )
                        message_list.append(x_upper_down)

                    if laplacians["up"] is not None:
                        x_upper_up = self.chebyshev_conv(
                            laplacians["up"], self.conv_order, x_upper_proj
                        )
                        message_list.append(x_upper_up)

        # Concatenate all messages
        messages = torch.cat(message_list, dim=2)

        return messages
