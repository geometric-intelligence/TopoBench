"""Pipeline-safe SMCN placeholder backbone."""

import torch


class SMCN(torch.nn.Module):
    """Skeleton Scalable Multi-Cellular Network backbone.

    This placeholder only applies rank-wise linear updates so the TopoBench
    pipeline can validate combinatorial model integration.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        neighborhoods=None,
        layers=1,
        activation="relu",
        use_subcomplex_signal=False,
    ):
        super().__init__()
        self.neighborhoods = neighborhoods or []
        self.layers = layers
        self.use_subcomplex_signal = use_subcomplex_signal

        activation_layer = self._get_activation(activation)
        self.rank_updates = torch.nn.ModuleDict(
            {
                str(rank): self._make_rank_update(
                    in_channels, hidden_channels, layers, activation_layer
                )
                for rank in range(3)
            }
        )

        self.rank02_tuple_encoder = torch.nn.Linear(
            2 * in_channels,
            hidden_channels,
        )

        self.rank02_tuple_update = self._make_rank_update(
            hidden_channels,
            hidden_channels,
            layers,
            activation_layer,
        )

    @staticmethod
    def _get_activation(name):
        activations = {
            "relu": torch.nn.ReLU,
            "gelu": torch.nn.GELU,
            "tanh": torch.nn.Tanh,
            "identity": torch.nn.Identity,
            None: torch.nn.Identity,
        }
        if name not in activations:
            raise ValueError(f"Unsupported activation: {name}")
        return activations[name]

    @staticmethod
    def _make_rank_update(
        in_channels, hidden_channels, layers, activation_layer
    ):
        modules = []
        current_channels = in_channels
        for layer_idx in range(max(layers, 1)):
            modules.append(torch.nn.Linear(current_channels, hidden_channels))
            current_channels = hidden_channels
            if activation_layer is not torch.nn.Identity:
                modules.append(activation_layer())
        return torch.nn.Sequential(*modules)

    def forward(self, batch):
        """Apply placeholder rank-wise updates to available cell features."""
        outputs = {}
        for rank in range(3):
            x = getattr(batch, f"x_{rank}", None)
            if x is not None:
                outputs[rank] = self.rank_updates[str(rank)](x)

        if (
            self.use_subcomplex_signal
            and 0 in outputs
            and hasattr(batch, "incidence_1")
            and hasattr(batch, "incidence_2")
            and hasattr(batch, "x_0")
            and hasattr(batch, "x_2")
        ):
            subcomplex = self.build_rank02_subcomplex(batch)
            pooled_features = self.pool_rank02_to_rank0(
                subcomplex, num_low_cells=batch.x_0.size(0)
            )
            if pooled_features.shape == outputs[0].shape:
                outputs[0] = outputs[0] + pooled_features

        return outputs

    def build_rank02_subcomplex(self, batch):
        """Build the rank-0/2 subcomplex from the rank-1/2 incidence matrices."""
        if not hasattr(batch, "incidence_1") or not hasattr(batch, "incidence_2"):
            raise ValueError(
                "Batch must have incidence_1 and incidence_2 attributes."
            )
        if not hasattr(batch, "x_0") or not hasattr(batch, "x_2"):
            raise ValueError("Batch must have x_0 and x_2 attributes.")

        incidence_0_2 = torch.sparse.mm(
            abs(batch.incidence_1).coalesce(),
            abs(batch.incidence_2).coalesce(),
        ).coalesce()
        if incidence_0_2._nnz() > 0:
            incidence_0_2 = torch.sparse_coo_tensor(
                incidence_0_2.indices(),
                torch.ones_like(incidence_0_2.values()),
                incidence_0_2.size(),
                device=incidence_0_2.device,
            ).coalesce()

        num_low_cells = batch.x_0.size(0)
        num_high_cells = batch.x_2.size(0)
        if (
            incidence_0_2.size(0) != num_low_cells
            or incidence_0_2.size(1) != num_high_cells
        ):
            raise ValueError(
                "Incidence matrix shape mismatch: "
                f"expected ({num_low_cells}, {num_high_cells}), "
                f"got {incidence_0_2.size()}"
            )

        device = batch.x_0.device
        low_indices = torch.arange(num_low_cells, device=device).repeat_interleave(
            num_high_cells
        )
        high_indices = torch.arange(num_high_cells, device=device).repeat(
            num_low_cells
        )

        if hasattr(batch, "batch_0") and hasattr(batch, "batch_2"):
            same_graph = batch.batch_0[low_indices] == batch.batch_2[high_indices]
            low_indices = low_indices[same_graph]
            high_indices = high_indices[same_graph]

        binary_marking = incidence_0_2.to_dense()[low_indices, high_indices]
        tuple_inputs = torch.cat(
            [batch.x_0[low_indices], batch.x_2[high_indices]], dim=-1
        )
        tuple_features = self.rank02_tuple_encoder(tuple_inputs)

        tuple_features = self.rank02_tuple_update(tuple_features)
        return {
            "incidence_0_2": incidence_0_2,
            "low_indices": low_indices,
            "high_indices": high_indices,
            "binary_marking": binary_marking,
            "tuple_features": tuple_features,
        }

    @staticmethod
    def pool_rank02_to_rank0(subcomplex, num_low_cells):
        """Pool rank-0/2 tuple features back to rank-0 cells by summation."""
        tuple_features = subcomplex["tuple_features"]
        low_indices = subcomplex["low_indices"]
        pooled = tuple_features.new_zeros(
            (num_low_cells, tuple_features.size(-1))
        )
        if tuple_features.numel() == 0:
            return pooled
        return pooled.index_add(0, low_indices, tuple_features)
