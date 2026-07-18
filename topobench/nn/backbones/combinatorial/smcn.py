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
        tuple_pooling="sum",
        tuple_selection="all",
        marking_embed_dim=0,
    ):
        super().__init__()
        self.neighborhoods = neighborhoods or []
        self.layers = layers
        self.use_subcomplex_signal = use_subcomplex_signal
        if tuple_pooling not in {"sum", "mean"}:
            raise ValueError(f"Unsupported tuple_pooling: {tuple_pooling}")
        self.tuple_pooling = tuple_pooling

        if tuple_selection not in {"all", "incident"}:
            raise ValueError(f"Unsupported tuple_selection: {tuple_selection}")
        self.tuple_selection = tuple_selection

        activation_layer = self._get_activation(activation)
        self.rank_updates = torch.nn.ModuleDict(
            {
                str(rank): self._make_rank_update(
                    in_channels, hidden_channels, layers, activation_layer
                )
                for rank in range(3)
            }
        )
        if marking_embed_dim < 0:
            raise ValueError(
                f"marking_embed_dim must be non-negative, got {marking_embed_dim}"
            )
        self.marking_embed_dim = marking_embed_dim

        marking_channels = marking_embed_dim if marking_embed_dim > 0 else 1
        self.rank02_marking_embed = (
            torch.nn.Embedding(2, marking_embed_dim)
            if marking_embed_dim > 0
            else None
        )

        self.rank02_tuple_encoder = torch.nn.Linear(
            2 * in_channels + marking_channels,
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

    @staticmethod
    def _lookup_sparse_binary_marking(incidence, low_indices, high_indices):
        """Look up binary incidence values for selected sparse matrix entries."""
        if low_indices.numel() == 0:
            return torch.empty(
                0, dtype=incidence.dtype, device=low_indices.device
            )

        incidence = incidence.coalesce()
        if incidence._nnz() == 0:
            return torch.zeros(
                low_indices.size(0),
                dtype=incidence.dtype,
                device=low_indices.device,
            )

        num_high_cells = incidence.size(1)
        tuple_keys = low_indices * num_high_cells + high_indices
        incident_indices = incidence.indices()
        incident_keys = (
            incident_indices[0] * num_high_cells + incident_indices[1]
        ).unique(sorted=True)
        positions = torch.searchsorted(incident_keys, tuple_keys)
        in_bounds = positions < incident_keys.numel()

        markings = torch.zeros(
            tuple_keys.size(0), dtype=incidence.dtype, device=tuple_keys.device
        )
        markings[in_bounds] = (
            incident_keys[positions[in_bounds]] == tuple_keys[in_bounds]
        ).to(incidence.dtype)
        return markings

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

        binary_marking = self._lookup_sparse_binary_marking(
            incidence_0_2, low_indices, high_indices
        )
        if self.tuple_selection == "incident":
            incident_tuples = binary_marking.bool()
            low_indices = low_indices[incident_tuples]
            high_indices = high_indices[incident_tuples]
            binary_marking = binary_marking[incident_tuples]

        tuple_features = self.encode_rank02_tuple_features(
            batch, low_indices, high_indices, binary_marking
        )
        subcomplex_edges = self.build_rank02_subcomplex_edges(
            low_indices, high_indices
        )
        return {
            "incidence_0_2": incidence_0_2,
            "low_indices": low_indices,
            "high_indices": high_indices,
            "binary_marking": binary_marking,
            "tuple_features": tuple_features,
            **subcomplex_edges,
        }

    def pool_rank02_to_rank0(self, subcomplex, num_low_cells):
        """Pool rank-0/2 tuple features back to rank-0 cells."""
        tuple_features = subcomplex["tuple_features"]
        low_indices = subcomplex["low_indices"]
        pooled = tuple_features.new_zeros(
            (num_low_cells, tuple_features.size(-1))
        )
        if tuple_features.numel() == 0:
            return pooled

        pooled = pooled.index_add(0, low_indices, tuple_features)
        if self.tuple_pooling == "mean":
            counts = tuple_features.new_zeros(num_low_cells)
            counts = counts.index_add(
                0,
                low_indices,
                tuple_features.new_ones(low_indices.size(0)),
            )
            pooled = pooled / counts.clamp_min(1).unsqueeze(-1)

        return pooled

    def encode_rank02_tuple_features(
        self, batch, low_indices, high_indices, binary_marking
    ):
        """Encode rank-0/2 tuple features from given indices and binary marking."""
        marking_features = self.encode_rank02_marking(binary_marking)
        tuple_inputs = torch.cat(
            [
                batch.x_0[low_indices],
                batch.x_2[high_indices],
                marking_features,
            ],
            dim=-1,
        )
        tuple_features = self.rank02_tuple_encoder(tuple_inputs)
        tuple_features = self.rank02_tuple_update(tuple_features)
        return tuple_features

    def encode_rank02_marking(self, binary_marking):
        """Encode rank-0/2 binary marking into a feature vector."""
        if self.rank02_marking_embed is None:
            return binary_marking.unsqueeze(-1).to(torch.float32)
        return self.rank02_marking_embed(binary_marking.long())

    def build_rank02_subcomplex_edges(self, low_indices, high_indices):
        """Build placeholder tuple-level edge indices for rank-0/2 subcomplexes."""
        device = low_indices.device
        num_tuples = low_indices.numel()
        empty_edge_index = torch.empty(
            (2, 0), dtype=torch.long, device=device
        )
        tuple_ids = torch.arange(num_tuples, device=device)
        edge_index_incidence = torch.stack([tuple_ids, tuple_ids])

        low_edges = []
        high_edges = []
        for i in range(num_tuples):
            for j in range(i + 1, num_tuples):
                if low_indices[i] == low_indices[j]:
                    low_edges.append(
                        torch.tensor([[i, j], [j, i]], device=device)
                    )
                if high_indices[i] == high_indices[j]:
                    high_edges.append(
                        torch.tensor([[i, j], [j, i]], device=device)
                    )

        edge_index_low_adjacency = (
            torch.cat(low_edges, dim=1) if low_edges else empty_edge_index
        )
        edge_index_high_adjacency = (
            torch.cat(high_edges, dim=1) if high_edges else empty_edge_index
        )
        return {
            "edge_index_low_adjacency": edge_index_low_adjacency,
            "edge_index_high_adjacency": edge_index_high_adjacency,
            "edge_index_incidence": edge_index_incidence,
        }
