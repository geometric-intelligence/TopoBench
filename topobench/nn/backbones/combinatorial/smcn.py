"""Pipeline-safe SMCN placeholder backbone."""

import torch


class SubComplexLayer(torch.nn.Module):
    """Placeholder layer for rank-0/2 subcomplexes.

    This layer aggregates tuple features over placeholder subcomplex edge
    indices, then applies separate relation-wise transforms and activation.
    """

    def __init__(
        self, channels, activation_layer=torch.nn.ReLU, aggregation="mean"
    ):
        super().__init__()
        if aggregation not in {"sum", "mean"}:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        self.aggregation = aggregation
        self.self_linear = torch.nn.Linear(channels, channels)
        self.low_linear = torch.nn.Linear(channels, channels)
        self.high_linear = torch.nn.Linear(channels, channels)
        self.incidence_linear = torch.nn.Linear(channels, channels)
        self.activation = activation_layer()
        self.low_bridge_linear = torch.nn.Linear(channels, channels)
        self.high_bridge_linear = torch.nn.Linear(channels, channels)

    def forward(
        self,
        tuple_features,
        edge_index_low_adjacency,
        edge_index_high_adjacency,
        edge_index_incidence,
        low_bridge_features=None,
        high_bridge_features=None,
    ):
        """Update tuple features using placeholder subcomplex edges."""
        low_messages = self._aggregate(tuple_features, edge_index_low_adjacency)
        high_messages = self._aggregate(tuple_features, edge_index_high_adjacency)
        if low_bridge_features is not None:
            low_messages = low_messages + self._aggregate_edge_features(
                self.low_bridge_linear(low_bridge_features),
                edge_index_low_adjacency,
                tuple_features.size(0),
            )
        if high_bridge_features is not None:
            high_messages = high_messages + self._aggregate_edge_features(
                self.high_bridge_linear(high_bridge_features),
                edge_index_high_adjacency,
                tuple_features.size(0),
            )
        incidence_messages = self._aggregate(tuple_features, edge_index_incidence)

        tuple_self = self.self_linear(tuple_features)
        low_msg = self.low_linear(low_messages)
        high_msg = self.high_linear(high_messages)
        incidence_msg = self.incidence_linear(incidence_messages)
        updates = tuple_self + low_msg + high_msg + incidence_msg

        return self.activation(updates)

    def _aggregate(self, features, edge_index):
        """Aggregate source tuple features into target tuple slots."""
        if edge_index.numel() == 0:
            return torch.zeros_like(features)

        source, target = edge_index
        messages = torch.zeros_like(features)
        messages.index_add_(0, target, features[source])
        if self.aggregation == "mean":
            counts = features.new_zeros(features.size(0))
            counts.index_add_(0, target, features.new_ones(target.size(0)))
            messages = messages / counts.clamp_min(1).unsqueeze(-1)

        return messages


    def _aggregate_edge_features(self, edge_features, edge_index, num_tuples):
        """Aggregate edge-aligned features into target tuple slots."""
        messages = edge_features.new_zeros((num_tuples, edge_features.size(-1)))
        if edge_index.numel() == 0 or edge_features.numel() == 0:
            return messages

        target = edge_index[1]
        messages.index_add_(0, target, edge_features)
        if self.aggregation == "mean":
            counts = edge_features.new_zeros(num_tuples)
            counts.index_add_(0, target, edge_features.new_ones(target.size(0)))
            messages = messages / counts.clamp_min(1).unsqueeze(-1)

        return messages

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
        subcomplex_aggregation="mean",
        max_rank02_tuples=None,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.neighborhoods = neighborhoods or []
        self.layers = layers
        self.use_subcomplex_signal = use_subcomplex_signal
        if tuple_pooling not in {"sum", "mean"}:
            raise ValueError(f"Unsupported tuple_pooling: {tuple_pooling}")
        self.tuple_pooling = tuple_pooling

        if tuple_selection not in {"all", "incident"}:
            raise ValueError(f"Unsupported tuple_selection: {tuple_selection}")
        self.tuple_selection = tuple_selection

        if subcomplex_aggregation not in {"sum", "mean"}:
            raise ValueError(
                f"Unsupported subcomplex_aggregation: {subcomplex_aggregation}"
            )
        self.subcomplex_aggregation = subcomplex_aggregation

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

        if use_subcomplex_signal:
            self.rank02_tuple_update = SubComplexLayer(
                hidden_channels,
                activation_layer,
                aggregation=subcomplex_aggregation,
            )
        else:
            self.rank02_tuple_update = self._make_rank_update(
                hidden_channels, hidden_channels, layers, activation_layer
            )

        if max_rank02_tuples is not None and max_rank02_tuples <= 0:
            raise ValueError(
                f"max_rank02_tuples must be positive, got {max_rank02_tuples}"
            )
        self.max_rank02_tuples = max_rank02_tuples

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
            subcomplex = self.forward_rank02_subcomplex(batch, subcomplex)
            pooled_rank0 = self.pool_rank02_to_rank0(
                subcomplex, num_low_cells=batch.x_0.size(0)
            )
            if pooled_rank0.shape == outputs[0].shape:
                outputs[0] = outputs[0] + pooled_rank0

            pooled_rank2 = self.pool_rank02_to_rank2(
                subcomplex, num_high_cells=batch.x_2.size(0)
            )
            if 2 in outputs and pooled_rank2.shape == outputs[2].shape:
                outputs[2] = outputs[2] + pooled_rank2

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

        if self.max_rank02_tuples is not None:
            low_indices = low_indices[: self.max_rank02_tuples]
            high_indices = high_indices[: self.max_rank02_tuples]
            binary_marking = binary_marking[: self.max_rank02_tuples]

        subcomplex_edges = self.build_rank02_subcomplex_edges(
            low_indices,
            high_indices,
            getattr(batch, "incidence_1", None),
            getattr(batch, "incidence_2", None),
        )

        return {
            "incidence_0_2": incidence_0_2,
            "low_indices": low_indices,
            "high_indices": high_indices,
            "binary_marking": binary_marking,
            **subcomplex_edges,
        }

    def forward_rank02_subcomplex(self, batch, subcomplex):
        """Encode and update rank-0/2 tuple features for a subcomplex."""
        low_indices = subcomplex["low_indices"]
        high_indices = subcomplex["high_indices"]
        binary_marking = subcomplex["binary_marking"]
        tuple_features = self.encode_rank02_tuple_features(
            batch, low_indices, high_indices, binary_marking
        )
        low_bridge_features = self._gather_bridge_features(
            batch,
            "x_1",
            subcomplex.get("bridge_index_low_adjacency"),
            subcomplex["edge_index_low_adjacency"],
        )
        high_bridge_features = self._gather_bridge_features(
            batch,
            "x_0",
            subcomplex.get("bridge_index_high_adjacency"),
            subcomplex["edge_index_high_adjacency"],
        )
        if self.use_subcomplex_signal:
            tuple_features = self.rank02_tuple_update(
                tuple_features,
                subcomplex["edge_index_low_adjacency"],
                subcomplex["edge_index_high_adjacency"],
                subcomplex["edge_index_incidence"],
                low_bridge_features=low_bridge_features,
                high_bridge_features=high_bridge_features,
            )
        else:
            tuple_features = self.rank02_tuple_update(tuple_features)

        return {
            **subcomplex,
            "tuple_features": tuple_features,
        }

    def _gather_bridge_features(
        self, batch, feature_name, bridge_indices, edge_index
    ):
        """Gather bridge-cell features when every tuple edge has a bridge."""
        if bridge_indices is None or not hasattr(batch, feature_name):
            return None
        if bridge_indices.numel() == 0 or bridge_indices.numel() != edge_index.size(1):
            return None
        bridge_features = getattr(batch, feature_name)[bridge_indices]
        if bridge_features.size(-1) != self.hidden_channels:
            return None
        return bridge_features

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

    def pool_rank02_to_rank2(self, subcomplex, num_high_cells):
        """Pool rank-0/2 tuple features back to rank-2 cells."""
        tuple_features = subcomplex["tuple_features"]
        high_indices = subcomplex["high_indices"]
        pooled = tuple_features.new_zeros(
            (num_high_cells, tuple_features.size(-1))
        )
        if tuple_features.numel() == 0:
            return pooled

        pooled = pooled.index_add(0, high_indices, tuple_features)
        if self.tuple_pooling == "mean":
            counts = tuple_features.new_zeros(num_high_cells)
            counts = counts.index_add(
                0,
                high_indices,
                tuple_features.new_ones(high_indices.size(0)),
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
        return tuple_features

    def encode_rank02_marking(self, binary_marking):
        """Encode rank-0/2 binary marking into a feature vector."""
        if self.rank02_marking_embed is None:
            return binary_marking.unsqueeze(-1).to(torch.float32)
        return self.rank02_marking_embed(binary_marking.long())

    def build_rank02_subcomplex_edges(
        self, low_indices, high_indices, incidence_1=None, incidence_2=None
    ):
        """Build tuple-level edge indices for rank-0/2 subcomplexes."""
        device = low_indices.device
        num_tuples = low_indices.numel()
        empty_edge_index = torch.empty(
            (2, 0), dtype=torch.long, device=device
        )
        empty_bridge_index = torch.empty(0, dtype=torch.long, device=device)
        tuple_ids = torch.arange(num_tuples, device=device)
        edge_index_incidence = torch.stack([tuple_ids, tuple_ids])

        def shared_cell_edges(cell_indices):
            edge_chunks = []
            for cell_id in cell_indices.unique():
                group = tuple_ids[cell_indices == cell_id]
                if group.numel() < 2:
                    continue
                pairs = torch.combinations(group, r=2).t()
                edge_chunks.append(
                    torch.cat([pairs, pairs.flip(0)], dim=1)
                )
            return torch.cat(edge_chunks, dim=1) if edge_chunks else empty_edge_index

        def low_adjacency_edges_with_bridges(incidence_1, incidence_2):
            if incidence_1 is None or incidence_2 is None:
                return shared_cell_edges(low_indices), empty_bridge_index

            incidence_1 = incidence_1.coalesce()
            incidence_2 = incidence_2.coalesce()
            incidence_1_indices = incidence_1.indices()
            incidence_2_indices = incidence_2.indices()
            vertices_by_edge = [
                incidence_1_indices[0, incidence_1_indices[1] == edge_id]
                for edge_id in range(incidence_1.size(1))
            ]
            edges_by_face = [
                incidence_2_indices[0, incidence_2_indices[1] == face_id]
                for face_id in range(incidence_2.size(1))
            ]

            tuple_lookup = {
                (int(low_id), int(high_id)): int(tuple_id)
                for tuple_id, (low_id, high_id) in enumerate(
                    zip(low_indices.tolist(), high_indices.tolist())
                )
            }
            edge_chunks = []
            bridge_chunks = []
            for face_id, edge_ids in enumerate(edges_by_face):
                for edge_id in edge_ids.tolist():
                    tuple_group = [
                        tuple_lookup[(int(vertex_id), face_id)]
                        for vertex_id in vertices_by_edge[edge_id].tolist()
                        if (int(vertex_id), face_id) in tuple_lookup
                    ]
                    if len(tuple_group) < 2:
                        continue
                    pairs = torch.combinations(
                        torch.tensor(tuple_group, dtype=torch.long, device=device),
                        r=2,
                    ).t()
                    directed_pairs = torch.cat([pairs, pairs.flip(0)], dim=1)
                    edge_chunks.append(directed_pairs)
                    bridge_chunks.append(
                        torch.full(
                            (directed_pairs.size(1),),
                            edge_id,
                            dtype=torch.long,
                            device=device,
                        )
                    )

            if not edge_chunks:
                return empty_edge_index, empty_bridge_index
            return torch.cat(edge_chunks, dim=1), torch.cat(bridge_chunks)

        def high_adjacency_edges_with_bridges(incidence_1, incidence_2):
            if incidence_1 is None or incidence_2 is None:
                return shared_cell_edges(high_indices), empty_bridge_index

            incidence_1 = incidence_1.coalesce()
            incidence_2 = incidence_2.coalesce()
            incidence_1_indices = incidence_1.indices()
            incidence_2_indices = incidence_2.indices()
            edges_by_vertex = [
                incidence_1_indices[1, incidence_1_indices[0] == vertex_id]
                for vertex_id in range(incidence_1.size(0))
            ]
            faces_by_edge = [
                incidence_2_indices[1, incidence_2_indices[0] == edge_id]
                for edge_id in range(incidence_2.size(0))
            ]

            tuple_lookup = {
                (int(low_id), int(high_id)): int(tuple_id)
                for tuple_id, (low_id, high_id) in enumerate(
                    zip(low_indices.tolist(), high_indices.tolist())
                )
            }
            edge_chunks = []
            bridge_chunks = []
            for vertex_id, edge_ids in enumerate(edges_by_vertex):
                if edge_ids.numel() == 0:
                    continue
                face_ids = torch.unique(
                    torch.cat(
                        [faces_by_edge[edge_id] for edge_id in edge_ids.tolist()]
                    )
                )
                tuple_group = [
                    tuple_lookup[(vertex_id, int(face_id))]
                    for face_id in face_ids.tolist()
                    if (vertex_id, int(face_id)) in tuple_lookup
                ]
                if len(tuple_group) < 2:
                    continue
                pairs = torch.combinations(
                    torch.tensor(tuple_group, dtype=torch.long, device=device),
                    r=2,
                ).t()
                directed_pairs = torch.cat([pairs, pairs.flip(0)], dim=1)
                edge_chunks.append(directed_pairs)
                bridge_chunks.append(
                    torch.full(
                        (directed_pairs.size(1),),
                        vertex_id,
                        dtype=torch.long,
                        device=device,
                    )
                )

            if not edge_chunks:
                return empty_edge_index, empty_bridge_index
            return torch.cat(edge_chunks, dim=1), torch.cat(bridge_chunks)

        (
            edge_index_low_adjacency,
            bridge_index_low_adjacency,
        ) = low_adjacency_edges_with_bridges(incidence_1, incidence_2)

        (
            edge_index_high_adjacency,
            bridge_index_high_adjacency,
        ) = high_adjacency_edges_with_bridges(incidence_1, incidence_2)

        return {
            "edge_index_low_adjacency": edge_index_low_adjacency,
            "edge_index_high_adjacency": edge_index_high_adjacency,
            "edge_index_incidence": edge_index_incidence,
            "bridge_index_low_adjacency": bridge_index_low_adjacency,
            "bridge_index_high_adjacency": bridge_index_high_adjacency,
        }
