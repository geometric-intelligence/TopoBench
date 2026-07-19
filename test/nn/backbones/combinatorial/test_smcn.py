"""Tests for the SMCN combinatorial backbone skeleton."""

import pytest
import torch
from torch_geometric.data import Data

from topobench.nn.backbones.combinatorial.smcn import SMCN, SubComplexLayer


def test_subcomplex_layer_has_relation_specific_transforms():
    """SubComplexLayer should use separate transforms for each relation type."""
    layer = SubComplexLayer(channels=2)

    assert isinstance(layer.self_linear, torch.nn.Linear)
    assert isinstance(layer.low_linear, torch.nn.Linear)
    assert isinstance(layer.high_linear, torch.nn.Linear)
    assert isinstance(layer.incidence_linear, torch.nn.Linear)

def test_subcomplex_layer_sum_aggregates_placeholder_edges():
    """SubComplexLayer should sum tuple features when aggregation is sum."""
    layer = SubComplexLayer(channels=2, aggregation="sum")
    with torch.no_grad():
        for linear in (
            layer.self_linear,
            layer.low_linear,
            layer.high_linear,
            layer.incidence_linear,
        ):
            linear.weight.copy_(torch.eye(2))
            linear.bias.zero_()

    tuple_features = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 0.0],
        ]
    )
    edge_index_low_adjacency = torch.tensor([[0, 1], [1, 0]])
    edge_index_high_adjacency = torch.tensor([[1, 2], [2, 1]])
    edge_index_incidence = torch.tensor([[0, 1, 2], [0, 1, 2]])

    out = layer(
        tuple_features,
        edge_index_low_adjacency,
        edge_index_high_adjacency,
        edge_index_incidence,
    )

    assert torch.equal(
        out,
        torch.tensor(
            [
                [2.0, 2.0],
                [4.0, 4.0],
                [6.0, 2.0],
            ]
        ),
    )

def test_subcomplex_layer_mean_aggregates_placeholder_edges():
    """SubComplexLayer should average incoming messages when aggregation is mean."""
    layer = SubComplexLayer(channels=2, aggregation="mean")
    with torch.no_grad():
        for linear in (
            layer.self_linear,
            layer.low_linear,
            layer.high_linear,
            layer.incidence_linear,
        ):
            linear.weight.copy_(torch.eye(2))
            linear.bias.zero_()

    tuple_features = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 0.0],
        ]
    )
    edge_index_low_adjacency = torch.tensor([[0, 2], [1, 1]])
    empty_edge_index = torch.empty(2, 0, dtype=torch.long)

    out = layer(
        tuple_features,
        edge_index_low_adjacency,
        empty_edge_index,
        empty_edge_index,
    )

    assert torch.equal(
        out,
        torch.tensor(
            [
                [1.0, 0.0],
                [2.0, 2.0],
                [3.0, 0.0],
            ]
        ),
    )


def test_subcomplex_layer_rejects_unknown_aggregation():
    """SubComplexLayer should fail clearly for unsupported aggregation."""
    with pytest.raises(ValueError, match="Unsupported aggregation"):
        SubComplexLayer(channels=2, aggregation="max")


def test_smcn_rejects_unknown_subcomplex_aggregation():
    """SMCN should fail clearly for unsupported subcomplex aggregation."""
    with pytest.raises(ValueError, match="Unsupported subcomplex_aggregation"):
        SMCN(
            in_channels=8,
            hidden_channels=16,
            subcomplex_aggregation="max",
        )

def test_smcn_forward_returns_rank_dict():
    """SMCN should return updated rank-wise features."""
    in_channels = 8
    hidden_channels = 16

    batch = Data(
        x_0=torch.randn(5, in_channels),
        x_1=torch.randn(7, in_channels),
        x_2=torch.randn(3, in_channels),
    )

    model = SMCN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        neighborhoods=[
            "up_adjacency-1",
            "up_incidence-0",
            "down_incidence-2",
        ],
        layers=1,
        activation="relu",
    )

    out = model(batch)

    assert set(out.keys()) == {0, 1, 2}
    assert out[0].shape == (5, hidden_channels)
    assert out[1].shape == (7, hidden_channels)
    assert out[2].shape == (3, hidden_channels)


def test_smcn_supports_multiple_layers():
    """SMCN should support more than one placeholder update layer."""
    in_channels = 8
    hidden_channels = 16

    batch = Data(
        x_0=torch.randn(5, in_channels),
        x_1=torch.randn(7, in_channels),
        x_2=torch.randn(3, in_channels),
    )

    model = SMCN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        layers=2,
        activation="relu",
    )

    out = model(batch)

    assert out[0].shape == (5, hidden_channels)
    assert out[1].shape == (7, hidden_channels)
    assert out[2].shape == (3, hidden_channels)


def test_smcn_rejects_unknown_activation():
    """SMCN should fail clearly for unsupported activations."""
    with pytest.raises(ValueError, match="Unsupported activation"):
        SMCN(
            in_channels=8,
            hidden_channels=16,
            activation="not_an_activation",
        )


def test_smcn_rejects_unknown_tuple_pooling():
    """SMCN should fail clearly for unsupported tuple pooling."""
    with pytest.raises(ValueError, match="Unsupported tuple_pooling"):
        SMCN(
            in_channels=8,
            hidden_channels=16,
            tuple_pooling="max",
        )


def test_smcn_rejects_unknown_tuple_selection():
    """SMCN should fail clearly for unsupported tuple selection."""
    with pytest.raises(ValueError, match="Unsupported tuple_selection"):
        SMCN(
            in_channels=8,
            hidden_channels=16,
            tuple_selection="max",
        )


def test_smcn_looks_up_sparse_binary_marking():
    """SMCN should look up tuple incidence markings from sparse indices."""
    incidence = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    ).to_sparse()
    low_indices = torch.tensor([0, 0, 1, 1])
    high_indices = torch.tensor([0, 1, 0, 1])

    markings = SMCN._lookup_sparse_binary_marking(
        incidence, low_indices, high_indices
    )

    assert torch.equal(markings, torch.tensor([1.0, 0.0, 0.0, 1.0]))


def test_smcn_looks_up_empty_sparse_binary_marking():
    """SMCN should return zeros when sparse tuple incidence has no entries."""
    incidence = torch.sparse_coo_tensor(size=(2, 2)).coalesce()
    low_indices = torch.tensor([0, 1])
    high_indices = torch.tensor([0, 1])

    markings = SMCN._lookup_sparse_binary_marking(
        incidence, low_indices, high_indices
    )

    assert torch.equal(markings, torch.zeros(2))


def test_smcn_builds_binary_rank02_incidence():
    """SMCN should compose rank 0-to-2 incidence from incidences 0-to-1 and 1-to-2."""
    incidence_1 = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    ).to_sparse()
    incidence_2 = torch.tensor([[1.0], [1.0], [1.0]]).to_sparse()
    batch = Data(
        x_0=torch.randn(3, 8),
        x_2=torch.randn(1, 8),
        incidence_1=incidence_1,
        incidence_2=incidence_2,
    )
    model = SMCN(in_channels=8, hidden_channels=16)

    subcomplex = model.build_rank02_subcomplex(batch)
    subcomplex = model.forward_rank02_subcomplex(batch, subcomplex)
    incidence_0_2 = subcomplex["incidence_0_2"]

    assert incidence_0_2.is_sparse
    assert incidence_0_2.shape == (3, 1)
    assert torch.equal(
        incidence_0_2.to_dense(),
        torch.ones(3, 1),
    )
    assert torch.equal(subcomplex["low_indices"], torch.tensor([0, 1, 2]))
    assert torch.equal(subcomplex["high_indices"], torch.tensor([0, 0, 0]))
    assert torch.equal(subcomplex["binary_marking"], torch.ones(3))
    assert subcomplex["tuple_features"].shape == (3, 16)


def test_smcn_builds_empty_rank02_subcomplex_when_no_rank2_cells():
    """SMCN should handle batches without rank-2 cells."""
    incidence_1 = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    ).to_sparse()
    incidence_2 = torch.sparse_coo_tensor(size=(3, 0)).coalesce()
    batch = Data(
        x_0=torch.randn(3, 8),
        x_2=torch.empty(0, 8),
        incidence_1=incidence_1,
        incidence_2=incidence_2,
    )
    model = SMCN(in_channels=8, hidden_channels=16)

    subcomplex = model.build_rank02_subcomplex(batch)
    subcomplex = model.forward_rank02_subcomplex(batch, subcomplex)
    incidence_0_2 = subcomplex["incidence_0_2"]

    assert incidence_0_2.is_sparse
    assert incidence_0_2.shape == (3, 0)
    assert incidence_0_2._nnz() == 0
    assert torch.equal(subcomplex["low_indices"], torch.empty(0, dtype=torch.long))
    assert torch.equal(subcomplex["high_indices"], torch.empty(0, dtype=torch.long))
    assert torch.equal(subcomplex["binary_marking"], torch.empty(0))
    assert subcomplex["tuple_features"].shape == (0, 16)


def test_smcn_pools_rank02_tuple_features_to_rank0():
    """SMCN should sum tuple features back onto their rank-0 cells."""
    subcomplex = {
        "tuple_features": torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ),
        "low_indices": torch.tensor([0, 1, 0]),
    }

    model = SMCN(in_channels=8, hidden_channels=16, tuple_pooling="sum")
    pooled = model.pool_rank02_to_rank0(subcomplex, num_low_cells=3)

    assert torch.equal(
        pooled,
        torch.tensor(
            [
                [6.0, 8.0],
                [3.0, 4.0],
                [0.0, 0.0],
            ]
        ),
    )


def test_smcn_mean_pools_rank02_tuple_features_to_rank0():
    """SMCN should average tuple features when tuple_pooling is mean."""
    subcomplex = {
        "tuple_features": torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ),
        "low_indices": torch.tensor([0, 1, 0]),
    }

    model = SMCN(in_channels=8, hidden_channels=16, tuple_pooling="mean")
    pooled = model.pool_rank02_to_rank0(subcomplex, num_low_cells=3)

    assert torch.equal(
        pooled,
        torch.tensor(
            [
                [3.0, 4.0],
                [3.0, 4.0],
                [0.0, 0.0],
            ]
        ),
    )


def test_smcn_rejects_negative_marking_embed_dim():
    """SMCN should fail clearly for negative marking embedding dimensions."""
    with pytest.raises(ValueError, match="marking_embed_dim"):
        SMCN(
            in_channels=8,
            hidden_channels=16,
            marking_embed_dim=-1,
        )


def test_smcn_encodes_scalar_rank02_marking_by_default():
    """SMCN should use scalar binary marking by default."""
    model = SMCN(in_channels=8, hidden_channels=16)

    marking_features = model.encode_rank02_marking(torch.tensor([0.0, 1.0]))

    assert torch.equal(marking_features, torch.tensor([[0.0], [1.0]]))


def test_smcn_embeds_rank02_marking_when_requested():
    """SMCN should embed binary marking when marking_embed_dim is positive."""
    model = SMCN(in_channels=8, hidden_channels=16, marking_embed_dim=4)

    marking_features = model.encode_rank02_marking(torch.tensor([0.0, 1.0]))

    assert marking_features.shape == (2, 4)


def test_smcn_encodes_rank02_tuple_features():
    """SMCN should encode selected rank-0/2 tuple features."""
    batch = Data(
        x_0=torch.ones(2, 8),
        x_2=2 * torch.ones(1, 8),
    )
    low_indices = torch.tensor([0, 1])
    high_indices = torch.tensor([0, 0])
    binary_marking = torch.tensor([1.0, 0.0])
    model = SMCN(in_channels=8, hidden_channels=16, marking_embed_dim=4)

    tuple_features = model.encode_rank02_tuple_features(
        batch, low_indices, high_indices, binary_marking
    )

    assert tuple_features.shape == (2, 16)


def test_smcn_uses_subcomplex_layer_when_enabled():
    """SMCN should update rank-0/2 tuple features with SubComplexLayer when enabled."""
    incidence_1 = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    ).to_sparse()
    incidence_2 = torch.tensor([[1.0], [1.0], [1.0]]).to_sparse()
    batch = Data(
        x_0=torch.ones(3, 8),
        x_1=torch.ones(3, 8),
        x_2=2 * torch.ones(1, 8),
        incidence_1=incidence_1,
        incidence_2=incidence_2,
    )
    model = SMCN(
        in_channels=8,
        hidden_channels=16,
        use_subcomplex_signal=True,
    )

    subcomplex = model.build_rank02_subcomplex(batch)
    subcomplex = model.forward_rank02_subcomplex(batch, subcomplex)
    out = model(batch)

    assert isinstance(model.rank02_tuple_update, SubComplexLayer)
    assert subcomplex["tuple_features"].shape == (3, 16)
    assert set(out.keys()) == {0, 1, 2}
    assert out[0].shape == (3, 16)

def test_smcn_rejects_non_positive_max_rank02_tuples():
    """SMCN should fail clearly for non-positive rank-0/2 tuple caps."""
    with pytest.raises(ValueError, match="max_rank02_tuples"):
        SMCN(in_channels=8, hidden_channels=16, max_rank02_tuples=0)


def test_smcn_caps_rank02_tuples_when_requested():
    """SMCN should keep only the first rank-0/2 tuples when capped."""
    incidence_1 = torch.eye(3).to_sparse()
    incidence_2 = torch.ones(3, 2).to_sparse()
    batch = Data(
        x_0=torch.ones(3, 8),
        x_2=2 * torch.ones(2, 8),
        incidence_1=incidence_1,
        incidence_2=incidence_2,
    )
    model = SMCN(in_channels=8, hidden_channels=16, max_rank02_tuples=4)

    subcomplex = model.build_rank02_subcomplex(batch)
    subcomplex = model.forward_rank02_subcomplex(batch, subcomplex)

    assert torch.equal(subcomplex["low_indices"], torch.tensor([0, 0, 1, 1]))
    assert torch.equal(subcomplex["high_indices"], torch.tensor([0, 1, 0, 1]))
    assert subcomplex["binary_marking"].shape == (4,)
    assert subcomplex["tuple_features"].shape == (4, 16)


def test_smcn_keeps_all_rank02_tuples_when_uncapped():
    """SMCN should keep all rank-0/2 tuples when no cap is configured."""
    incidence_1 = torch.eye(3).to_sparse()
    incidence_2 = torch.ones(3, 2).to_sparse()
    batch = Data(
        x_0=torch.ones(3, 8),
        x_2=2 * torch.ones(2, 8),
        incidence_1=incidence_1,
        incidence_2=incidence_2,
    )
    model = SMCN(in_channels=8, hidden_channels=16)

    subcomplex = model.build_rank02_subcomplex(batch)
    subcomplex = model.forward_rank02_subcomplex(batch, subcomplex)

    assert subcomplex["low_indices"].shape == (6,)
    assert subcomplex["high_indices"].shape == (6,)
    assert subcomplex["tuple_features"].shape == (6, 16)

def test_smcn_builds_and_pools_rank02_subcomplex():
    """SMCN should build rank-0/2 tuples and pool them back to rank 0."""
    incidence_1 = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    ).to_sparse()
    incidence_2 = torch.tensor([[1.0], [1.0], [1.0]]).to_sparse()
    batch = Data(
        x_0=torch.ones(3, 8),
        x_2=2 * torch.ones(1, 8),
        incidence_1=incidence_1,
        incidence_2=incidence_2,
    )
    model = SMCN(in_channels=8, hidden_channels=16)

    subcomplex = model.build_rank02_subcomplex(batch)
    subcomplex = model.forward_rank02_subcomplex(batch, subcomplex)
    pooled = model.pool_rank02_to_rank0(
        subcomplex,
        num_low_cells=batch.x_0.size(0),
    )

    assert pooled.shape == (3, 16)


def test_smcn_filters_rank02_tuples_across_batched_graphs():
    """SMCN should not create node-face tuples across different graphs."""
    incidence_1 = torch.eye(4).to_sparse()
    incidence_2 = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    ).to_sparse()
    batch = Data(
        x_0=torch.ones(4, 8),
        x_2=2 * torch.ones(2, 8),
        incidence_1=incidence_1,
        incidence_2=incidence_2,
        batch_0=torch.tensor([0, 0, 1, 1]),
        batch_2=torch.tensor([0, 1]),
    )
    model = SMCN(in_channels=8, hidden_channels=16)

    subcomplex = model.build_rank02_subcomplex(batch)
    subcomplex = model.forward_rank02_subcomplex(batch, subcomplex)

    assert torch.equal(subcomplex["low_indices"], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(subcomplex["high_indices"], torch.tensor([0, 0, 1, 1]))
    assert torch.equal(
        subcomplex["binary_marking"], torch.tensor([1.0, 0.0, 1.0, 1.0])
    )
    assert subcomplex["tuple_features"].shape == (4, 16)


def test_smcn_keeps_all_rank02_tuples_by_default():
    """SMCN should keep same-graph non-incident tuples by default."""
    incidence_1 = torch.eye(2).to_sparse()
    incidence_2 = torch.tensor([[1.0], [0.0]]).to_sparse()
    batch = Data(
        x_0=torch.ones(2, 8),
        x_2=2 * torch.ones(1, 8),
        incidence_1=incidence_1,
        incidence_2=incidence_2,
    )
    model = SMCN(in_channels=8, hidden_channels=16)

    subcomplex = model.build_rank02_subcomplex(batch)
    subcomplex = model.forward_rank02_subcomplex(batch, subcomplex)

    assert torch.equal(subcomplex["low_indices"], torch.tensor([0, 1]))
    assert torch.equal(subcomplex["high_indices"], torch.tensor([0, 0]))
    assert torch.equal(subcomplex["binary_marking"], torch.tensor([1.0, 0.0]))
    assert subcomplex["tuple_features"].shape == (2, 16)


def test_smcn_filters_to_incident_rank02_tuples_when_requested():
    """SMCN should keep only incident tuples in incident selection mode."""
    incidence_1 = torch.eye(2).to_sparse()
    incidence_2 = torch.tensor([[1.0], [0.0]]).to_sparse()
    batch = Data(
        x_0=torch.ones(2, 8),
        x_2=2 * torch.ones(1, 8),
        incidence_1=incidence_1,
        incidence_2=incidence_2,
    )
    model = SMCN(
        in_channels=8, hidden_channels=16, tuple_selection="incident"
    )

    subcomplex = model.build_rank02_subcomplex(batch)
    subcomplex = model.forward_rank02_subcomplex(batch, subcomplex)

    assert torch.equal(subcomplex["low_indices"], torch.tensor([0]))
    assert torch.equal(subcomplex["high_indices"], torch.tensor([0]))
    assert torch.equal(subcomplex["binary_marking"], torch.tensor([1.0]))
    assert subcomplex["tuple_features"].shape == (1, 16)


def test_smcn_builds_rank02_subcomplex_edges():
    """SMCN should build tuple-level low, high, and incidence edge indices."""
    model = SMCN(in_channels=8, hidden_channels=16)
    low_indices = torch.tensor([0, 0, 1])
    high_indices = torch.tensor([0, 1, 1])

    edges = model.build_rank02_subcomplex_edges(low_indices, high_indices)

    assert torch.equal(
        edges["edge_index_low_adjacency"], torch.tensor([[0, 1], [1, 0]])
    )
    assert torch.equal(
        edges["edge_index_high_adjacency"], torch.tensor([[1, 2], [2, 1]])
    )
    assert torch.equal(
        edges["edge_index_incidence"], torch.tensor([[0, 1, 2], [0, 1, 2]])
    )


def test_smcn_builds_empty_rank02_subcomplex_edges():
    """SMCN should return empty edge indices when there are no tuples."""
    model = SMCN(in_channels=8, hidden_channels=16)
    low_indices = torch.empty(0, dtype=torch.long)
    high_indices = torch.empty(0, dtype=torch.long)

    edges = model.build_rank02_subcomplex_edges(low_indices, high_indices)

    assert edges["edge_index_low_adjacency"].shape == (2, 0)
    assert edges["edge_index_high_adjacency"].shape == (2, 0)
    assert edges["edge_index_incidence"].shape == (2, 0)


def test_smcn_rank02_subcomplex_includes_edge_indices():
    """SMCN rank-0/2 subcomplex output should include placeholder edge indices."""
    incidence_1 = torch.eye(2).to_sparse()
    incidence_2 = torch.tensor([[1.0], [0.0]]).to_sparse()
    batch = Data(
        x_0=torch.ones(2, 8),
        x_2=2 * torch.ones(1, 8),
        incidence_1=incidence_1,
        incidence_2=incidence_2,
    )
    model = SMCN(in_channels=8, hidden_channels=16)

    subcomplex = model.build_rank02_subcomplex(batch)

    assert "edge_index_low_adjacency" in subcomplex
    assert "edge_index_high_adjacency" in subcomplex
    assert "edge_index_incidence" in subcomplex
    assert "tuple_features" not in subcomplex


def test_smcn_pools_empty_rank02_tuple_features_to_rank0():
    """SMCN should return zeros when there are no rank-0/2 tuples."""
    subcomplex = {
        "tuple_features": torch.empty(0, 2),
        "low_indices": torch.empty(0, dtype=torch.long),
    }

    model = SMCN(in_channels=8, hidden_channels=16, tuple_pooling="sum")
    pooled = model.pool_rank02_to_rank0(subcomplex, num_low_cells=3)

    assert torch.equal(pooled, torch.zeros(3, 2))
