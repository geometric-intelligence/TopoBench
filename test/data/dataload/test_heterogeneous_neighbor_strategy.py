"""Deterministic materialized heterogeneous NeighborLoader strategy tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from test.data.dataload.test_disk_graph_datamodule import (
    assert_heterogeneous_exact,
)
from topobench.dataloader.disk_graph import (
    HeterogeneousNeighborStrategy,
    SamplingCapabilityError,
)

WRITES = ("paper", "writes", "author")
WRITTEN_BY = ("author", "written_by", "paper")
CITES = ("paper", "cites", "paper")


def _choice_rich_data() -> HeteroData:
    """Build duplicates, an isolated target, high degree, and explicit reverse."""
    data = HeteroData()
    data["paper"].x = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    data["paper"].n_id = torch.arange(6)
    data["author"].x = torch.arange(4, dtype=torch.float32).reshape(4, 1)
    data["author"].y = torch.tensor([3, 2, 1, 0])
    data["author"].n_id = torch.arange(4)
    data["author"].train_mask = torch.tensor([True, True, True, False])
    data["author"].val_mask = torch.tensor([False, False, False, True])
    data["author"].test_mask = torch.tensor([False, True, False, False])

    # Author 0 has five incoming rows, including duplicate endpoint 0 -> 0.
    data[WRITES].edge_index = torch.tensor(
        [[0, 0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 2, 3]],
        dtype=torch.long,
    )
    data[WRITES].edge_id = torch.tensor([10, 11, 12, 13, 14, 15, 16])
    data[WRITES].weight = torch.tensor([1, 2, 3, 4, 5, 6, 7])
    data[WRITTEN_BY].edge_index = torch.tensor(
        [[0, 2, 3], [0, 4, 5]], dtype=torch.long
    )
    data[WRITTEN_BY].edge_id = torch.tensor([30, 31, 32])
    data[CITES].edge_index = torch.empty((2, 0), dtype=torch.long)
    data[CITES].edge_id = torch.empty(0, dtype=torch.long)
    data.content_sha256 = "2" * 64
    data.active_split_tag = "primary"
    data.target_node_type = "author"
    return data


def _fanout() -> dict[tuple[str, str, str], list[int]]:
    """Traverse writes exhaustively while keeping other directions disabled."""
    return {WRITES: [-1], WRITTEN_BY: [0], CITES: [0]}


def test_descriptors_preserve_phase_seed_order_and_final_short_batch() -> None:
    """Target batches are exact ordered slices of the named active phase."""
    data = _choice_rich_data()
    strategy = HeterogeneousNeighborStrategy(
        batch_size=2,
        num_neighbors=_fanout(),
        seed=41,
    )
    descriptors = strategy.setup(
        data,
        phase="train",
        active_split_tag="primary",
        shuffle=False,
    )
    assert tuple(descriptor.target_seed_ids for descriptor in descriptors) == (
        (0, 1),
        (2,),
    )
    assert all(not descriptor.participant_counts for descriptor in descriptors)
    assert descriptors == strategy.setup(
        data,
        phase="train",
        active_split_tag="primary",
        shuffle=False,
    )
    assert strategy.sampler_state() == {
        "format_version": "graph-sampling-state-v1",
        "seed": 41,
        "strategy": "heterogeneous-neighbor",
    }


def test_materialization_matches_installed_neighborloader_in_exact_order() -> None:
    """PyG owns exhaustive traversal, duplicate edges, fields, and hop metadata."""
    data = _choice_rich_data()
    strategy = HeterogeneousNeighborStrategy(
        batch_size=2,
        num_neighbors=_fanout(),
        seed=43,
    )
    descriptor = strategy.setup(
        data,
        phase="train",
        active_split_tag="primary",
        shuffle=False,
    )[0]
    oracle = next(
        iter(
            NeighborLoader(
                data,
                input_nodes=("author", torch.tensor(descriptor.target_seed_ids)),
                num_neighbors=_fanout(),
                batch_size=2,
                shuffle=False,
                replace=False,
                subgraph_type="directional",
                generator=torch.Generator().manual_seed(
                    descriptor.generator_seed
                ),
            )
        )
    )
    expected_content_sha256 = descriptor.content_sha256
    assert expected_content_sha256 != data.content_sha256
    data["paper"].x.add_(1000)
    data[WRITES].edge_id.add_(1000)
    assert (
        strategy.setup(
            data,
            phase="train",
            active_split_tag="primary",
            shuffle=False,
        )[0]
        == descriptor
    )

    torch.manual_seed(999)
    global_state = torch.random.get_rng_state().clone()
    actual = strategy.materialize(data, descriptor)
    assert torch.equal(torch.random.get_rng_state(), global_state)
    assert_heterogeneous_exact(actual, oracle)
    assert actual["author"].batch_size == 2
    assert actual["author"].n_id[:2].tolist() == [0, 1]
    assert actual[WRITES].edge_id.tolist() == [10, 11, 12, 13, 14]
    assert actual[WRITES].weight.tolist() == [1, 2, 3, 4, 5]
    assert actual[WRITTEN_BY].edge_index.numel() == 0
    assert actual[CITES].edge_index.numel() == 0
    assert actual.participant_counts == {
        node_type: len(actual[node_type].n_id)
        for node_type in actual.node_types
    }
    assert actual.sampling_descriptor != descriptor
    assert actual.sampling_descriptor.participant_counts == tuple(
        sorted(actual.participant_counts.items())
    )
    assert torch.equal(
        actual["author"].supervised_mask,
        actual["author"].train_mask,
    )


def test_truncated_sampling_is_serialized_deterministic_and_rng_neutral() -> None:
    """Finite PyG sampling is deterministic under concurrent strategy calls."""
    data = _choice_rich_data()
    stochastic = _fanout() | {WRITES: [2]}
    strategy = HeterogeneousNeighborStrategy(
        batch_size=1,
        num_neighbors=stochastic,
        seed=47,
    )
    descriptor = strategy.setup(
        data,
        phase="train",
        active_split_tag="primary",
        shuffle=False,
    )[0]
    torch.manual_seed(999)
    global_state = torch.random.get_rng_state().clone()
    expected = strategy.materialize(data, descriptor)

    with ThreadPoolExecutor(max_workers=4) as executor:
        observed = tuple(
            executor.map(
                lambda _: strategy.materialize(data, descriptor),
                range(8),
            )
        )

    assert torch.equal(torch.random.get_rng_state(), global_state)
    assert expected[WRITES].edge_index.shape[1] <= 2
    for batch in observed:
        assert_heterogeneous_exact(batch, expected)


def test_replace_sampling_is_rejected_without_local_generator_support() -> None:
    data = _choice_rich_data()
    with pytest.raises(
        SamplingCapabilityError,
        match="replace=True.*deterministic",
    ):
        HeterogeneousNeighborStrategy(
            batch_size=1,
            num_neighbors=_fanout(),
            replace=True,
            seed=47,
        ).setup(
            data,
            phase="train",
            active_split_tag="primary",
            shuffle=False,
        )


def test_induced_sampling_is_rejected_without_native_hop_counts() -> None:
    data = _choice_rich_data()
    with pytest.raises(
        SamplingCapabilityError,
        match="induced.*hop counts",
    ):
        HeterogeneousNeighborStrategy(
            batch_size=1,
            num_neighbors=_fanout(),
            subgraph_type="induced",
        ).setup(
            data,
            phase="train",
            active_split_tag="primary",
            shuffle=False,
        )


def test_neighbor_option_validation_rejects_bad_fanout_and_direction() -> None:
    """Fanout and direction options reject bools, ranges, and relation mismatch."""
    with pytest.raises(TypeError, match="fanout.*non-boolean integers"):
        HeterogeneousNeighborStrategy(
            batch_size=1,
            num_neighbors={WRITES: [True]},
        )
    with pytest.raises(ValueError, match="fanout.*at least -1"):
        HeterogeneousNeighborStrategy(
            batch_size=1,
            num_neighbors={WRITES: [-2]},
        )
    with pytest.raises(ValueError, match="sample_direction"):
        HeterogeneousNeighborStrategy(
            batch_size=1,
            num_neighbors={WRITES: [-1]},
            sample_direction="reverse",
        )
