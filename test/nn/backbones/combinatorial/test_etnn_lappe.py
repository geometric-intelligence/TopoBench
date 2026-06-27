"""Unit tests for the LapPE structural-coordinate ETNN backbone.

The tests focus on the extra contract introduced by ``ETNNLapPE`` relative to
the coordinate-free ETNN baseline: rank-0 LapPE coordinates must exist, they
must align with rank-0 cells, they must lift cleanly through TopoBench
incidence matrices, and the resulting message distance must be invariant to
rigid transformations of the structural coordinate frame.
"""

import pytest
import torch

from test.nn.backbones.combinatorial.test_etnn import (
    create_mock_complex_batch,
)
from topobench.nn.backbones.combinatorial.etnn_lappe import (
    ETNNLapPE,
    _build_lappe_cell_coordinates,
    _squared_coordinate_distances,
)


def create_lappe_complex_batch():
    """Create a mock lifted complex with coordinate-lifting incidences.

    The baseline ETNN tests already construct directional neighborhoods for
    message passing. ETNN-LapPE additionally needs the canonical
    ``incidence_r`` matrices because it recursively lifts rank-0 coordinates
    to rank-1 and rank-2 cells.
    """
    batch = create_mock_complex_batch()

    # The coordinate-enabled backbone needs the base incidence matrices because
    # it lifts rank-0 LapPE coordinates to higher-rank cells. The baseline ETNN
    # only consumes the directional neighborhood aliases.
    batch.incidence_1 = batch["down_incidence-1"]
    batch.incidence_2 = batch["down_incidence-2"]
    return batch


def create_lappe_etnn():
    """Instantiate ETNN-LapPE with the same relations as the baseline ETNN."""
    return ETNNLapPE(
        in_channels=16,
        hidden_channels=8,
        out_channels=16,
        neighborhoods=[
            "up_adjacency-0",
            "up_adjacency-1",
            "up_adjacency-2",
            "up_incidence-0",
            "down_incidence-1",
            "up_incidence-1",
            "down_incidence-2",
        ],
        num_layers=2,
        coordinate_attr="LapPE",
    )


def test_lappe_coordinate_etnn_runs_without_physical_positions():
    """ETNN-LapPE should run from structural coordinates without ``pos``.

    GraphUniverse does not provide physical Euclidean positions. This test
    verifies that the coordinate-enabled variant consumes ``LapPE`` instead of
    requiring a PyG-style ``pos`` tensor, while still returning one embedding
    tensor per visible cell rank.
    """
    batch = create_lappe_complex_batch()
    batch.LapPE = torch.randn(batch.x_0.shape[0], 3)
    assert "pos" not in batch

    out = create_lappe_etnn()(batch)

    assert set(out) == {0, 1, 2}
    assert out[0].shape == batch.x_0.shape
    assert out[1].shape == batch.x_1.shape
    assert out[2].shape == batch.x_2.shape


def test_lappe_coordinate_etnn_requires_lappe_attribute():
    """Coordinate mode should fail clearly when LapPE is unavailable.

    The model intentionally requires an explicit structural coordinate source.
    A missing coordinate attribute should not silently fall back to zeros,
    random coordinates, or coordinate-free behavior.
    """
    batch = create_lappe_complex_batch()

    with pytest.raises(AttributeError, match="LapPE"):
        create_lappe_etnn()(batch)


def test_lappe_coordinate_etnn_validates_rank_0_coordinate_count():
    """LapPE rows must align one-to-one with rank-0 cells.

    A row-count mismatch would attach coordinates to the wrong vertices and
    corrupt every higher-rank coordinate produced by incidence averaging.
    """
    batch = create_lappe_complex_batch()
    batch.LapPE = torch.randn(batch.x_0.shape[0] + 1, 3)

    with pytest.raises(ValueError, match="one rank-0 coordinate row"):
        create_lappe_etnn()(batch)


def test_lappe_cell_coordinates_are_barycentric_by_rank():
    """Higher-rank structural coordinates are incidence-weighted averages.

    The implementation uses recursive incidence averaging rather than direct
    vertex-to-cell membership. This test checks the intended rank-by-rank
    policy on a small complex where the expected edge and face coordinates can
    be computed by hand.
    """
    batch = create_lappe_complex_batch()
    batch.LapPE = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ]
    )

    coordinates = _build_lappe_cell_coordinates(
        batch=batch,
        coordinate_attr="LapPE",
        max_rank=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    expected_rank_1 = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [0.0, 1.0],
        ]
    )
    expected_rank_2 = torch.tensor(
        [
            [4.0 / 3.0, 1.0],
            [1.0, 4.0 / 3.0],
        ]
    )
    assert torch.allclose(coordinates[0], batch.LapPE)
    assert torch.allclose(coordinates[1], expected_rank_1)
    assert torch.allclose(coordinates[2], expected_rank_2)


def test_lappe_cell_coordinates_handle_empty_rank():
    """Empty ranks should receive empty coordinate tensors.

    Some lifted mini-batches can contain no cells at a visible higher rank.
    The coordinate construction should preserve that empty rank with shape
    ``[0, coordinate_dim]`` instead of creating placeholder coordinates.
    """
    batch = create_lappe_complex_batch()
    batch.LapPE = torch.randn(batch.x_0.shape[0], 3)
    batch.x_2 = torch.empty(0, 16)
    batch.batch_2 = torch.empty(0, dtype=torch.long)
    batch.incidence_2 = torch.sparse_coo_tensor(
        indices=torch.empty(2, 0, dtype=torch.long),
        values=torch.empty(0),
        size=(batch.x_1.shape[0], 0),
    ).coalesce()

    coordinates = _build_lappe_cell_coordinates(
        batch=batch,
        coordinate_attr="LapPE",
        max_rank=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert coordinates[2].shape == (0, 3)


def test_lappe_cell_coordinates_validate_incidence_source_axis():
    """Incidence source rows must align with lower-rank coordinates.

    TopoBench incidence matrices should have rows for rank ``r-1`` cells and
    columns for rank ``r`` cells. A source-axis mismatch means the coordinate
    lift is ill-defined, so the helper should raise before indexing.
    """
    batch = create_lappe_complex_batch()
    batch.LapPE = torch.randn(batch.x_0.shape[0], 3)
    batch.incidence_1 = torch.sparse_coo_tensor(
        indices=torch.empty(2, 0, dtype=torch.long),
        values=torch.empty(0),
        size=(batch.x_0.shape[0] + 1, batch.x_1.shape[0]),
    ).coalesce()

    with pytest.raises(ValueError, match="source rows"):
        _build_lappe_cell_coordinates(
            batch=batch,
            coordinate_attr="LapPE",
            max_rank=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_squared_lappe_distance_is_rigid_motion_invariant():
    """Distance features should ignore rigid coordinate-frame choices.

    LapPE coordinates are structural pseudo-coordinates, so their absolute
    orientation is arbitrary. The squared-distance message feature should be
    unchanged by translations, rotations, and reflections of that structural
    coordinate frame.
    """
    src = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    dst = torch.tensor([[1.0, 1.0], [3.0, 0.0]])
    edge_index = torch.tensor([[0, 1], [0, 1]])
    base = _squared_coordinate_distances(src, dst, edge_index, torch.float32)

    rotation_reflection = torch.tensor([[0.0, -1.0], [-1.0, 0.0]])
    translation = torch.tensor([4.0, -2.0])
    transformed_src = src @ rotation_reflection.T + translation
    transformed_dst = dst @ rotation_reflection.T + translation
    transformed = _squared_coordinate_distances(
        transformed_src,
        transformed_dst,
        edge_index,
        torch.float32,
    )

    assert torch.allclose(base, transformed)
