"""Native batch-isolation gates for the GCN-DGM candidate."""

import pytest
import torch

from topobench.nn.backbones.graph import GCNDGM


def test_gcn_dgm_returns_only_tensor_embeddings_and_isolated_edges() -> None:
    torch.manual_seed(5)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    model = GCNDGM(
        in_channels=4,
        hidden_channels=8,
        num_layers=2,
        k=3,
    )

    output = model(
        torch.randn(7, 4),
        torch.empty((2, 0), dtype=torch.long),
        batch=batch,
    )

    assert isinstance(output, torch.Tensor)
    assert output.shape == (7, 8)
    assert model.last_auxiliary_edge_index is not None
    source, target = model.last_auxiliary_edge_index
    assert torch.equal(batch[source], batch[target])


def test_gcn_dgm_large_graph_is_invariant_to_singleton_cobatch() -> None:
    torch.manual_seed(7)
    model = GCNDGM(
        in_channels=4,
        hidden_channels=8,
        num_layers=2,
        k=3,
    )
    large_x = torch.randn(5, 4)

    large_output = model(
        large_x,
        torch.empty((2, 0), dtype=torch.long),
    )
    assert model.last_auxiliary_edge_index is not None
    assert model.last_auxiliary_logprobs is not None
    large_edges = model.last_auxiliary_edge_index.clone()
    large_logprobs = model.last_auxiliary_logprobs.clone()

    batched_output = model(
        torch.cat((large_x, torch.randn(1, 4))),
        torch.empty((2, 0), dtype=torch.long),
        batch=torch.tensor([0, 0, 0, 0, 0, 1]),
    )

    assert model.last_auxiliary_edge_index is not None
    assert model.last_auxiliary_logprobs is not None
    assert model.last_auxiliary_logprobs.ndim == 1
    assert model.last_auxiliary_logprobs.numel() == 16
    torch.testing.assert_close(
        model.last_auxiliary_edge_index[:, :15],
        large_edges,
    )
    torch.testing.assert_close(
        model.last_auxiliary_logprobs[:15],
        large_logprobs,
    )
    torch.testing.assert_close(batched_output[:5], large_output)
    assert torch.equal(
        model.last_auxiliary_edge_index[:, -1],
        torch.tensor([5, 5]),
    )


def test_gcn_dgm_structure_parameters_receive_supervised_gradient() -> None:
    torch.manual_seed(9)
    model = GCNDGM(
        in_channels=4,
        hidden_channels=8,
        num_layers=1,
        k=2,
    )
    output = model(
        torch.randn(6, 4),
        torch.empty((2, 0), dtype=torch.long),
        batch=torch.tensor([0, 0, 0, 1, 1, 1]),
    )

    output.square().mean().backward()

    gradient = model.structure_encoder.linear.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient)


@pytest.mark.parametrize("k", (2, 3))
def test_gcn_dgm_supported_k_trains_structure_and_temperature(k: int) -> None:
    torch.manual_seed(13)
    model = GCNDGM(
        in_channels=2,
        hidden_channels=4,
        num_layers=1,
        k=k,
    )
    x = torch.tensor(
        [[0.0, 0.0], [0.5, 1.0], [2.0, -0.5], [4.0, 2.0], [9.0, -1.0]]
    )
    structure = model.structure_encoder(x)
    _, weights, _ = model._learned_edges(
        structure,
        torch.zeros(x.size(0), dtype=torch.long),
    )

    coefficients = torch.linspace(
        0.25,
        2.0,
        weights.numel(),
        dtype=weights.dtype,
    )
    (weights * coefficients).sum().backward()

    structure_gradient = model.structure_encoder.linear.weight.grad
    temperature_gradient = model.log_temperature.grad
    assert structure_gradient is not None
    assert torch.count_nonzero(structure_gradient)
    assert temperature_gradient is not None
    assert torch.count_nonzero(temperature_gradient)


def test_gcn_dgm_edges_are_selected_neighbor_to_query_and_incoming_normalized() -> (
    None
):
    model = GCNDGM(
        in_channels=1,
        hidden_channels=2,
        num_layers=1,
        k=2,
    )
    structure = torch.tensor([[0.0], [1.0], [4.0], [10.0]])

    edge_index, weights, logprobs = model._learned_edges(
        structure,
        torch.zeros(4, dtype=torch.long),
    )

    expected_neighbors = torch.tensor(
        [[1, 2], [0, 2], [1, 0], [2, 1]],
        dtype=torch.long,
    )
    expected_queries = torch.arange(4).unsqueeze(1).expand(-1, 2)
    torch.testing.assert_close(
        edge_index[0].reshape(4, 2).sort(dim=1).values,
        expected_neighbors.sort(dim=1).values,
    )
    torch.testing.assert_close(
        edge_index[1].reshape(4, 2),
        expected_queries,
    )
    incoming_weights = weights.reshape(4, 2)
    torch.testing.assert_close(
        incoming_weights.sum(dim=1),
        torch.ones(4),
    )
    torch.testing.assert_close(logprobs.exp(), weights)


@pytest.mark.parametrize("k", (True, False, 2.5, "2"))
def test_gcn_dgm_k_requires_an_integer_not_bool(k: object) -> None:
    with pytest.raises(TypeError, match="k must be an integer"):
        GCNDGM(
            in_channels=4,
            hidden_channels=8,
            num_layers=1,
            k=k,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("k", (-1, 0, 1))
def test_gcn_dgm_k_requires_at_least_two(k: int) -> None:
    with pytest.raises(ValueError, match="k must be at least 2"):
        GCNDGM(
            in_channels=4,
            hidden_channels=8,
            num_layers=1,
            k=k,
        )


def test_gcn_dgm_requires_valid_batch_membership() -> None:
    model = GCNDGM(
        in_channels=4,
        hidden_channels=8,
        num_layers=1,
        k=2,
    )

    with pytest.raises(ValueError, match="batch must be contiguous"):
        model(
            torch.randn(4, 4),
            torch.empty((2, 0), dtype=torch.long),
            batch=torch.tensor([0, 0, 2, 2]),
        )


def _dense_exact_neighbors(
    structure: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.cdist(structure, structure)
    distances = distances.masked_fill(
        torch.eye(
            structure.size(0),
            dtype=torch.bool,
            device=structure.device,
        ),
        torch.inf,
    )
    indices = torch.argsort(distances, dim=1, stable=True)[:, :k]
    return distances.gather(1, indices), indices


@pytest.mark.parametrize("query_chunk_size", (1, 2, 4))
def test_gcn_dgm_chunked_topk_matches_dense_scores_edges_and_gradients(
    query_chunk_size: int,
) -> None:
    base_structure = torch.tensor(
        [[0.0], [1.0], [-1.0], [2.0], [-2.0], [4.0]],
        dtype=torch.float64,
    )
    max_workspace_bytes = GCNDGM.estimate_workspace_bytes(
        node_count=6,
        query_chunk_size=query_chunk_size,
        k=3,
        feature_dim=2,
        element_size=8,
    )
    model = GCNDGM(
        in_channels=1,
        hidden_channels=2,
        num_layers=1,
        k=3,
        query_chunk_size=query_chunk_size,
        max_nodes=6,
        max_workspace_bytes=max_workspace_bytes,
    ).double()
    with torch.no_grad():
        model.log_temperature.fill_(-0.25)
    structure = base_structure.clone().requires_grad_()

    chunked_distances, chunked_neighbors = model._chunked_neighbors(structure)
    edge_index, weights, logprobs = model._learned_edges(
        structure,
        torch.zeros(6, dtype=torch.long),
    )

    reference_structure = base_structure.clone().requires_grad_()
    reference_log_temperature = (
        model.log_temperature.detach().clone().requires_grad_()
    )
    dense_distances, dense_neighbors = _dense_exact_neighbors(
        reference_structure,
        3,
    )
    dense_logprobs = torch.log_softmax(
        -dense_distances / reference_log_temperature.exp().clamp_min(1e-6),
        dim=1,
    ).reshape(-1)
    expected_queries = torch.arange(6).unsqueeze(1).expand(-1, 3)
    expected_edges = torch.stack(
        (dense_neighbors.reshape(-1), expected_queries.reshape(-1))
    )
    coefficients = torch.linspace(
        0.5,
        2.5,
        logprobs.numel(),
        dtype=logprobs.dtype,
    )
    actual_gradients = torch.autograd.grad(
        (logprobs * coefficients).sum(),
        (structure, model.log_temperature),
    )
    reference_gradients = torch.autograd.grad(
        (dense_logprobs * coefficients).sum(),
        (reference_structure, reference_log_temperature),
    )

    torch.testing.assert_close(chunked_distances, dense_distances)
    torch.testing.assert_close(chunked_neighbors, dense_neighbors)
    torch.testing.assert_close(edge_index, expected_edges)
    torch.testing.assert_close(logprobs, dense_logprobs)
    torch.testing.assert_close(weights, dense_logprobs.exp())
    torch.testing.assert_close(actual_gradients[0], reference_gradients[0])
    torch.testing.assert_close(actual_gradients[1], reference_gradients[1])
    assert torch.equal(
        edge_index[0].reshape(6, 3)[0],
        torch.tensor([1, 2, 3]),
    )
    assert all(torch.isfinite(gradient).all() for gradient in actual_gradients)
    assert all(torch.count_nonzero(gradient) for gradient in actual_gradients)


@pytest.mark.parametrize(
    ("node_count", "query_chunk_size"),
    ((7, 6), (257, 32)),
)
def test_gcn_dgm_never_requests_a_full_pairwise_distance_matrix(
    monkeypatch: pytest.MonkeyPatch,
    node_count: int,
    query_chunk_size: int,
) -> None:
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    original_cdist = torch.cdist

    def recording_cdist(
        queries: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        calls.append((tuple(queries.shape), tuple(candidates.shape)))
        return original_cdist(queries, candidates)

    monkeypatch.setattr(torch, "cdist", recording_cdist)
    model = GCNDGM(
        in_channels=2,
        hidden_channels=4,
        num_layers=1,
        k=4,
        query_chunk_size=query_chunk_size,
        max_nodes=node_count,
        max_workspace_bytes=GCNDGM.estimate_workspace_bytes(
            node_count=node_count,
            query_chunk_size=query_chunk_size,
            k=4,
            feature_dim=4,
            element_size=8,
        ),
    )

    model._learned_edges(
        torch.randn(node_count, 2),
        torch.zeros(node_count, dtype=torch.long),
    )

    assert calls
    assert all(query[0] <= query_chunk_size for query, _ in calls)
    assert all(candidate[0] == node_count for _, candidate in calls)
    assert ((node_count, 2), (node_count, 2)) not in calls


def test_gcn_dgm_selects_without_grad_then_recomputes_pair_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grad_modes: list[bool] = []
    original_cdist = torch.cdist

    def recording_cdist(
        queries: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        grad_modes.append(torch.is_grad_enabled())
        return original_cdist(queries, candidates)

    monkeypatch.setattr(torch, "cdist", recording_cdist)
    model = GCNDGM(
        in_channels=4,
        hidden_channels=4,
        num_layers=1,
        k=3,
        query_chunk_size=2,
        max_nodes=8,
        max_workspace_bytes=GCNDGM.estimate_workspace_bytes(
            node_count=8,
            query_chunk_size=2,
            k=3,
            feature_dim=4,
            element_size=8,
        ),
    ).double()
    structure = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)

    selected_distances, _ = model._chunked_neighbors(structure)
    coefficients = torch.linspace(
        0.5,
        2.0,
        selected_distances.numel(),
        dtype=selected_distances.dtype,
    ).reshape_as(selected_distances)
    gradient = torch.autograd.grad(
        (selected_distances * coefficients).sum(),
        structure,
    )[0]

    assert grad_modes and not any(grad_modes)
    assert selected_distances.requires_grad
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient)


def test_gcn_dgm_workspace_bound_includes_k_feature_state_and_large_integers() -> (
    None
):
    small_k = GCNDGM.estimate_workspace_bytes(
        node_count=64,
        query_chunk_size=1,
        k=2,
        feature_dim=128,
        element_size=8,
    )
    near_dense_k = GCNDGM.estimate_workspace_bytes(
        node_count=64,
        query_chunk_size=1,
        k=63,
        feature_dim=128,
        element_size=8,
    )
    wide_features = GCNDGM.estimate_workspace_bytes(
        node_count=64,
        query_chunk_size=1,
        k=63,
        feature_dim=256,
        element_size=8,
    )
    huge = GCNDGM.estimate_workspace_bytes(
        node_count=2**40,
        query_chunk_size=1,
        k=2**20,
        feature_dim=2**20,
        element_size=8,
    )

    assert near_dense_k > small_k
    assert wide_features > near_dense_k
    assert huge > 2**63


def test_gcn_dgm_rejects_node_bound_before_distance_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_cdist(*_: object, **__: object) -> torch.Tensor:
        raise AssertionError("torch.cdist reached before node admission")

    monkeypatch.setattr(torch, "cdist", unexpected_cdist)
    model = GCNDGM(
        in_channels=2,
        hidden_channels=4,
        num_layers=1,
        k=4,
        query_chunk_size=8,
        max_nodes=16,
        max_workspace_bytes=GCNDGM.estimate_workspace_bytes(
            node_count=16,
            query_chunk_size=8,
            k=4,
            feature_dim=4,
            element_size=8,
        ),
    )

    with pytest.raises(
        ValueError, match=r"node count 17 exceeds max_nodes=16"
    ):
        model._learned_edges(
            torch.randn(17, 2),
            torch.zeros(17, dtype=torch.long),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("query_chunk_size", True),
        ("query_chunk_size", 2.5),
        ("max_nodes", False),
        ("max_nodes", 8.5),
        ("max_workspace_bytes", True),
        ("max_workspace_bytes", 1024.5),
    ),
)
def test_gcn_dgm_scale_limits_require_integers_not_bool(
    field: str,
    value: object,
) -> None:
    parameters: dict[str, object] = {
        "in_channels": 2,
        "hidden_channels": 4,
        "num_layers": 1,
        "k": 2,
        "query_chunk_size": 2,
        "max_nodes": 8,
        "max_workspace_bytes": 4096,
    }
    parameters[field] = value

    with pytest.raises(TypeError, match=rf"{field} must be an integer"):
        GCNDGM(**parameters)  # type: ignore[arg-type]


def test_gcn_dgm_rejects_infeasible_neighbor_and_workspace_limits() -> None:
    with pytest.raises(ValueError, match="max_nodes must be greater than k"):
        GCNDGM(
            in_channels=2,
            hidden_channels=4,
            num_layers=1,
            k=4,
            query_chunk_size=2,
            max_nodes=4,
            max_workspace_bytes=4096,
        )

    required = GCNDGM.estimate_workspace_bytes(
        node_count=8,
        query_chunk_size=2,
        k=2,
        feature_dim=4,
        element_size=8,
    )
    with pytest.raises(ValueError, match="max_workspace_bytes"):
        GCNDGM(
            in_channels=2,
            hidden_channels=4,
            num_layers=1,
            k=2,
            query_chunk_size=2,
            max_nodes=8,
            max_workspace_bytes=required - 1,
        )
