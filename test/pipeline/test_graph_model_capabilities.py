"""Network-free lifecycle gates for every declared graph model contract."""

from __future__ import annotations

import hydra
import pytest
import torch
from omegaconf import DictConfig, open_dict
from torch_geometric.data import Data
from torch_geometric.nn.models import GCN

from test._utils.simplified_pipeline import run
from topobench.data.capabilities import GRAPH_DATASET_MANIFEST
from topobench.loss.model import GraphMLPLoss
from topobench.nn.backbones.graph import GCNDGM
from topobench.nn.capabilities import (
    GRAPH_MODEL_CAPABILITIES,
    compatible_graph_models,
    validate_capability_composition,
    validate_graph_composition,
)
from topobench.nn.readouts import NoReadOut
from topobench.nn.wrappers.graph import GNNWrapper
from topobench.utils.config_resolvers import register_all_resolvers
from topobench.utils.model_instantiation import instantiate_model

SYNTHETIC_SELECTORS = (
    "SyntheticGraph",
    "SyntheticGraphRegression",
    "SyntheticNodeGraph",
)
LIFECYCLE_PAIRS = tuple(
    (dataset_selector, model.selector)
    for dataset_selector in SYNTHETIC_SELECTORS
    for model in compatible_graph_models(
        GRAPH_DATASET_MANIFEST[dataset_selector]
    )
)


def _compose(
    dataset: str,
    model: str,
    *,
    batch_size: int | None = None,
    epochs: int = 1,
) -> DictConfig:
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    register_all_resolvers()
    overrides = [
        f"dataset=graph/{dataset}",
        f"model=graph/{model}",
        f"trainer.max_epochs={epochs}",
        "trainer.min_epochs=1",
        "trainer.check_val_every_n_epoch=1",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        "paths=test",
        "callbacks=model_checkpoint",
    ]
    if batch_size is not None:
        overrides.append(f"dataset.dataloader_params.batch_size={batch_size}")
    with hydra.initialize(
        version_base="1.3",
        config_path="../../configs",
        job_name="graph_capability_lifecycle",
    ):
        return hydra.compose(config_name="run.yaml", overrides=overrides)


def _build(cfg: DictConfig):
    pipeline = hydra.utils.instantiate(cfg.data_pipeline)
    output = pipeline.build(cfg)
    capability_validation = validate_capability_composition(
        cfg,
        observed=output.capability_spec,
    )
    model = instantiate_model(
        cfg,
        data_spec=None,
        capability_validation=capability_validation,
    )
    model.eval()
    model.state_str = "Training"
    return output.datamodule, model


@pytest.mark.parametrize(
    ("dataset_selector", "model_selector"),
    LIFECYCLE_PAIRS,
    ids=lambda value: value,
)
def test_declared_synthetic_contract_completes_one_epoch_and_final_test(
    dataset_selector: str,
    model_selector: str,
) -> None:
    result = run(_compose(dataset_selector, model_selector))

    assert result["epochs_completed"] >= 1
    assert {"train/loss", "val/loss"} <= result["fit_metrics"].keys()
    assert result["test_results"]
    if GRAPH_DATASET_MANIFEST[dataset_selector].task_level == "graph":
        assert result["observed_train_batch_size"] > 1


def test_scalar_regression_preserves_exact_batch_target_shape_including_remainder() -> (
    None
):
    cfg = _compose("SyntheticGraphRegression", "gcn", batch_size=3)
    datamodule, model = _build(cfg)
    observed_shapes: list[tuple[int, int]] = []

    model.on_train_epoch_start()
    try:
        for batch in datamodule.train_dataloader():
            model_out = model.model_step(batch)
            expected = (int(batch.num_graphs), 1)
            assert tuple(model_out["logits"].shape) == expected
            assert tuple(model_out["labels"].shape) == expected
            assert model_out["loss"].ndim == 0
            assert torch.isfinite(model_out["loss"])
            observed_shapes.append(expected)
    finally:
        model.abort_evaluation()

    assert observed_shapes == [(3, 1), (3, 1), (2, 1)]


@pytest.mark.parametrize("model_selector", ("gat", "gin", "gps", "nsd"))
def test_every_declared_non_gcn_regressor_preserves_scalar_shape(
    model_selector: str,
) -> None:
    cfg = _compose("SyntheticGraphRegression", model_selector)
    datamodule, model = _build(cfg)
    batch = next(iter(datamodule.train_dataloader()))
    model.on_train_epoch_start()
    try:
        model_out = model.model_step(batch)
        assert model_out["logits"].shape == model_out["labels"].shape
        assert tuple(model_out["logits"].shape) == (
            int(batch.num_graphs),
            1,
        )
        assert torch.isfinite(model_out["loss"])
    finally:
        model.abort_evaluation()


@pytest.mark.parametrize("model_selector", ("gps", "nsd"))
@pytest.mark.parametrize(
    ("dataset_selector", "expected_task_level"),
    (("SyntheticGraph", "graph"), ("SyntheticNodeGraph", "node")),
)
def test_gps_and_nsd_preserve_native_outputs_at_both_classification_levels(
    model_selector: str,
    dataset_selector: str,
    expected_task_level: str,
) -> None:
    cfg = _compose(dataset_selector, model_selector)
    datamodule, model = _build(cfg)
    batch = next(iter(datamodule.train_dataloader()))

    model_out = model.forward(batch)
    assert set(model_out) == {"x", "labels", "batch", "logits"}
    assert model.task_level == expected_task_level
    expected_examples = (
        int(batch.num_graphs)
        if expected_task_level == "graph"
        else int(batch.num_nodes)
    )
    assert model_out["logits"].shape == (
        expected_examples,
        int(cfg.dataset.parameters.num_classes),
    )


@pytest.mark.parametrize("model_selector", tuple(GRAPH_MODEL_CAPABILITIES))
@pytest.mark.parametrize("field", ("edge_attr", "edge_weight"))
def test_every_model_enforces_its_declared_optional_edge_mode(
    model_selector: str,
    field: str,
) -> None:
    cfg = _compose("SyntheticNodeGraph", model_selector)
    datamodule, model = _build(cfg)
    batch = next(iter(datamodule.train_dataloader())).clone()
    edge_count = int(batch.edge_index.size(1))
    batch[field] = (
        torch.randn(edge_count, 2)
        if field == "edge_attr"
        else torch.linspace(0.25, 1.25, edge_count)
    )
    mode = getattr(
        GRAPH_MODEL_CAPABILITIES[model_selector],
        f"{field}_mode",
    )

    assert model.backbone.edge_modes[field] == mode
    if mode == "reject":
        with pytest.raises(ValueError, match=f"{field} is unsupported"):
            model.forward(batch)
    else:
        model_out = model.forward(batch)
        assert set(model_out) == {"x", "labels", "batch", "logits"}


def test_gcn_edge_weights_change_logits_and_zero_matches_edge_removal() -> (
    None
):
    """The declared consumed weights control native GCN messages."""
    assert GRAPH_MODEL_CAPABILITIES["gcn"].edge_weight_mode == "consume"
    backbone = GCN(
        in_channels=2,
        hidden_channels=2,
        num_layers=2,
        out_channels=2,
        dropout=0.0,
    )
    with torch.no_grad():
        for index, parameter in enumerate(backbone.parameters(), start=1):
            parameter.copy_(
                torch.linspace(
                    0.1 * index,
                    0.1 * index + 0.05,
                    parameter.numel(),
                    dtype=parameter.dtype,
                ).reshape_as(parameter)
            )
    backbone.eval()
    wrapper = GNNWrapper(
        backbone,
        edge_attr_mode="ignore",
        edge_weight_mode="consume",
    )
    readout = NoReadOut(
        hidden_dim=2,
        out_channels=2,
        task_level="node",
        logits_linear_layer=False,
    )
    x = torch.tensor([[1.0, 0.25], [0.5, 2.0], [3.0, 0.75]])
    labels = torch.tensor([0, 1, 0])
    edge_index = torch.tensor([[0, 1], [1, 2]])

    def logits(edges: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        data = Data(
            x=x.clone(),
            edge_index=edges.clone(),
            edge_weight=weights.clone(),
            y=labels.clone(),
        )
        return readout(wrapper(data), data)["logits"]

    weighted = logits(edge_index, torch.tensor([0.25, 2.0]))
    unweighted = logits(edge_index, torch.ones(2))
    zero_bridge = logits(edge_index, torch.tensor([1.0, 0.0]))
    removed_bridge = logits(edge_index[:, :1], torch.ones(1))

    assert not torch.allclose(weighted, unweighted)
    torch.testing.assert_close(zero_bridge, removed_bridge)


def _cycle_edges(node_count: int, offset: int = 0) -> torch.Tensor:
    nodes = torch.arange(node_count, dtype=torch.long)
    successors = (nodes + 1) % node_count
    return (
        torch.cat(
            (
                torch.stack((nodes, successors)),
                torch.stack((successors, nodes)),
            ),
            dim=1,
        )
        + offset
    )


@pytest.mark.parametrize("model_state", ("Validation", "Test"))
def test_graph_mlp_auxiliary_loss_is_zero_outside_training(
    model_state: str,
) -> None:
    embeddings = torch.randn(3, 5, dtype=torch.float64)
    loss = GraphMLPLoss()(
        {"x": embeddings},
        Data(model_state=model_state),
    )

    assert loss.shape == torch.Size([])
    assert loss.dtype == embeddings.dtype
    assert loss.device == embeddings.device
    assert loss.item() == 0.0


@pytest.mark.parametrize(
    "batch",
    (Data(), Data(model_state="training")),
)
def test_graph_mlp_auxiliary_loss_requires_valid_model_state(
    batch: Data,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"batch\['model_state'\] must be one of",
    ):
        GraphMLPLoss()({"x": torch.randn(3, 5)}, batch)


def test_graph_mlp_contrastive_loss_isolates_disjoint_graphs() -> None:
    torch.manual_seed(7)
    first_x = torch.randn(3, 5)
    second_x = torch.randn(4, 5)
    first = Data(
        x=first_x,
        edge_index=_cycle_edges(3),
        batch=torch.zeros(3, dtype=torch.long),
        model_state="Training",
    )
    second = Data(
        x=second_x,
        edge_index=_cycle_edges(4),
        batch=torch.zeros(4, dtype=torch.long),
        model_state="Training",
    )
    combined = Data(
        x=torch.cat((first_x, second_x)),
        edge_index=torch.cat((_cycle_edges(3), _cycle_edges(4, 3)), dim=1),
        batch=torch.tensor([0, 0, 0, 1, 1, 1, 1]),
        model_state="Training",
    )
    loss = GraphMLPLoss(r_adj_power=2)

    actual = loss({"x": combined.x}, combined)
    expected = (
        3 * loss({"x": first.x}, first) + 4 * loss({"x": second.x}, second)
    ) / 7

    torch.testing.assert_close(actual, expected)


def test_gcn_dgm_auxiliary_edges_masks_and_gradients_are_batch_isolated() -> (
    None
):
    torch.manual_seed(11)
    batch_index = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    train_mask = torch.tensor([True, False, True, True, False, True, False])
    model = GCNDGM(
        in_channels=4,
        hidden_channels=8,
        num_layers=1,
        k=3,
    )

    output = model(
        torch.randn(7, 4),
        torch.empty((2, 0), dtype=torch.long),
        batch=batch_index,
    )
    auxiliary_edges = model.last_auxiliary_edge_index
    auxiliary_logprobs = model.last_auxiliary_logprobs

    assert output.shape == (7, 8)
    assert auxiliary_edges is not None
    assert auxiliary_logprobs is not None
    assert auxiliary_logprobs.ndim == 1
    assert auxiliary_logprobs.numel() == auxiliary_edges.size(1)
    assert torch.equal(
        batch_index[auxiliary_edges[0]],
        batch_index[auxiliary_edges[1]],
    )
    for graph_id in (0, 1):
        local_mask = train_mask & batch_index.eq(graph_id)
        assert not torch.any(local_mask & batch_index.ne(graph_id))
        incident = local_mask[auxiliary_edges[0]]
        assert torch.all(batch_index[auxiliary_edges[1, incident]] == graph_id)

    output.square().mean().backward()
    assert model.structure_encoder.linear.weight.grad is not None
    assert torch.count_nonzero(model.structure_encoder.linear.weight.grad)


def test_gcn_dgm_config_records_exact_search_and_scale_provenance() -> None:
    cfg = _compose("SyntheticNodeGraph", "gcn_dgm")

    assert cfg.model.backbone.query_chunk_size == 256
    assert cfg.model.backbone.max_nodes == 20_000
    assert cfg.model.backbone.max_workspace_bytes == 768 * 1024**2
    assert GRAPH_DATASET_MANIFEST["SyntheticNodeGraph"].num_nodes == 18


def test_gcn_dgm_requires_qualified_node_count_evidence() -> None:
    compatible = {
        capability.selector
        for capability in compatible_graph_models(
            GRAPH_DATASET_MANIFEST["roman_empire"]
        )
    }

    assert "gcn_dgm" not in compatible


def test_gcn_dgm_rejects_infeasible_k_before_model_construction() -> None:
    cfg = _compose("SyntheticNodeGraph", "gcn_dgm")
    with open_dict(cfg.model.backbone):
        cfg.model.backbone.k = 18

    with pytest.raises(
        ValueError,
        match=r"model\.backbone\.k=18 must be less than qualified "
        r"dataset node count 18",
    ):
        validate_graph_composition(cfg.dataset, cfg.model)


def test_gcn_dgm_rejects_dataset_above_node_limit_before_training() -> None:
    cfg = _compose("SyntheticNodeGraph", "gcn_dgm")
    with open_dict(cfg.model.backbone):
        cfg.model.backbone.max_nodes = 17

    with pytest.raises(
        ValueError,
        match=r"qualified dataset node count 18 exceeds "
        r"model\.backbone\.max_nodes=17",
    ):
        validate_graph_composition(cfg.dataset, cfg.model)


def test_gcn_dgm_rejects_workspace_limit_before_training() -> None:
    cfg = _compose("SyntheticNodeGraph", "gcn_dgm")
    with open_dict(cfg.model.backbone):
        cfg.model.backbone.query_chunk_size = 4
        cfg.model.backbone.max_workspace_bytes = 1

    with pytest.raises(
        ValueError,
        match=r"model\.backbone\.max_workspace_bytes=1 cannot admit "
        r"qualified dataset node count 18",
    ):
        validate_graph_composition(cfg.dataset, cfg.model)
