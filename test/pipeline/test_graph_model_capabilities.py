"""Network-free lifecycle gates for every declared graph model contract."""

from __future__ import annotations

import hydra
import pytest
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data

from test._utils.simplified_pipeline import run
from topobench.data.capabilities import GRAPH_DATASET_MANIFEST
from topobench.loss.model import GraphMLPLoss
from topobench.nn.backbones.graph import GCNDGM
from topobench.nn.capabilities import (
    GRAPH_MODEL_CAPABILITIES,
    compatible_graph_models,
)
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
    model = instantiate_model(cfg, data_spec=None)
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


def test_scalar_regression_preserves_exact_batch_target_shape_including_remainder() -> None:
    cfg = _compose("SyntheticGraphRegression", "gcn", batch_size=3)
    datamodule, model = _build(cfg)
    observed_shapes: list[tuple[int, int]] = []

    for batch in datamodule.train_dataloader():
        model_out = model.model_step(batch)
        expected = (int(batch.num_graphs), 1)
        assert tuple(model_out["logits"].shape) == expected
        assert tuple(model_out["labels"].shape) == expected
        assert model_out["loss"].ndim == 0
        assert torch.isfinite(model_out["loss"])
        observed_shapes.append(expected)

    assert observed_shapes == [(3, 1), (3, 1), (2, 1)]


@pytest.mark.parametrize("model_selector", ("gat", "gin", "gps", "nsd"))
def test_every_declared_non_gcn_regressor_preserves_scalar_shape(
    model_selector: str,
) -> None:
    cfg = _compose("SyntheticGraphRegression", model_selector)
    datamodule, model = _build(cfg)
    batch = next(iter(datamodule.train_dataloader()))

    model_out = model.model_step(batch)
    assert model_out["logits"].shape == model_out["labels"].shape
    assert tuple(model_out["logits"].shape) == (int(batch.num_graphs), 1)
    assert torch.isfinite(model_out["loss"])


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


def _cycle_edges(node_count: int, offset: int = 0) -> torch.Tensor:
    nodes = torch.arange(node_count, dtype=torch.long)
    successors = (nodes + 1) % node_count
    return torch.cat(
        (
            torch.stack((nodes, successors)),
            torch.stack((successors, nodes)),
        ),
        dim=1,
    ) + offset


def test_graph_mlp_contrastive_loss_isolates_disjoint_graphs() -> None:
    torch.manual_seed(7)
    first_x = torch.randn(3, 5)
    second_x = torch.randn(4, 5)
    first = Data(
        x=first_x,
        edge_index=_cycle_edges(3),
        batch=torch.zeros(3, dtype=torch.long),
    )
    second = Data(
        x=second_x,
        edge_index=_cycle_edges(4),
        batch=torch.zeros(4, dtype=torch.long),
    )
    combined = Data(
        x=torch.cat((first_x, second_x)),
        edge_index=torch.cat((_cycle_edges(3), _cycle_edges(4, 3)), dim=1),
        batch=torch.tensor([0, 0, 0, 1, 1, 1, 1]),
    )
    loss = GraphMLPLoss(r_adj_power=2)

    actual = loss({"x": combined.x}, combined)
    expected = (
        3 * loss({"x": first.x}, first)
        + 4 * loss({"x": second.x}, second)
    ) / 7

    torch.testing.assert_close(actual, expected)


def test_gcn_dgm_auxiliary_edges_masks_and_gradients_are_batch_isolated() -> None:
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
    assert auxiliary_logprobs.size(0) == batch_index.numel()
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
