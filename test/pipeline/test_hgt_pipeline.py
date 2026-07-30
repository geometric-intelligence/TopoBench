"""Pipeline configuration tests for the cell-complex HGT model."""

import math
from pathlib import Path

import hydra
import pytest
import torch

from test._utils.simplified_pipeline import run
from topobench.nn.backbones.combinatorial.hgt import CellHGT
from topobench.utils.config_resolvers import register_all_resolvers

NEIGHBORHOODS = [
    "up_incidence-0",
    "down_incidence-1",
    "up_incidence-1",
    "down_incidence-2",
]
EDGE_TYPES = [
    ("rank_0", "up_incidence-0", "rank_1"),
    ("rank_1", "down_incidence-1", "rank_0"),
    ("rank_1", "up_incidence-1", "rank_2"),
    ("rank_2", "down_incidence-2", "rank_1"),
]


@pytest.mark.parametrize(
    ("experiment", "expected_dataset", "expected_batch_size"),
    [
        ("cell_hgt_mutag_debug", "MUTAG", 16),
        ("cell_hgt_proteins_debug", "PROTEINS", 32),
        ("cell_hgt_zinc", "ZINC", 128),
    ],
)
def test_hgt_experiment_composes(
    experiment: str,
    expected_dataset: str,
    expected_batch_size: int,
) -> None:
    """Compose each HGT experiment with its intended dataset and batching."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    register_all_resolvers()
    with hydra.initialize(version_base="1.3", config_path="../../configs"):
        cfg = hydra.compose(
            config_name="run.yaml",
            overrides=[f"experiment={experiment}"],
            return_hydra_config=True,
        )

    assert cfg.model._target_ == "topobench.model.TBModel"
    assert (
        cfg.model.backbone._target_
        == "topobench.nn.backbones.combinatorial.hgt.CellHGT"
    )
    assert cfg.model.model_name == "hgt"
    assert list(cfg.model.backbone.neighborhoods) == NEIGHBORHOODS
    assert cfg.dataset.loader.parameters.data_name == expected_dataset
    assert cfg.dataset.dataloader_params.batch_size == expected_batch_size
    assert cfg.dataset.dataloader_params.batch_size > 1


@pytest.mark.parametrize(
    ("dataset", "expected_in_channels", "expected_out_channels"),
    [
        ("MUTAG", [7, 4, 4], 2),
        ("PROTEINS", [3, 3, 3], 2),
        ("ZINC", [21, 21, 21], 1),
    ],
    ids=["mutag", "proteins", "zinc"],
)
def test_hgt_model_config_composes_and_instantiates(
    dataset: str,
    expected_in_channels: list[int],
    expected_out_channels: int,
) -> None:
    """Compose and instantiate CellHGT through the standard model config."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    register_all_resolvers()
    with hydra.initialize(version_base="1.3", config_path="../../configs"):
        cfg = hydra.compose(
            config_name="run.yaml",
            overrides=[
                "model=cell/hgt",
                f"dataset=graph/{dataset}",
                "trainer.accelerator=cpu",
                "trainer.devices=1",
            ],
        )

    model = hydra.utils.instantiate(
        cfg.model,
        evaluator=cfg.evaluator,
        optimizer=cfg.optimizer,
        loss=cfg.loss,
    )

    assert isinstance(model.backbone.backbone, CellHGT)
    assert list(model.feature_encoder.in_channels) == expected_in_channels
    assert model.backbone.backbone.neighborhoods == NEIGHBORHOODS
    assert model.backbone.backbone.edge_types == EDGE_TYPES
    assert model.backbone.backbone.metadata[0] == [
        "rank_0",
        "rank_1",
        "rank_2",
    ]
    assert cfg.transforms.graph2cell_lifting.transform_name == (
        "CellCycleLifting"
    )
    assert cfg.transforms.graph2cell_lifting.max_cell_length == 10
    assert cfg.dataset.dataloader_params.batch_size > 1
    assert cfg.model.backbone_wrapper.num_cell_dimensions == 3
    assert cfg.model.readout.num_cell_dimensions == 3
    assert list(model.backbone.dimensions) == [0, 1, 2]
    assert list(model.readout.dimensions) == [2, 1]
    assert model.hparams.model_name == "hgt"

    logits = model.readout.compute_logits(
        torch.zeros(3, model.readout.hidden_dim),
        torch.tensor([0, 0, 1]),
    )
    assert logits.shape == (2, expected_out_channels)

    if dataset == "ZINC":
        assert (
            cfg.transforms.one_hot_node_degree_features.transform_name
            == "OneHotDegreeFeatures"
        )


@pytest.mark.parametrize(
    ("dataset", "batch_size"),
    [("graph/MUTAG", 16), ("graph/PROTEINS", 32)],
)
def test_hgt_two_epoch_batched_pipeline(
    dataset: str,
    batch_size: int,
    tmp_path: Path,
) -> None:
    """Train, validate, and test batched CellHGT for two epochs."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    dataset_name = dataset.rsplit("/", maxsplit=1)[-1].lower()
    output_dir = tmp_path / dataset_name
    with hydra.initialize(version_base="1.3", config_path="../../configs"):
        cfg = hydra.compose(
            config_name="run.yaml",
            overrides=[
                "model=cell/hgt",
                f"dataset={dataset}",
                f"dataset.dataloader_params.batch_size={batch_size}",
                "model.feature_encoder.out_channels=32",
                "model.backbone.heads=4",
                "model.backbone.num_layers=2",
                "model.backbone.dropout=0.0",
                "trainer.max_epochs=2",
                "trainer.min_epochs=1",
                "trainer.check_val_every_n_epoch=1",
                "trainer.accelerator=cpu",
                "trainer.devices=1",
                "paths=test",
                "callbacks=model_checkpoint",
                f"trainer.default_root_dir={output_dir}",
                f"callbacks.model_checkpoint.dirpath={output_dir / 'checkpoints'}",
            ],
            return_hydra_config=True,
        )

    result = run(cfg)

    assert result["epochs_completed"] == 2
    assert result["observed_train_batch_size"] == batch_size
    assert result["observed_train_batch_size"] > 1

    for metric_name in ("train/loss", "val/loss"):
        assert metric_name in result["fit_metrics"], (
            f"Missing fit metric: {metric_name}"
        )
        assert math.isfinite(float(result["fit_metrics"][metric_name]))

    assert result["test_results"], "trainer.test returned no results"
    test_metrics = result["test_results"][0]
    assert "test/loss" in test_metrics, "Missing test metric: test/loss"
    assert math.isfinite(float(test_metrics["test/loss"]))
