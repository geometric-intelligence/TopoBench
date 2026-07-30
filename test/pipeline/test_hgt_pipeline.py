"""Pipeline configuration tests for the cell-complex HGT model."""

import hydra
import pytest
import torch

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
