"""Pipeline configuration tests for the cell-complex HGT model."""

import hydra

from topobench.nn.backbones.combinatorial.hgt import CellHGT
from topobench.utils.config_resolvers import register_all_resolvers


def test_hgt_model_config_composes_and_instantiates():
    """Compose and instantiate CellHGT through the standard model config."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    register_all_resolvers()
    with hydra.initialize(version_base="1.3", config_path="../../configs"):
        cfg = hydra.compose(
            config_name="run.yaml",
            overrides=[
                "model=cell/hgt",
                "dataset=graph/MUTAG",
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
    assert model.backbone.backbone.metadata[0] == [
        "rank_0",
        "rank_1",
        "rank_2",
    ]
    assert model.hparams.model_name == "hgt"
