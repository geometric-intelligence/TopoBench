"""Pipeline smoke checks for the combinatorial ETNN model family.

The fast test composes Hydra config only. The actual one-epoch training smoke
tests are kept in this file but skipped by default because they may download
and preprocess MUTAG or QM9. That gives us ready manual/CI checks without
making normal unit-test runs depend on network/data availability.
"""

from __future__ import annotations

import hydra
import pytest

from test._utils.simplified_pipeline import run


class TestETNNPipeline:
    """End-to-end checks for ETNN-family combinatorial model configs."""

    def setup_method(self):
        """Reset Hydra between tests so overrides stay isolated."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    def test_etnn_config_composes_with_graph_to_combinatorial_lifting(self):
        """ETNN config should select the expected TopoBench lifting path."""
        # This is the cheapest integration check: it verifies Hydra can resolve
        # the graph dataset, combinatorial ETNN model, and required lifting
        # without instantiating datasets or trainers.
        with hydra.initialize(version_base="1.3", config_path="../../configs"):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "model=combinatorial/etnn",
                    "dataset=graph/MUTAG",
                ],
                return_hydra_config=False,
            )

        # The model config should resolve to the new combinatorial ETNN
        # backbone.
        assert cfg.model.model_name == "etnn"
        assert (
            cfg.model.backbone._target_
            == "topobench.nn.backbones.combinatorial.etnn.ETNN"
        )

        # Graph datasets need a graph-to-combinatorial lifting before ETNN can
        # consume rank-wise cell features and neighborhoods.
        assert "graph2combinatorial_lifting" in cfg.transforms
        lifting = cfg.transforms.graph2combinatorial_lifting
        assert lifting.transform_name == "GraphTriangleInducedCC"

        # Hydra resolves the lifting interpolation to the ETNN neighborhood
        # list. This ensures preprocessing creates exactly the sparse relations
        # the backbone will consume.
        assert list(lifting.neighborhoods) == list(
            cfg.model.backbone.neighborhoods
        )

    def test_etnn_lappe_config_composes_with_lappe_before_lifting(self):
        """LapPE ETNN config should compute coordinates before lifting."""
        with hydra.initialize(version_base="1.3", config_path="../../configs"):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "model=combinatorial/etnn_lappe",
                    "dataset=graph/MUTAG",
                ],
                return_hydra_config=False,
            )

        assert cfg.model.model_name == "etnn_lappe"
        assert (
            cfg.model.backbone._target_
            == "topobench.nn.backbones.combinatorial.etnn_lappe.ETNNLapPE"
        )
        assert cfg.model.backbone.coordinate_attr == "LapPE"

        # The model-specific transform default should run LapPE first, storing
        # rank-0 structural coordinates separately from node features.
        assert "lappe_coordinates" in cfg.transforms
        lappe = cfg.transforms.lappe_coordinates
        assert lappe.transform_name == "LapPE"
        assert lappe.concat_to_x is False
        assert lappe.max_pe_dim == 3

        # The required graph-to-combinatorial lifting should still be present
        # and should use the neighborhoods consumed by the ETNN backbone.
        assert "graph2combinatorial_lifting" in cfg.transforms
        lifting = cfg.transforms.graph2combinatorial_lifting
        assert lifting.transform_name == "GraphTriangleInducedCC"
        assert list(lifting.neighborhoods) == list(
            cfg.model.backbone.neighborhoods
        )

    def test_coordinate_policy_none_config_uses_lifting_only(self):
        """Coordinate-policy none config should require no coordinates."""
        with hydra.initialize(version_base="1.3", config_path="../../configs"):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "model=combinatorial/etnn_coordinate_policy_none",
                    "dataset=graph/MUTAG",
                ],
                return_hydra_config=False,
            )

        # This consolidated config should use the new coordinate-policy
        # backbone, but keep the same coordinate-free behavior as ETNN.
        assert cfg.model.model_name == "etnn_coordinate_policy_none"
        assert (
            cfg.model.backbone._target_
            == "topobench.nn.backbones.combinatorial."
            "etnn_coordinate_policy.ETNNCoordinatePolicy"
        )
        assert cfg.model.backbone.coordinate_policy == "none"

        # No coordinate preprocessing should be introduced for the none policy.
        # Graph datasets still need the combinatorial lifting that creates
        # rank-wise cell features and sparse ETNN neighborhoods.
        assert list(cfg.transforms.keys()) == ["graph2combinatorial_lifting"]
        lifting = cfg.transforms.graph2combinatorial_lifting
        assert lifting.transform_name == "GraphTriangleInducedCC"
        assert list(lifting.neighborhoods) == list(
            cfg.model.backbone.neighborhoods
        )

    def test_coordinate_policy_lappe_config_adds_lappe_before_lifting(self):
        """Coordinate-policy LapPE config should run LapPE before lifting."""
        with hydra.initialize(version_base="1.3", config_path="../../configs"):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "model=combinatorial/etnn_coordinate_policy_lappe",
                    "dataset=graph/MUTAG",
                ],
                return_hydra_config=False,
            )

        assert cfg.model.model_name == "etnn_coordinate_policy_lappe"
        assert (
            cfg.model.backbone._target_
            == "topobench.nn.backbones.combinatorial."
            "etnn_coordinate_policy.ETNNCoordinatePolicy"
        )
        assert cfg.model.backbone.coordinate_policy == "structural_lappe"
        assert cfg.model.backbone.structural_coordinate_attr == "LapPE"

        # The structural policy must compute LapPE separately from node
        # features, then lift the graph to a combinatorial complex. Keeping
        # concat_to_x=false ensures LapPE is treated as coordinates, not as
        # ordinary node features.
        assert list(cfg.transforms.keys()) == [
            "lappe_coordinates",
            "graph2combinatorial_lifting",
        ]
        lappe = cfg.transforms.lappe_coordinates
        assert lappe.transform_name == "LapPE"
        assert lappe.concat_to_x is False
        assert lappe.max_pe_dim == 3

        lifting = cfg.transforms.graph2combinatorial_lifting
        assert lifting.transform_name == "GraphTriangleInducedCC"
        assert list(lifting.neighborhoods) == list(
            cfg.model.backbone.neighborhoods
        )

    def test_coordinate_policy_physical_config_uses_qm9_positions(self):
        """Physical coordinate-policy config should target datasets with pos."""
        with hydra.initialize(version_base="1.3", config_path="../../configs"):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "model=combinatorial/etnn_coordinate_policy_physical",
                    "dataset=graph/QM9",
                ],
                return_hydra_config=False,
            )

        assert cfg.model.model_name == "etnn_coordinate_policy_physical"
        assert (
            cfg.model.backbone._target_
            == "topobench.nn.backbones.combinatorial."
            "etnn_coordinate_policy.ETNNCoordinatePolicy"
        )
        assert cfg.model.backbone.coordinate_policy == "physical"
        assert cfg.model.backbone.physical_coordinate_attr == "pos"
        assert cfg.model.backbone.pos_update is True
        assert (
            cfg.model.backbone.coordinate_update_neighborhood
            == "up_adjacency-0"
        )
        assert cfg.model.backbone.hausdorff_dists is True
        assert cfg.model.backbone.normalize_invariants is True
        assert cfg.model.backbone.invariant_normalization == "batch_norm"
        assert cfg.model.backbone.invariant_normalization_eps == 1e-8

        # QM9 is still loaded as a graph dataset, so ETNN needs the same
        # graph-to-combinatorial lifting. The difference from the none policy
        # is that the backbone requires preserved rank-0 `pos` values at runtime,
        # derives invariant physical relation attributes from them, and updates
        # rank-0 coordinates internally between ETNN layers.
        assert list(cfg.transforms.keys()) == ["graph2combinatorial_lifting"]
        lifting = cfg.transforms.graph2combinatorial_lifting
        assert lifting.transform_name == "GraphTriangleInducedCC"
        assert list(lifting.neighborhoods) == list(
            cfg.model.backbone.neighborhoods
        )

    @pytest.mark.skip(
        reason=(
            "One-epoch ETNN pipeline run may download/process MUTAG. "
            "Run manually when network/data setup is available."
        )
    )
    def test_etnn_one_epoch_pipeline_smoke(self):
        """Exercise lifting -> encoder -> ETNN -> wrapper -> readout -> loss."""
        # This is the real end-to-end check to run when data/network access is
        # available. It intentionally stays here as executable documentation for
        # the exact command path ETNN must support.
        with hydra.initialize(version_base="1.3", config_path="../../configs"):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "model=combinatorial/etnn",
                    "dataset=graph/MUTAG",
                    "trainer.max_epochs=1",
                    "trainer.min_epochs=1",
                    "trainer.check_val_every_n_epoch=1",
                    "trainer.accelerator=cpu",
                    "trainer.devices=1",
                    "paths=test",
                    "callbacks=model_checkpoint",
                ],
                return_hydra_config=True,
            )

            # The simplified pipeline instantiates the dataset loader,
            # preprocessing/lifting, datamodule, TBModel, trainer, and test
            # loop. Keeping this as an explicit smoke test protects the ETNN
            # config from drifting away from TopoBench's normal execution path.
            run(cfg)

    @pytest.mark.skip(
        reason=(
            "One-epoch physical ETNN QM9 run may download/process QM9. "
            "Run manually when network/data setup is available."
        )
    )
    def test_coordinate_policy_physical_qm9_one_epoch_pipeline_smoke(self):
        """Exercise QM9 -> lifting -> physical ETNN -> readout -> loss."""
        # Physical coordinate-policy ETNN is the consolidated PR's new
        # paper-faithful path. This skipped smoke documents the exact
        # end-to-end command path without making ordinary CI process QM9.
        with hydra.initialize(version_base="1.3", config_path="../../configs"):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "model=combinatorial/etnn_coordinate_policy_physical",
                    "dataset=graph/QM9",
                    "trainer.max_epochs=1",
                    "trainer.min_epochs=1",
                    "trainer.check_val_every_n_epoch=1",
                    "trainer.accelerator=cpu",
                    "trainer.devices=1",
                    "paths=test",
                    "callbacks=model_checkpoint",
                ],
                return_hydra_config=True,
            )

            # The simplified pipeline should instantiate QM9, preserve rank-0
            # physical coordinates through graph-to-combinatorial lifting, and
            # run physical invariant ETNN message passing with coordinate
            # updates enabled by the model config.
            run(cfg)
