"""Required challenge pipeline test for the coordinate-free ETNN baseline.

The challenge asks each submission to fill this file with a compatible dataset
and model config. This test exercises the normal TopoBench training path for
``model=combinatorial/etnn`` on MUTAG: Hydra composition, graph-to-combinatorial
lifting, feature encoding, ETNN message passing, readout, loss, and trainer.
"""

import hydra

from test._utils.simplified_pipeline import run

DATASET = "graph/MUTAG"
MODELS = ["combinatorial/etnn"]


class TestPipeline:
    """Run the submitted ETNN config through the standard TopoBench pipeline."""

    def setup_method(self):
        """Reset Hydra so this test is isolated from previous compositions."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    def test_pipeline(self):
        """Verify that ETNN trains end-to-end on a compatible graph dataset."""
        with hydra.initialize(config_path="../../configs", job_name="job"):
            for MODEL in MODELS:
                # MUTAG is intentionally small, so this is a practical CI smoke
                # test while still covering the full graph -> combinatorial
                # lifting -> ETNN -> readout execution path required by the
                # challenge.
                cfg = hydra.compose(
                    config_name="run.yaml",
                    overrides=[
                        f"model={MODEL}",
                        f"dataset={DATASET}",
                        "trainer.max_epochs=2",
                        "trainer.min_epochs=1",
                        "trainer.check_val_every_n_epoch=1",
                        "trainer.accelerator=cpu",
                        "trainer.devices=1",
                        "paths=test",
                        "callbacks=model_checkpoint",
                    ],
                    return_hydra_config=True,
                )
                run(cfg)
