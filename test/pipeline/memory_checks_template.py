"""Test pipeline for a particular dataset and model."""

import hydra

from test._utils.simplified_pipeline import run

# Use your contributed dataset + a lightweight, well-supported model
DATASET = "placeholder/dataset"
MODELS = ["placeholder/model"]

class TestPipeline:
    """Test pipeline for a particular dataset and model."""

    def setup_method(pythself):
        """Setup method."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    def test_pipeline(self):
        """Test pipeline."""
        # The config path is relative to this file: ../../configs
        with hydra.initialize(config_path="../../configs", job_name="job"):
            for MODEL in MODELS:
                cfg = hydra.compose(
                    config_name="run.yaml",
                    overrides=[
                        f"model={MODEL}",
                        f"dataset={DATASET}",
                        # keep CI fast & deterministic
                        "trainer.max_epochs=2",
                        "trainer.min_epochs=1",
                        "trainer.check_val_every_n_epoch=1",
                        "trainer.accelerator=cpu",
                        "trainer.devices=1",
                        "seed=42",
                        # write under a tmp test path
                        "paths=test",
                        # reuse the same callback config used elsewhere in tests
                        "callbacks=model_checkpoint"
                    ],
                    return_hydra_config=True,
                )
                # print(cfg)
                run(cfg)
