"""Test pipeline for the Metamath dataset and a simple GNN model."""

import hydra
from test._utils.simplified_pipeline import run
from hydra.core.global_hydra import GlobalHydra

# Your dataset + a simple graph model from TopoBench
DATASET = "graph/metamath"
MODELS  = ["graph/gcn"]   # could also try ["graph/gin"] if that config exists


class TestPipeline:
    """End-to-end pipeline test for Metamath."""

    def setup_method(self):
        """Reset Hydra between tests."""
        GlobalHydra.instance().clear()

    def test_pipeline(self):
        """Run a very short training job and ensure it completes."""
        with hydra.initialize(config_path="../../configs", job_name="metamath_test"):
            for MODEL in MODELS:
                cfg = hydra.compose(
                    config_name="run.yaml",
                    overrides=[
                        f"model={MODEL}",
                        f"dataset={DATASET}",
                        "trainer.max_epochs=2",
                        "trainer.min_epochs=1",
                        "trainer.check_val_every_n_epoch=1",
                        "paths=test",
                        "callbacks=model_checkpoint", 
                    ],
                    return_hydra_config=True,
                )
