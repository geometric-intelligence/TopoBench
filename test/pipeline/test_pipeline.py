"""Test pipeline for a particular dataset and model."""

import hydra
import pytest

from test._utils.simplified_pipeline import run


DATASET = "graph/MUTAG"  # ADD YOUR DATASET HERE
MODELS = ["graph/gcn", "cell/topotune", "simplicial/topotune"]  # ADD ONE OR SEVERAL MODELS


class TestPipeline:
    """Test pipeline for a particular dataset and model."""

    def setup_method(self):
        """Setup method."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    @pytest.mark.parametrize("model", MODELS)
    def test_pipeline(self, model, tmp_path, monkeypatch):
        """Test pipeline."""
        monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
        with hydra.initialize(
            version_base="1.3", config_path="../../configs", job_name="job"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    f"model={model}",
                    f"dataset={DATASET}",
                    "trainer.max_epochs=2",
                    "trainer.min_epochs=1",
                    "trainer.check_val_every_n_epoch=1",
                    "paths=test",
                    "callbacks=model_checkpoint",
                    "trainer.accelerator=cpu",
                    "trainer.devices=1",
                ],
                return_hydra_config=True,
            )
            run(cfg)
        hydra.core.global_hydra.GlobalHydra.instance().clear()
