"""Test pipeline for OC20/OC22 datasets."""

import hydra
import pytest
from test._utils.simplified_pipeline import run


class TestPipeline:
    """Test pipeline for OC20 and OC22 datasets."""

    def setup_method(self):
        """Setup method."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    
    # IS2RE and OC22 tests commented out to prevent large dataset downloads during testing
    # def test_pipeline_oc20_is2re(self):
    #     """Test pipeline with OC20 IS2RE dataset."""
    #     dataset = "graph/OC20_IS2RE"
    #     model = "graph/gcn"
    #     
    #     with hydra.initialize(config_path="../../configs", job_name="job"):
    #         cfg = hydra.compose(
    #             config_name="run.yaml",
    #             overrides=[
    #                 f"model={model}",
    #                 f"dataset={dataset}",
    #                 "trainer.max_epochs=2",
    #                 "trainer.min_epochs=1",
    #                 "trainer.check_val_every_n_epoch=1",
    #                 "paths=test",
    #                 "callbacks=model_checkpoint",
    #                 "dataset.dataloader_params.num_workers=0",
    #                 "dataset.dataloader_params.persistent_workers=false",
    #             ],
    #             return_hydra_config=True
    #         )
    #         run(cfg)

    # def test_pipeline_oc22_is2re(self):
    #     """Test pipeline with OC22 IS2RE dataset."""
    #     dataset = "graph/OC22_IS2RE"
    #     model = "graph/gcn"
    #     
    #     with hydra.initialize(config_path="../../configs", job_name="job"):
    #         cfg = hydra.compose(
    #             config_name="run.yaml",
    #             overrides=[
    #                 f"model={model}",
    #                 f"dataset={dataset}",
    #                 "trainer.max_epochs=2",
    #                 "trainer.min_epochs=1",
    #                 "trainer.check_val_every_n_epoch=1",
    #                 "paths=test",
    #                 "callbacks=model_checkpoint",
    #                 "dataset.dataloader_params.num_workers=0",
    #                 "dataset.dataloader_params.persistent_workers=false",
    #             ],
    #             return_hydra_config=True
    #         )
    #         run(cfg)

    def test_pipeline_oc20_s2ef(self):
        """Test pipeline with OC20 S2EF dataset."""
        dataset = "graph/OC20_S2EF_200K_mock"
        model = "graph/gcn"
        
        with hydra.initialize(config_path="../../configs", job_name="job"):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    f"model={model}",
                    f"dataset={dataset}",
                    "trainer.max_epochs=2",
                    "trainer.min_epochs=1",
                    "trainer.check_val_every_n_epoch=1",
                    "paths=test",
                    "callbacks=model_checkpoint",
                    "dataset.dataloader_params.num_workers=0",
                    "dataset.dataloader_params.persistent_workers=false",
                ],
                return_hydra_config=True
            )
            run(cfg)
