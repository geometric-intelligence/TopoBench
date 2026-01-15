"""Test pipeline for a particular dataset and model."""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import hydra
from test._utils.simplified_pipeline import run
import argparse

class TestPipeline:
    """Test pipeline for a particular dataset and model.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing dataset and models configuration.
    """

    def __init__(self, args):
        self.dataset = args.dataset
        self.models = args.models if isinstance(args.models, list) else [args.models]

    def setup_method(self):
        """Setup method."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    
    def test_pipeline(self):
        """Test pipeline."""
        with hydra.initialize(config_path="../../configs", job_name="job"):
            for MODEL in self.models:
                cfg = hydra.compose(
                    config_name="run.yaml",
                    overrides=[
                        f"model={MODEL}",
                        f"dataset={self.dataset}", # IF YOU IMPLEMENT A LARGE DATASET WITH AN OPTION TO USE A SLICE OF IT, ADD BELOW THE CORRESPONDING OPTION
                        "trainer.max_epochs=2",
                        "trainer.min_epochs=1",
                        "trainer.check_val_every_n_epoch=1",
                        "paths=test",
                        "callbacks=model_checkpoint",
                    ],
                    return_hydra_config=True
                )
                run(cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Pipeline")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to data root directory",
        choices = ['graph/MUTAG', 'graph/a123']  # ADD YOUR DATASET HERE
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        required=False,
        default=["graph/gcn"],
        help="Model(s) to use in the pipeline",
    )
    args = parser.parse_args()

    test_pipeline = TestPipeline(args)
    test_pipeline.setup_method()
    test_pipeline.test_pipeline()