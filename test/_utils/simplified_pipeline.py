"""Test pipeline for a particular dataset and model."""

from collections.abc import Mapping
from typing import TypedDict

import hydra
from lightning import seed_everything
from omegaconf import DictConfig

from topobench.utils import instantiate_callbacks
from topobench.utils.config_resolvers import register_all_resolvers

register_all_resolvers()


class PipelineRunResult(TypedDict):
    """Structured evidence from a simplified pipeline run."""

    epochs_completed: int
    observed_train_batch_size: int
    fit_metrics: dict[str, float]
    test_results: list[Mapping[str, float]]


def run(cfg: DictConfig) -> PipelineRunResult:
    """Run pipeline with given configuration.

    Parameters
    ----------
    cfg : DictConfig
        Configuration.

    Returns
    -------
    PipelineRunResult
        Evidence captured from training, validation, and testing.
    """
    seed_everything(cfg.seed, workers=True)

    pipeline = hydra.utils.instantiate(cfg.data_pipeline)
    pipeline_output = pipeline.build(cfg)
    datamodule = pipeline_output.datamodule

    first_train_batch = next(iter(datamodule.train_dataloader()))
    observed_train_batch_size = int(first_train_batch.num_graphs)

    # Model for us is Network + logic: inputs backbone, readout, losses
    model = hydra.utils.instantiate(
        cfg.model,
        evaluator=cfg.evaluator,
        optimizer=cfg.optimizer,
        loss=cfg.loss,
    )
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(
        model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
    )
    fit_metrics = {
        metric_name: float(
            trainer.callback_metrics[metric_name].detach().cpu().item()
        )
        for metric_name in ("train/loss", "val/loss")
        if metric_name in trainer.callback_metrics
    }
    epochs_completed = int(trainer.current_epoch)
    ckpt_path = trainer.checkpoint_callback.best_model_path
    test_results = trainer.test(
        model=model, datamodule=datamodule, ckpt_path=ckpt_path
    )

    return {
        "epochs_completed": epochs_completed,
        "observed_train_batch_size": observed_train_batch_size,
        "fit_metrics": fit_metrics,
        "test_results": test_results,
    }
