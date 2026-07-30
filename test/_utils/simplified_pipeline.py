"""Test pipeline for a particular dataset and model."""

from collections.abc import Mapping
from typing import TypedDict

import hydra
from lightning import seed_everything
from omegaconf import DictConfig

from topobench.data.preprocessor import PreProcessor
from topobench.dataloader import TBDataloader
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

    # Instantiate and load dataset
    dataset_loader = hydra.utils.instantiate(cfg.dataset.loader)
    dataset, dataset_dir = dataset_loader.load()
    # Preprocess dataset and load the splits
    transform_config = cfg.get("transforms", None)
    preprocessor = PreProcessor(dataset, dataset_dir, transform_config)
    dataset_train, dataset_val, dataset_test = (
        preprocessor.load_dataset_splits(cfg.dataset.split_params)
    )
    # Prepare datamodule
    if cfg.dataset.parameters.task_level in ["node", "graph"]:
        datamodule = TBDataloader(
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            dataset_test=dataset_test,
            **cfg.dataset.get("dataloader_params", {}),
        )
    else:
        raise ValueError("Invalid task_level")

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
