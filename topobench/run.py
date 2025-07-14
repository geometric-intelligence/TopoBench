"""Main entry point for training and testing models."""

import random
from typing import Any

import hydra
import lightning as L
import numpy as np
import rootutils
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from topobench.data.preprocessor import PreProcessor
from topobench.dataloader import TBDataloader
from topobench.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from topobench.utils.config_resolvers import (
    get_default_metrics,
    get_default_trainer,
    get_default_transform,
    get_monitor_metric,
    get_monitor_mode,
    get_required_lifting,
    infer_in_channels,
    infer_num_cell_dimensions,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #


OmegaConf.register_new_resolver(
    "get_default_metrics", get_default_metrics, replace=True
)
OmegaConf.register_new_resolver(
    "get_default_trainer", get_default_trainer, replace=True
)
OmegaConf.register_new_resolver(
    "get_default_transform", get_default_transform, replace=True
)
OmegaConf.register_new_resolver(
    "get_required_lifting", get_required_lifting, replace=True
)
OmegaConf.register_new_resolver(
    "get_monitor_metric", get_monitor_metric, replace=True
)
OmegaConf.register_new_resolver(
    "get_monitor_mode", get_monitor_mode, replace=True
)
OmegaConf.register_new_resolver(
    "infer_in_channels", infer_in_channels, replace=True
)
OmegaConf.register_new_resolver(
    "infer_num_cell_dimensions", infer_num_cell_dimensions, replace=True
)
OmegaConf.register_new_resolver(
    "parameter_multiplication", lambda x, y: int(int(x) * int(y)), replace=True
)


def initialize_hydra() -> DictConfig:
    """Initialize Hydra when main is not an option (e.g. tests).

    Returns
    -------
    DictConfig
        A DictConfig object containing the config tree.
    """
    hydra.initialize(
        version_base="1.3", config_path="../configs", job_name="run"
    )
    cfg = hydra.compose(config_name="run.yaml")
    return cfg


torch.set_num_threads(1)
log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def run(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train the model.

    Can additionally evaluate on a testset, using best weights obtained during training.

    This method is wrapped in optional @task_wrapper decorator, that controls
    the behavior during failure. Useful for multiruns, saving info about the
    crash, etc.

    Parameters
    ----------
    cfg : DictConfig
        Configuration composed by Hydra.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        A tuple with metrics and dict with all instantiated objects.
    """
    # Set seed for random number generators in pytorch, numpy and python.random
    # if cfg.get("seed"):
    L.seed_everything(cfg.seed, workers=True)
    # Seed for torch
    torch.manual_seed(cfg.seed)
    # Seed for numpy
    np.random.seed(cfg.seed)
    # Seed for python random
    random.seed(cfg.seed)

    # Instantiate and load dataset
    log.info(f"Instantiating loader <{cfg.dataset.loader._target_}>")
    dataset_loader = hydra.utils.instantiate(cfg.dataset.loader)
    dataset, dataset_dir = dataset_loader.load()
    # Preprocess dataset and load the splits
    log.info("Instantiating preprocessor...")
    transform_config = cfg.get("transforms", None)
    preprocessor = PreProcessor(dataset, dataset_dir, transform_config)
    dataset_train, dataset_val, dataset_test = (
        preprocessor.load_dataset_splits(cfg.dataset.split_params)
    )
    # Prepare datamodule
    log.info("Instantiating datamodule...")
    if cfg.dataset.parameters.task_level in ["node", "graph"]:
        datamodule = TBDataloader(
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            dataset_test=dataset_test,
            **cfg.dataset.get("dataloader_params", {}),
        )
    else:
        raise ValueError("Invalid task_level")

    # Model for us is Network + logic: inputs backbone, readout, losses
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.model,
        evaluator=cfg.evaluator,
        optimizer=cfg.optimizer,
        loss=cfg.loss,
    )

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
        )
        # Log the best model checkpoint path into wandb
        for logger_elem in logger:
            if isinstance(
                logger_elem, L.pytorch.loggers.wandb.WandbLogger
            ) and hasattr(logger_elem, "experiment"):
                logger_elem.experiment.log(
                    {"checkpoint": trainer.checkpoint_callback.best_model_path}
                )

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        test_best_model_path = True
        if cfg.get("ckpt_path"):
            ckpt_path = cfg.ckpt_path
            log.info(
                f"Attempting to load weights from the provided ckpt_path: {ckpt_path}"
            )
            try:
                trainer.test(
                    model=model, datamodule=datamodule, ckpt_path=ckpt_path
                )
                test_best_model_path = False  # do not test "best model" if a valid ckpt_path is provided
            except FileNotFoundError:
                log.warning(
                    f"No checkpoint file found at the provided ckpt_path: {ckpt_path}."
                )
                log.info("Trying with best model instead...")
        if test_best_model_path:
            ckpt_path = trainer.checkpoint_callback.best_model_path
            if ckpt_path == "":
                log.warning(
                    "Best ckpt not found! Using current weights for testing..."
                )
                ckpt_path = None
            trainer.test(
                model=model, datamodule=datamodule, ckpt_path=ckpt_path
            )

    test_metrics = trainer.callback_metrics

    # Merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


def count_number_of_parameters(
    model: torch.nn.Module, only_trainable: bool = True
) -> int:
    """Count the number of trainable params.

    If all params, specify only_trainable = False.

    Ref:
        - https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9?u=brando_miranda
        - https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model/62764464#62764464

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    only_trainable : bool, optional
        If True, only count trainable parameters (default: True).

    Returns
    -------
    int
        The number of parameters.
    """
    if only_trainable:
        num_params: int = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    else:  # counts trainable and none-traibale
        num_params: int = sum(p.numel() for p in model.parameters() if p)
    assert num_params > 0, f"Err: {num_params=}"
    return int(num_params)


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="run.yaml"
)
def main(cfg: DictConfig) -> float | None:
    """Main entry point for training.

    Parameters
    ----------
    cfg : DictConfig
        Configuration composed by Hydra.

    Returns
    -------
    float | None
        Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = run(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
