"""Main entry point for training and testing models."""

import hashlib
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from types import MappingProxyType
from typing import Any

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from topobench.callbacks.input_pipeline import (
    InputPipelineCallback,
    create_input_monitor,
)
from topobench.domains import SUPPORTED_DOMAINS
from topobench.evaluator import EvaluationResult
from topobench.nn.capabilities import validate_graph_composition
from topobench.preflight import PreflightRunner
from topobench.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    instantiate_model,
    log_hyperparameters,
    task_wrapper,
)
from topobench.utils.config_resolvers import register_all_resolvers

_DOMAIN_PIPELINE_TARGETS = {
    "graph": "topobench.data.pipelines.DefaultDataPipeline",
    "heterogeneous": (
        "topobench.data.pipelines.HeterogeneousNodeDataPipeline"
    ),
    "hypergraph": "topobench.data.pipelines.HypergraphNodeDataPipeline",
}

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


# Register custom resolvers before Hydra initialization
register_all_resolvers()


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


def validate_domain_composition(cfg: DictConfig) -> str:
    """Reject unsupported or cross-domain dataset, model, and pipeline compositions."""
    if not isinstance(cfg, DictConfig):
        raise TypeError("cfg must be a DictConfig")

    dataset_path = "cfg.dataset.loader.parameters.data_domain"
    model_path = "cfg.model.model_domain"
    dataset_domain = OmegaConf.select(
        cfg,
        "dataset.loader.parameters.data_domain",
        default=None,
    )
    model_domain = OmegaConf.select(cfg, "model.model_domain", default=None)
    for path, domain in (
        (dataset_path, dataset_domain),
        (model_path, model_domain),
    ):
        if domain is None:
            raise ValueError(f"{path} is required")
        if not isinstance(domain, str):
            raise TypeError(f"{path} must be a string")
        if domain not in SUPPORTED_DOMAINS:
            raise ValueError(
                f"{path}={domain!r} is unsupported; "
                f"expected one of {SUPPORTED_DOMAINS}"
            )
    if dataset_domain != model_domain:
        raise ValueError(
            "Cross-domain composition is unsupported: "
            f"{dataset_path}={dataset_domain!r} does not match "
            f"{model_path}={model_domain!r}"
        )
    pipeline_path = "cfg.data_pipeline._target_"
    pipeline_target = OmegaConf.select(
        cfg,
        "data_pipeline._target_",
        default=None,
    )
    if pipeline_target is None:
        raise ValueError(f"{pipeline_path} is required")
    if not isinstance(pipeline_target, str):
        raise TypeError(f"{pipeline_path} must be a string")
    expected_pipeline_target = _DOMAIN_PIPELINE_TARGETS[dataset_domain]
    if pipeline_target != expected_pipeline_target:
        raise ValueError(
            "Cross-domain pipeline composition is unsupported: "
            f"{dataset_path}={dataset_domain!r} requires "
            f"{pipeline_path}={expected_pipeline_target!r}, "
            f"got {pipeline_target!r}"
        )
    if dataset_domain == "graph":
        validate_graph_composition(cfg.dataset, cfg.model)
    return dataset_domain


def _instantiate_execution_monitor(
    callbacks_cfg: DictConfig | None,
) -> object | None:
    """Construct only the input monitor before pipeline conversion begins."""

    if callbacks_cfg is None:
        return None
    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")
    target = "topobench.callbacks.input_pipeline.InputPipelineCallback"
    monitor_configs = [
        callback
        for callback in callbacks_cfg.values()
        if isinstance(callback, DictConfig)
        and callback.get("_target_") == target
    ]
    if len(monitor_configs) > 1:
        raise ValueError(
            "At most one InputPipelineCallback may own execution evidence"
        )
    if not monitor_configs:
        return None
    resolved = OmegaConf.to_container(monitor_configs[0], resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(
            "InputPipelineCallback config must resolve to a mapping"
        )
    monitor_keys = {
        "event_log_path",
        "event_capacity",
        "pending_cuda_capacity",
        "overflow_policy",
        "sample_every_n",
        "sample_offset",
        "warmup_steps",
        "rolling_window_steps",
        "max_input_stall_fraction",
        "max_consecutive_starved_steps",
        "patience_windows",
        "stall_action",
    }
    monitor_kwargs = {
        key: value for key, value in resolved.items() if key in monitor_keys
    }
    return create_input_monitor(**monitor_kwargs)


def _shared_execution_monitor(
    callbacks: list[Callback],
) -> object | None:
    """Return the sole callback-owned monitor for pre-training pipeline work."""
    owners = [
        callback
        for callback in callbacks
        if isinstance(callback, InputPipelineCallback)
    ]
    if len(owners) > 1:
        raise ValueError(
            "At most one InputPipelineCallback may own execution evidence"
        )
    return None if not owners else owners[0].monitor


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
    # Lightning is the single authority for Python, NumPy, torch, and workers.
    L.seed_everything(cfg.seed, workers=True)

    if cfg.get("deterministic", False):
        # Enable cudnn deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        log.info(
            "Enabled cudnn.deterministic and torch.use_deterministic_algorithms"
        )

    execution_monitor = _instantiate_execution_monitor(cfg.get("callbacks"))
    log.info(f"Instantiating data pipeline <{cfg.data_pipeline._target_}>")
    pipeline = hydra.utils.instantiate(
        cfg.data_pipeline,
        execution_monitor=execution_monitor,
    )
    pipeline_output = pipeline.build(cfg)
    datamodule = pipeline_output.datamodule

    def model_factory() -> LightningModule:
        return instantiate_model(
            cfg,
            data_spec=pipeline_output.data_spec,
        )

    preflight = PreflightRunner(cfg, pipeline_output)
    static_preflight = preflight.validate_static()
    preflight_result = preflight.run_probe(
        model_factory=model_factory,
        static_result=static_preflight,
    )

    # Discard probe RNG effects before constructing any production object.
    L.seed_everything(cfg.seed, workers=True)

    # Model for us is Network + logic: inputs backbone, readout, losses
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = model_factory()

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(
        cfg.get("callbacks"),
        input_pipeline_monitor=execution_monitor,
    )
    callback_monitor = _shared_execution_monitor(callbacks)
    if callback_monitor is not execution_monitor:
        raise RuntimeError(
            "production InputPipelineCallback did not adopt the preflight monitor"
        )
    if execution_monitor is not None:
        if hasattr(datamodule, "execution_monitor"):
            datamodule.execution_monitor = execution_monitor
        model.execution_monitor = execution_monitor

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))
    if not logger:
        callbacks = [
            callback
            for callback in callbacks
            if not isinstance(callback, LearningRateMonitor)
        ]

    # Log to wandb preprocessor time
    if logger:
        for log_temp in logger:
            if isinstance(log_temp, L.pytorch.loggers.wandb.WandbLogger):
                log_temp.log_metrics(
                    {
                        "preprocessor_time": pipeline_output.preprocessing_time,
                    }
                )

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,
        log_every_n_steps=1,  # Log metrics every step (Lightning requires >=1)
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
        "data_spec": pipeline_output.data_spec,
        "pipeline_output": pipeline_output,
        "preflight": preflight_result,
        "selected_checkpoint_results": MappingProxyType({}),
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
        )

    train_metrics = trainer.callback_metrics
    selected_checkpoint_results: Mapping[str, EvaluationResult] = (
        MappingProxyType({})
    )
    if cfg.get("test"):
        log.info("Starting testing!")
        selected_checkpoint_results = rerun_best_model_checkpoint(
            checkpoint_model=model,
            cfg=cfg,
            datamodule=datamodule,
            device=model.device,
            callbacks=callbacks,
            logger=logger,
        )
        object_dict["selected_checkpoint_results"] = (
            selected_checkpoint_results
        )

    selected_metrics = {
        name: value
        for result in selected_checkpoint_results.values()
        for name, value in _selected_checkpoint_payload(result).items()
    }
    # The qualification bit remains authoritative even when preflight is
    # explicitly disabled under the experimental profile.
    metric_dict = {
        **train_metrics,
        **selected_metrics,
        "qualified": preflight_result.qualified,
    }

    return metric_dict, object_dict


def _selected_checkpoint_payload(
    result: EvaluationResult,
) -> dict[str, object]:
    """Flatten only one authoritative result's scalar logger boundary."""
    namespace = f"evaluations/best_checkpoint/{result.context.split}/"
    return {
        **{
            f"{namespace}{name}": value
            for name, value in result.metrics.items()
        },
        f"{namespace}num_examples": int(result.num_examples),
    }


def _publish_selected_checkpoint_result(
    result: EvaluationResult,
    loggers: list[Logger],
) -> None:
    """Publish the same result-derived payload to every configured logger."""
    payload = _selected_checkpoint_payload(result)
    log.info(payload)
    for configured_logger in loggers:
        log_metrics = getattr(configured_logger, "log_metrics", None)
        if not callable(log_metrics):
            raise TypeError(
                "configured logger must expose a callable log_metrics boundary"
            )
        log_metrics(payload)


def _take_selected_checkpoint_result(
    checkpoint_model: LightningModule,
    *,
    split: str,
    checkpoint_id: str,
) -> EvaluationResult:
    take_result = getattr(
        checkpoint_model,
        "take_selected_checkpoint_result",
        None,
    )
    if not callable(take_result):
        raise TypeError(
            "checkpoint model must expose take_selected_checkpoint_result"
        )
    result = take_result(split)
    if not isinstance(result, EvaluationResult):
        raise TypeError("checkpoint model must return an EvaluationResult")
    if (
        result.context.split != split
        or result.context.pass_kind != "selected_checkpoint"
        or result.context.checkpoint_id != checkpoint_id
    ):
        raise RuntimeError(
            "selected-checkpoint result context does not match the rerun"
        )
    if result.num_examples <= 0:
        raise RuntimeError(
            f"selected-checkpoint {split} result must not be empty"
        )
    return result


def rerun_best_model_checkpoint(
    checkpoint_model: LightningModule,
    cfg: DictConfig,
    datamodule: LightningDataModule,
    device: torch.device,
    callbacks: list[Callback],
    logger: list[Logger],
) -> Mapping[str, EvaluationResult]:
    """Load one validation-selected checkpoint and evaluate val and test."""
    checkpoint_callbacks = [
        callback
        for callback in callbacks
        if isinstance(callback, ModelCheckpoint)
    ]
    if len(checkpoint_callbacks) != 1:
        raise RuntimeError(
            "selected-checkpoint evaluation requires exactly one "
            f"ModelCheckpoint callback, found {len(checkpoint_callbacks)}"
        )
    selected_checkpoint = checkpoint_callbacks[0]
    if not selected_checkpoint.best_model_path:
        raise RuntimeError(
            "validation did not select a non-empty checkpoint path"
        )
    model_path = Path(selected_checkpoint.best_model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"selected checkpoint does not exist: {model_path}"
        )
    with model_path.open("rb") as checkpoint_stream:
        checkpoint_id = hashlib.file_digest(
            checkpoint_stream,
            "sha256",
        ).hexdigest()

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("selected checkpoint must contain a mapping")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError(
            "selected checkpoint must contain a non-empty state_dict"
        )
    checkpoint_model.load_state_dict(state_dict, strict=True)
    checkpoint_model.to(device)

    bind_checkpoint_id = getattr(
        checkpoint_model,
        "set_selected_checkpoint_id",
        None,
    )
    if not callable(bind_checkpoint_id):
        raise TypeError(
            "checkpoint model must expose set_selected_checkpoint_id"
        )
    bind_checkpoint_id(checkpoint_id)

    checkpoint_trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        num_sanity_val_steps=0,
        enable_progress_bar=cfg.get("enable_progress_bar", True),
        logger=False,
    )
    set_validation_pass_kind = getattr(
        checkpoint_model,
        "set_next_validation_pass_kind",
        None,
    )
    if not callable(set_validation_pass_kind):
        raise TypeError(
            "checkpoint model must expose set_next_validation_pass_kind"
        )
    set_test_pass_kind = getattr(
        checkpoint_model,
        "set_next_test_pass_kind",
        None,
    )
    if not callable(set_test_pass_kind):
        raise TypeError("checkpoint model must expose set_next_test_pass_kind")

    log.info(
        "Re-evaluating validation-selected checkpoint %s on validation",
        checkpoint_id,
    )
    val_loader = datamodule.val_dataloader()
    set_validation_pass_kind("selected_checkpoint")
    try:
        checkpoint_trainer.validate(
            model=checkpoint_model,
            dataloaders=val_loader,
        )
    except BaseException:
        with suppress(RuntimeError):
            set_validation_pass_kind("fit_epoch")
        raise
    val_result = _take_selected_checkpoint_result(
        checkpoint_model,
        split="val",
        checkpoint_id=checkpoint_id,
    )
    _publish_selected_checkpoint_result(val_result, logger)

    log.info(
        "Re-evaluating validation-selected checkpoint %s on test",
        checkpoint_id,
    )
    test_loader = datamodule.test_dataloader()
    set_test_pass_kind("selected_checkpoint")
    try:
        checkpoint_trainer.test(
            model=checkpoint_model,
            dataloaders=test_loader,
        )
    except BaseException:
        with suppress(RuntimeError):
            set_test_pass_kind("fit_epoch")
        raise
    test_result = _take_selected_checkpoint_result(
        checkpoint_model,
        split="test",
        checkpoint_id=checkpoint_id,
    )
    _publish_selected_checkpoint_result(test_result, logger)

    results: dict[str, EvaluationResult] = {
        "val": val_result,
        "test": test_result,
    }
    immutable_results: Mapping[str, EvaluationResult] = MappingProxyType(
        results
    )

    if cfg.get("delete_checkpoint_after_test", False):
        log.info("Cleaning up: Deleting checkpoint at %s", model_path)
        try:
            model_path.unlink()
        except OSError as error:
            log.warning(
                "Failed to delete checkpoint at %s. Error: %s",
                model_path,
                error,
            )
    return immutable_results


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
    validate_domain_composition(cfg)

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
