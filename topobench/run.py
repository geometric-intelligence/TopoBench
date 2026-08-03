"""Main entry point for training and testing models."""

import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import replace
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

from topobench.callbacks import (
    BestEpochMetricsCallback,
    SelectedCheckpointArtifactCallback,
    SplitPublication,
)
from topobench.callbacks.input_pipeline import (
    InputPipelineCallback,
    create_input_monitor,
)
from topobench.data.loaders.base import canonical_sha256, resolve_cache_config
from topobench.evaluator import EvaluationResult
from topobench.nn.capabilities import CapabilityValidation
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
from topobench.utils.artifact_logging import ArtifactLoggerAdapter
from topobench.utils.checkpoint_io import (
    LoadedSelectedCheckpoint,
    TrustedCheckpointIO,
    checkpoint_manifest_path,
    checkpoint_state_path,
    load_selected_checkpoint,
    validate_trusted_resume,
)
from topobench.utils.config_resolvers import register_all_resolvers
from topobench.utils.instantiators import (
    ExecutionProfileRecord,
    validate_execution_profile,
    validate_profile_capability,
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
    """Return the dataset domain after profile-aware capability validation."""
    profile_record = validate_execution_profile(cfg)
    return validate_profile_capability(
        cfg,
        profile_record=profile_record,
    ).model.data_domain


def _profile_provenance(
    profile_record: ExecutionProfileRecord,
) -> dict[str, object]:
    """Serialize the immutable execution profile for runtime provenance."""
    return {
        "profile": profile_record.profile,
        "qualified": profile_record.qualified,
        "targets": tuple(
            {"path": path, "import_path": import_path}
            for path, import_path in profile_record.targets
        ),
        "custom_targets": tuple(
            {"path": path, "import_path": import_path}
            for path, import_path in profile_record.custom_targets
        ),
    }


def _profiled_provenance_input(
    provenance_input: Mapping[str, object] | None,
    *,
    profile_record: ExecutionProfileRecord,
) -> dict[str, object]:
    """Add execution qualification without mutating pipeline provenance."""
    provenance = {} if provenance_input is None else dict(provenance_input)
    provenance["execution_profile"] = _profile_provenance(profile_record)
    return provenance


def _live_pipeline_provenance(
    datamodule: LightningDataModule,
    provenance_input: Mapping[str, object] | None,
) -> dict[str, object]:
    """Copy provenance and bind any fitted transform's published state."""

    provenance = {} if provenance_input is None else dict(provenance_input)
    fitted_transform = getattr(datamodule, "fitted_transform", None)
    if fitted_transform is None:
        return provenance
    state_key = getattr(fitted_transform, "state_key", None)
    if not _is_sha256(state_key):
        raise RuntimeError(
            "fitted transform must expose a valid immutable state_key "
            "before selected artifact publication"
        )
    recorded_state_key = provenance.get("fitted_transform_state_key")
    if recorded_state_key is not None and recorded_state_key != state_key:
        raise RuntimeError(
            "fitted transform immutable state_key changed after provenance "
            "binding"
        )
    provenance["fitted_transform_state_key"] = state_key
    return provenance


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
    profile_record = validate_execution_profile(cfg)
    if not profile_record.qualified:
        log.warning(
            "Experimental execution profile selected; outputs are unqualified."
        )
    capability_validation: CapabilityValidation = validate_profile_capability(
        cfg,
        profile_record=profile_record,
    )
    checkpoint_path = cfg.get("ckpt_path")
    if profile_record.qualified and checkpoint_path is not None:
        validate_trusted_resume(
            checkpoint_path,
            output_root=cfg.paths.output_dir,
            checkpoint_root=cfg.paths.checkpoint_dir,
        )
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
    capability_validation = validate_profile_capability(
        cfg,
        profile_record=profile_record,
        observed=pipeline_output.capability_spec,
    )
    datamodule = pipeline_output.datamodule

    def model_factory() -> LightningModule:
        return instantiate_model(
            cfg,
            data_spec=pipeline_output.data_spec,
            capability_validation=capability_validation,
            profile_record=profile_record,
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
    artifacts_enabled = bool(
        OmegaConf.select(
            cfg,
            "evaluation_artifacts.enabled",
            default=False,
        )
    )
    artifact_callback = (
        _selected_checkpoint_artifact_callback(callbacks)
        if artifacts_enabled
        else None
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
    trainer_kwargs: dict[str, object] = {
        "callbacks": callbacks,
        "logger": logger,
        "num_sanity_val_steps": 0,
        "log_every_n_steps": 1,
    }
    if profile_record.qualified:
        trainer_kwargs["plugins"] = TrustedCheckpointIO(
            output_root=cfg.paths.output_dir,
            checkpoint_root=cfg.paths.checkpoint_dir,
        )
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        **trainer_kwargs,
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
        "execution_profile": profile_record,
        "selected_checkpoint_results": MappingProxyType({}),
        "selected_checkpoint_publications": MappingProxyType({}),
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
        )

    original_provenance = getattr(pipeline_output, "provenance_input", None)
    live_provenance = _live_pipeline_provenance(
        datamodule,
        original_provenance,
    )
    if live_provenance != (
        {} if original_provenance is None else dict(original_provenance)
    ):
        pipeline_output = replace(
            pipeline_output,
            provenance_input=live_provenance,
        )
        object_dict["pipeline_output"] = pipeline_output
    profiled_provenance = _profiled_provenance_input(
        live_provenance,
        profile_record=profile_record,
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
            prediction_row_adapter=pipeline_output.prediction_row_adapter,
            supervision_counts=pipeline_output.supervision_counts,
            provenance_input=profiled_provenance,
            source_graph_id=pipeline_output.source_graph_id,
        )
        object_dict["selected_checkpoint_results"] = (
            selected_checkpoint_results
        )
        if artifact_callback is not None:
            object_dict["selected_checkpoint_publications"] = (
                artifact_callback.publications
            )

    selected_metrics: dict[str, object] = {}
    for split, result in selected_checkpoint_results.items():
        selected_metrics.update(_selected_checkpoint_payload(result))
        if artifact_callback is not None:
            publication = artifact_callback.publications.get(split)
            if publication is None:
                raise RuntimeError(
                    f"selected-checkpoint {split} publication is missing"
                )
            selected_metrics.update(_selected_publication_payload(publication))
    # The qualification bit remains authoritative even when preflight is
    # explicitly disabled under the experimental profile.
    metric_dict = {
        **train_metrics,
        **_best_epoch_payload(callbacks),
        **selected_metrics,
        "qualified": (profile_record.qualified and preflight_result.qualified),
    }

    return metric_dict, object_dict


def _selected_checkpoint_artifact_callback(
    callbacks: list[Callback],
) -> SelectedCheckpointArtifactCallback:
    """Return the sole configured owner of selected prediction artifacts."""

    matches = [
        callback
        for callback in callbacks
        if isinstance(callback, SelectedCheckpointArtifactCallback)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "enabled prediction artifacts require exactly one "
            "SelectedCheckpointArtifactCallback, "
            f"found {len(matches)}"
        )
    return matches[0]


def _best_epoch_payload(callbacks: list[Callback]) -> dict[str, object]:
    """Expose the callback-owned best epoch without scalar conversion."""

    matches = [
        callback
        for callback in callbacks
        if isinstance(callback, BestEpochMetricsCallback)
    ]
    if len(matches) > 1:
        raise RuntimeError(
            "best-epoch result authority requires at most one "
            f"BestEpochMetricsCallback, found {len(matches)}"
        )
    if not matches:
        return {}
    callback = matches[0]
    payload = {
        f"best_epoch/{name}": value
        for name, value in callback.best_epoch_metrics.items()
    }
    if callback.best_epoch_number is not None:
        payload["best_epoch"] = callback.best_epoch_number
    return payload


def _selected_publication_payload(
    publication: SplitPublication,
) -> dict[str, object]:
    """Expose only the canonical manifest locator at the metric boundary."""

    namespace = f"evaluations/best_checkpoint/{publication.split}/"
    return {
        f"{namespace}predictions/manifest_path": publication.manifest_file.path,
        (
            f"{namespace}predictions/manifest_sha256"
        ): publication.manifest_file.sha256,
    }


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
        f"{namespace}num_examples": result.num_examples,
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


def _resolved_config_section(cfg: DictConfig, name: str) -> object:
    """Resolve one config section for deterministic identity hashing."""

    value = OmegaConf.select(cfg, name, default=None)
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(
            value,
            resolve=True,
            throw_on_missing=True,
        )
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _selected_provenance_by_split(
    cfg: DictConfig,
    *,
    source_graph_id: str | None,
    provenance_input: Mapping[str, object] | None,
    expected_num_examples: Mapping[str, int],
    checkpoint_sha256: str,
) -> dict[str, dict[str, str]]:
    """Bind selected results to resolved data, split, transforms, and model."""

    provenance = {} if provenance_input is None else dict(provenance_input)
    normalized_provenance = resolve_cache_config(
        provenance,
        context="selected checkpoint pipeline provenance",
    )
    source_candidate = provenance.get("source_fingerprint", source_graph_id)
    source_fingerprint = (
        source_candidate
        if _is_sha256(source_candidate)
        else canonical_sha256(
            {
                "source_graph_id": source_graph_id,
                "loader": _resolved_config_section(cfg, "dataset.loader"),
            }
        )
    )
    dataset_candidate = provenance.get("dataset_fingerprint")
    dataset_fingerprint = (
        dataset_candidate
        if _is_sha256(dataset_candidate)
        else canonical_sha256(
            {
                "dataset": _resolved_config_section(cfg, "dataset"),
                "source_fingerprint": source_fingerprint,
                "pipeline_provenance": normalized_provenance,
            }
        )
    )
    transform_candidate = provenance.get("transform_fingerprint")
    fitted_state_key = provenance.get("fitted_transform_state_key")
    transform_fingerprint = (
        transform_candidate
        if _is_sha256(transform_candidate) and fitted_state_key is None
        else canonical_sha256(
            {
                "declared_transform_fingerprint": (
                    transform_candidate
                    if _is_sha256(transform_candidate)
                    else None
                ),
                "transforms": _resolved_config_section(cfg, "transforms"),
                "fitted_transform": provenance.get("fitted_transform"),
                "fitted_transform_state_key": fitted_state_key,
            }
        )
    )
    model_fingerprint = canonical_sha256(
        {
            "model": _resolved_config_section(cfg, "model"),
            "checkpoint_sha256": checkpoint_sha256,
        }
    )
    split_base = provenance.get("split_fingerprint")
    split_configuration = _resolved_config_section(
        cfg,
        "dataset.split_params",
    )
    split_provenance = dict(normalized_provenance)
    split_provenance.pop("execution_profile", None)
    return {
        split: {
            "source_fingerprint": source_fingerprint,
            "dataset_fingerprint": dataset_fingerprint,
            "split_fingerprint": canonical_sha256(
                {
                    "partition_fingerprint": (
                        split_base if _is_sha256(split_base) else None
                    ),
                    "split_configuration": split_configuration,
                    "pipeline_provenance": split_provenance,
                    "split": split,
                    "num_examples": expected_num_examples[split],
                }
            ),
            "model_fingerprint": model_fingerprint,
            "transform_fingerprint": transform_fingerprint,
        }
        for split in ("val", "test")
    }


def _checkpoint_counter(checkpoint: Mapping[str, object], name: str) -> int:
    value = checkpoint.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"selected checkpoint {name} must be a non-negative integer"
        )
    return value


def _restore_quarantined_checkpoint_artifact(
    quarantine_path: Path,
    public_path: Path,
) -> bool:
    """Restore without overwriting a newer object at the public path."""
    try:
        os.link(
            quarantine_path,
            public_path,
            follow_symlinks=False,
        )
    except FileExistsError:
        return False
    quarantine_path.unlink()
    return True


def _remove_loaded_checkpoint_artifacts(
    model_path: Path,
    loaded_checkpoint: LoadedSelectedCheckpoint,
    *,
    checkpoint_root: str | Path,
) -> None:
    """Delete only artifact inodes observed by trusted checkpoint loading."""
    artifact_paths = {
        "checkpoint": model_path,
        "manifest": checkpoint_manifest_path(model_path),
        "state": checkpoint_state_path(model_path),
    }
    quarantine_root = Path(
        tempfile.mkdtemp(
            prefix=".selected-checkpoint-cleanup-",
            dir=Path(checkpoint_root),
        )
    )
    try:
        for role in ("checkpoint", "manifest", "state"):
            expected_identity = loaded_checkpoint.artifact_identities.get(role)
            if expected_identity is None:
                continue
            public_path = artifact_paths[role]
            quarantine_path = quarantine_root / role
            try:
                os.replace(public_path, quarantine_path)
            except FileNotFoundError:
                continue
            if loaded_checkpoint.matches_artifact(role, quarantine_path):
                quarantine_path.unlink()
                continue
            restored = _restore_quarantined_checkpoint_artifact(
                quarantine_path,
                public_path,
            )
            recovery = (
                str(public_path)
                if restored
                else f"{quarantine_path} (public path already occupied)"
            )
            log.warning(
                "Skipped deletion of replaced selected-checkpoint "
                f"{role}; replacement retained at {recovery}"
            )
    finally:
        loaded_checkpoint.close()
        with suppress(OSError):
            quarantine_root.rmdir()


def rerun_best_model_checkpoint(
    checkpoint_model: LightningModule,
    cfg: DictConfig,
    datamodule: LightningDataModule,
    device: torch.device,
    callbacks: list[Callback],
    logger: list[Logger],
    prediction_row_adapter: object | None = None,
    supervision_counts: Mapping[str, int] | None = None,
    provenance_input: Mapping[str, object] | None = None,
    source_graph_id: str | None = None,
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
    loaded_checkpoint = load_selected_checkpoint(
        model_path,
        output_root=cfg.paths.output_dir,
        checkpoint_root=cfg.paths.checkpoint_dir,
    )
    checkpoint = loaded_checkpoint.checkpoint
    checkpoint_id = loaded_checkpoint.checkpoint_id
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError(
            "selected checkpoint must contain a non-empty state_dict"
        )
    checkpoint_model.load_state_dict(state_dict, strict=True)
    checkpoint_model.to(device)
    cleanup_checkpoint = bool(cfg.get("delete_checkpoint_after_test", False))
    if not cleanup_checkpoint:
        loaded_checkpoint.close()

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

    artifacts_enabled = bool(
        OmegaConf.select(
            cfg,
            "evaluation_artifacts.enabled",
            default=False,
        )
    )
    artifact_callback = (
        _selected_checkpoint_artifact_callback(callbacks)
        if artifacts_enabled
        else None
    )
    if artifact_callback is not None:
        provenance_input = _live_pipeline_provenance(
            datamodule,
            provenance_input,
        )
    expected_num_examples: dict[str, int] | None = None
    provenance_by_split: dict[str, dict[str, str]] | None = None
    artifact_logger: ArtifactLoggerAdapter | None = None
    if artifact_callback is not None:
        artifact_callback.configure_slice_evaluator_factory(
            lambda: hydra.utils.instantiate(cfg.evaluator)
        )
        checkpoint_epoch = _checkpoint_counter(checkpoint, "epoch")
        checkpoint_global_step = _checkpoint_counter(
            checkpoint,
            "global_step",
        )
        if prediction_row_adapter is None:
            raise RuntimeError(
                "enabled prediction artifacts require a prediction row adapter"
            )
        if supervision_counts is None:
            raise RuntimeError(
                "enabled prediction artifacts require supervision counts"
            )
        expected_num_examples = {}
        for split in ("val", "test"):
            count = supervision_counts.get(split)
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
            ):
                raise ValueError(
                    "selected-checkpoint artifact supervision count for "
                    f"{split} must be a positive integer"
                )
            expected_num_examples[split] = count
        provenance_by_split = _selected_provenance_by_split(
            cfg,
            source_graph_id=source_graph_id,
            provenance_input=provenance_input,
            expected_num_examples=expected_num_examples,
            checkpoint_sha256=checkpoint_id,
        )
        artifact_logger = ArtifactLoggerAdapter(
            logger,
            run_root=artifact_callback.run_root,
        )

    trainer_kwargs: dict[str, object] = {
        "num_sanity_val_steps": 0,
        "enable_progress_bar": cfg.get("enable_progress_bar", True),
        "logger": False,
    }
    if artifact_callback is not None:
        trainer_kwargs["callbacks"] = [artifact_callback]
    checkpoint_trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        **trainer_kwargs,
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

    configure_capture = None
    clear_capture = None
    selected_context = None
    capture_bound = False
    if artifact_callback is not None:
        configure_capture = getattr(
            checkpoint_model,
            "configure_prediction_artifact_capture",
            None,
        )
        clear_capture = getattr(
            checkpoint_model,
            "clear_prediction_artifact_capture",
            None,
        )
        selected_context = getattr(
            checkpoint_model,
            "selected_checkpoint_context",
            None,
        )
        if not all(
            callable(boundary)
            for boundary in (
                configure_capture,
                clear_capture,
                selected_context,
            )
        ):
            raise TypeError(
                "checkpoint model must expose selected artifact capture "
                "configuration, clearing, and context boundaries"
            )
        assert expected_num_examples is not None
        assert provenance_by_split is not None
        configure_capture(
            prediction_row_adapter,
            artifact_callback,
            expected_num_examples,
            provenance_by_split=provenance_by_split,
        )
        capture_bound = True

    try:
        log.info(
            "Re-evaluating validation-selected checkpoint "
            f"{checkpoint_id} on validation"
        )
        val_loader = datamodule.val_dataloader()
        if artifact_callback is not None:
            assert selected_context is not None
            artifact_callback.begin(
                selected_context("val"),
                checkpoint_path=model_path,
                checkpoint_sha256=checkpoint_id,
                checkpoint_epoch=checkpoint_epoch,
                checkpoint_global_step=checkpoint_global_step,
                world_size=checkpoint_trainer.world_size,
                global_rank=checkpoint_trainer.global_rank,
            )
        try:
            set_validation_pass_kind("selected_checkpoint")
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
        if artifact_callback is None:
            _publish_selected_checkpoint_result(val_result, logger)
        else:
            val_publication = artifact_callback.finalize(val_result)
            assert artifact_logger is not None
            artifact_logger.register(val_publication)

        log.info(
            "Re-evaluating validation-selected checkpoint "
            f"{checkpoint_id} on test"
        )
        test_loader = datamodule.test_dataloader()
        if artifact_callback is not None:
            assert selected_context is not None
            artifact_callback.begin(
                selected_context("test"),
                checkpoint_path=model_path,
                checkpoint_sha256=checkpoint_id,
                checkpoint_epoch=checkpoint_epoch,
                checkpoint_global_step=checkpoint_global_step,
                world_size=checkpoint_trainer.world_size,
                global_rank=checkpoint_trainer.global_rank,
            )
        try:
            set_test_pass_kind("selected_checkpoint")
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
        if artifact_callback is None:
            _publish_selected_checkpoint_result(test_result, logger)
        else:
            test_publication = artifact_callback.finalize(test_result)
            assert artifact_logger is not None
            artifact_logger.register(test_publication)
    except BaseException:
        with suppress(BaseException):
            abort_evaluation = getattr(
                checkpoint_model,
                "abort_evaluation",
                None,
            )
            if callable(abort_evaluation):
                abort_evaluation()
        if artifact_callback is not None:
            with suppress(BaseException):
                artifact_callback.abort()
        if capture_bound:
            if clear_capture is not None:
                with suppress(BaseException):
                    clear_capture()
            capture_bound = False
        loaded_checkpoint.close()
        raise
    finally:
        if capture_bound:
            assert clear_capture is not None
            clear_capture()

    results: dict[str, EvaluationResult] = {
        "val": val_result,
        "test": test_result,
    }
    immutable_results: Mapping[str, EvaluationResult] = MappingProxyType(
        results
    )

    if cleanup_checkpoint:
        log.info(f"Cleaning up: Deleting checkpoint at {model_path}")
        try:
            _remove_loaded_checkpoint_artifacts(
                model_path,
                loaded_checkpoint,
                checkpoint_root=cfg.paths.checkpoint_dir,
            )
        except OSError as error:
            log.warning(
                f"Failed to delete checkpoint at {model_path}. Error: {error}"
            )
        finally:
            loaded_checkpoint.close()
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
