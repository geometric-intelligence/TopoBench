"""Hydra instantiation and exact execution-profile validation."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from topobench.utils import pylogger
from topobench.utils.artifact_logging import ARTIFACT_LOGGER_TARGETS

if TYPE_CHECKING:
    from topobench.nn.capabilities import CapabilityValidation

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

TargetRecord = tuple[str, str]
TargetRecords = tuple[TargetRecord, ...]

_QUALIFIED_PROFILE = "qualified"
_EXPERIMENTAL_PROFILE = "experimental"
_SUPPORTED_PROFILES = frozenset({_QUALIFIED_PROFILE, _EXPERIMENTAL_PROFILE})

_FIXED_TARGETS: Mapping[str, str] = MappingProxyType(
    {
        "trainer._target_": "lightning.pytorch.trainer.Trainer",
        "evaluator._target_": "topobench.evaluator.evaluator.TBEvaluator",
        "loss._target_": "topobench.loss.TBLoss",
        "optimizer._target_": "topobench.optimizer.TBOptimizer",
    }
)
_CALLBACK_TARGETS: Mapping[str, str] = MappingProxyType(
    {
        "best_epoch_metrics": ("topobench.callbacks.BestEpochMetricsCallback"),
        "dataloader_commit": (
            "topobench.callbacks.dataloader_commit.DataloaderCommitCallback"
        ),
        "early_stopping": "lightning.pytorch.callbacks.EarlyStopping",
        "input_pipeline": (
            "topobench.callbacks.input_pipeline.InputPipelineCallback"
        ),
        "learning_rate_monitor": (
            "lightning.pytorch.callbacks.LearningRateMonitor"
        ),
        "model_checkpoint": ("lightning.pytorch.callbacks.ModelCheckpoint"),
        "model_summary": "lightning.pytorch.callbacks.RichModelSummary",
        "model_timer": "topobench.callbacks.timer_callback.PipelineTimer",
        "prediction_artifacts": (
            "topobench.callbacks.SelectedCheckpointArtifactCallback"
        ),
        "rich_progress_bar": ("lightning.pytorch.callbacks.RichProgressBar"),
    }
)
_LOGGER_TARGETS: Mapping[str, str] = ARTIFACT_LOGGER_TARGETS


@dataclass(frozen=True, slots=True)
class ExecutionProfileRecord:
    """Immutable audit record for every configured Hydra executable."""

    profile: str
    qualified: bool
    targets: TargetRecords
    custom_targets: TargetRecords

    def __post_init__(self) -> None:
        if not isinstance(self.profile, str):
            raise TypeError("profile must be a string")
        if self.profile not in _SUPPORTED_PROFILES:
            raise ValueError("profile must be 'qualified' or 'experimental'")
        if not isinstance(self.qualified, bool):
            raise TypeError("qualified must be bool")
        if self.qualified != (self.profile == _QUALIFIED_PROFILE):
            raise ValueError("qualified must agree with profile")
        _validate_target_records("targets", self.targets)
        _validate_target_records("custom_targets", self.custom_targets)
        if tuple(sorted(self.targets)) != self.targets:
            raise ValueError("targets must be deterministically sorted")
        if tuple(sorted(self.custom_targets)) != self.custom_targets:
            raise ValueError("custom_targets must be deterministically sorted")
        if not set(self.custom_targets).issubset(self.targets):
            raise ValueError("custom_targets must be a subset of targets")


def _validate_target_records(name: str, records: object) -> None:
    """Validate one immutable, unique target-record tuple."""
    if not isinstance(records, tuple):
        raise TypeError(f"{name} must be a tuple")
    for record in records:
        if (
            not isinstance(record, tuple)
            or len(record) != 2
            or not all(isinstance(value, str) for value in record)
        ):
            raise TypeError(f"{name} entries must be (str, str) tuples")
    paths = tuple(path for path, _ in records)
    if len(paths) != len(set(paths)):
        raise ValueError(f"{name} paths must be unique")


def _unresolved_container(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a traversal view without resolving unrelated config values."""
    if isinstance(cfg, DictConfig):
        container = OmegaConf.to_container(cfg, resolve=False)
        if not isinstance(container, Mapping):
            raise TypeError("cfg must be a mapping")
        return container
    return cfg


def _collect_targets(
    value: object,
    *,
    path: str = "",
) -> TargetRecords:
    """Collect every recursively configured ``_target_`` deterministically."""
    collected: list[TargetRecord] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not path and key == "hydra":
                continue
            key_path = f"{path}.{key}" if path else str(key)
            if key == "_target_":
                if not isinstance(child, str):
                    raise TypeError(
                        f"cfg.{key_path} must be a string, got "
                        f"{type(child).__name__}"
                    )
                if not child:
                    raise ValueError(
                        f"cfg.{key_path} must be a non-empty import path"
                    )
                collected.append((key_path, child))
            else:
                collected.extend(_collect_targets(child, path=key_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_path = f"{path}.{index}" if path else str(index)
            collected.extend(_collect_targets(child, path=child_path))
    return tuple(sorted(collected))


def _configured_targets(
    cfg: Mapping[str, Any],
    unresolved_cfg: Mapping[str, Any],
) -> TargetRecords:
    """Resolve only target values so records match Hydra execution."""
    targets = _collect_targets(unresolved_cfg)
    if not isinstance(cfg, DictConfig):
        return targets
    resolved: list[TargetRecord] = []
    for path, _ in targets:
        target = OmegaConf.select(cfg, path)
        if not isinstance(target, str):
            raise TypeError(
                f"cfg.{path} must resolve to a string, got "
                f"{type(target).__name__}"
            )
        if not target:
            raise ValueError(
                f"cfg.{path} must resolve to a non-empty import path"
            )
        resolved.append((path, target))
    return tuple(resolved)


def _validate_artifact_logger_targets(targets: TargetRecords) -> None:
    """Reject qualified loggers that cannot publish selected artifacts."""
    supported = frozenset(ARTIFACT_LOGGER_TARGETS.values())
    for path, target in targets:
        if (
            path.startswith("logger.")
            and path.endswith("._target_")
            and path.count(".") == 2
            and target not in supported
        ):
            raise ValueError(
                f"cfg.{path} target {target!r} is not supported by "
                "selected-checkpoint artifact publication"
            )


def _required_mapping(
    parent: Mapping[str, Any],
    key: str,
    *,
    path: str,
) -> Mapping[str, Any]:
    """Read a required nested mapping with a path-specific error."""
    value = parent.get(key)
    child_path = f"{path}.{key}" if path else key
    if not isinstance(value, Mapping):
        raise TypeError(f"{child_path} must be a mapping")
    return value


def _required_string(
    parent: Mapping[str, Any],
    key: str,
    *,
    path: str,
) -> str:
    """Read a required non-empty selector string."""
    value = parent.get(key)
    child_path = f"{path}.{key}" if path else key
    if not isinstance(value, str):
        raise TypeError(f"{child_path} must be a string")
    if not value:
        raise ValueError(f"{child_path} must be non-empty")
    return value


def _canonical_targets(
    cfg: Mapping[str, Any],
) -> tuple[dict[str, str], frozenset[str]]:
    """Derive exact targets from qualification and model authorities."""
    from topobench.data.capabilities import qualify_dataset
    from topobench.data.qualification import (
        DATASET_QUALIFICATION_MANIFEST,
    )
    from topobench.nn.capabilities import (
        GRAPH_MODEL_CAPABILITIES,
        MODEL_CAPABILITY_MANIFEST,
    )

    dataset_cfg = _required_mapping(cfg, "dataset", path="")
    try:
        qualification = qualify_dataset(dataset_cfg)
    except (TypeError, ValueError) as error:
        qualifications = {}
        loader_targets = sorted(
            {
                row.loader_family
                for row in DATASET_QUALIFICATION_MANIFEST.values()
            }
        )
        for loader_target in loader_targets:
            candidate = deepcopy(dataset_cfg)
            if not isinstance(candidate, (DictConfig, MutableMapping)):
                raise TypeError(
                    "dataset deep copy must be a mutable mapping"
                ) from error
            _replace_target(candidate, "loader._target_", loader_target)
            try:
                row = qualify_dataset(candidate)
            except (TypeError, ValueError):
                continue
            qualifications[row.selector] = row
        if len(qualifications) != 1:
            raise
        qualification = next(iter(qualifications.values()))

    model_cfg = _required_mapping(cfg, "model", path="")
    model_domain = _required_string(
        model_cfg,
        "model_domain",
        path="model",
    )
    model_name = _required_string(model_cfg, "model_name", path="model")
    model_selector = f"{model_domain}/{model_name}"
    model = MODEL_CAPABILITY_MANIFEST.get(model_selector)
    if model is None:
        raise ValueError(
            "model.model_domain and model.model_name form unknown packaged "
            f"selector {model_selector!r}"
        )

    expected = dict(_FIXED_TARGETS)
    selector_owned = {
        "dataset.loader._target_",
        "data_pipeline._target_",
        "model._target_",
        "model.feature_encoder._target_",
        "model.backbone._target_",
        "model.backbone_wrapper._target_",
        "model.readout._target_",
    }
    expected.update(
        {
            "dataset.loader._target_": qualification.loader_family,
            "data_pipeline._target_": model.pipeline_target,
            "model._target_": model.model_target,
            "model.feature_encoder._target_": (model.feature_encoder_target),
            "model.backbone._target_": model.backbone_target,
            "model.backbone_wrapper._target_": model.wrapper_target,
            "model.readout._target_": model.readout_target,
        }
    )
    if model.supervision_adapter_target is not None:
        path = "model.supervision_adapter._target_"
        expected[path] = model.supervision_adapter_target
        selector_owned.add(path)
    if model_domain == "graph":
        graph_model = GRAPH_MODEL_CAPABILITIES[model_name]
        if graph_model.backbone_loss_target is not None:
            path = "model.backbone.loss._target_"
            expected[path] = graph_model.backbone_loss_target
            selector_owned.add(path)

    for group_name, authorities in (
        ("callbacks", _CALLBACK_TARGETS),
        ("logger", _LOGGER_TARGETS),
    ):
        group = cfg.get(group_name)
        if group is None:
            continue
        if not isinstance(group, Mapping):
            raise TypeError(f"{group_name} must be a mapping")
        for name, target in authorities.items():
            if isinstance(group.get(name), Mapping):
                expected[f"{group_name}.{name}._target_"] = target
    return expected, frozenset(selector_owned)


def _execution_profile_record(
    cfg: Mapping[str, Any],
) -> ExecutionProfileRecord:
    """Build one deterministic execution-profile record."""
    if not isinstance(cfg, Mapping):
        raise TypeError("cfg must be a mapping")
    raw_cfg = _unresolved_container(cfg)
    profile = raw_cfg.get("execution_profile", _QUALIFIED_PROFILE)
    if not isinstance(profile, str):
        raise TypeError("execution_profile must be a string")
    if profile not in _SUPPORTED_PROFILES:
        raise ValueError(
            "execution_profile must be 'qualified' or 'experimental'"
        )
    evaluation_artifacts = raw_cfg.get("evaluation_artifacts")
    if profile == _QUALIFIED_PROFILE:
        if not isinstance(evaluation_artifacts, Mapping):
            raise TypeError(
                "cfg.evaluation_artifacts must be a mapping in qualified "
                "execution"
            )
        if evaluation_artifacts.get("enabled") is not True:
            raise ValueError(
                "cfg.evaluation_artifacts.enabled must be exactly true in "
                "qualified execution"
            )

    targets = _configured_targets(cfg, raw_cfg)
    if (
        isinstance(evaluation_artifacts, Mapping)
        and evaluation_artifacts.get("enabled") is True
    ):
        _validate_artifact_logger_targets(targets)
    configured = dict(targets)
    expected, _ = _canonical_targets(cfg)
    for path in sorted(expected):
        if path not in configured:
            raise ValueError(f"cfg.{path} is required")
    custom_targets = tuple(
        (path, target)
        for path, target in targets
        if expected.get(path) != target
    )
    if profile == _QUALIFIED_PROFILE and custom_targets:
        path, target = custom_targets[0]
        packaged = expected.get(path)
        if packaged is None:
            raise ValueError(
                f"cfg.{path} is not a packaged target path in qualified "
                "execution"
            )
        raise ValueError(
            f"cfg.{path} must be exact packaged target {packaged!r}, "
            f"got {target!r}"
        )
    return ExecutionProfileRecord(
        profile=profile,
        qualified=profile == _QUALIFIED_PROFILE,
        targets=targets,
        custom_targets=custom_targets,
    )


def validate_execution_profile(
    cfg: Mapping[str, Any],
) -> ExecutionProfileRecord:
    """Audit every Hydra target under qualified or experimental execution."""
    return _execution_profile_record(cfg)


def _replace_target(
    cfg: MutableMapping[str, Any],
    path: str,
    target: str,
) -> None:
    """Replace one existing target on a validation-only deep copy."""
    if isinstance(cfg, DictConfig):
        OmegaConf.update(cfg, path, target, merge=False)
        return
    parts = path.split(".")
    current: MutableMapping[str, Any] = cfg
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, MutableMapping):
            raise TypeError(f"{'.'.join(parts[:-1])} must be a mapping")
        current = child
    current[parts[-1]] = target


def validate_profile_capability(
    cfg: Mapping[str, Any],
    *,
    profile_record: ExecutionProfileRecord,
    observed: object | None = None,
) -> CapabilityValidation:
    """Validate packaged selectors while preserving experimental targets."""
    from topobench.nn.capabilities import validate_capability_composition

    if not isinstance(profile_record, ExecutionProfileRecord):
        raise TypeError("profile_record must be an ExecutionProfileRecord")
    current_record = _execution_profile_record(cfg)
    if current_record != profile_record:
        raise ValueError("profile_record does not match the current config")
    if profile_record.qualified:
        return validate_capability_composition(cfg, observed=observed)

    validation_cfg = deepcopy(cfg)
    if not isinstance(validation_cfg, (DictConfig, MutableMapping)):
        raise TypeError("cfg deep copy must be a mutable mapping")
    raw_cfg = _unresolved_container(cfg)
    expected, selector_owned = _canonical_targets(raw_cfg)
    for path, _ in profile_record.custom_targets:
        if path in selector_owned:
            _replace_target(validation_cfg, path, expected[path])
    return validate_capability_composition(
        validation_cfg,
        observed=observed,
    )


def instantiate_callbacks(
    callbacks_cfg: DictConfig,
    *,
    input_pipeline_monitor: object | None = None,
) -> list[Callback]:
    r"""Instantiate callbacks from config.

    Parameters
    ----------
    callbacks_cfg : DictConfig
        A DictConfig object containing callback configurations.

    Returns
    -------
    list[Callback]
        A list of instantiated callbacks.
    """
    callbacks: list[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for cb_conf in callbacks_cfg.values():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            kwargs = {}
            if (
                cb_conf._target_
                == "topobench.callbacks.input_pipeline.InputPipelineCallback"
                and input_pipeline_monitor is not None
            ):
                kwargs["monitor"] = input_pipeline_monitor
            callbacks.append(hydra.utils.instantiate(cb_conf, **kwargs))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    r"""Instantiate loggers from config.

    Parameters
    ----------
    logger_cfg : DictConfig
        A DictConfig object containing logger configurations.

    Returns
    -------
    list[Logger]
        A list of instantiated loggers.
    """
    logger: list[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for lg_conf in logger_cfg.values():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger
