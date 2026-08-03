"""Automatic static and isolated execution qualification before training."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from lightning import LightningDataModule, LightningModule
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data, HeteroData

from topobench.evaluator import EvaluationResult, UndefinedMetricError
from topobench.evaluator.base import AbstractEvaluator
from topobench.evaluator.registry import (
    BUILTIN_METRIC_SPECS,
    resolve_metric_specs,
)

_ALLOWED_PROFILES = frozenset({"qualified", "experimental"})
_ALLOWED_POLICIES = frozenset({"online", "exact", "audit"})
_ALLOWED_MONITOR_NAN_POLICIES = frozenset({"error", "skip"})
_ALLOWED_PARTITION_BACKENDS = frozenset({"pyg", "external"})
_ALLOWED_SAMPLER_BACKENDS = frozenset(
    {"pyg", "external", "native", "materialized"}
)
_SUPPORTED_ARTIFACT_LOGGERS = frozenset(
    {
        "lightning.pytorch.loggers.CSVLogger",
        "lightning.pytorch.loggers.WandbLogger",
        "lightning.pytorch.loggers.csv_logs.CSVLogger",
        "lightning.pytorch.loggers.wandb.WandbLogger",
    }
)
_PHASES = ("train", "val", "test")
_SHA256 = re.compile(r"[0-9a-f]{64}", re.IGNORECASE)
_MISSING = object()


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    """One immutable, provenance-ready preflight observation."""

    check_id: str
    passed: bool
    detail: str
    remediation: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.check_id, str) or not self.check_id:
            raise ValueError("check_id must be a non-empty string")
        if type(self.passed) is not bool:
            raise TypeError("passed must be a boolean")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("detail must be a non-empty string")
        if not isinstance(self.remediation, str):
            raise TypeError("remediation must be a string")

    def as_record(self) -> dict[str, object]:
        """Return the non-executable serialization used by later provenance."""
        return {
            "check_id": self.check_id,
            "passed": self.passed,
            "detail": self.detail,
            "remediation": self.remediation,
        }


@dataclass(frozen=True, slots=True)
class PreflightResult:
    """Complete immutable result of the configured pre-training gate."""

    enabled: bool
    qualified: bool
    checks: tuple[PreflightCheck, ...]

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        if type(self.qualified) is not bool:
            raise TypeError("qualified must be a boolean")
        if not isinstance(self.checks, tuple) or any(
            not isinstance(check, PreflightCheck) for check in self.checks
        ):
            raise TypeError("checks must be a tuple of PreflightCheck values")
        if not self.checks:
            raise ValueError("preflight result must contain at least one check")
        if self.qualified and (not self.enabled or not self.passed):
            raise ValueError("only an enabled, passing preflight can be qualified")

    @property
    def passed(self) -> bool:
        """Whether every performed check passed."""
        return all(check.passed for check in self.checks)

    def as_record(self) -> dict[str, object]:
        """Return a stable record without exporting mutable internals."""
        return {
            "enabled": self.enabled,
            "passed": self.passed,
            "qualified": self.qualified,
            "checks": [check.as_record() for check in self.checks],
        }


class PreflightError(RuntimeError):
    """Stop a run at the gate while retaining its structured failure."""

    def __init__(self, result: PreflightResult) -> None:
        if not isinstance(result, PreflightResult) or result.passed:
            raise ValueError("PreflightError requires a failing PreflightResult")
        self.result = result
        failure = next(check for check in result.checks if not check.passed)
        super().__init__(f"{failure.check_id}: {failure.detail}")


class PreflightRunner:
    """Own the single static and representative-batch preflight path."""

    def __init__(
        self,
        cfg: DictConfig,
        pipeline_output: object | None,
    ) -> None:
        if not isinstance(cfg, DictConfig):
            raise TypeError("cfg must be an OmegaConf DictConfig")
        self._cfg = cfg
        self._pipeline_output = pipeline_output
        self._resolved: Mapping[str, Any] | None = None

    def validate_static(self) -> PreflightResult:
        """Resolve and validate all checks that precede model construction."""
        checks: list[PreflightCheck] = []
        try:
            resolved = {
                key: (
                    OmegaConf.to_container(
                        self._cfg[key],
                        resolve=True,
                        throw_on_missing=True,
                    )
                    if OmegaConf.is_config(self._cfg[key])
                    else self._cfg[key]
                )
                for key in self._cfg
                if key != "hydra"
            }
        except Exception as error:
            self._fail(
                checks,
                "static.interpolation",
                f"configuration interpolation could not be resolved: {error}",
                "define every referenced OmegaConf value before starting the run",
                enabled=True,
            )
        if not isinstance(resolved, Mapping):
            self._fail(
                checks,
                "static.configuration",
                "resolved run configuration must be a mapping",
                "compose the run from the packaged run.yaml root",
                enabled=True,
            )
        self._resolved = resolved
        checks.append(
            PreflightCheck(
                "static.interpolation",
                True,
                "all OmegaConf interpolations resolved",
            )
        )

        profile = _select(resolved, "execution_profile", "qualified")
        if not isinstance(profile, str) or profile not in _ALLOWED_PROFILES:
            self._fail(
                checks,
                "static.execution_profile",
                f"unknown execution profile {profile!r}",
                "select qualified or experimental",
                enabled=True,
            )

        enabled = _select(resolved, "preflight.enabled", True)
        if type(enabled) is not bool:
            self._fail(
                checks,
                "preflight.enabled",
                "preflight.enabled must be boolean",
                "set preflight.enabled to true or false",
                enabled=True,
            )
        if not enabled:
            if profile != "experimental":
                self._fail(
                    checks,
                    "preflight.enabled",
                    "disabling preflight requires execution_profile=experimental",
                    "enable preflight or explicitly select the experimental profile",
                    enabled=False,
                )
            checks.append(
                PreflightCheck(
                    "preflight.enabled",
                    True,
                    "preflight explicitly disabled under the experimental profile",
                )
            )
            return PreflightResult(
                enabled=False,
                qualified=False,
                checks=tuple(checks),
            )

        checks.append(
            PreflightCheck(
                "preflight.enabled",
                True,
                "automatic preflight is enabled",
            )
        )
        execution_probe = _select(resolved, "preflight.execution_probe", True)
        if type(execution_probe) is not bool:
            self._fail(
                checks,
                "preflight.execution_probe",
                "preflight.execution_probe must be boolean",
                "set execution_probe to true or false",
                enabled=True,
            )
        if not execution_probe and profile == "qualified":
            self._fail(
                checks,
                "preflight.execution_probe",
                "a qualified run cannot disable the isolated execution probe",
                "enable the probe or select execution_profile=experimental",
                enabled=True,
            )

        self._validate_metrics(resolved, checks)
        self._validate_ceilings(resolved, checks)
        self._validate_splits(resolved, checks, profile=profile)
        self._validate_pipeline_identities(resolved, checks, profile=profile)
        self._validate_validation_schedule(resolved, checks, profile=profile)
        self._validate_distributed(resolved, checks, profile=profile)
        self._validate_checkpoint_monitor(resolved, checks)
        self._validate_artifact_loggers(resolved, checks)
        self._validate_reproducibility_bundle(resolved, checks, profile=profile)
        checks.append(
            PreflightCheck(
                "static.configuration",
                True,
                "static preflight configuration and identities are qualified",
            )
        )
        return PreflightResult(
            enabled=True,
            qualified=profile == "qualified" and execution_probe,
            checks=tuple(checks),
        )

    def run_probe(
        self,
        *,
        model_factory: Callable[[], LightningModule],
        static_result: PreflightResult,
    ) -> PreflightResult:
        """Exercise real production model boundaries with throwaway state."""
        if not isinstance(static_result, PreflightResult) or not static_result.passed:
            raise ValueError("static_result must be a passing PreflightResult")
        if static_result.enabled is False:
            return static_result
        if not callable(model_factory):
            raise TypeError("model_factory must be callable")
        if self._resolved is None:
            raise RuntimeError("validate_static must run before run_probe")
        if not _select(self._resolved, "preflight.execution_probe", True):
            return static_result
        phases = _enabled_probe_phases(self._resolved)
        if not phases:
            return PreflightResult(
                enabled=True,
                qualified=static_result.qualified,
                checks=static_result.checks
                + (
                    PreflightCheck(
                        "execution.not_requested",
                        True,
                        "execution probe not executed because train and test are disabled",
                    ),
                ),
            )
        if self._pipeline_output is None:
            self._probe_failure(
                static_result,
                "the data pipeline did not return a probeable output",
                "return a DataPipelineOutput with a Lightning data module",
            )

        datamodule = getattr(self._pipeline_output, "datamodule", None)
        if not isinstance(datamodule, LightningDataModule):
            self._probe_failure(
                static_result,
                "pipeline_output.datamodule is not a LightningDataModule",
                "return the production LightningDataModule from the data pipeline",
            )

        model: LightningModule | None = None
        try:
            model = model_factory()
            if not isinstance(model, LightningModule):
                raise TypeError("model factory must return a LightningModule")
            if not isinstance(getattr(model, "evaluator", None), AbstractEvaluator):
                raise TypeError("throwaway model must own an AbstractEvaluator")
            if not callable(getattr(model, "model_step", None)):
                raise TypeError("throwaway model must expose model_step")
            if not callable(getattr(model, "abort_evaluation", None)):
                raise TypeError("throwaway model must expose abort_evaluation")

            device = _configured_probe_device(self._resolved)
            model.to(device)
            with _representative_batches(datamodule, phases) as batches:
                for phase in phases:
                    batch = batches[phase]
                    if not isinstance(batch, (Data, HeteroData)):
                        raise TypeError(
                            f"{phase} representative batch must be native Data or HeteroData"
                        )
                    transferred = model.transfer_batch_to_device(batch, device, 0)
                    _start_evaluation_phase(model, phase)
                    model.train(phase == "train")
                    try:
                        context = (
                            torch.enable_grad()
                            if phase == "train"
                            else torch.no_grad()
                        )
                        with context:
                            model_out = model.model_step(transferred)
                        if not isinstance(model_out, Mapping):
                            raise TypeError("model_step must return a mapping")
                        _require_finite_loss(model_out.get("loss"), phase)
                        try:
                            snapshot = model.evaluator.snapshot()
                        except UndefinedMetricError as error:
                            if error.num_examples <= 0 or error.reason not in {
                                "binary_target_single_class",
                                "macro_target_missing_class",
                                "multiclass_target_missing_class",
                                "r2_constant_target",
                                "r2_too_few_examples",
                            }:
                                raise
                        else:
                            if not isinstance(snapshot, EvaluationResult):
                                raise TypeError(
                                    "typed evaluator snapshot must be an EvaluationResult"
                                )
                            if snapshot.num_examples <= 0:
                                raise ValueError(
                                    f"{phase} evaluator snapshot has no supervised examples"
                                )
                    finally:
                        model.abort_evaluation()
        except Exception as error:
            if model is not None:
                abort = getattr(model, "abort_evaluation", None)
                if callable(abort):
                    try:
                        abort()
                    except Exception:
                        pass
            self._probe_failure(
                static_result,
                f"isolated execution probe failed: {type(error).__name__}: {error}",
                "repair the reported data/model/supervision/loss/evaluator boundary",
                cause=error,
            )

        return PreflightResult(
            enabled=True,
            qualified=static_result.qualified,
            checks=static_result.checks
            + (
                PreflightCheck(
                    "execution.representative_batch",
                    True,
                    "throwaway forward, supervision, loss, and typed metric update passed",
                ),
            ),
        )

    def _validate_metrics(
        self,
        cfg: Mapping[str, Any],
        checks: list[PreflightCheck],
    ) -> None:
        task = _select(cfg, "evaluator.task", _MISSING)
        dataset_task = _select(cfg, "dataset.parameters.task", _MISSING)
        if task is _MISSING:
            task = dataset_task
        if not isinstance(task, str) or task not in {"classification", "regression"}:
            self._fail(
                checks,
                "static.evaluator.task",
                f"unknown evaluator task {task!r}",
                "select classification or regression",
                enabled=True,
            )
        if dataset_task is not _MISSING and dataset_task != task:
            self._fail(
                checks,
                "static.evaluator.task",
                "dataset and evaluator task selectors disagree",
                "use the dataset task as the single evaluator task authority",
                enabled=True,
            )
        num_classes = _select(
            cfg,
            "evaluator.num_classes",
            _select(cfg, "dataset.parameters.num_classes", _MISSING),
        )
        if (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, int)
            or num_classes < 1
        ):
            self._fail(
                checks,
                "static.evaluator.num_classes",
                "evaluator num_classes must be a positive integer",
                "derive num_classes from the dataset contract",
                enabled=True,
            )

        metrics = _select(cfg, "evaluator.metrics", _MISSING)
        if (
            metrics is _MISSING
            or isinstance(metrics, (str, bytes))
            or not isinstance(metrics, Sequence)
            or not metrics
            or any(not isinstance(name, str) or not name for name in metrics)
        ):
            self._fail(
                checks,
                "static.evaluator.metrics",
                "metrics must be a non-empty ordered sequence of names",
                "configure explicit TopoBench metric names",
                enabled=True,
            )
        metric_names = tuple(metrics)
        custom = _custom_metric_declarations(cfg, checks, self)
        unknown = sorted(
            set(metric_names) - set(BUILTIN_METRIC_SPECS) - set(custom)
        )
        if unknown:
            self._fail(
                checks,
                "static.evaluator.metrics",
                f"unknown metric names: {unknown}",
                "select a builtin metric or declare one complete custom metric",
                enabled=True,
            )

        policies = _select(cfg, "evaluator.policy", {})
        if not isinstance(policies, Mapping):
            self._fail(
                checks,
                "static.evaluator.policy",
                "evaluator.policy must map train, val, and test policies",
                "configure one policy per phase",
                enabled=True,
            )
        resolved_policies = {
            phase: policies.get(
                phase,
                "online" if phase == "train" else "exact",
            )
            for phase in _PHASES
        }
        for phase, policy in resolved_policies.items():
            if not isinstance(policy, str) or policy not in _ALLOWED_POLICIES:
                self._fail(
                    checks,
                    "static.evaluator.policy",
                    f"unknown {phase} policy {policy!r}",
                    "select online, exact, or audit",
                    enabled=True,
                )
            builtins = tuple(
                name for name in metric_names if name in BUILTIN_METRIC_SPECS
            )
            if builtins:
                try:
                    resolve_metric_specs(
                        builtins,
                        task=task,
                        num_classes=num_classes,
                        policy=policy,
                    )
                except (TypeError, ValueError) as error:
                    self._fail(
                        checks,
                        "static.evaluator.metric_policy",
                        str(error),
                        "select metrics compatible with the configured phase policy",
                        enabled=True,
                    )
            capability = {
                "online": "supports_online",
                "exact": "supports_exact",
                "audit": "supports_audit",
            }[policy]
            for name in metric_names:
                if name in custom and custom[name].get(capability) is not True:
                    self._fail(
                        checks,
                        "static.evaluator.custom_metric_policy",
                        f"custom metric {name!r} does not support {policy} policy",
                        "change the phase policy or provide the required backend",
                        enabled=True,
                    )

    def _validate_ceilings(
        self,
        cfg: Mapping[str, Any],
        checks: list[PreflightCheck],
    ) -> None:
        for path, label in (
            ("evaluator.exact.max_ranking_bytes", "exact max_ranking_bytes"),
            (
                "dataset.loader.parameters.partition.memory_limit_bytes",
                "topology partition memory ceiling",
            ),
            (
                "preflight.prefetch_memory_ceiling_bytes",
                "prefetch memory ceiling",
            ),
        ):
            value = _select(cfg, path, None)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                self._fail(
                    checks,
                    "static.memory_ceiling",
                    f"{label} must be a positive integer byte ceiling",
                    "set a finite positive byte limit",
                    enabled=True,
                )

    def _validate_splits(
        self,
        cfg: Mapping[str, Any],
        checks: list[PreflightCheck],
        *,
        profile: str,
    ) -> None:
        registry = _select(
            cfg,
            "dataset.loader.parameters.supervision.splits",
            None,
        )
        if registry is None:
            return
        if not isinstance(registry, Mapping):
            self._split_failure(checks, "named split registry must be a mapping")
        active = registry.get("active")
        sets = registry.get("sets")
        if not isinstance(active, str) or not active:
            self._split_failure(checks, "named split registry has no active tag")
        if not isinstance(sets, Mapping) or not sets:
            self._split_failure(checks, "named split registry has no split tags")
        if active not in sets:
            self._split_failure(
                checks,
                f"active split tag {active!r} is not registered",
            )
        for tag, split in sets.items():
            if not isinstance(tag, str) or not tag or not isinstance(split, Mapping):
                self._split_failure(checks, "named split tag is malformed")
            phase_sources: dict[str, frozenset[str]] = {}
            for phase in _PHASES:
                try:
                    phase_sources[phase] = _split_sources(split[phase])
                except (KeyError, TypeError, ValueError) as error:
                    self._split_failure(
                        checks,
                        f"split tag {tag!r} has malformed {phase}: {error}",
                    )
            for index, left in enumerate(_PHASES):
                for right in _PHASES[index + 1 :]:
                    overlap = phase_sources[left].intersection(phase_sources[right])
                    if overlap:
                        self._split_failure(
                            checks,
                            f"split tag {tag!r} internally overlaps {left}/{right}: "
                            f"{sorted(overlap)!r}",
                        )
        selected = _select(cfg, "data_pipeline.active_split_tag", None)
        if selected is not None and selected != active:
            self._split_failure(
                checks,
                f"selected split tag {selected!r} differs from active tag {active!r}",
            )
        active_record = sets[active]
        if profile == "qualified" and active_record.get("qualified", True) is not True:
            self._split_failure(checks, f"active split tag {active!r} is unqualified")

    def _validate_validation_schedule(
        self,
        cfg: Mapping[str, Any],
        checks: list[PreflightCheck],
        *,
        profile: str,
    ) -> None:
        interval = _select(cfg, "trainer.val_check_interval", None)
        if profile == "qualified" and interval is not None and (
            isinstance(interval, bool)
            or not isinstance(interval, float)
            or interval != 1.0
        ):
            self._fail(
                checks,
                "static.validation_schedule",
                (
                    "qualified typed evaluation cannot safely resume a train "
                    f"context after mid-epoch validation ({interval!r})"
                ),
                (
                    "use epoch-end validation by omitting "
                    "trainer.val_check_interval or setting it to 1.0"
                ),
                enabled=True,
            )
        checks.append(
            PreflightCheck(
                "static.validation_schedule",
                True,
                "validation is restricted to the typed evaluator's epoch-end boundary",
            )
        )


    def _validate_pipeline_identities(
        self,
        cfg: Mapping[str, Any],
        checks: list[PreflightCheck],
        *,
        profile: str,
    ) -> None:
        partition_backend = _select(
            cfg,
            "dataset.loader.parameters.partition.backend",
            None,
        )
        if partition_backend is not None and (
            not isinstance(partition_backend, str)
            or partition_backend not in _ALLOWED_PARTITION_BACKENDS
        ):
            self._fail(
                checks,
                "static.sampler_backend",
                f"unsupported partition/sampler backend {partition_backend!r}",
                "select the qualified pyg or external partition backend",
                enabled=True,
            )
        output = self._pipeline_output
        if output is None:
            return
        report = getattr(output, "qualification_report", None)
        if report is not None and getattr(report, "passed", None) is not True:
            self._fail(
                checks,
                "static.store_qualification",
                "pipeline store qualification report did not pass",
                "rebuild and requalify the immutable typed graph store",
                enabled=True,
            )
        provenance = getattr(output, "provenance_input", None)
        if provenance is None:
            return
        if not isinstance(provenance, Mapping):
            self._fail(
                checks,
                "static.pipeline_identity",
                "pipeline provenance identities must be a mapping",
                "return immutable pipeline provenance identity values",
                enabled=True,
            )
        source_identity = getattr(output, "source_graph_id", None)
        recorded_source = provenance.get("source_graph_id")
        if source_identity is not None:
            if not _is_sha256(source_identity) or recorded_source != source_identity:
                self._fail(
                    checks,
                    "static.store_identity",
                    "store identity is malformed or stale against pipeline provenance",
                    "reopen the exact content-addressed qualified store",
                    enabled=True,
                )
        partition_identity = provenance.get("partition_book_identity")
        if partition_identity is not None and not _is_sha256(partition_identity):
            self._fail(
                checks,
                "static.partition_book_identity",
                "partition book identity is malformed or stale",
                "rebuild the accepted typed partition book",
                enabled=True,
            )
        active_tag = getattr(output, "active_split_tag", None)
        recorded_tag = provenance.get("active_split_tag")
        if active_tag is not None and active_tag != recorded_tag:
            self._split_failure(
                checks,
                "pipeline active split identity is stale against provenance",
            )
        expected_sampler_backend = _select(
            cfg,
            "dataset.loader.parameters.partition.backend",
            None,
        )
        sampler_backend = provenance.get("sampler_backend")
        if not isinstance(sampler_backend, str) or (
            sampler_backend not in _ALLOWED_SAMPLER_BACKENDS
            or (
                isinstance(expected_sampler_backend, str)
                and sampler_backend != expected_sampler_backend
            )
        ):
            self._fail(
                checks,
                "static.sampler_identity",
                (
                    "sampler backend identity is missing, unsupported, or stale: "
                    f"expected {expected_sampler_backend!r}, observed "
                    f"{sampler_backend!r}"
                ),
                "rebuild the pipeline with the configured qualified sampler backend",
                enabled=True,
            )
        partition_strategy = _select(
            cfg,
            "dataset.loader.parameters.partition.strategy",
            None,
        )
        output_kind = _select(
            cfg,
            "dataset.loader.parameters.output_kind",
            None,
        )
        if output_kind in {"homogeneous", "heterogeneous"} and partition_strategy in {
            "cluster",
            "neighbor",
        }:
            expected_strategy = f"{output_kind}-{partition_strategy}"
        sampling_strategy = provenance.get("sampling_strategy")
        if (
            not isinstance(sampling_strategy, str)
            or not sampling_strategy
            or (
                isinstance(expected_strategy, str)
                and sampling_strategy != expected_strategy
            )
        ):
            self._fail(
                checks,
                "static.sampler_identity",
                (
                    "sampling strategy identity is missing or stale: "
                    f"expected {expected_strategy!r}, observed "
                    f"{sampling_strategy!r}"
                ),
                "rebuild the pipeline with the configured qualified sampling strategy",
                enabled=True,
            )
        transform = getattr(output, "fitted_transform", None)
        if transform is not None:
            recorded_state = provenance.get("fitted_transform_state_key")
            actual_state = getattr(transform, "state_key", None)
            if (
                not isinstance(recorded_state, str)
                or not recorded_state
                or actual_state != recorded_state
            ):
                self._fail(
                    checks,
                    "static.fitted_transform_identity",
                    "fitted-transform identity is missing or stale against pipeline provenance",
                    "load the training-only fitted state bound to this store and split",
                    enabled=True,
                )
        if profile == "qualified" and provenance.get("qualified_profile", True) is not True:
            self._fail(
                checks,
                "static.pipeline_identity",
                "pipeline identities were produced under an unqualified profile",
                "rebuild the pipeline with qualified_profile=true",
                enabled=True,
            )

    def _validate_distributed(
        self,
        cfg: Mapping[str, Any],
        checks: list[PreflightCheck],
        *,
        profile: str,
    ) -> None:
        if profile != "qualified":
            return
        devices = _select(cfg, "trainer.devices", 1)
        nodes = _select(cfg, "trainer.num_nodes", 1)
        strategy = _select(cfg, "trainer.strategy", None)
        multi_device = (
            isinstance(devices, int)
            and not isinstance(devices, bool)
            and devices > 1
        ) or (
            isinstance(devices, Sequence)
            and not isinstance(devices, (str, bytes))
            and len(devices) > 1
        )
        multi_node = (
            isinstance(nodes, int)
            and not isinstance(nodes, bool)
            and nodes > 1
        )
        distributed_strategy = "ddp" in str(strategy).lower()
        if multi_device or multi_node or distributed_strategy:
            self._fail(
                checks,
                "static.distributed",
                "unsupported multi-rank qualified evaluator run",
                "use one device/rank or select execution_profile=experimental",
                enabled=True,
            )

    def _validate_checkpoint_monitor(
        self,
        cfg: Mapping[str, Any],
        checks: list[PreflightCheck],
    ) -> None:
        policy = _select(cfg, "preflight.monitor_nan_policy", None)
        if policy is not None and (
            not isinstance(policy, str)
            or policy not in _ALLOWED_MONITOR_NAN_POLICIES
        ):
            self._fail(
                checks,
                "static.monitor_nan_policy",
                f"unknown checkpoint monitor NaN policy {policy!r}",
                "select error or skip",
                enabled=True,
            )
        callbacks = _select(cfg, "callbacks", None)
        if not isinstance(callbacks, Mapping):
            return
        for callback in callbacks.values():
            if not isinstance(callback, Mapping):
                continue
            target = callback.get("_target_", "")
            if not isinstance(target, str) or not target.endswith("ModelCheckpoint"):
                continue
            monitor = callback.get("monitor")
            if not isinstance(monitor, str) or not monitor:
                continue
            metric_name = monitor.rsplit("/", 1)[-1]
            spec = BUILTIN_METRIC_SPECS.get(metric_name)
            undefined = (
                frozenset()
                if spec is None
                else spec.undefined_reasons - {"empty_evaluation"}
            )
            if undefined and policy is None:
                self._fail(
                    checks,
                    "static.monitor_nan_policy",
                    f"checkpoint monitor {monitor!r} can produce NaN without a monitor policy",
                    "set preflight.monitor_nan_policy to error or skip",
                    enabled=True,
                )

    def _validate_artifact_loggers(
        self,
        cfg: Mapping[str, Any],
        checks: list[PreflightCheck],
    ) -> None:
        artifacts = _select(cfg, "evaluation_artifacts", None)
        if not isinstance(artifacts, Mapping) or artifacts.get("enabled") is not True:
            return
        adapter = artifacts.get("adapter")
        if adapter is not None and (
            not isinstance(adapter, str)
            or adapter not in {"csv", "local", "wandb"}
        ):
            self._fail(
                checks,
                "static.artifact_adapter",
                f"unsupported evaluation artifact adapter {adapter!r}",
                "select the local/CSV or W&B artifact adapter",
                enabled=True,
            )
        loggers = _select(cfg, "logger", None)
        if not isinstance(loggers, Mapping):
            return
        unsupported: list[str] = []
        for logger in loggers.values():
            if not isinstance(logger, Mapping):
                continue
            target = logger.get("_target_")
            if isinstance(target, str) and target not in _SUPPORTED_ARTIFACT_LOGGERS:
                unsupported.append(target)
        if unsupported:
            self._fail(
                checks,
                "static.artifact_adapter",
                f"configured logger has no artifact adapter: {sorted(unsupported)!r}",
                "disable evaluation artifacts or select a supported CSV/W&B logger",
                enabled=True,
            )

    def _validate_reproducibility_bundle(
        self,
        cfg: Mapping[str, Any],
        checks: list[PreflightCheck],
        *,
        profile: str,
    ) -> None:
        if profile != "qualified":
            return
        configured = _select(
            cfg,
            (
                "dataset.loader.parameters.reproducibility."
                "save_reproducibility_bundle"
            ),
            None,
        )
        output = self._pipeline_output
        policy = (
            None
            if output is None
            else getattr(output, "reproducibility_policy", None)
        )
        runtime = (
            None
            if policy is None
            else getattr(policy, "save_reproducibility_bundle", None)
        )
        provenance = (
            None if output is None else getattr(output, "provenance_input", None)
        )
        recorded = (
            None
            if not isinstance(provenance, Mapping)
            else provenance.get("save_reproducibility_bundle")
        )
        if configured is False or runtime is False or recorded is False:
            self._fail(
                checks,
                "static.reproducibility_bundle",
                "qualified execution cannot disable the reproducibility bundle",
                "set save_reproducibility_bundle=true",
                enabled=True,
            )

    def _split_failure(
        self,
        checks: list[PreflightCheck],
        detail: str,
    ) -> None:
        self._fail(
            checks,
            "static.named_splits",
            detail,
            "declare one active, disjoint train/val/test named split tag",
            enabled=True,
        )

    @staticmethod
    def _fail(
        checks: list[PreflightCheck],
        check_id: str,
        detail: str,
        remediation: str,
        *,
        enabled: bool,
    ) -> None:
        failed = PreflightCheck(check_id, False, detail, remediation)
        result = PreflightResult(
            enabled=enabled,
            qualified=False,
            checks=tuple(checks) + (failed,),
        )
        raise PreflightError(result)

    @staticmethod
    def _probe_failure(
        static_result: PreflightResult,
        detail: str,
        remediation: str,
        *,
        cause: Exception | None = None,
    ) -> None:
        result = PreflightResult(
            enabled=static_result.enabled,
            qualified=False,
            checks=static_result.checks
            + (PreflightCheck("execution.representative_batch", False, detail, remediation),),
        )
        error = PreflightError(result)
        if cause is None:
            raise error
        raise error from cause


def _select(cfg: Mapping[str, Any], path: str, default: Any) -> Any:
    current: Any = cfg
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _custom_metric_declarations(
    cfg: Mapping[str, Any],
    checks: list[PreflightCheck],
    runner: PreflightRunner,
) -> dict[str, Mapping[str, Any]]:
    configured = _select(cfg, "evaluator.custom_metrics", ())
    if configured is None:
        return {}
    if isinstance(configured, (str, bytes)) or not isinstance(configured, Sequence):
        runner._fail(
            checks,
            "static.evaluator.custom_metrics",
            "custom_metrics must be an ordered sequence of declarations",
            "declare immutable custom metric capability records",
            enabled=True,
        )
    result: dict[str, Mapping[str, Any]] = {}
    for declaration in configured:
        if not isinstance(declaration, Mapping):
            runner._fail(
                checks,
                "static.evaluator.custom_metrics",
                "custom metric declaration must be a mapping",
                "declare name and exact/online/audit support",
                enabled=True,
            )
        name = declaration.get("name")
        if not isinstance(name, str) or not name or name in result:
            runner._fail(
                checks,
                "static.evaluator.custom_metrics",
                f"custom metric name is missing or duplicated: {name!r}",
                "use one unique non-empty custom metric name",
                enabled=True,
            )
        result[name] = declaration
    return result


def _split_sources(value: object) -> frozenset[str]:
    if isinstance(value, str) and value:
        return frozenset({value})
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and value
        and all(isinstance(item, str) and item for item in value)
    ):
        return frozenset(value)
    raise ValueError("phase source must be a path or non-empty path sequence")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _configured_probe_device(cfg: Mapping[str, Any]) -> torch.device:
    accelerator = _select(cfg, "trainer.accelerator", "cpu")
    if accelerator == "auto":
        if torch.cuda.is_available():
            accelerator = "cuda"
        elif torch.backends.mps.is_available():
            accelerator = "mps"
        else:
            accelerator = "cpu"
    if accelerator in {"gpu", "cuda"}:
        if not torch.cuda.is_available():
            raise RuntimeError("configured CUDA probe device is unavailable")
        devices = _select(cfg, "trainer.devices", 1)
        index = devices[0] if isinstance(devices, Sequence) else 0
        return torch.device("cuda", int(index))
    if accelerator == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("configured MPS probe device is unavailable")
        return torch.device("mps")
    if accelerator != "cpu":
        raise ValueError(f"unsupported probe accelerator {accelerator!r}")
    return torch.device("cpu")


def _enabled_probe_phases(cfg: Mapping[str, Any]) -> tuple[str, ...]:
    phases: list[str] = []
    if _select(cfg, "train", True):
        phases.extend(("train", "val"))
    if _select(cfg, "test", True):
        phases.append("test")
    return tuple(phases)


@contextmanager
def _representative_batches(
    datamodule: LightningDataModule,
    phases: Sequence[str],
) -> Iterator[dict[str, Data | HeteroData]]:
    state = _snapshot_datamodule(datamodule)
    iterators: list[Iterator[Any]] = []
    loaders: list[object] = []
    batches: dict[str, Data | HeteroData] = {}
    try:
        if any(phase in {"train", "val"} for phase in phases):
            datamodule.setup("fit")
        if "test" in phases:
            datamodule.setup("test")
        for phase in phases:
            loader_factory = getattr(datamodule, f"{phase}_dataloader", None)
            if not callable(loader_factory):
                raise TypeError(f"datamodule has no {phase}_dataloader")
            loader = _one_loader(loader_factory(), phase)
            loaders.append(loader)
            iterator = iter(loader)
            iterators.append(iterator)
            try:
                batches[phase] = next(iterator)
            except StopIteration as error:
                raise ValueError(
                    f"{phase} dataloader has no representative batch"
                ) from error
        yield batches
    finally:
        for iterator in iterators:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
        for loader in loaders:
            close = getattr(loader, "close", None)
            if callable(close):
                close()
        _restore_datamodule(datamodule, state)


def _snapshot_datamodule(datamodule: LightningDataModule) -> object:
    state_dict = getattr(datamodule, "state_dict", None)
    load_state_dict = getattr(datamodule, "load_state_dict", None)
    if not callable(state_dict) or not callable(load_state_dict):
        return _MISSING
    return copy.deepcopy(state_dict())


def _restore_datamodule(
    datamodule: LightningDataModule,
    state: object,
) -> None:
    if state is _MISSING:
        return
    load_state_dict = getattr(datamodule, "load_state_dict", None)
    if not callable(load_state_dict):
        raise RuntimeError("datamodule lost load_state_dict during probe")
    load_state_dict(state)


def _one_loader(value: object, phase: str) -> object:
    if isinstance(value, Mapping):
        values = tuple(value.values())
        if len(values) != 1:
            raise ValueError(f"{phase} requires exactly one representative loader")
        return values[0]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 1:
            raise ValueError(f"{phase} requires exactly one representative loader")
        return value[0]
    return value


def _start_evaluation_phase(model: LightningModule, phase: str) -> None:
    hook_name = {
        "train": "on_train_epoch_start",
        "val": "on_validation_epoch_start",
        "test": "on_test_epoch_start",
    }[phase]
    hook = getattr(model, hook_name, None)
    if not callable(hook):
        raise TypeError(f"throwaway model has no {hook_name}")
    hook()


def _require_finite_loss(value: object, phase: str) -> None:
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise TypeError(f"{phase} loss must be a scalar tensor")
    if not math.isfinite(float(value.detach().cpu().item())):
        raise ValueError(f"{phase} loss is not finite")


__all__ = [
    "PreflightCheck",
    "PreflightError",
    "PreflightResult",
    "PreflightRunner",
]
