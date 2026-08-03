"""Automatic static and isolated execution qualification before training."""

from __future__ import annotations

import copy
import math
import random
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from lightning import LightningDataModule, LightningModule
from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, HeteroData

from topobench.evaluator import EvaluationResult, UndefinedMetricError
from topobench.evaluator.base import AbstractEvaluator
from topobench.evaluator.prediction import PredictionPayload
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
            raise ValueError(
                "preflight result must contain at least one check"
            )
        if self.qualified and (not self.enabled or not self.passed):
            raise ValueError(
                "only an enabled, passing preflight can be qualified"
            )

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
            raise ValueError(
                "PreflightError requires a failing PreflightResult"
            )
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
        self._validate_reproducibility_bundle(
            resolved, checks, profile=profile
        )
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
        """Exercise production boundaries with fully disposable runtime state."""
        if (
            not isinstance(static_result, PreflightResult)
            or not static_result.passed
        ):
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

        runtime_state = _snapshot_runtime_state()
        model: LightningModule | None = None
        optimizer: Optimizer | None = None
        scheduler: object | None = None
        scheduler_config: Mapping[str, Any] = {}
        scheduler_stepped = False
        phase_losses: dict[str, torch.Tensor] = {}
        payloads: list[Mapping[str, object]] = []
        try:
            model = model_factory()
            if not isinstance(model, LightningModule):
                raise TypeError("model factory must return a LightningModule")
            if not isinstance(
                getattr(model, "evaluator", None), AbstractEvaluator
            ):
                raise TypeError(
                    "throwaway model must own an AbstractEvaluator"
                )
            if not callable(getattr(model, "model_step", None)):
                raise TypeError("throwaway model must expose model_step")
            if not callable(getattr(model, "abort_evaluation", None)):
                raise TypeError("throwaway model must expose abort_evaluation")

            device = _configured_probe_device(self._resolved)
            model.to(device)
            if "train" in phases:
                configure_optimizers = getattr(
                    model, "configure_optimizers", None
                )
                if not callable(configure_optimizers):
                    raise TypeError(
                        "throwaway model must configure an optimizer"
                    )
                optimizer, scheduler, scheduler_config = _optimizer_components(
                    configure_optimizers()
                )
            setup = getattr(model, "setup", None)
            if callable(setup):
                setup("fit" if "train" in phases else "test")

            with _representative_batches(datamodule, phases) as batches:
                for phase in phases:
                    batch = batches[phase]
                    if not isinstance(batch, (Data, HeteroData)):
                        raise TypeError(
                            f"{phase} representative batch must be native Data or HeteroData"
                        )
                    transferred = model.transfer_batch_to_device(
                        batch, device, 0
                    )
                    _start_evaluation_phase(model, phase)
                    model.train(phase == "train")
                    try:
                        if phase == "train":
                            if optimizer is None:
                                raise RuntimeError(
                                    "training probe has no throwaway optimizer"
                                )
                            optimizer.zero_grad(set_to_none=True)
                        context = (
                            torch.enable_grad()
                            if phase == "train"
                            else torch.no_grad()
                        )
                        with context:
                            model_out = model.model_step(transferred)
                        if not isinstance(model_out, Mapping):
                            raise TypeError("model_step must return a mapping")
                        loss = _require_finite_loss(
                            model_out.get("loss"), phase
                        )
                        phase_losses[phase] = loss.detach()
                        payloads.append(
                            _validate_nonpublishing_prediction_payload(
                                model_out,
                                phase=phase,
                                device=device,
                                pipeline_output=self._pipeline_output,
                            )
                        )
                        if phase == "train":
                            loss.backward()
                            _require_finite_gradients(model)

                            def optimizer_closure(
                                loss_value: torch.Tensor = loss,
                            ) -> torch.Tensor:
                                return loss_value

                            success_token_before = getattr(
                                model,
                                "dataloader_optimizer_success_token",
                                None,
                            )
                            model.optimizer_step(
                                epoch=0,
                                batch_idx=0,
                                optimizer=optimizer,
                                optimizer_closure=optimizer_closure,
                            )
                            if isinstance(success_token_before, int):
                                success_token_after = getattr(
                                    model,
                                    "dataloader_optimizer_success_token",
                                    None,
                                )
                                if (
                                    success_token_after
                                    != success_token_before + 1
                                ):
                                    raise RuntimeError(
                                        "model optimizer hook did not prove one successful step"
                                    )
                            if (
                                scheduler is not None
                                and scheduler_config.get("interval", "epoch")
                                == "step"
                            ):
                                _step_probe_scheduler(scheduler, loss.detach())
                                scheduler_stepped = True
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

            if (
                scheduler is not None
                and not scheduler_stepped
                and "train" in phases
            ):
                scheduler_loss = phase_losses.get(
                    "val",
                    phase_losses.get("train"),
                )
                if scheduler_loss is None:
                    raise RuntimeError(
                        "scheduler probe requires a semantically valid loss"
                    )
                _step_probe_scheduler(scheduler, scheduler_loss)
                scheduler_stepped = True
            _validate_structured_probe_payloads(
                payloads,
                pipeline_output=self._pipeline_output,
            )
        except Exception as error:
            if model is not None:
                abort = getattr(model, "abort_evaluation", None)
                if callable(abort):
                    with suppress(Exception):
                        abort()
            self._probe_failure(
                static_result,
                f"isolated execution probe failed: {type(error).__name__}: {error}",
                "repair the reported data/model/supervision/loss/evaluator/optimizer boundary",
                cause=error,
            )
        finally:
            if model is not None:
                with suppress(Exception):
                    model.to(torch.device("cpu"))
            scheduler = None
            optimizer = None
            model = None
            optimizer_closure = None
            scheduler_loss = None
            phase_losses.clear()
            loss = None
            model_out = None
            transferred = None
            batch = None
            batches = None
            with suppress(Exception):
                _purge_probe_device_cache(self._resolved)
            _restore_runtime_state(runtime_state)

        configured_compile = bool(
            "train" in phases
            and _select(self._resolved, "model.compile", False)
        )
        training_checks = (
            (
                PreflightCheck(
                    "execution.gradient",
                    True,
                    "training loss produced nonempty finite gradients",
                ),
                PreflightCheck(
                    "execution.optimizer",
                    True,
                    "throwaway optimizer constructed and completed one step",
                ),
                PreflightCheck(
                    "execution.scheduler",
                    True,
                    (
                        "configured scheduler constructed and stepped"
                        if scheduler_stepped
                        else "no scheduler was configured"
                    ),
                ),
            )
            if "train" in phases
            else ()
        )
        return PreflightResult(
            enabled=True,
            qualified=static_result.qualified,
            checks=static_result.checks
            + (
                PreflightCheck(
                    "execution.representative_batch",
                    True,
                    "one non-committing native batch per enabled phase passed",
                ),
            )
            + training_checks
            + (
                PreflightCheck(
                    "execution.compile",
                    True,
                    (
                        "configured compile path executed"
                        if configured_compile
                        else "compile disabled and no compile path requested"
                    ),
                ),
                PreflightCheck(
                    "execution.structured_checks",
                    True,
                    "bounded structured execution payload validated without emission",
                ),
                PreflightCheck(
                    "execution.reproducibility_payload",
                    True,
                    "reproducibility payload validated without publication",
                ),
                PreflightCheck(
                    "execution.prediction_payload",
                    True,
                    "prediction payload schema validated without publication",
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
        if not isinstance(task, str) or task not in {
            "classification",
            "regression",
        }:
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
            self._split_failure(
                checks, "named split registry must be a mapping"
            )
        active = registry.get("active")
        sets = registry.get("sets")
        if not isinstance(active, str) or not active:
            self._split_failure(
                checks, "named split registry has no active tag"
            )
        if not isinstance(sets, Mapping) or not sets:
            self._split_failure(
                checks, "named split registry has no split tags"
            )
        if active not in sets:
            self._split_failure(
                checks,
                f"active split tag {active!r} is not registered",
            )
        for tag, split in sets.items():
            if (
                not isinstance(tag, str)
                or not tag
                or not isinstance(split, Mapping)
            ):
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
                    overlap = phase_sources[left].intersection(
                        phase_sources[right]
                    )
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
        if (
            profile == "qualified"
            and active_record.get("qualified", True) is not True
        ):
            self._split_failure(
                checks, f"active split tag {active!r} is unqualified"
            )

    def _validate_validation_schedule(
        self,
        cfg: Mapping[str, Any],
        checks: list[PreflightCheck],
        *,
        profile: str,
    ) -> None:
        interval = _select(cfg, "trainer.val_check_interval", None)
        if (
            profile == "qualified"
            and interval is not None
            and (
                isinstance(interval, bool)
                or not isinstance(interval, float)
                or interval != 1.0
            )
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
        if source_identity is not None and recorded_source != source_identity:
            self._fail(
                checks,
                "static.store_identity",
                "pipeline store identity is stale against pipeline provenance",
                "bind provenance to the exact source opened by the pipeline",
                enabled=True,
            )
        if (
            report is not None
            and source_identity is not None
            and not _is_sha256(source_identity)
        ):
            self._fail(
                checks,
                "static.store_identity",
                "qualified store identity is malformed",
                "reopen the exact content-addressed qualified store",
                enabled=True,
            )
        if report is None:
            return
        partition_identity = provenance.get("partition_book_identity")
        if partition_identity is not None and not _is_sha256(
            partition_identity
        ):
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
        if output_kind in {
            "homogeneous",
            "heterogeneous",
        } and partition_strategy in {
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
        if (
            profile == "qualified"
            and provenance.get("qualified_profile", True) is not True
        ):
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
            if not isinstance(target, str) or not target.endswith(
                "ModelCheckpoint"
            ):
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
        if (
            not isinstance(artifacts, Mapping)
            or artifacts.get("enabled") is not True
        ):
            return
        metadata_value = artifacts.get("metadata_fields", ())
        if (
            not isinstance(metadata_value, Sequence)
            or isinstance(metadata_value, (str, bytes))
            or any(
                not isinstance(field, str) or not field
                for field in metadata_value
            )
        ):
            self._fail(
                checks,
                "static.evaluation_slices",
                "evaluation artifact metadata_fields must be non-empty strings",
                "declare every captured metadata column by name",
                enabled=True,
            )
            metadata_fields = frozenset()
        else:
            metadata_fields = frozenset(metadata_value)
        slice_value = artifacts.get("evaluation_slices", {})
        if not isinstance(slice_value, Mapping):
            self._fail(
                checks,
                "static.evaluation_slices",
                "evaluation_slices must be a mapping",
                "map each bounded metadata field to max_categories and min_rows",
                enabled=True,
            )
            slice_value = {}
        for field, spec in slice_value.items():
            if not isinstance(field, str) or not field:
                self._fail(
                    checks,
                    "static.evaluation_slices",
                    "evaluation slice field names must be non-empty strings",
                    "name a captured metadata column",
                    enabled=True,
                )
                continue
            if field not in metadata_fields:
                self._fail(
                    checks,
                    "static.evaluation_slices",
                    f"evaluation slice field {field!r} is absent from metadata_fields",
                    "capture the slice field in evaluation_artifacts.metadata_fields",
                    enabled=True,
                )
            if not isinstance(spec, Mapping):
                self._fail(
                    checks,
                    "static.evaluation_slices",
                    f"evaluation slice {field!r} must be a mapping",
                    "set max_categories and min_rows",
                    enabled=True,
                )
                continue
            unknown = set(spec) - {
                "max_categories",
                "min_rows",
                "vocabulary",
            }
            if unknown:
                self._fail(
                    checks,
                    "static.evaluation_slices",
                    f"evaluation slice {field!r} has unknown settings {sorted(unknown)!r}",
                    "use max_categories, min_rows, and optional vocabulary",
                    enabled=True,
                )
            for name in ("max_categories", "min_rows"):
                value = spec.get(name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                ):
                    self._fail(
                        checks,
                        "static.evaluation_slices",
                        f"evaluation slice {field!r} {name} must be a positive integer",
                        f"set {name} to an explicit positive bound",
                        enabled=True,
                    )
            vocabulary = spec.get("vocabulary")
            if vocabulary is None:
                continue
            if (
                isinstance(vocabulary, (str, bytes))
                or not isinstance(vocabulary, Sequence)
                or not vocabulary
                or any(
                    not isinstance(item, str) or not item
                    for item in vocabulary
                )
            ):
                self._fail(
                    checks,
                    "static.evaluation_slices",
                    f"evaluation slice {field!r} vocabulary must contain non-empty strings",
                    "declare a finite vocabulary of non-empty category names",
                    enabled=True,
                )
            if len(vocabulary) != len(set(vocabulary)):
                self._fail(
                    checks,
                    "static.evaluation_slices",
                    f"evaluation slice {field!r} vocabulary contains duplicates",
                    "declare each category exactly once",
                    enabled=True,
                )
            if len(vocabulary) > spec["max_categories"]:
                self._fail(
                    checks,
                    "static.evaluation_slices",
                    f"evaluation slice {field!r} vocabulary exceeds max_categories",
                    "increase max_categories or reduce the declared vocabulary",
                    enabled=True,
                )
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
            if (
                isinstance(target, str)
                and target not in _SUPPORTED_ARTIFACT_LOGGERS
            ):
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
            None
            if output is None
            else getattr(output, "provenance_input", None)
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
            + (
                PreflightCheck(
                    "execution.representative_batch",
                    False,
                    detail,
                    remediation,
                ),
            ),
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
    if isinstance(configured, (str, bytes)) or not isinstance(
        configured, Sequence
    ):
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


@dataclass(frozen=True, slots=True)
class _RuntimeState:
    python_rng: object
    numpy_rng: object
    torch_cpu_rng: torch.Tensor
    torch_cuda_rng: tuple[torch.Tensor, ...]
    default_dtype: torch.dtype
    grad_enabled: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    deterministic_algorithms: bool
    deterministic_warn_only: bool
    float32_matmul_precision: str


def _snapshot_runtime_state() -> _RuntimeState:
    return _RuntimeState(
        python_rng=random.getstate(),
        numpy_rng=copy.deepcopy(np.random.get_state()),
        torch_cpu_rng=torch.random.get_rng_state().clone(),
        torch_cuda_rng=tuple(
            state.clone() for state in torch.cuda.get_rng_state_all()
        ),
        default_dtype=torch.get_default_dtype(),
        grad_enabled=torch.is_grad_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
    )


def _restore_runtime_state(state: _RuntimeState) -> None:
    random.setstate(state.python_rng)
    np.random.set_state(state.numpy_rng)
    torch.random.set_rng_state(state.torch_cpu_rng)
    if state.torch_cuda_rng:
        torch.cuda.set_rng_state_all(list(state.torch_cuda_rng))
    torch.set_default_dtype(state.default_dtype)
    torch.set_grad_enabled(state.grad_enabled)
    torch.backends.cudnn.benchmark = state.cudnn_benchmark
    torch.backends.cudnn.deterministic = state.cudnn_deterministic
    torch.use_deterministic_algorithms(
        state.deterministic_algorithms,
        warn_only=state.deterministic_warn_only,
    )
    torch.set_float32_matmul_precision(state.float32_matmul_precision)


def _optimizer_components(
    configured: object,
) -> tuple[Optimizer, object | None, Mapping[str, Any]]:
    if isinstance(configured, Optimizer):
        return configured, None, {}
    if not isinstance(configured, Mapping):
        raise TypeError(
            "configure_optimizers must return one optimizer or a mapping"
        )
    optimizer = configured.get("optimizer")
    if not isinstance(optimizer, Optimizer):
        raise TypeError("configure_optimizers did not return one optimizer")
    scheduler_value = configured.get("lr_scheduler")
    if scheduler_value is None:
        return optimizer, None, {}
    if isinstance(scheduler_value, Mapping):
        scheduler = scheduler_value.get("scheduler")
        scheduler_config = scheduler_value
    else:
        scheduler = scheduler_value
        scheduler_config = {}
    if scheduler is None or not callable(getattr(scheduler, "step", None)):
        raise TypeError("configured scheduler must expose step")
    return optimizer, scheduler, scheduler_config


def _step_probe_scheduler(
    scheduler: object,
    metric: torch.Tensor,
) -> None:
    step = getattr(scheduler, "step", None)
    if not callable(step):
        raise TypeError("configured scheduler must expose step")
    if isinstance(scheduler, ReduceLROnPlateau):
        step(float(metric.detach().cpu().item()))
    else:
        step()


def _require_finite_gradients(model: LightningModule) -> None:
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        raise ValueError("training probe produced no gradients")
    if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise ValueError("training probe gradients are not finite")


def _validate_nonpublishing_prediction_payload(
    model_out: Mapping[str, Any],
    *,
    phase: str,
    device: torch.device,
    pipeline_output: object,
) -> Mapping[str, object]:
    logits = model_out.get("logits")
    labels = model_out.get("labels")
    if not isinstance(logits, torch.Tensor) or not isinstance(
        labels,
        torch.Tensor,
    ):
        raise TypeError("prediction payload requires tensor logits and labels")
    if logits.ndim == 0 or labels.ndim == 0:
        raise ValueError(
            "prediction payload tensors require a leading dimension"
        )
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("prediction payload logits and labels are misaligned")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("prediction payload logits are not finite")
    payload: dict[str, object] = {
        "phase": phase,
        "device": str(device),
        "num_examples": int(labels.shape[0]),
        "logit_shape": tuple(int(value) for value in logits.shape),
        "label_shape": tuple(int(value) for value in labels.shape),
        "logit_dtype": str(logits.dtype),
        "label_dtype": str(labels.dtype),
    }
    adapter = getattr(pipeline_output, "prediction_row_adapter", None)
    prediction_payload = model_out.get("prediction_payload")
    if prediction_payload is not None:
        if adapter is None:
            raise TypeError(
                "prediction payload requires the pipeline prediction row adapter"
            )
        if not isinstance(prediction_payload, PredictionPayload):
            raise TypeError(
                "model prediction_payload must be a PredictionPayload"
            )
        if prediction_payload.num_rows != int(labels.shape[0]):
            raise ValueError(
                "prediction payload does not align with supervised outputs"
            )
        identity_names = tuple(prediction_payload.identity.columns)
        if not identity_names or identity_names[0] != "split_ordinal":
            raise ValueError(
                "prediction payload must prepend split_ordinal identity"
            )
        if "split_ordinal" in prediction_payload.identity.key:
            raise ValueError(
                "split_ordinal must not replace the domain identity key"
            )
        payload["canonical_identity_count"] = prediction_payload.num_rows
    return payload


def _canonical_probe_value(value: object) -> None:
    if value is None or type(value) in {bool, int, float, str}:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                "structured probe payload contains a nonfinite value"
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(
                    "structured probe payload keys must be strings"
                )
            _canonical_probe_value(item)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _canonical_probe_value(item)
        return
    raise TypeError(
        f"structured probe payload contains {type(value).__name__}"
    )


def _validate_structured_probe_payloads(
    payloads: Sequence[Mapping[str, object]],
    *,
    pipeline_output: object,
) -> None:
    if not payloads:
        raise ValueError("execution probe produced no structured payloads")
    reproducibility = {
        "source_graph_id": getattr(pipeline_output, "source_graph_id", None),
        "active_split_tag": getattr(
            pipeline_output,
            "active_split_tag",
            None,
        ),
        "phase_count": len(payloads),
    }
    _canonical_probe_value(tuple(payloads))
    _canonical_probe_value(reproducibility)


def _purge_probe_device_cache(cfg: Mapping[str, Any]) -> None:
    accelerator = _select(cfg, "trainer.accelerator", "cpu")
    if accelerator == "auto":
        if torch.cuda.is_available():
            accelerator = "cuda"
        elif torch.backends.mps.is_available():
            accelerator = "mps"
        else:
            accelerator = "cpu"
    if accelerator in {"gpu", "cuda"}:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return
    if accelerator == "mps" and torch.backends.mps.is_available():
        mps = getattr(torch, "mps", None)
        empty_cache = getattr(mps, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()


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
    provider = getattr(datamodule, "noncommitting_probe_batches", None)
    if callable(provider):
        with provider(tuple(phases)) as provided:
            if not isinstance(provided, Mapping):
                raise TypeError(
                    "noncommitting probe provider must yield a phase mapping"
                )
            if tuple(provided) != tuple(phases):
                raise ValueError(
                    "noncommitting probe provider returned unexpected phases"
                )
            batches = dict(provided)
            if any(
                not isinstance(batch, (Data, HeteroData))
                for batch in batches.values()
            ):
                raise TypeError(
                    "noncommitting probe provider must yield native batches"
                )
            yield batches
        return
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
            raise ValueError(
                f"{phase} requires exactly one representative loader"
            )
        return values[0]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 1:
            raise ValueError(
                f"{phase} requires exactly one representative loader"
            )
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


def _require_finite_loss(value: object, phase: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise TypeError(f"{phase} loss must be a scalar tensor")
    if not math.isfinite(float(value.detach().cpu().item())):
        raise ValueError(f"{phase} loss is not finite")
    return value


__all__ = [
    "PreflightCheck",
    "PreflightError",
    "PreflightResult",
    "PreflightRunner",
]
