"""Static validation coverage for qualified automatic preflight."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import hydra
import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from topobench.preflight import (
    PreflightCheck,
    PreflightError,
    PreflightResult,
    PreflightRunner,
)
from topobench.utils.config_resolvers import register_all_resolvers


def _compose_tiny_config() -> DictConfig:
    GlobalHydra.instance().clear()
    register_all_resolvers()
    with hydra.initialize(
        version_base="1.3",
        config_path="../../configs",
        job_name="preflight_static_validation",
    ):
        cfg = hydra.compose(
            config_name="run.yaml",
            overrides=[
                "dataset=graph/SyntheticGraph",
                "model=graph/gcn",
                "paths=test",
                "logger=csv",
                "callbacks=model_checkpoint",
                "trainer=cpu",
            ],
        )
    OmegaConf.set_struct(cfg, False)
    OmegaConf.update(
        cfg,
        "paths.root_dir",
        "test-root",
        merge=False,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "paths.work_dir",
        "test-work",
        merge=False,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "evaluator.metrics",
        ["accuracy"],
        merge=False,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "evaluator.policy",
        {"train": "online", "val": "exact", "test": "exact"},
        merge=False,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "evaluator.exact",
        {"max_ranking_bytes": 1048576},
        merge=False,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "dataset.loader.parameters.partition",
        {
            "strategy": "cluster",
            "backend": "pyg",
            "memory_limit_bytes": 1048576,
        },
        merge=False,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "dataset.loader.parameters.output_kind",
        "homogeneous",
        merge=False,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "dataset.loader.parameters.reproducibility",
        {"save_reproducibility_bundle": True},
        merge=False,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "dataset.loader.parameters.supervision.splits",
        {
            "active": "canonical",
            "within_tag_phases": "disjoint",
            "sets": {
                "canonical": {
                    "train": "splits/train.parquet",
                    "val": "splits/val.parquet",
                    "test": "splits/test.parquet",
                    "qualified": True,
                }
            },
        },
        merge=False,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "data_pipeline.active_split_tag",
        "canonical",
        merge=False,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "preflight.prefetch_memory_ceiling_bytes",
        1048576,
        merge=False,
        force_add=True,
    )
    return cfg


def _qualified_pipeline_output():
    source_identity = "a" * 64
    partition_identity = "b" * 64
    transform = SimpleNamespace(state_key="fit-state-v1")
    return SimpleNamespace(
        source_graph_id=source_identity,
        active_split_tag="canonical",
        qualification_report=SimpleNamespace(passed=True),
        fitted_transform=transform,
        reproducibility_policy=SimpleNamespace(
            save_reproducibility_bundle=True
        ),
        provenance_input={
            "source_graph_id": source_identity,
            "partition_book_identity": partition_identity,
            "active_split_tag": "canonical",
            "sampling_strategy": "homogeneous-cluster",
            "sampler_backend": "pyg",
            "fitted_transform": "identity",
            "fitted_transform_state_key": "fit-state-v1",
            "save_reproducibility_bundle": True,
            "qualified_profile": True,
        },
    )


def test_existing_qualified_config_and_matching_pipeline_evidence_pass() -> (
    None
):
    result = PreflightRunner(
        _compose_tiny_config(),
        _qualified_pipeline_output(),
    ).validate_static()

    assert result.passed is True
    assert result.qualified is True


@pytest.mark.parametrize("value", (0.5, 1, True, "00:10:00"))
def test_static_validation_rejects_mid_epoch_validation_schedules(
    value: object,
) -> None:
    cfg = _compose_tiny_config()
    OmegaConf.update(
        cfg,
        "trainer.val_check_interval",
        value,
        merge=False,
        force_add=True,
    )

    with pytest.raises(PreflightError) as error:
        PreflightRunner(cfg, _qualified_pipeline_output()).validate_static()

    check = error.value.result.checks[-1]
    assert check.check_id == "static.validation_schedule"
    assert "epoch-end" in check.remediation


def test_static_validation_accepts_explicit_epoch_end_validation() -> None:
    cfg = _compose_tiny_config()
    OmegaConf.update(
        cfg,
        "trainer.val_check_interval",
        1.0,
        merge=False,
        force_add=True,
    )

    result = PreflightRunner(
        cfg, _qualified_pipeline_output()
    ).validate_static()

    assert result.passed is True


@pytest.mark.parametrize(
    ("path", "value", "match"),
    (
        ("evaluator.task", "ranking", "task"),
        ("evaluator.metrics", ["mystery_metric"], "metric"),
        ("evaluator.policy.val", "eventual", "policy"),
        ("evaluator.exact.max_ranking_bytes", 0, "max_ranking_bytes"),
        (
            "dataset.loader.parameters.partition.memory_limit_bytes",
            -1,
            "partition",
        ),
        (
            "preflight.prefetch_memory_ceiling_bytes",
            0,
            "prefetch",
        ),
    ),
)
def test_static_validation_rejects_unknown_contracts_and_invalid_ceilings(
    path: str,
    value: object,
    match: str,
) -> None:
    cfg = _compose_tiny_config()
    OmegaConf.update(cfg, path, value, merge=False, force_add=True)

    with pytest.raises(PreflightError, match=match):
        PreflightRunner(cfg, _qualified_pipeline_output()).validate_static()


@pytest.mark.parametrize(
    "mutate",
    (
        lambda cfg: OmegaConf.update(
            cfg,
            "dataset.loader.parameters.supervision.splits.sets.canonical.val",
            "splits/train.parquet",
            merge=False,
            force_add=True,
        ),
        lambda cfg: OmegaConf.update(
            cfg,
            "dataset.loader.parameters.supervision.splits.active",
            "missing",
            merge=False,
            force_add=True,
        ),
        lambda cfg: OmegaConf.update(
            cfg,
            "data_pipeline.active_split_tag",
            "missing",
            merge=False,
            force_add=True,
        ),
    ),
)
def test_static_validation_rejects_overlapping_or_missing_named_split_tags(
    mutate: Callable[[DictConfig], None],
) -> None:
    cfg = _compose_tiny_config()
    mutate(cfg)

    with pytest.raises(PreflightError, match="split"):
        PreflightRunner(cfg, _qualified_pipeline_output()).validate_static()


def test_static_validation_rejects_unqualified_store_report() -> None:
    cfg = _compose_tiny_config()
    output = _qualified_pipeline_output()
    output.qualification_report = SimpleNamespace(passed=False)

    with pytest.raises(PreflightError, match="qualification"):
        PreflightRunner(cfg, output).validate_static()


def test_static_validation_rejects_stale_store_identity() -> None:
    cfg = _compose_tiny_config()
    output = _qualified_pipeline_output()
    output.source_graph_id = "c" * 64

    with pytest.raises(PreflightError, match="store identity"):
        PreflightRunner(cfg, output).validate_static()


def test_static_validation_rejects_malformed_partition_book_identity() -> None:
    cfg = _compose_tiny_config()
    output = _qualified_pipeline_output()
    output.provenance_input["partition_book_identity"] = "stale"

    with pytest.raises(PreflightError, match="partition book"):
        PreflightRunner(cfg, output).validate_static()


@pytest.mark.parametrize(
    "missing_key", ("sampling_strategy", "sampler_backend")
)
def test_static_validation_rejects_missing_sampler_identity(
    missing_key: str,
) -> None:
    cfg = _compose_tiny_config()
    output = _qualified_pipeline_output()
    del output.provenance_input[missing_key]

    with pytest.raises(PreflightError, match="sampler"):
        PreflightRunner(cfg, output).validate_static()


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("sampling_strategy", "heterogeneous-neighbor"),
        ("sampler_backend", "metis"),
    ),
)
def test_static_validation_rejects_stale_sampler_identity(
    key: str,
    value: str,
) -> None:
    cfg = _compose_tiny_config()
    output = _qualified_pipeline_output()
    output.provenance_input[key] = value

    with pytest.raises(PreflightError, match="sampler"):
        PreflightRunner(cfg, output).validate_static()


def test_static_validation_rejects_unsupported_sampler_backend() -> None:
    cfg = _compose_tiny_config()
    OmegaConf.update(
        cfg,
        "dataset.loader.parameters.partition.backend",
        "unqualified_backend",
        merge=False,
        force_add=True,
    )

    with pytest.raises(PreflightError, match="backend"):
        PreflightRunner(cfg, _qualified_pipeline_output()).validate_static()


def test_static_validation_rejects_stale_fitted_transform_identity() -> None:
    cfg = _compose_tiny_config()
    output = _qualified_pipeline_output()
    output.fitted_transform.state_key = "foreign-state"

    with pytest.raises(PreflightError, match="fitted-transform"):
        PreflightRunner(cfg, output).validate_static()


def test_static_validation_rejects_missing_fitted_transform_state_identity() -> (
    None
):
    cfg = _compose_tiny_config()
    output = _qualified_pipeline_output()
    del output.provenance_input["fitted_transform_state_key"]

    with pytest.raises(PreflightError, match="fitted-transform"):
        PreflightRunner(cfg, output).validate_static()


def test_static_validation_rejects_custom_metric_policy_incompatibility() -> (
    None
):
    cfg = _compose_tiny_config()
    OmegaConf.update(
        cfg,
        "evaluator.metrics",
        ["custom_streaming"],
        merge=False,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "evaluator.custom_metrics",
        [
            {
                "name": "custom_streaming",
                "supports_online": True,
                "supports_exact": False,
                "supports_audit": False,
            }
        ],
        merge=False,
        force_add=True,
    )

    with pytest.raises(PreflightError, match="custom metric.*exact"):
        PreflightRunner(cfg, _qualified_pipeline_output()).validate_static()


def test_static_validation_rejects_qualified_multi_rank_execution() -> None:
    cfg = _compose_tiny_config()
    OmegaConf.update(cfg, "trainer.devices", 2, merge=False, force_add=True)

    with pytest.raises(PreflightError, match="multi-rank"):
        PreflightRunner(cfg, _qualified_pipeline_output()).validate_static()


def test_static_validation_rejects_nan_capable_checkpoint_monitor_without_policy() -> (
    None
):
    cfg = _compose_tiny_config()
    OmegaConf.update(cfg, "evaluator.metrics", ["auroc"], merge=False)
    OmegaConf.update(
        cfg,
        "callbacks.model_checkpoint.monitor",
        "val/auroc",
        merge=False,
    )
    OmegaConf.update(
        cfg,
        "preflight.monitor_nan_policy",
        None,
        merge=False,
        force_add=True,
    )

    with pytest.raises(PreflightError, match="monitor.*NaN"):
        PreflightRunner(cfg, _qualified_pipeline_output()).validate_static()


def test_static_validation_rejects_logger_without_artifact_adapter() -> None:
    cfg = _compose_tiny_config()
    OmegaConf.update(
        cfg,
        "evaluation_artifacts",
        {"enabled": True},
        merge=False,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "logger.csv._target_",
        "lightning.pytorch.loggers.NeptuneLogger",
        merge=False,
    )

    with pytest.raises(PreflightError, match="artifact adapter"):
        PreflightRunner(cfg, _qualified_pipeline_output()).validate_static()


def test_static_validation_rejects_slice_field_not_captured_as_metadata() -> (
    None
):
    cfg = _compose_tiny_config()
    OmegaConf.update(
        cfg,
        "evaluation_artifacts",
        {
            "enabled": True,
            "metadata_fields": [],
            "evaluation_slices": {
                "source": {"max_categories": 8, "min_rows": 1}
            },
        },
        merge=False,
        force_add=True,
    )

    with pytest.raises(PreflightError, match="slice field.*metadata_fields"):
        PreflightRunner(cfg, _qualified_pipeline_output()).validate_static()


def test_static_validation_accepts_bounded_evaluation_slice_vocabulary() -> (
    None
):
    cfg = _compose_tiny_config()
    OmegaConf.update(
        cfg,
        "evaluation_artifacts",
        {
            "enabled": True,
            "metadata_fields": ["source"],
            "evaluation_slices": {
                "source": {
                    "max_categories": 2,
                    "min_rows": 1,
                    "vocabulary": ["alpha", "beta"],
                }
            },
        },
        merge=False,
        force_add=True,
    )

    result = PreflightRunner(
        cfg, _qualified_pipeline_output()
    ).validate_static()

    assert result.passed is True


def test_static_validation_rejects_unbounded_evaluation_slice() -> None:
    cfg = _compose_tiny_config()
    OmegaConf.update(
        cfg,
        "evaluation_artifacts",
        {
            "enabled": True,
            "metadata_fields": ["source"],
            "evaluation_slices": {"source": {"min_rows": 1}},
        },
        merge=False,
        force_add=True,
    )

    with pytest.raises(PreflightError, match="max_categories"):
        PreflightRunner(cfg, _qualified_pipeline_output()).validate_static()


def test_static_validation_rejects_disabled_reproducibility_bundle() -> None:
    cfg = _compose_tiny_config()
    OmegaConf.update(
        cfg,
        (
            "dataset.loader.parameters.reproducibility."
            "save_reproducibility_bundle"
        ),
        False,
        merge=False,
    )

    with pytest.raises(PreflightError, match="reproducibility bundle"):
        PreflightRunner(cfg, _qualified_pipeline_output()).validate_static()


def test_static_validation_rejects_unresolved_interpolation() -> None:
    cfg = _compose_tiny_config()
    OmegaConf.update(
        cfg,
        "preflight.prefetch_memory_ceiling_bytes",
        "${missing.ceiling}",
        merge=False,
    )

    with pytest.raises(PreflightError, match="interpolation"):
        PreflightRunner(cfg, _qualified_pipeline_output()).validate_static()


def test_structured_preflight_values_are_immutable() -> None:
    check = PreflightCheck("static.configuration", True, "resolved")
    result = PreflightResult(enabled=True, qualified=True, checks=(check,))

    with pytest.raises(FrozenInstanceError):
        check.passed = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.qualified = False  # type: ignore[misc]
    assert result.passed is True
