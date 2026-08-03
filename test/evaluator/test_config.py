"""Authoritative evaluator configuration and dependency contracts."""

from __future__ import annotations

import importlib
import inspect
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import hydra
import pytest
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationResolutionError
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from topobench.evaluator import TBEvaluator
from topobench.utils.config_resolvers import register_all_resolvers

PROJECT_ROOT = Path(__file__).parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"
EXPECTED_POLICY = {"train": "online", "val": "exact", "test": "exact"}
EXPECTED_ONLINE_RESOURCES = {"ranking_thresholds": 512}
EXPECTED_EXACT_RESOURCES = {
    "max_ranking_bytes": 536870912,
    "buffer_device": "cpu",
}
REMOVED_FIELDS = frozenset(
    {"multioutput_classes", "auroc_thresholds", "max_auroc_bytes"}
)


def _compose(*overrides: str) -> DictConfig:
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    register_all_resolvers()
    with hydra.initialize_config_dir(
        version_base="1.3",
        config_dir=str(CONFIG_ROOT),
        job_name="evaluator_config_contract",
    ):
        return hydra.compose(
            config_name="run.yaml",
            overrides=list(overrides),
        )


def _nested_mapping_keys(value: Any) -> frozenset[str]:
    if isinstance(value, dict):
        return frozenset(value) | frozenset(
            key
            for child in value.values()
            for key in _nested_mapping_keys(child)
        )
    if isinstance(value, list):
        return frozenset(
            key for child in value for key in _nested_mapping_keys(child)
        )
    return frozenset()


def _resolved_evaluator(cfg: DictConfig) -> dict[str, Any]:
    evaluator = OmegaConf.to_container(
        cfg.evaluator,
        resolve=True,
        throw_on_missing=True,
    )
    assert isinstance(evaluator, dict)
    return evaluator


@pytest.mark.parametrize(
    ("evaluator_selector", "dataset_selector"),
    [
        ("default", "graph/SyntheticGraph"),
        ("classification", "graph/IMDB-MULTI"),
        ("regression", "graph/SyntheticGraphRegression"),
    ],
)
def test_every_evaluator_selector_has_the_authoritative_policy_surface(
    evaluator_selector: str,
    dataset_selector: str,
) -> None:
    cfg = _compose(
        f"evaluator={evaluator_selector}",
        f"dataset={dataset_selector}",
    )
    evaluator = _resolved_evaluator(cfg)

    assert evaluator.get("policy") == EXPECTED_POLICY
    assert evaluator.get("online") == EXPECTED_ONLINE_RESOURCES
    assert evaluator.get("exact") == EXPECTED_EXACT_RESOURCES
    assert evaluator.get("undefined_metric_policy") == "error"
    assert REMOVED_FIELDS.isdisjoint(_nested_mapping_keys(evaluator))
    assert cfg.preflight.enabled is True


def test_evaluator_uses_authoritative_resource_defaults_without_aliases() -> (
    None
):
    evaluator = TBEvaluator(
        "classification",
        num_classes=2,
        metrics=["accuracy"],
    )

    assert evaluator.metric_backend.ranking_thresholds == 512
    assert evaluator.metric_backend.max_exact_ranking_bytes == 536870912


@pytest.mark.parametrize(
    ("legacy_name", "legacy_value"),
    [
        ("ranking_thresholds", 1024),
        ("max_exact_ranking_bytes", 1073741824),
    ],
)
def test_evaluator_rejects_explicit_legacy_resource_aliases(
    legacy_name: str,
    legacy_value: int,
) -> None:
    with pytest.raises(ValueError, match=legacy_name):
        TBEvaluator(
            "classification",
            num_classes=2,
            metrics=["accuracy"],
            **{legacy_name: legacy_value},
        )


def test_dataset_metric_override_is_resolved_through_the_active_registry() -> (
    None
):
    cfg = _compose(
        "dataset=graph/SyntheticGraph",
        "+dataset.parameters.metrics=[accuracy,auprc,somers_d]",
    )

    assert _resolved_evaluator(cfg)["metrics"] == [
        "accuracy",
        "auprc",
        "somers_d",
    ]


@pytest.mark.parametrize("metric", ["auprc", "somers_d"])
def test_dataset_binary_metric_override_rejects_multiclass(
    metric: str,
) -> None:
    cfg = _compose(
        "dataset=graph/IMDB-MULTI",
        f'+dataset.parameters.metrics=["{metric}"]',
    )

    with pytest.raises(
        InterpolationResolutionError,
        match="binary classification",
    ):
        _resolved_evaluator(cfg)


@pytest.mark.parametrize(
    "metric",
    [
        "example",
        "confusion_matrix",
        "f1_macro",
        "f1_weighted",
        "accuracy-0",
        "accuracy_0",
    ],
)
def test_dataset_metric_override_rejects_removed_names(metric: str) -> None:
    cfg = _compose(
        "dataset=graph/SyntheticGraph",
        f'+dataset.parameters.metrics=["{metric}"]',
    )

    with pytest.raises(
        InterpolationResolutionError,
        match="Unsupported metric",
    ):
        _resolved_evaluator(cfg)


def _project_requirements() -> dict[str, Requirement]:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    requirements = [Requirement(value) for value in project["dependencies"]]
    return {canonicalize_name(item.name): item for item in requirements}


def _locked_package(name: str) -> dict[str, Any]:
    with (PROJECT_ROOT / "uv.lock").open("rb") as stream:
        lock = tomllib.load(stream)
    matches = [
        package
        for package in lock["package"]
        if canonicalize_name(package["name"]) == canonicalize_name(name)
    ]
    assert len(matches) == 1, name
    return matches[0]


def test_torchmetrics_is_a_direct_constrained_locked_dependency() -> None:
    requirements = _project_requirements()
    assert "torchmetrics" in requirements
    torchmetrics_requirement = requirements["torchmetrics"]
    locked_torchmetrics = _locked_package("torchmetrics")
    locked_root = _locked_package("topobench")

    assert str(torchmetrics_requirement.specifier)
    assert Version(locked_torchmetrics["version"]) in (
        torchmetrics_requirement.specifier
    )
    assert any(
        canonicalize_name(dependency["name"]) == "torchmetrics"
        for dependency in locked_root["dependencies"]
    )
    locked_requirement = next(
        requirement
        for requirement in locked_root["metadata"]["requires-dist"]
        if canonicalize_name(requirement["name"]) == "torchmetrics"
    )
    assert locked_requirement["specifier"]
    assert (
        Version(locked_torchmetrics["version"])
        in Requirement(
            f"torchmetrics{locked_requirement['specifier']}"
        ).specifier
    )
    assert str(requirements["torch"].specifier) == "==2.3.0"
    assert str(requirements["lightning"].specifier) == "==2.4.0"


def test_locked_metrics_stack_imports_in_an_isolated_process(
    tmp_path: Path,
) -> None:
    requirements = _project_requirements()
    locked_torchmetrics = _locked_package("torchmetrics")["version"]
    script = """
import importlib
import importlib.metadata
import json

for name in ("torch", "lightning", "torchmetrics"):
    importlib.import_module(name)
versions = {
    name: importlib.metadata.version(name)
    for name in ("torch", "lightning", "torchmetrics")
}
print("TOPOBENCH_IMPORT_VERSIONS=" + json.dumps(versions, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    version_line = next(
        line
        for line in completed.stdout.splitlines()
        if line.startswith("TOPOBENCH_IMPORT_VERSIONS=")
    )
    versions = json.loads(version_line.partition("=")[2])
    assert Version(versions["torch"]) in requirements["torch"].specifier
    assert Version(versions["lightning"]) in (
        requirements["lightning"].specifier
    )
    assert versions["torchmetrics"] == locked_torchmetrics


def test_public_evaluator_exports_no_torchmetrics_class() -> None:
    evaluator_api = importlib.import_module("topobench.evaluator")

    for name in evaluator_api.__all__:
        exported = getattr(evaluator_api, name)
        if inspect.isclass(exported):
            assert all(
                base.__module__.partition(".")[0] != "torchmetrics"
                for base in exported.__mro__
            ), name
