"""Tests for the closed public domain, config, and neural-surface contract."""

import re
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from test.architecture.verify_forbidden_imports import FORBIDDEN_DEPENDENCIES
from topobench import SUPPORTED_DOMAINS, require_supported_domain
from topobench.data.capabilities import (
    GRAPH_DATASET_MANIFEST,
    qualify_graph_dataset,
)
from topobench.domains import __all__ as domain_exports
from topobench.nn import backbones, encoders, readouts, wrappers
from topobench.nn.backbones import graph as graph_backbones
from topobench.nn.backbones import heterogeneous as heterogeneous_backbones
from topobench.nn.backbones import hypergraph as hypergraph_backbones
from topobench.nn.wrappers import graph as graph_wrappers
from topobench.nn.wrappers import heterogeneous as heterogeneous_wrappers
from topobench.nn.wrappers import hypergraph as hypergraph_wrappers

EXPECTED_BACKBONES = (
    "EDGNN",
    "GCNDGM",
    "GPSEncoder",
    "GraphMLP",
    "HGTBackbone",
    "HeteroSAGEBackbone",
    "HypergraphConvBackbone",
    "NSDEncoder",
)
EXPECTED_WRAPPERS = (
    "GNNWrapper",
    "GraphMLPWrapper",
    "HeterogeneousWrapper",
    "HypergraphWrapper",
)
EXPECTED_ENCODERS = (
    "DGMStructureFeatureEncoder",
    "GraphNodeFeatureEncoder",
    "HeterogeneousNodeFeatureEncoder",
)
EXPECTED_READOUTS = (
    "HeterogeneousNodeReadout",
    "MLPReadout",
    "NoReadOut",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"
PRODUCTION_ROOT = PROJECT_ROOT / "topobench"
HISTORICAL_PLAN_ROOT = PROJECT_ROOT / "docs" / "plans"
SUPPORTED_TASK_KINDS = frozenset({"classification", "regression"})
REMOVED_GRAPH_DATASETS = (
    "US-county-demos",
    "graphuniverse_inductive",
    "ogbg-molpcba",
    "manual_dataset",
)
REMOVED_SOURCE_ONLY_PATHS = (
    PRODUCTION_ROOT / "data/loaders/graph/manual_graph_dataset_loader.py",
    PRODUCTION_ROOT / "data/loaders/graph/us_county_demos_dataset_loader.py",
    PRODUCTION_ROOT / "data/datasets/us_county_demos_dataset.py",
)
FORBIDDEN_TOKENS = (
    "x_0",
    "x_1",
    "x_2",
    "batch_0",
    "batch_1",
    "batch_2",
    "incidence_1",
    "incidence_2",
    "incidence_hyperedges",
    "num_cell_dimensions",
    *FORBIDDEN_DEPENDENCIES,
)
FORBIDDEN_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])("
    + "|".join(re.escape(token) for token in FORBIDDEN_TOKENS)
    + r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


def _domain_directories(package_file: str) -> tuple[str, ...]:
    package_root = Path(package_file).parent
    return tuple(
        sorted(
            path.name
            for path in package_root.iterdir()
            if path.is_dir() and (path / "__init__.py").is_file()
        )
    )


def _config_domain_directories(group: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.name
            for path in (CONFIG_ROOT / group).iterdir()
            if path.is_dir()
        )
    )


def _surviving_yaml_paths() -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for pattern in ("*.yaml", "*.yml")
            for path in PROJECT_ROOT.rglob(pattern)
            if not path.is_relative_to(HISTORICAL_PLAN_ROOT)
        )
    )


def _selector_pattern(selector: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<![A-Za-z0-9_-]){re.escape(selector)}(?![A-Za-z0-9_-])",
        re.IGNORECASE,
    )


def _assert_registry(
    registry: dict[str, type],
    expected: tuple[str, ...],
) -> None:
    assert registry.__class__ is dict
    assert tuple(registry) == expected
    assert all(
        name == registered_class.__name__
        for name, registered_class in registry.items()
    )


def test_supported_domains_are_closed_and_ordered():
    """The public domain set has an explicit, stable order."""
    assert SUPPORTED_DOMAINS == ("graph", "heterogeneous", "hypergraph")


@pytest.mark.parametrize("domain", SUPPORTED_DOMAINS)
def test_supported_domains_are_returned_unchanged(domain):
    """Every supported domain passes through validation unchanged."""
    assert require_supported_domain(domain) == domain


@pytest.mark.parametrize(
    "domain",
    ["cell", "simplicial", "combinatorial", "pointcloud"],
)
def test_legacy_domains_are_rejected(domain):
    """Legacy domains are outside the closed public contract."""
    expected = (
        f"Unsupported domain {domain!r}; expected one of {SUPPORTED_DOMAINS}"
    )
    with pytest.raises(ValueError, match=re.escape(expected)):
        require_supported_domain(domain)


@pytest.mark.parametrize("domain", [None, 1, ("graph",)])
def test_non_string_domains_are_rejected(domain):
    """Domain validation rejects non-string inputs explicitly."""
    with pytest.raises(TypeError, match="domain must be a string"):
        require_supported_domain(domain)


def test_domain_module_exports_only_the_public_contract():
    """The domain module exposes no additional public symbols."""
    assert domain_exports == ["SUPPORTED_DOMAINS", "require_supported_domain"]


def test_neural_domain_directories_match_the_public_contract() -> None:
    """Backbone and wrapper packages exist only for supported domains."""
    assert _domain_directories(backbones.__file__) == SUPPORTED_DOMAINS
    assert _domain_directories(wrappers.__file__) == SUPPORTED_DOMAINS


def test_config_domain_directories_match_the_public_contract() -> None:
    """Dataset and model selectors exist only for supported domains."""
    assert _config_domain_directories("dataset") == SUPPORTED_DOMAINS
    assert _config_domain_directories("model") == SUPPORTED_DOMAINS


def test_graph_dataset_yaml_selectors_exactly_match_the_manifest() -> None:
    """The immutable graph manifest and surviving config selectors are equal."""
    yaml_selectors = {
        path.stem
        for path in (CONFIG_ROOT / "dataset" / "graph").glob("*.yaml")
    }

    assert yaml_selectors == set(GRAPH_DATASET_MANIFEST)


def test_surviving_dataset_configs_use_only_supported_task_kinds() -> None:
    """No dataset YAML can request a removed task contract."""
    tasks = {
        path.relative_to(CONFIG_ROOT).as_posix(): OmegaConf.select(
            OmegaConf.load(path),
            "parameters.task",
        )
        for domain in SUPPORTED_DOMAINS
        for path in (CONFIG_ROOT / "dataset" / domain).glob("*.yaml")
    }
    unsupported = {
        path: task
        for path, task in tasks.items()
        if task not in SUPPORTED_TASK_KINDS
    }

    assert tasks
    assert not unsupported


@pytest.mark.parametrize("selector", REMOVED_GRAPH_DATASETS)
def test_removed_graph_selectors_are_rejected_without_source_only_paths(
    selector: str,
) -> None:
    """Removed graph products have neither configs nor hidden source selectors."""
    assert selector not in GRAPH_DATASET_MANIFEST
    assert not (
        CONFIG_ROOT / "dataset" / "graph" / f"{selector}.yaml"
    ).exists()
    with pytest.raises(
        ValueError,
        match="does not match an exact graph manifest selector",
    ):
        qualify_graph_dataset(
            {
                "loader": {
                    "parameters": {
                        "data_domain": "graph",
                        "data_name": selector,
                    }
                }
            }
        )

    selector_pattern = _selector_pattern(selector)
    assert not any(
        selector_pattern.search(source_path.read_text(encoding="utf-8"))
        for source_path in PRODUCTION_ROOT.rglob("*.py")
    )
    assert not any(path.exists() for path in REMOVED_SOURCE_ONLY_PATHS)


def test_production_python_and_surviving_yaml_have_no_forbidden_tokens() -> (
    None
):
    """Runtime source and live YAML cannot retain removed topology contracts."""
    source_paths = (
        *PRODUCTION_ROOT.rglob("*.py"),
        *_surviving_yaml_paths(),
    )
    violations = []
    for source_path in source_paths:
        source = source_path.read_text(encoding="utf-8")
        for match in FORBIDDEN_TOKEN_PATTERN.finditer(source):
            line = source.count("\n", 0, match.start()) + 1
            relative_path = source_path.relative_to(PROJECT_ROOT)
            violations.append(f"{relative_path}:{line}: {match.group(0)}")

    assert not violations, "\n".join(violations)


def test_neural_registries_are_exact_and_deterministic() -> None:
    """Every public neural registry contains only proven native classes."""
    _assert_registry(backbones.MODEL_CLASSES, EXPECTED_BACKBONES)
    _assert_registry(
        graph_backbones.BACKBONE_CLASSES,
        ("GCNDGM", "GPSEncoder", "GraphMLP", "NSDEncoder"),
    )
    _assert_registry(
        heterogeneous_backbones.BACKBONE_CLASSES,
        ("HGTBackbone", "HeteroSAGEBackbone"),
    )
    _assert_registry(
        hypergraph_backbones.BACKBONE_CLASSES,
        ("EDGNN", "HypergraphConvBackbone"),
    )
    _assert_registry(wrappers.WRAPPER_CLASSES, EXPECTED_WRAPPERS)
    _assert_registry(
        graph_wrappers.WRAPPER_CLASSES,
        ("GNNWrapper", "GraphMLPWrapper"),
    )
    _assert_registry(
        heterogeneous_wrappers.WRAPPER_CLASSES,
        ("HeterogeneousWrapper",),
    )
    _assert_registry(
        hypergraph_wrappers.WRAPPER_CLASSES,
        ("HypergraphWrapper",),
    )
    _assert_registry(encoders.FEATURE_ENCODERS, EXPECTED_ENCODERS)
    _assert_registry(readouts.READOUT_CLASSES, EXPECTED_READOUTS)


def test_neural_registries_have_narrow_public_exports() -> None:
    """Registry modules export no discovery or compatibility symbols."""
    assert backbones.__all__ == [*EXPECTED_BACKBONES, "MODEL_CLASSES"]
    assert wrappers.__all__ == [*EXPECTED_WRAPPERS, "WRAPPER_CLASSES"]
    assert encoders.__all__ == [*EXPECTED_ENCODERS, "FEATURE_ENCODERS"]
    assert readouts.__all__ == [*EXPECTED_READOUTS, "READOUT_CLASSES"]


def test_surviving_neural_components_have_no_rank_based_bases() -> None:
    """Native components do not import removed topology bases."""
    forbidden_tokens = {
        "AbstractZeroCellReadOut",
        "AbstractWrapper",
        "AllCellFeatureEncoder",
        "BaseEncoder",
        "FlatEncoder",
        "HOPSEFeatureEncoder",
        "HOPSEReadout",
        "PropagateSignalDown",
        "all_cell_encoder",
        "flat_encoder",
        "hopse_encoder",
        "propagate_signal_down",
    }
    package_paths = (
        Path(encoders.__file__).parent,
        Path(readouts.__file__).parent,
        Path(wrappers.__file__).parent,
    )

    for package_path in package_paths:
        for source_path in package_path.glob("*.py"):
            source = source_path.read_text(encoding="utf-8")
            assert not any(token in source for token in forbidden_tokens), (
                source_path
            )
