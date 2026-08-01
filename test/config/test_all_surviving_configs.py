"""Exhaustive composition contract for every supported configuration selector."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from typing import Any

import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from topobench.data.capabilities import GRAPH_DATASET_MANIFEST
from topobench.data.heterogeneous import HeterogeneousDataSpec
from topobench.data.qualification import (
    DATASET_QUALIFICATION_MANIFEST,
    DatasetQualification,
)
from topobench.nn.capabilities import (
    GRAPH_MODEL_CAPABILITIES,
    compatible_graph_models,
)
from topobench.run import validate_domain_composition
from topobench.utils.config_resolvers import register_all_resolvers
from topobench.utils.model_instantiation import instantiate_model

PROJECT_ROOT = Path(__file__).parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"
SUPPORTED_DOMAINS = ("graph", "heterogeneous", "hypergraph")
PACKAGED_DATASETS = frozenset(
    {
        "graph/SyntheticGraph",
        "graph/SyntheticGraphRegression",
        "graph/SyntheticNodeGraph",
        "heterogeneous/SyntheticHeterogeneous",
        "hypergraph/SyntheticHypergraph",
    }
)
EXPECTED_MODELS: Mapping[str, frozenset[str]] = {
    "graph": frozenset(
        {"gat", "gcn", "gcn_dgm", "gin", "gps", "graph_mlp", "nsd"}
    ),
    "heterogeneous": frozenset({"heterosage", "hgt"}),
    "hypergraph": frozenset({"edgnn", "hypergraph_conv"}),
}
EXPECTED_HETEROGENEOUS_TRANSFORMS: Mapping[str, str] = {
    "heterogeneous/DBLP": "dataset_defaults/DBLP",
    "heterogeneous/OGB_MAG": "dataset_defaults/OGB_MAG",
    "heterogeneous/SyntheticHeterogeneous": (
        "dataset_defaults/SyntheticHeterogeneous"
    ),
}
EXPECTED_EXPERIMENTS: Mapping[str, tuple[str, str, str]] = {
    "example": ("graph/SyntheticGraph", "graph/gcn", "default"),
    "graph_synthetic_regression": (
        "graph/SyntheticGraphRegression",
        "graph/gcn",
        "default",
    ),
    "hypergraph_synthetic_edgnn": (
        "hypergraph/SyntheticHypergraph",
        "hypergraph/edgnn",
        "hypergraph_node",
    ),
    "hypergraph_synthetic_hypergraph_conv": (
        "hypergraph/SyntheticHypergraph",
        "hypergraph/hypergraph_conv",
        "hypergraph_node",
    ),
    "heterogeneous_synthetic_heterosage_full": (
        "heterogeneous/SyntheticHeterogeneous",
        "heterogeneous/heterosage",
        "heterogeneous_node",
    ),
    "heterogeneous_synthetic_heterosage_neighbor": (
        "heterogeneous/SyntheticHeterogeneous",
        "heterogeneous/heterosage",
        "heterogeneous_node",
    ),
    "heterogeneous_synthetic_hgt_full": (
        "heterogeneous/SyntheticHeterogeneous",
        "heterogeneous/hgt",
        "heterogeneous_node",
    ),
    "heterogeneous_synthetic_hgt_neighbor": (
        "heterogeneous/SyntheticHeterogeneous",
        "heterogeneous/hgt",
        "heterogeneous_node",
    ),
    "heterogeneous_dblp_heterosage": (
        "heterogeneous/DBLP",
        "heterogeneous/heterosage",
        "heterogeneous_node",
    ),
    "heterogeneous_dblp_hgt": (
        "heterogeneous/DBLP",
        "heterogeneous/hgt",
        "heterogeneous_node",
    ),
    "heterogeneous_ogb_mag_heterosage": (
        "heterogeneous/OGB_MAG",
        "heterogeneous/heterosage",
        "heterogeneous_node",
    ),
    "heterogeneous_ogb_mag_hgt": (
        "heterogeneous/OGB_MAG",
        "heterogeneous/hgt",
        "heterogeneous_node",
    ),
}
NON_GRAPH_MODEL_ROWS: Mapping[str, tuple[str, ...]] = {
    "heterogeneous/SyntheticHeterogeneous": (
        "heterogeneous/heterosage",
        "heterogeneous/hgt",
    ),
    "heterogeneous/DBLP": (
        "heterogeneous/heterosage",
        "heterogeneous/hgt",
    ),
    "heterogeneous/OGB_MAG": (
        "heterogeneous/heterosage",
        "heterogeneous/hgt",
    ),
    "hypergraph/SyntheticHypergraph": (
        "hypergraph/edgnn",
        "hypergraph/hypergraph_conv",
    ),
    "hypergraph/20newsgroup": (
        "hypergraph/edgnn",
        "hypergraph/hypergraph_conv",
    ),
    "hypergraph/ModelNet40": (
        "hypergraph/edgnn",
        "hypergraph/hypergraph_conv",
    ),
    "hypergraph/Mushroom": (
        "hypergraph/edgnn",
        "hypergraph/hypergraph_conv",
    ),
    "hypergraph/NTU2012": (
        "hypergraph/edgnn",
        "hypergraph/hypergraph_conv",
    ),
    "hypergraph/coauthorship_cora": (
        "hypergraph/edgnn",
        "hypergraph/hypergraph_conv",
    ),
    "hypergraph/coauthorship_dblp": (
        "hypergraph/edgnn",
        "hypergraph/hypergraph_conv",
    ),
    "hypergraph/cocitation_citeseer": (
        "hypergraph/edgnn",
        "hypergraph/hypergraph_conv",
    ),
    "hypergraph/cocitation_cora": (
        "hypergraph/edgnn",
        "hypergraph/hypergraph_conv",
    ),
    "hypergraph/cocitation_pubmed": (
        "hypergraph/edgnn",
        "hypergraph/hypergraph_conv",
    ),
    "hypergraph/zoo": (
        "hypergraph/edgnn",
        "hypergraph/hypergraph_conv",
    ),
}
REQUIRED_QUALIFICATION_FIELDS = frozenset(
    {
        "selector",
        "loader_family",
        "gate",
        "task",
        "task_level",
        "split_mode",
        "feature_policy",
        "edge_policy",
        "compatible_model",
        "evidence_test",
    }
)


def _graph_pairs() -> tuple[tuple[str, str], ...]:
    return tuple(
        (f"graph/{dataset.selector}", f"graph/{model.selector}")
        for dataset in GRAPH_DATASET_MANIFEST.values()
        for model in compatible_graph_models(dataset)
    )


def _non_graph_pairs() -> tuple[tuple[str, str], ...]:
    return tuple(
        (dataset_selector, model_selector)
        for dataset_selector, model_selectors in NON_GRAPH_MODEL_ROWS.items()
        for model_selector in model_selectors
    )


VALID_PAIRS = tuple(sorted((*_graph_pairs(), *_non_graph_pairs())))
VALID_PAIR_SET = frozenset(VALID_PAIRS)
ALL_MODEL_SELECTORS = frozenset(
    f"{domain}/{selector}"
    for domain, selectors in EXPECTED_MODELS.items()
    for selector in selectors
)
CROSS_DOMAIN_PAIRS = tuple(
    sorted(
        (dataset, model)
        for dataset in DATASET_QUALIFICATION_MANIFEST
        for model in ALL_MODEL_SELECTORS
        if dataset.split("/", maxsplit=1)[0] != model.split("/", maxsplit=1)[0]
    )
)


def _dataset_param(
    *values: Any,
    dataset_selector: str,
    id: str,
) -> Any:
    qualification = DATASET_QUALIFICATION_MANIFEST[dataset_selector]
    if qualification.gate == "packaged":
        return pytest.param(*values, id=id)
    return pytest.param(
        *values,
        id=id,
        marks=(pytest.mark.integration, pytest.mark.download),
    )


def _compose(*overrides: str) -> DictConfig:
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    register_all_resolvers()
    with hydra.initialize_config_dir(
        version_base="1.3",
        config_dir=str(CONFIG_ROOT),
        job_name="all_surviving_configs",
    ):
        cfg = hydra.compose(
            config_name="run.yaml",
            overrides=list(overrides),
            return_hydra_config=True,
        )
    HydraConfig.instance().set_config(cfg)
    return cfg


def _compose_pair(dataset_selector: str, model_selector: str) -> DictConfig:
    domain = dataset_selector.split("/", maxsplit=1)[0]
    pipeline_selector = {
        "graph": "default",
        "heterogeneous": "heterogeneous_node",
        "hypergraph": "hypergraph_node",
    }[domain]
    return _compose(
        f"dataset={dataset_selector}",
        f"model={model_selector}",
        f"data_pipeline={pipeline_selector}",
        "transforms=no_transform",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
    )


def _fully_resolve(cfg: DictConfig) -> dict[str, Any]:
    application_cfg = deepcopy(cfg)
    del application_cfg.hydra
    resolved = OmegaConf.to_container(
        application_cfg,
        resolve=True,
        throw_on_missing=True,
    )
    assert isinstance(resolved, dict)
    return resolved


def _pytest_node_prefixes(path: Path) -> frozenset[str]:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    prefixes: set[str] = set()
    relative_path = path.relative_to(PROJECT_ROOT).as_posix()
    for node in module.body:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and node.name.startswith("test_"):
            prefixes.add(f"{relative_path}::{node.name}")
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and child.name.startswith("test_"):
                    prefixes.add(f"{relative_path}::{node.name}::{child.name}")
    return frozenset(prefixes)


def _discover_selectors(group: str) -> frozenset[str]:
    return frozenset(
        f"{domain}/{path.stem}"
        for domain in SUPPORTED_DOMAINS
        for path in (CONFIG_ROOT / group / domain).glob("*.yaml")
    )


def test_default_run_is_explicit_native_graph_and_network_free() -> None:
    cfg = _compose()
    choices = cfg.hydra.runtime.choices

    assert choices.dataset == "graph/SyntheticGraph"
    assert choices.model == "graph/gcn"
    assert choices.data_pipeline == "default"
    assert choices.transforms == "no_transform"
    assert choices.logger == "csv"
    assert choices.trainer == "cpu"
    assert cfg.train is True
    assert cfg.test is True
    _fully_resolve(cfg)


def test_dataset_manifest_exactly_matches_supported_yaml_selectors() -> None:
    discovered = _discover_selectors("dataset")

    assert discovered == frozenset(DATASET_QUALIFICATION_MANIFEST)
    assert frozenset(DATASET_QUALIFICATION_MANIFEST) == (
        frozenset(f"graph/{selector}" for selector in GRAPH_DATASET_MANIFEST)
        | frozenset(NON_GRAPH_MODEL_ROWS)
    )


def test_model_directories_are_exact_and_match_capabilities() -> None:
    for domain, expected in EXPECTED_MODELS.items():
        discovered = frozenset(
            path.stem
            for path in (CONFIG_ROOT / "model" / domain).glob("*.yaml")
        )
        assert discovered == expected, domain
    assert frozenset(GRAPH_MODEL_CAPABILITIES) == EXPECTED_MODELS["graph"]

    assert _discover_selectors("model") == ALL_MODEL_SELECTORS


def test_declared_pair_matrix_is_total_and_exact() -> None:
    assert len(_graph_pairs()) == 155
    assert len(_non_graph_pairs()) == 28
    assert len(VALID_PAIRS) == 183
    assert len(VALID_PAIR_SET) == len(VALID_PAIRS)
    assert {dataset for dataset, _ in VALID_PAIRS} == set(
        DATASET_QUALIFICATION_MANIFEST
    )
    assert len(CROSS_DOMAIN_PAIRS) == 242


@pytest.mark.parametrize(
    "selector",
    [
        _dataset_param(selector, dataset_selector=selector, id=selector)
        for selector in sorted(DATASET_QUALIFICATION_MANIFEST)
    ],
)
def test_every_dataset_has_complete_resolvable_qualification(
    selector: str,
) -> None:
    qualification = DATASET_QUALIFICATION_MANIFEST[selector]
    qualification_fields = {
        field.name for field in fields(DatasetQualification)
    }

    assert qualification_fields == REQUIRED_QUALIFICATION_FIELDS
    assert qualification.selector == selector
    assert qualification.gate in {"packaged", "download"}
    assert (qualification.gate == "packaged") == (
        selector in PACKAGED_DATASETS
    )
    for field_name in REQUIRED_QUALIFICATION_FIELDS - {"gate"}:
        assert getattr(qualification, field_name), f"{selector}: {field_name}"

    model_selector = qualification.compatible_model
    assert model_selector in ALL_MODEL_SELECTORS
    assert (
        model_selector.split("/", maxsplit=1)[0]
        == selector.split("/", maxsplit=1)[0]
    )
    if selector.startswith("graph/"):
        dataset = GRAPH_DATASET_MANIFEST[selector.split("/", maxsplit=1)[1]]
        compatible = {
            f"graph/{model.selector}"
            for model in compatible_graph_models(dataset)
        }
        assert model_selector in compatible
    else:
        assert model_selector in NON_GRAPH_MODEL_ROWS[selector]

    evidence_path_text, separator, _ = qualification.evidence_test.partition(
        "::"
    )
    assert separator, selector
    evidence_path = PROJECT_ROOT / evidence_path_text
    assert evidence_path.is_file(), qualification.evidence_test
    assert any(
        prefix.startswith(qualification.evidence_test)
        for prefix in _pytest_node_prefixes(evidence_path)
    ), qualification.evidence_test


def _heterogeneous_spec(cfg: DictConfig) -> HeterogeneousDataSpec:
    target_node_type = str(cfg.dataset.parameters.target_node_type)
    context_node_type = "paper" if target_node_type != "paper" else "author"
    return HeterogeneousDataSpec(
        node_types=(target_node_type, context_node_type),
        edge_types=((context_node_type, "relates_to", target_node_type),),
        target_node_type=target_node_type,
        num_classes=int(cfg.dataset.parameters.num_classes),
        input_channels=((target_node_type, 4), (context_node_type, 3)),
    )


@pytest.mark.parametrize(
    ("dataset_selector", "model_selector"),
    [
        _dataset_param(
            dataset_selector,
            model_selector,
            dataset_selector=dataset_selector,
            id=f"{dataset_selector}--{model_selector}",
        )
        for dataset_selector, model_selector in VALID_PAIRS
    ],
)
def test_every_declared_pair_composes_resolves_validates_and_builds_models(
    dataset_selector: str,
    model_selector: str,
) -> None:
    cfg = _compose_pair(dataset_selector, model_selector)

    validate_domain_composition(cfg)
    _fully_resolve(cfg)
    data_spec = (
        _heterogeneous_spec(cfg)
        if dataset_selector.startswith("heterogeneous/")
        else None
    )
    model = instantiate_model(cfg, data_spec=data_spec)
    assert model is not None


@pytest.mark.parametrize(
    "dataset_selector",
    [
        pytest.param(selector, id=selector)
        for selector in sorted(PACKAGED_DATASETS)
    ],
)
def test_every_packaged_dataset_loader_constructs_without_network(
    dataset_selector: str,
    tmp_path: Path,
) -> None:
    model_selector = DATASET_QUALIFICATION_MANIFEST[
        dataset_selector
    ].compatible_model
    cfg = _compose_pair(dataset_selector, model_selector)
    with open_dict(cfg.paths):
        cfg.paths.data_dir = str(tmp_path / "data")

    loader = hydra.utils.instantiate(cfg.dataset.loader)
    assert loader is not None


def test_undeclared_same_domain_graph_pair_is_rejected_with_paths() -> None:
    rejected_pair = ("graph/SyntheticGraph", "graph/graph_mlp")
    assert rejected_pair not in VALID_PAIR_SET
    cfg = _compose_pair(*rejected_pair)

    with pytest.raises(ValueError) as error:
        validate_domain_composition(cfg)

    message = str(error.value)
    assert "dataset.parameters.task" in message
    assert "model.model_name" in message


def test_mutated_unsupported_task_contract_is_rejected_with_paths() -> None:
    cfg = _compose_pair("graph/SyntheticGraph", "graph/gcn")
    with open_dict(cfg.dataset.parameters):
        cfg.dataset.parameters.task = "regression"

    with pytest.raises(ValueError) as error:
        validate_domain_composition(cfg)

    message = str(error.value)
    assert "dataset.parameters.task" in message


@pytest.mark.parametrize(
    ("dataset_selector", "model_selector"),
    [
        _dataset_param(
            dataset_selector,
            model_selector,
            dataset_selector=dataset_selector,
            id=f"{dataset_selector}--{model_selector}",
        )
        for dataset_selector, model_selector in CROSS_DOMAIN_PAIRS
    ],
)
def test_every_cross_domain_pair_is_rejected_by_the_central_gate(
    dataset_selector: str,
    model_selector: str,
) -> None:
    cfg = _compose_pair(dataset_selector, model_selector)

    with pytest.raises(ValueError) as error:
        validate_domain_composition(cfg)

    message = str(error.value)
    assert "cfg.dataset.loader.parameters.data_domain" in message
    assert "cfg.model.model_domain" in message
    assert "cross-domain" in message.lower()


def _experiment_parameters() -> Iterable[Any]:
    for experiment, (dataset, _, _) in EXPECTED_EXPERIMENTS.items():
        yield _dataset_param(
            experiment,
            dataset_selector=dataset,
            id=experiment,
        )


def test_experiment_directory_is_exact() -> None:
    discovered = frozenset(
        path.stem for path in (CONFIG_ROOT / "experiment").glob("*.yaml")
    )
    assert discovered == frozenset(EXPECTED_EXPERIMENTS)


@pytest.mark.parametrize("experiment", list(_experiment_parameters()))
def test_every_experiment_composes_resolves_and_references_existing_selectors(
    experiment: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_DISABLED", "true")
    cfg = _compose(f"experiment={experiment}")
    choices = cfg.hydra.runtime.choices
    expected_dataset, expected_model, expected_pipeline = EXPECTED_EXPERIMENTS[
        experiment
    ]

    assert choices.dataset == expected_dataset
    assert choices.model == expected_model
    assert choices.data_pipeline == expected_pipeline
    if expected_dataset.startswith("heterogeneous/"):
        assert (
            choices.transforms
            == EXPECTED_HETEROGENEOUS_TRANSFORMS[expected_dataset]
        )
        assert len(cfg.transforms) > 0
    else:
        assert choices.transforms == "no_transform"
    expected_logger = (
        "heterogeneous_wandb"
        if expected_dataset.startswith("heterogeneous/")
        else "csv"
    )
    assert choices.logger == expected_logger
    assert choices.dataset in DATASET_QUALIFICATION_MANIFEST
    assert choices.model in ALL_MODEL_SELECTORS
    validate_domain_composition(cfg)
    _fully_resolve(cfg)
