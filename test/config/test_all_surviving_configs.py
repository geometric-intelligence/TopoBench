"""Exhaustive composition contract for every supported configuration selector."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from topobench.data.capabilities import (
    GRAPH_DATASET_MANIFEST,
    RuntimeDataCapability,
    qualify_dataset,
    qualify_graph_dataset,
)
from topobench.data.heterogeneous import HeterogeneousDataSpec
from topobench.data.qualification import (
    DATASET_QUALIFICATION_MANIFEST,
    DatasetQualification,
)
from topobench.nn.capabilities import (
    GRAPH_MODEL_CAPABILITIES,
    MODEL_CAPABILITY_MANIFEST,
    compatible_graph_models,
    validate_capability_composition,
)
from topobench.run import validate_domain_composition
from topobench.utils.config_resolvers import register_all_resolvers
from topobench.utils.instantiators import validate_execution_profile
from topobench.utils.model_instantiation import instantiate_model

PROJECT_ROOT = Path(__file__).parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"
SUPPORTED_DOMAINS = ("graph", "heterogeneous", "hypergraph")
EXPECTED_PIPELINE_TARGETS: Mapping[str, str] = {
    "graph": "topobench.data.pipelines.DefaultDataPipeline",
    "heterogeneous": (
        "topobench.data.pipelines.HeterogeneousNodeDataPipeline"
    ),
    "hypergraph": "topobench.data.pipelines.HypergraphNodeDataPipeline",
}
PACKAGED_DATASETS = frozenset(
    {
        "graph/SyntheticGraph",
        "graph/ParquetTypedGraph",
        "graph/SyntheticGraphRegression",
        "graph/SyntheticNodeGraph",
        "heterogeneous/ParquetTypedGraph",
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
    "heterogeneous/ParquetTypedGraph": (
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
}
EXPECTED_NON_GRAPH_PAIRS = (
    ("heterogeneous/DBLP", "heterogeneous/heterosage"),
    ("heterogeneous/DBLP", "heterogeneous/hgt"),
    ("heterogeneous/OGB_MAG", "heterogeneous/heterosage"),
    ("heterogeneous/OGB_MAG", "heterogeneous/hgt"),
    (
        "heterogeneous/ParquetTypedGraph",
        "heterogeneous/heterosage",
    ),
    ("heterogeneous/ParquetTypedGraph", "heterogeneous/hgt"),
    (
        "heterogeneous/SyntheticHeterogeneous",
        "heterogeneous/heterosage",
    ),
    ("heterogeneous/SyntheticHeterogeneous", "heterogeneous/hgt"),
    ("hypergraph/SyntheticHypergraph", "hypergraph/edgnn"),
    (
        "hypergraph/SyntheticHypergraph",
        "hypergraph/hypergraph_conv",
    ),
)
REQUIRED_QUALIFICATION_FIELDS = frozenset(
    {
        "selector",
        "loader_family",
        "gate",
        "task",
        "task_level",
        "split_mode",
        "split_type",
        "feature_policy",
        "num_classes",
        "target_node_type",
        "feature_widths",
        "edge_policy",
        "compatible_model",
        "evidence_test",
    }
)
EXPECTED_EVALUATOR_POLICY: Mapping[str, str] = {
    "train": "online",
    "val": "exact",
    "test": "exact",
}
EXPECTED_ONLINE_RESOURCES: Mapping[str, int] = {
    "ranking_thresholds": 512,
}
EXPECTED_EXACT_RESOURCES: Mapping[str, int | str] = {
    "max_ranking_bytes": 536870912,
    "buffer_device": "cpu",
}
REMOVED_EVALUATOR_FIELDS = frozenset(
    {"multioutput_classes", "auroc_thresholds", "max_auroc_bytes"}
)


def _is_runnable_graph_dataset(dataset: Any) -> bool:
    if not dataset.descriptor_only:
        return True
    source = OmegaConf.load(
        CONFIG_ROOT / "dataset" / "graph" / f"{dataset.selector}.yaml"
    )
    output_kind = source.loader.parameters.output_kind
    strategy = source.loader.parameters.partition.strategy
    backend = source.loader.parameters.partition.backend
    return output_kind == "homogeneous" and dataset.supports_source(
        domain="graph",
        output_kind=output_kind,
        strategy=strategy,
        backend=backend,
    )


def _graph_pairs() -> tuple[tuple[str, str], ...]:
    return tuple(
        (f"graph/{dataset.selector}", f"graph/{model.selector}")
        for dataset in GRAPH_DATASET_MANIFEST.values()
        if _is_runnable_graph_dataset(dataset)
        for model in compatible_graph_models(dataset)
    )


def _descriptor_only_dataset_selectors() -> frozenset[str]:
    return frozenset(
        f"{domain}/{capability.selector}"
        for capability in GRAPH_DATASET_MANIFEST.values()
        if capability.descriptor_only
        for domain, _, _, _ in capability.source_capabilities
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
        "trainer.accelerator=cpu",
        "trainer.devices=1",
    )


@pytest.mark.parametrize(
    ("dataset_selector", "model_selector"),
    [
        ("graph/SyntheticGraph", "graph/gcn"),
        (
            "heterogeneous/SyntheticHeterogeneous",
            "heterogeneous/hgt",
        ),
        ("hypergraph/SyntheticHypergraph", "hypergraph/edgnn"),
    ],
)
def test_packaged_compositions_have_a_qualified_execution_profile(
    dataset_selector: str,
    model_selector: str,
) -> None:
    """Every packaged domain composes only its exact executable targets."""
    cfg = _compose_pair(dataset_selector, model_selector)

    record = validate_execution_profile(cfg)

    assert cfg.execution_profile == "qualified"
    assert record.profile == "qualified"
    assert record.qualified is True
    assert record.custom_targets == ()
    assert record.targets


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


def _nested_mapping_keys(value: Any) -> frozenset[str]:
    if isinstance(value, Mapping):
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


def _assert_authoritative_evaluator_config(cfg: DictConfig) -> None:
    evaluator = OmegaConf.to_container(
        cfg.evaluator,
        resolve=True,
        throw_on_missing=True,
    )
    assert isinstance(evaluator, dict)
    assert evaluator.get("policy") == EXPECTED_EVALUATOR_POLICY
    assert evaluator.get("online") == EXPECTED_ONLINE_RESOURCES
    assert evaluator.get("exact") == EXPECTED_EXACT_RESOURCES
    assert evaluator.get("undefined_metric_policy") == "error"
    assert REMOVED_EVALUATOR_FIELDS.isdisjoint(_nested_mapping_keys(evaluator))
    assert cfg.preflight.enabled is True


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
    assert "input_pipeline" not in cfg.callbacks
    _fully_resolve(cfg)


def test_dataset_manifest_exactly_matches_supported_yaml_selectors() -> None:
    discovered = _discover_selectors("dataset")
    descriptor_selectors = _descriptor_only_dataset_selectors()
    runnable_graph_selectors = frozenset(
        f"graph/{selector}"
        for selector, capability in GRAPH_DATASET_MANIFEST.items()
        if _is_runnable_graph_dataset(capability)
    )

    assert discovered == (
        frozenset(DATASET_QUALIFICATION_MANIFEST) | descriptor_selectors
    )
    assert frozenset(DATASET_QUALIFICATION_MANIFEST) == (
        runnable_graph_selectors | frozenset(NON_GRAPH_MODEL_ROWS)
    )


@pytest.mark.parametrize(
    ("domain", "model", "pipeline", "output_kind", "strategy"),
    [
        ("graph", "graph/gcn", "default", "homogeneous", "cluster"),
        (
            "heterogeneous",
            "heterogeneous/hgt",
            "heterogeneous_node",
            "heterogeneous",
            "neighbor",
        ),
    ],
)
def test_typed_parquet_selectors_compose_through_the_shared_capability(
    domain: str,
    model: str,
    pipeline: str,
    output_kind: str,
    strategy: str,
) -> None:
    capability = GRAPH_DATASET_MANIFEST["ParquetTypedGraph"]
    selector = f"{domain}/ParquetTypedGraph"
    cfg = _compose(
        f"dataset={selector}",
        f"model={model}",
        f"data_pipeline={pipeline}",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
    )
    loader_parameters = OmegaConf.to_container(
        cfg.dataset.loader.parameters,
        resolve=True,
        throw_on_missing=True,
    )

    assert cfg.dataset.loader._target_ == (
        "topobench.data.loaders.parquet.ParquetTypedGraphLoader"
    )
    assert cfg.data_pipeline._target_ == EXPECTED_PIPELINE_TARGETS[domain]
    assert cfg.dataset.loader.parameters.output_kind == output_kind
    assert cfg.dataset.loader.parameters.partition.strategy == strategy
    assert capability.supports_source(
        domain=domain,
        output_kind=output_kind,
        strategy=strategy,
        backend=str(cfg.dataset.loader.parameters.partition.backend),
    )
    loader = hydra.utils.instantiate(cfg.dataset.loader)
    assert type(loader).__module__ == "topobench.data.loaders.parquet"
    if domain == "graph":
        assert qualify_graph_dataset(cfg.dataset) is capability

    assert "sql" not in loader_parameters
    assert "python" not in loader_parameters


def test_packaged_parquet_configs_compose_one_qualified_runtime_contract() -> (
    None
):
    graph = _compose_pair("graph/ParquetTypedGraph", "graph/gcn")
    heterogeneous = _compose_pair(
        "heterogeneous/ParquetTypedGraph",
        "heterogeneous/hgt",
    )

    for cfg in (graph, heterogeneous):
        assert cfg.data_pipeline.parquet_store_root
        assert cfg.data_pipeline.parquet_store_path is None
        assert cfg.data_pipeline.active_split_tag is None
        assert cfg.data_pipeline.qualified_profile is True
        assert cfg.data_pipeline.execution_monitor is None
        assert cfg.dataset.loader.parameters.profiling.enabled is True
        assert (
            cfg.dataset.loader.parameters.reproducibility.save_reproducibility_bundle
            is True
        )
        assert cfg.callbacks.input_pipeline._target_ == (
            "topobench.callbacks.input_pipeline.InputPipelineCallback"
        )
        assert cfg.callbacks.dataloader_commit._target_ == (
            "topobench.callbacks.dataloader_commit.DataloaderCommitCallback"
        )

    assert graph.dataset.dataloader_params.clusters_per_batch == 1
    assert graph.dataset.dataloader_params.train_shuffle is True
    assert heterogeneous.dataset.dataloader_params.mode == "neighbor"
    assert heterogeneous.dataset.dataloader_params.batch_size > 0
    assert list(heterogeneous.dataset.dataloader_params.num_neighbors) == [
        -1,
        -1,
    ]
    assert (
        len(heterogeneous.dataset.dataloader_params.num_neighbors)
        == heterogeneous.model.backbone.num_layers
    )


def test_graph_configs_do_not_advertise_unqualified_kfold() -> None:
    """Surviving graph selectors expose only qualified split parameters."""
    for config_path in sorted(
        (CONFIG_ROOT / "dataset" / "graph").glob("*.yaml")
    ):
        config = OmegaConf.load(config_path)
        split_params = config.split_params
        assert split_params.split_type not in {"k-fold", "k-fold-fixed"}, (
            config_path
        )
        assert "k" not in split_params, config_path
        assert (
            "k-fold" not in config_path.read_text(encoding="utf-8").lower()
        ), config_path


def test_model_directories_are_exact_and_match_capabilities() -> None:
    for domain, expected in EXPECTED_MODELS.items():
        discovered = frozenset(
            path.stem
            for path in (CONFIG_ROOT / "model" / domain).glob("*.yaml")
        )
        assert discovered == expected, domain
    assert frozenset(GRAPH_MODEL_CAPABILITIES) == EXPECTED_MODELS["graph"]

    assert _discover_selectors("model") == ALL_MODEL_SELECTORS


def test_model_capability_records_match_every_exact_component_target() -> None:
    assert frozenset(MODEL_CAPABILITY_MANIFEST) == ALL_MODEL_SELECTORS
    for selector, capability in MODEL_CAPABILITY_MANIFEST.items():
        domain, model_name = selector.split("/", maxsplit=1)
        model_cfg = OmegaConf.load(
            CONFIG_ROOT / "model" / domain / f"{model_name}.yaml"
        )

        assert capability.pipeline_target == EXPECTED_PIPELINE_TARGETS[domain]
        assert capability.model_target == "topobench.model.TBModel"
        assert model_cfg._target_ == capability.model_target
        assert (
            model_cfg.feature_encoder._target_
            == capability.feature_encoder_target
        )
        assert model_cfg.backbone._target_ == capability.backbone_target
        assert model_cfg.backbone_wrapper._target_ == capability.wrapper_target
        assert model_cfg.readout._target_ == capability.readout_target
        if capability.supervision_adapter_target is None:
            assert "supervision_adapter" not in model_cfg
        else:
            assert (
                model_cfg.supervision_adapter._target_
                == capability.supervision_adapter_target
            )


def test_declared_pair_matrix_is_total_and_exact() -> None:
    assert len(_graph_pairs()) == 155
    assert tuple(sorted(_non_graph_pairs())) == EXPECTED_NON_GRAPH_PAIRS
    assert len(VALID_PAIRS) == len(_graph_pairs()) + len(
        EXPECTED_NON_GRAPH_PAIRS
    )
    assert len(VALID_PAIR_SET) == len(VALID_PAIRS)
    assert {dataset for dataset, _ in VALID_PAIRS} == set(
        DATASET_QUALIFICATION_MANIFEST
    )
    expected_cross_domain_count = sum(
        sum(
            len(model_selectors)
            for model_domain, model_selectors in EXPECTED_MODELS.items()
            if model_domain != dataset_selector.partition("/")[0]
        )
        for dataset_selector in DATASET_QUALIFICATION_MANIFEST
    )
    assert len(CROSS_DOMAIN_PAIRS) == expected_cross_domain_count


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
    for field_name in REQUIRED_QUALIFICATION_FIELDS - {
        "gate",
        "num_classes",
        "target_node_type",
    }:
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
    qualification = qualify_dataset(cfg.dataset)
    assert qualification.target_node_type is not None
    assert qualification.num_classes is not None
    node_types = tuple(
        node_type for node_type, _ in qualification.feature_widths
    )
    edge_types = tuple(
        (node_type, "relates_to", qualification.target_node_type)
        for node_type in node_types
        if node_type != qualification.target_node_type
    )
    return HeterogeneousDataSpec(
        node_types=node_types,
        edge_types=edge_types,
        target_node_type=qualification.target_node_type,
        num_classes=qualification.num_classes,
        input_channels=qualification.feature_widths,
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
    _assert_authoritative_evaluator_config(cfg)

    static_validation = validate_capability_composition(cfg)
    assert (
        validate_domain_composition(cfg) == dataset_selector.partition("/")[0]
    )
    assert (
        static_validation.dataset
        is DATASET_QUALIFICATION_MANIFEST[dataset_selector]
    )
    assert static_validation.model is MODEL_CAPABILITY_MANIFEST[model_selector]
    assert static_validation.observed is None
    _fully_resolve(cfg)
    data_spec = (
        _heterogeneous_spec(cfg)
        if dataset_selector.startswith("heterogeneous/")
        else None
    )
    qualification = static_validation.dataset
    observed = RuntimeDataCapability(
        selector=qualification.selector,
        data_domain=dataset_selector.partition("/")[0],
        output_kind=(
            "graph"
            if qualification.task_level == "graph"
            else static_validation.model.output_kind
        ),
        feature_widths=(
            (("node", cfg.model.feature_encoder.in_channels),)
            if dataset_selector.startswith("graph/")
            else qualification.feature_widths
        ),
        num_classes=(
            qualification.num_classes
            if qualification.task == "classification"
            else None
        ),
        target_node_type=qualification.target_node_type,
    )
    observed_validation = validate_capability_composition(
        cfg,
        observed=observed,
    )
    model = instantiate_model(
        cfg,
        data_spec=data_spec,
        capability_validation=observed_validation,
    )
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
def test_every_cross_domain_pair_is_rejected_during_default_composition(
    dataset_selector: str,
    model_selector: str,
) -> None:
    with pytest.raises(hydra.errors.ConfigCompositionException) as error:
        _compose_pair(dataset_selector, model_selector)

    messages: list[str] = []
    cause: BaseException | None = error.value
    while cause is not None:
        messages.append(str(cause))
        cause = cause.__cause__
    message = " ".join(messages)
    assert "get_default_transform" in message


@pytest.mark.parametrize(
    ("dataset_selector", "model_selector"),
    [
        (
            "heterogeneous/SyntheticHeterogeneous",
            "heterogeneous/heterosage",
        ),
        ("hypergraph/SyntheticHypergraph", "hypergraph/edgnn"),
    ],
)
def test_bare_non_graph_overrides_reject_the_default_graph_pipeline(
    dataset_selector: str,
    model_selector: str,
) -> None:
    cfg = _compose(
        f"dataset={dataset_selector}",
        f"model={model_selector}",
        "transforms=no_transform",
    )

    with pytest.raises(ValueError) as error:
        validate_domain_composition(cfg)

    domain = dataset_selector.split("/", maxsplit=1)[0]
    message = str(error.value)
    assert "cfg.data_pipeline._target_" in message
    assert EXPECTED_PIPELINE_TARGETS[domain] in message
    assert EXPECTED_PIPELINE_TARGETS["graph"] in message


def test_missing_pipeline_target_is_rejected_with_its_config_path() -> None:
    cfg = _compose_pair("graph/SyntheticGraph", "graph/gcn")
    with open_dict(cfg.data_pipeline):
        del cfg.data_pipeline._target_

    with pytest.raises(ValueError) as error:
        validate_domain_composition(cfg)

    assert "cfg.data_pipeline._target_ is required" in str(error.value)


def test_non_string_pipeline_target_is_rejected_with_its_config_path() -> None:
    cfg = _compose_pair("graph/SyntheticGraph", "graph/gcn")
    with open_dict(cfg.data_pipeline):
        cfg.data_pipeline._target_ = 42

    with pytest.raises(TypeError) as error:
        validate_domain_composition(cfg)

    assert "cfg.data_pipeline._target_ must be a string" in str(error.value)


def test_graph_dataset_rejects_a_non_graph_pipeline() -> None:
    cfg = _compose_pair("graph/SyntheticGraph", "graph/gcn")
    with open_dict(cfg.data_pipeline):
        cfg.data_pipeline._target_ = EXPECTED_PIPELINE_TARGETS["heterogeneous"]

    with pytest.raises(ValueError) as error:
        validate_domain_composition(cfg)

    message = str(error.value)
    assert "cfg.data_pipeline._target_" in message
    assert EXPECTED_PIPELINE_TARGETS["graph"] in message
    assert EXPECTED_PIPELINE_TARGETS["heterogeneous"] in message


_REPRESENTATIVE_COMPOSITIONS: Mapping[str, str] = {
    "graph/SyntheticGraph": "graph/gcn",
    "heterogeneous/SyntheticHeterogeneous": "heterogeneous/hgt",
    "hypergraph/SyntheticHypergraph": "hypergraph/edgnn",
}
_COMMON_COMPOSITION_MUTATIONS = (
    ("model.model_name", "not_qualified", "model.model_name"),
    ("model.model_domain", "other", "model.model_domain"),
    ("model._target_", "tests.UnqualifiedTarget", "model._target_"),
    (
        "model.feature_encoder._target_",
        "tests.UnqualifiedTarget",
        "model.feature_encoder._target_",
    ),
    (
        "model.backbone._target_",
        "tests.UnqualifiedTarget",
        "model.backbone._target_",
    ),
    (
        "model.backbone_wrapper._target_",
        "tests.UnqualifiedTarget",
        "model.backbone_wrapper._target_",
    ),
    (
        "model.readout._target_",
        "tests.UnqualifiedTarget",
        "model.readout._target_",
    ),
    (
        "data_pipeline._target_",
        "tests.UnqualifiedTarget",
        "cfg.data_pipeline._target_",
    ),
)
_COMPOSITION_MUTATIONS = tuple(
    (
        dataset_selector,
        model_selector,
        path,
        value,
        expected_path,
    )
    for dataset_selector, model_selector in _REPRESENTATIVE_COMPOSITIONS.items()
    for path, value, expected_path in (
        *_COMMON_COMPOSITION_MUTATIONS,
        *(
            (
                (
                    "model.supervision_adapter._target_",
                    "tests.UnqualifiedTarget",
                    "model.supervision_adapter._target_",
                ),
            )
            if dataset_selector.startswith("heterogeneous/")
            else ()
        ),
    )
)


@pytest.mark.parametrize(
    (
        "dataset_selector",
        "model_selector",
        "path",
        "value",
        "expected_path",
    ),
    [
        pytest.param(
            *mutation,
            id=f"{mutation[0]}-{mutation[2]}",
        )
        for mutation in _COMPOSITION_MUTATIONS
    ],
)
def test_each_composition_target_and_selector_mutation_is_rejected(
    dataset_selector: str,
    model_selector: str,
    path: str,
    value: str,
    expected_path: str,
) -> None:
    cfg = _compose_pair(dataset_selector, model_selector)
    with open_dict(cfg):
        OmegaConf.update(cfg, path, value, merge=False)

    with pytest.raises((TypeError, ValueError)) as error:
        validate_capability_composition(cfg)

    message = str(error.value)
    assert expected_path in message
    assert value in message
    if path.endswith("._target_"):
        capability = MODEL_CAPABILITY_MANIFEST[model_selector]
        expected_target = {
            "model._target_": capability.model_target,
            "model.feature_encoder._target_": (
                capability.feature_encoder_target
            ),
            "model.backbone._target_": capability.backbone_target,
            "model.backbone_wrapper._target_": capability.wrapper_target,
            "model.readout._target_": capability.readout_target,
            "model.supervision_adapter._target_": (
                capability.supervision_adapter_target
            ),
            "data_pipeline._target_": capability.pipeline_target,
        }[path]
        assert expected_target is not None
        assert expected_target in message


def _runtime_capability(cfg: DictConfig) -> RuntimeDataCapability:
    validation = validate_capability_composition(cfg)
    qualification = validation.dataset
    return RuntimeDataCapability(
        selector=qualification.selector,
        data_domain=validation.model.data_domain,
        output_kind=(
            "graph"
            if qualification.task_level == "graph"
            else validation.model.output_kind
        ),
        feature_widths=qualification.feature_widths,
        num_classes=(
            qualification.num_classes
            if qualification.task == "classification"
            else None
        ),
        target_node_type=qualification.target_node_type,
    )


_RUNTIME_MISMATCHES = {
    "graph/SyntheticGraph": (
        ("feature_widths", (("node", 5),)),
        ("num_classes", 3),
        ("target_node_type", "author"),
        ("selector", "graph/SyntheticGraphRegression"),
    ),
    "heterogeneous/SyntheticHeterogeneous": (
        (
            "feature_widths",
            (("author", 9), ("paper", 5), ("venue", 1)),
        ),
        ("num_classes", 3),
        ("target_node_type", "paper"),
        ("selector", "heterogeneous/DBLP"),
    ),
    "hypergraph/SyntheticHypergraph": (
        ("feature_widths", (("node", 5),)),
        ("num_classes", 3),
        ("target_node_type", "author"),
        (
            "selector",
            RuntimeDataCapability(
                selector="graph/SyntheticNodeGraph",
                data_domain="graph",
                output_kind="homogeneous",
                feature_widths=(("node", 4),),
                num_classes=2,
                target_node_type=None,
            ),
        ),
    ),
}


@pytest.mark.parametrize(
    ("dataset_selector", "field_name", "mismatched_value"),
    [
        pytest.param(
            dataset_selector,
            field_name,
            mismatched_value,
            id=f"{dataset_selector}-observed-{field_name}",
        )
        for dataset_selector, mismatches in _RUNTIME_MISMATCHES.items()
        for field_name, mismatched_value in mismatches
    ],
)
def test_runtime_observation_mismatches_are_rejected_with_exact_paths(
    dataset_selector: str,
    field_name: str,
    mismatched_value: object,
) -> None:
    model_selector = _REPRESENTATIVE_COMPOSITIONS[dataset_selector]
    cfg = _compose_pair(dataset_selector, model_selector)
    baseline = _runtime_capability(cfg)
    observed = (
        mismatched_value
        if isinstance(mismatched_value, RuntimeDataCapability)
        else replace(
            baseline,
            **{field_name: mismatched_value},
        )
    )

    with pytest.raises(ValueError) as error:
        validate_capability_composition(cfg, observed=observed)

    message = str(error.value)
    assert f"observed.{field_name}" in message


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
    _assert_authoritative_evaluator_config(cfg)
    choices = cfg.hydra.runtime.choices
    expected_dataset, expected_model, expected_pipeline = EXPECTED_EXPERIMENTS[
        experiment
    ]

    assert choices.dataset == expected_dataset
    assert choices.model == expected_model
    assert choices.data_pipeline == expected_pipeline
    expected_domain = expected_dataset.split("/", maxsplit=1)[0]
    assert (
        cfg.data_pipeline._target_
        == EXPECTED_PIPELINE_TARGETS[expected_domain]
    )
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
