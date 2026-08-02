"""Exact manifest gates for surviving native graph datasets."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from topobench.data.capabilities import (
    GRAPH_DATASET_MANIFEST,
    GraphDatasetCapability,
    qualify_graph_dataset,
)
from topobench.data.qualification import DATASET_QUALIFICATION_MANIFEST
from topobench.nn.capabilities import compatible_graph_models

GRAPH_CONFIG_DIR = Path("configs/dataset/graph")
EXPECTED_SELECTORS = frozenset(
    {
        "AQSOL",
        "BBB_Martins",
        "CYP3A4_Veith",
        "Caco2_Wang",
        "Clearance_Hepatocyte_AZ",
        "IMDB-BINARY",
        "IMDB-MULTI",
        "MUTAG",
        "NCI1",
        "NCI109",
        "PROTEINS",
        "ParquetTypedGraph",
        "QM9",
        "REDDIT-BINARY",
        "SyntheticGraph",
        "SyntheticGraphRegression",
        "SyntheticNodeGraph",
        "ZINC",
        "ZINC_OGB",
        "amazon_ratings",
        "cocitation_citeseer",
        "cocitation_cora",
        "cocitation_pubmed",
        "graphuniverse_inductive_triangle",
        "graphuniverse_transductive",
        "minesweeper",
        "ogbg-molhiv",
        "questions",
        "roman_empire",
        "tolokers",
    }
)
EDGE_ATTR_SELECTORS = frozenset(
    {
        "AQSOL",
        "BBB_Martins",
        "CYP3A4_Veith",
        "Caco2_Wang",
        "Clearance_Hepatocyte_AZ",
        "MUTAG",
        "QM9",
        "ZINC",
        "ZINC_OGB",
        "ogbg-molhiv",
    }
)


def _load_dataset_config(selector: str) -> DictConfig:
    raw = OmegaConf.load(GRAPH_CONFIG_DIR / f"{selector}.yaml")
    root = OmegaConf.create(
        {"dataset": OmegaConf.to_container(raw, resolve=False)}
    )
    return root.dataset


def _at_path(config: DictConfig, path: str) -> object:
    value: object = config
    for component in path.split("."):
        assert isinstance(value, DictConfig), path
        assert component in value.keys(), path
        value = value[component]
    return value


def test_manifest_selectors_exactly_equal_surviving_yaml_files() -> None:
    yaml_selectors = frozenset(path.stem for path in GRAPH_CONFIG_DIR.glob("*.yaml"))

    assert yaml_selectors == EXPECTED_SELECTORS
    assert frozenset(GRAPH_DATASET_MANIFEST) == EXPECTED_SELECTORS


@pytest.mark.parametrize(
    ("selector", "num_classes", "target_node_type"),
    [
        ("heterogeneous/DBLP", 4, "author"),
        ("heterogeneous/OGB_MAG", 349, "paper"),
        ("heterogeneous/SyntheticHeterogeneous", 2, "author"),
    ],
)
def test_heterogeneous_qualification_anchors_supervision_contract(
    selector: str,
    num_classes: int,
    target_node_type: str,
) -> None:
    qualification = DATASET_QUALIFICATION_MANIFEST[selector]

    assert qualification.num_classes == num_classes
    assert qualification.target_node_type == target_node_type


@pytest.mark.parametrize(
    "selector",
    sorted(EXPECTED_SELECTORS),
)
def test_manifest_entry_exactly_matches_qualified_yaml(selector: str) -> None:
    config = _load_dataset_config(selector)
    capability = GRAPH_DATASET_MANIFEST[selector]

    assert capability.selector == selector
    assert capability.task == config.parameters.task
    assert capability.task_level == config.parameters.task_level
    assert capability.learning_setting == config.split_params.learning_setting
    assert capability.feature_policy == config.parameters.feature_policy
    assert capability.num_classes == config.parameters.num_classes
    expected_edge_fields = (
        frozenset({"edge_attr"})
        if selector in EDGE_ATTR_SELECTORS
        else frozenset()
    )
    assert capability.edge_fields == expected_edge_fields
    assert capability.qualification
    for path, expected in capability.qualification:
        assert _at_path(config, path) == expected, f"{selector}: {path}"
    assert qualify_graph_dataset(config) is capability


@pytest.mark.parametrize(
    "selector",
    sorted(EXPECTED_SELECTORS),
)
def test_every_dataset_has_qualification_evidence_and_a_compatible_model(
    selector: str,
) -> None:
    capability = GRAPH_DATASET_MANIFEST[selector]
    evidence_paths = {path for path, _ in capability.qualification}
    assert "loader._target_" in evidence_paths

    assert "loader.parameters.data_name" in evidence_paths
    assert "parameters.task" in evidence_paths
    assert "parameters.feature_policy" in evidence_paths
    assert "split_params.learning_setting" in evidence_paths
    assert compatible_graph_models(capability), selector


def test_manifest_and_entries_are_immutable() -> None:
    capability = GRAPH_DATASET_MANIFEST["SyntheticGraph"]

    with pytest.raises(TypeError):
        GRAPH_DATASET_MANIFEST["new"] = capability  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        capability.task = "regression"  # type: ignore[misc]


def test_graphuniverse_uses_resolved_configured_class_count() -> None:
    config = _load_dataset_config("graphuniverse_transductive")
    assert config.parameters.num_classes == 10
    assert (
        config.loader.parameters.generation_parameters.universe_parameters.K
        == 10
    )
    config.parameters.num_classes = 11

    with pytest.raises(ValueError) as error:
        qualify_graph_dataset(config)

    message = str(error.value)
    assert "graphuniverse_transductive" in message
    assert "dataset.parameters.num_classes=11" in message
    assert "manifest expects 10" in message


def test_unqualified_dataset_reports_the_failing_paths() -> None:
    config = OmegaConf.load(GRAPH_CONFIG_DIR / "SyntheticGraph.yaml")
    config.loader.parameters.data_name = "unknown"

    with pytest.raises(ValueError) as error:
        qualify_graph_dataset(config)
    message = str(error.value)
    assert "dataset.loader.parameters.data_name" in message
    assert "dataset.parameters.task" in message
    assert "dataset.split_params.learning_setting" in message



@pytest.mark.parametrize(
    "configured_num_classes",
    [True, 1.0],
    ids=["bool", "integral-float"],
)
def test_manifest_rejects_nonintegral_class_count_types(
    configured_num_classes: object,
) -> None:
    config = _load_dataset_config("SyntheticGraphRegression")
    config.parameters.num_classes = configured_num_classes

    with pytest.raises(TypeError) as error:
        qualify_graph_dataset(config)

    message = str(error.value)
    assert "SyntheticGraphRegression" in message
    assert "dataset.parameters.num_classes" in message
    assert type(configured_num_classes).__name__ in message
    assert "integer" in message




@pytest.mark.parametrize(
    "configured_num_classes",
    [1, 3],
    ids=["missing-class", "extra-class"],
)
def test_manifest_rejects_configured_class_count_mismatch(
    configured_num_classes: int,
) -> None:
    config = OmegaConf.load(GRAPH_CONFIG_DIR / "SyntheticGraph.yaml")
    config.parameters.num_classes = configured_num_classes

    with pytest.raises(ValueError) as error:
        qualify_graph_dataset(config)

    message = str(error.value)
    assert "SyntheticGraph" in message
    assert (
        f"dataset.parameters.num_classes={configured_num_classes}"
        in message
    )
    assert "manifest expects 2" in message


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("loader.parameters.output_kind", "heterogeneous"),
        ("loader.parameters.output_kind", "unknown"),
        ("loader.parameters.output_kind", None),
        ("loader.parameters.partition.strategy", "neighbor"),
        ("loader.parameters.partition.backend", "unsupported"),
    ],
)
def test_parquet_descriptor_requires_an_exact_homogeneous_source_capability(
    path: str,
    value: object,
) -> None:
    config = _load_dataset_config("ParquetTypedGraph")
    OmegaConf.update(config, path, value)

    with pytest.raises(
        ValueError,
        match="does not match an exact graph manifest selector",
    ) as error:
        qualify_graph_dataset(config)

    assert f"dataset.{path}" in str(error.value)


def test_parquet_heterogeneous_source_remains_outside_graph_qualification() -> (
    None
):
    config = _load_dataset_config("ParquetTypedGraph")
    config.loader.parameters.data_domain = "heterogeneous"
    config.loader.parameters.output_kind = "heterogeneous"
    config.loader.parameters.partition.strategy = "neighbor"

    with pytest.raises(ValueError, match="must be 'graph'"):
        qualify_graph_dataset(config)


def test_dataset_capability_constructor_cannot_hide_missing_evidence() -> None:
    with pytest.raises(ValueError, match="qualification"):
        GraphDatasetCapability(
            selector="invalid",
            task="classification",
            task_level="graph",
            learning_setting="inductive",
            feature_policy="continuous",
            edge_fields=frozenset(),
            qualification=(),
        )
