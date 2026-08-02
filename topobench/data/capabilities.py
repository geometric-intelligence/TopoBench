"""Immutable qualification manifest for native homogeneous graph datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

TaskKind = Literal["classification", "regression"]
TaskLevel = Literal["graph", "node"]
LearningSetting = Literal["inductive", "transductive"]
FeaturePolicy = Literal[
    "continuous",
    "categorical_one_hot",
    "degree",
    "constant",
]
EdgeField = Literal["edge_attr", "edge_weight"]
QualificationValue = str | int | float | bool


@dataclass(frozen=True, slots=True)
class GraphTaskContract:
    """One coupled task, supervision level, and learning setting."""

    task: TaskKind
    task_level: TaskLevel
    learning_setting: LearningSetting


@dataclass(frozen=True, slots=True)
class GraphDatasetCapability:
    """Static evidence and supported native fields for one YAML selector."""

    selector: str
    task: TaskKind
    task_level: TaskLevel
    learning_setting: LearningSetting
    feature_policy: FeaturePolicy
    edge_fields: frozenset[EdgeField]
    qualification: tuple[tuple[str, QualificationValue], ...]
    feature_width: int = 1
    feature_transforms: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if isinstance(self.feature_width, bool) or self.feature_width <= 0:
            raise ValueError("feature_width must be a positive integer")
        if not all(
            isinstance(name, str) and name for name in self.feature_transforms
        ):
            raise ValueError("feature_transforms must contain non-empty names")
        if not self.qualification:
            raise ValueError("qualification evidence is required")
        paths = [path for path, _ in self.qualification]
        if len(paths) != len(set(paths)):
            raise ValueError("qualification paths must be unique")
        if not self.edge_fields <= {"edge_attr", "edge_weight"}:
            raise ValueError(
                "edge_fields contains an unsupported native field"
            )

    @property
    def task_contract(self) -> GraphTaskContract:
        """Return the coupled task contract declared by the selector."""
        return GraphTaskContract(
            task=self.task,
            task_level=self.task_level,
            learning_setting=self.learning_setting,
        )


def _loader_target(selector: str) -> str:
    """Return the exact surviving YAML loader target for one selector."""
    if selector in {"AQSOL", "QM9"}:
        return "topobench.data.loaders.graph.MoleculeDatasetLoader"
    if selector in {
        "BBB_Martins",
        "CYP3A4_Veith",
        "Caco2_Wang",
        "Clearance_Hepatocyte_AZ",
    }:
        return "topobench.data.loaders.ADMEDatasetLoader"
    if selector in {
        "IMDB-BINARY",
        "IMDB-MULTI",
        "MUTAG",
        "NCI1",
        "NCI109",
        "PROTEINS",
        "REDDIT-BINARY",
    }:
        return "topobench.data.loaders.TUDatasetLoader"
    if selector in {
        "SyntheticGraph",
        "SyntheticGraphRegression",
        "SyntheticNodeGraph",
    }:
        return "topobench.data.loaders.SyntheticGraphDatasetLoader"
    if selector in {"ZINC", "ZINC_OGB"}:
        return "topobench.data.loaders.MoleculeDatasetLoader"
    if selector in {
        "amazon_ratings",
        "minesweeper",
        "questions",
        "roman_empire",
        "tolokers",
    }:
        return "topobench.data.loaders.HeterophilousGraphDatasetLoader"
    if selector in {
        "cocitation_citeseer",
        "cocitation_cora",
        "cocitation_pubmed",
    }:
        return "topobench.data.loaders.PlanetoidDatasetLoader"
    if selector in {
        "graphuniverse_inductive_triangle",
        "graphuniverse_transductive",
    }:
        return "topobench.data.loaders.GraphUniverseDatasetLoader"
    if selector == "ogbg-molhiv":
        return "topobench.data.loaders.OGBGDatasetLoader"
    raise ValueError(f"missing loader qualification for {selector!r}")


def _capability(
    selector: str,
    *,
    data_name: str,
    task: TaskKind,
    task_level: TaskLevel,
    learning_setting: LearningSetting,
    feature_policy: FeaturePolicy,
    feature_width: int,
    feature_transforms: frozenset[str] = frozenset(),
    edge_fields: frozenset[EdgeField] = frozenset(),
    extra_evidence: tuple[tuple[str, QualificationValue], ...] = (),
) -> GraphDatasetCapability:
    """Construct one manifest row with common path-level evidence."""
    return GraphDatasetCapability(
        selector=selector,
        task=task,
        task_level=task_level,
        learning_setting=learning_setting,
        feature_policy=feature_policy,
        feature_width=feature_width,
        feature_transforms=feature_transforms,
        edge_fields=edge_fields,
        qualification=(
            ("loader._target_", _loader_target(selector)),
            ("loader.parameters.data_name", data_name),
            ("parameters.task", task),
            ("parameters.task_level", task_level),
            ("parameters.feature_policy", feature_policy),
            ("split_params.learning_setting", learning_setting),
            *extra_evidence,
        ),
    )


_EDGE_ATTR = frozenset({"edge_attr"})
_DEGREE_FEATURE_TRANSFORMS = frozenset(
    {"NodeDegrees", "OneHotDegreeFeatures"}
)
_CATEGORICAL_FEATURE_TRANSFORMS = frozenset({"OneHotDegreeFeatures"})
_CONSTANT_FEATURE_TRANSFORMS = frozenset({"ConstantNodeFeatures"})
_ROWS = (
    _capability(
        "AQSOL",
        data_name="AQSOL",
        task="regression",
        task_level="graph",
        learning_setting="inductive",
        feature_policy="degree",
        feature_width=21,
        feature_transforms=_DEGREE_FEATURE_TRANSFORMS,
        edge_fields=_EDGE_ATTR,
    ),
    _capability(
        "BBB_Martins",
        data_name="BBB_Martins",
        task="classification",
        task_level="graph",
        learning_setting="inductive",
        feature_policy="categorical_one_hot",
        feature_width=174,
        edge_fields=_EDGE_ATTR,
    ),
    _capability(
        "CYP3A4_Veith",
        data_name="CYP3A4_Veith",
        task="classification",
        task_level="graph",
        learning_setting="inductive",
        feature_policy="categorical_one_hot",
        feature_width=174,
        edge_fields=_EDGE_ATTR,
    ),
    _capability(
        "Caco2_Wang",
        data_name="Caco2_Wang",
        task="regression",
        task_level="graph",
        learning_setting="inductive",
        feature_policy="categorical_one_hot",
        feature_width=174,
        edge_fields=_EDGE_ATTR,
    ),
    _capability(
        "Clearance_Hepatocyte_AZ",
        data_name="Clearance_Hepatocyte_AZ",
        task="regression",
        task_level="graph",
        learning_setting="inductive",
        feature_policy="categorical_one_hot",
        feature_width=174,
        edge_fields=_EDGE_ATTR,
    ),
    _capability(
        "IMDB-BINARY",
        data_name="IMDB-BINARY",
        task="classification",
        task_level="graph",
        learning_setting="inductive",
        feature_policy="degree",
        feature_width=136,
        feature_transforms=_DEGREE_FEATURE_TRANSFORMS,
    ),
    _capability(
        "IMDB-MULTI",
        data_name="IMDB-MULTI",
        task="classification",
        task_level="graph",
        learning_setting="inductive",
        feature_policy="degree",
        feature_width=89,
        feature_transforms=_DEGREE_FEATURE_TRANSFORMS,
    ),
    _capability(
        "MUTAG",
        data_name="MUTAG",
        task="classification",
        task_level="graph",
        learning_setting="inductive",
        feature_policy="continuous",
        feature_width=7,
        edge_fields=_EDGE_ATTR,
    ),
    _capability(
        "NCI1",
        data_name="NCI1",
        task="classification",
        task_level="graph",
        learning_setting="inductive",
        feature_policy="continuous",
        feature_width=37,
    ),
    _capability(
        "NCI109",
        data_name="NCI109",
        task="classification",
        task_level="graph",
        learning_setting="inductive",
        feature_policy="continuous",
        feature_width=38,
    ),
    _capability(
        "PROTEINS",
        data_name="PROTEINS",
        task="classification",
        task_level="graph",
        learning_setting="inductive",
        feature_policy="continuous",
        feature_width=3,
    ),
    _capability(
        "QM9",
        data_name="QM9",
        task="regression",
        task_level="graph",
        learning_setting="inductive",
        feature_policy="continuous",
        feature_width=11,
        edge_fields=_EDGE_ATTR,
    ),
    _capability(
        "REDDIT-BINARY",
        data_name="REDDIT-BINARY",
        task="classification",
        task_level="graph",
        learning_setting="inductive",
        feature_policy="constant",
        feature_width=10,
        feature_transforms=_CONSTANT_FEATURE_TRANSFORMS,
    ),
    _capability(
        "SyntheticGraph",
        data_name="SyntheticGraph",
        task="classification",
        task_level="graph",
        learning_setting="inductive",
        feature_policy="continuous",
        feature_width=4,
    ),
    _capability(
        "SyntheticGraphRegression",
        data_name="SyntheticGraphRegression",
        task="regression",
        task_level="graph",
        learning_setting="inductive",
        feature_policy="continuous",
        feature_width=4,
    ),
    _capability(
        "SyntheticNodeGraph",
        data_name="SyntheticNodeGraph",
        task="classification",
        task_level="node",
        learning_setting="transductive",
        feature_policy="continuous",
        feature_width=4,
    ),
    _capability(
        "ZINC",
        data_name="ZINC",
        task="regression",
        task_level="graph",
        learning_setting="inductive",
        feature_policy="categorical_one_hot",
        feature_width=21,
        feature_transforms=_CATEGORICAL_FEATURE_TRANSFORMS,
        edge_fields=_EDGE_ATTR,
        extra_evidence=(("parameters.loss_type", "mse"),),
    ),
    _capability(
        "ZINC_OGB",
        data_name="ZINC",
        task="regression",
        task_level="graph",
        learning_setting="inductive",
        feature_policy="categorical_one_hot",
        feature_width=21,
        feature_transforms=_CATEGORICAL_FEATURE_TRANSFORMS,
        edge_fields=_EDGE_ATTR,
        extra_evidence=(("parameters.loss_type", "mae"),),
    ),
    _capability(
        "amazon_ratings",
        data_name="amazon_ratings",
        task="classification",
        task_level="node",
        learning_setting="transductive",
        feature_policy="continuous",
        feature_width=300,
    ),
    _capability(
        "cocitation_citeseer",
        data_name="citeseer",
        task="classification",
        task_level="node",
        learning_setting="transductive",
        feature_policy="continuous",
        feature_width=3703,
    ),
    _capability(
        "cocitation_cora",
        data_name="Cora",
        task="classification",
        task_level="node",
        learning_setting="transductive",
        feature_policy="continuous",
        feature_width=1433,
    ),
    _capability(
        "cocitation_pubmed",
        data_name="PubMed",
        task="classification",
        task_level="node",
        learning_setting="transductive",
        feature_policy="continuous",
        feature_width=500,
    ),
    _capability(
        "graphuniverse_inductive_triangle",
        data_name="GraphUniverse",
        task="regression",
        task_level="graph",
        learning_setting="inductive",
        feature_policy="continuous",
        feature_width=15,
        extra_evidence=(
            (
                "loader.parameters.generation_parameters.task",
                "triangle_counting",
            ),
        ),
    ),
    _capability(
        "graphuniverse_transductive",
        data_name="GraphUniverse",
        task="classification",
        task_level="node",
        learning_setting="transductive",
        feature_policy="continuous",
        feature_width=15,
        extra_evidence=(
            (
                "loader.parameters.generation_parameters.task",
                "community_detection",
            ),
        ),
    ),
    _capability(
        "minesweeper",
        data_name="minesweeper",
        task="classification",
        task_level="node",
        learning_setting="transductive",
        feature_policy="continuous",
        feature_width=7,
    ),
    _capability(
        "ogbg-molhiv",
        data_name="ogbg-molhiv",
        task="classification",
        task_level="graph",
        learning_setting="inductive",
        feature_policy="categorical_one_hot",
        feature_width=174,
        edge_fields=_EDGE_ATTR,
    ),
    _capability(
        "questions",
        data_name="questions",
        task="classification",
        task_level="node",
        learning_setting="transductive",
        feature_policy="continuous",
        feature_width=301,
    ),
    _capability(
        "roman_empire",
        data_name="roman_empire",
        task="classification",
        task_level="node",
        learning_setting="transductive",
        feature_policy="continuous",
        feature_width=300,
    ),
    _capability(
        "tolokers",
        data_name="tolokers",
        task="classification",
        task_level="node",
        learning_setting="transductive",
        feature_policy="continuous",
        feature_width=10,
    ),
)
GRAPH_DATASET_MANIFEST: Mapping[str, GraphDatasetCapability] = (
    MappingProxyType({row.selector: row for row in _ROWS})
)


def _value_at_path(config: Mapping[str, Any], path: str) -> object:
    """Read a dotted mapping path without resolving unrelated values."""
    value: object = config
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return None
        value = value[component]
    return value


def configured_graph_feature_width(dataset: Mapping[str, Any]) -> int:
    """Return the declared node width without accepting ambiguous values."""
    value = _value_at_path(dataset, "parameters.num_features")
    if isinstance(value, bool):
        raise TypeError(
            "dataset.parameters.num_features must declare an integer width"
        )
    if isinstance(value, int):
        width = value
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if (
            not value
            or isinstance(value[0], bool)
            or not isinstance(value[0], int)
        ):
            raise TypeError(
                "dataset.parameters.num_features[0] must declare an integer "
                "node width"
            )
        width = value[0]
    else:
        raise TypeError(
            "dataset.parameters.num_features must declare an integer width"
        )
    if width <= 0:
        raise ValueError("dataset.parameters.num_features must be positive")
    return width


def qualify_graph_dataset(
    dataset: Mapping[str, Any],
) -> GraphDatasetCapability:
    """Qualify exactly one manifest row from selector-specific evidence."""
    if not isinstance(dataset, Mapping):
        raise TypeError("dataset must be a mapping")
    domain_path = "dataset.loader.parameters.data_domain"
    domain = _value_at_path(dataset, "loader.parameters.data_domain")
    if domain != "graph":
        raise ValueError(f"{domain_path} must be 'graph', got {domain!r}")

    matches = [
        capability
        for capability in GRAPH_DATASET_MANIFEST.values()
        if all(
            _value_at_path(dataset, path) == expected
            for path, expected in capability.qualification
        )
    ]
    if len(matches) == 1:
        return matches[0]

    observed_paths = (
        "loader.parameters.data_name",
        "parameters.task",
        "parameters.task_level",
        "parameters.feature_policy",
        "split_params.learning_setting",
    )
    observed = ", ".join(
        f"dataset.{path}={_value_at_path(dataset, path)!r}"
        for path in observed_paths
    )
    if not matches:
        raise ValueError(
            "dataset does not match an exact graph manifest selector: "
            f"{observed}"
        )
    selectors = ", ".join(sorted(row.selector for row in matches))
    raise ValueError(
        f"dataset qualification is ambiguous for [{selectors}]: {observed}"
    )


__all__ = [
    "EdgeField",
    "FeaturePolicy",
    "GRAPH_DATASET_MANIFEST",
    "GraphDatasetCapability",
    "GraphTaskContract",
    "configured_graph_feature_width",
    "LearningSetting",
    "TaskKind",
    "TaskLevel",
    "qualify_graph_dataset",
]
