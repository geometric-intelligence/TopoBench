"""Immutable qualification manifest for native homogeneous graph datasets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import Tensor
from topobench.data.qualification import (
    DATASET_QUALIFICATION_MANIFEST,
    DatasetQualification,
)

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
    """Static evidence and supported fields for one YAML selector."""

    selector: str
    task: TaskKind
    task_level: TaskLevel
    learning_setting: LearningSetting
    feature_policy: FeaturePolicy
    edge_fields: frozenset[EdgeField]
    qualification: tuple[tuple[str, QualificationValue], ...]
    feature_width: int = 1
    feature_transforms: frozenset[str] = frozenset()
    num_classes: int = 2
    num_nodes: int | None = None
    allow_incomplete_class_vocabulary: bool = False
    descriptor_only: bool = False
    source_capabilities: frozenset[tuple[str, str, str, str]] = frozenset()

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_classes, bool)
            or not isinstance(self.num_classes, int)
            or self.num_classes <= 0
        ):
            raise ValueError("num_classes must be a positive integer")
        if self.task == "classification" and self.num_classes < 2:
            raise ValueError(
                "classification num_classes must be at least 2"
            )
        if self.task == "regression" and self.num_classes != 1:
            raise ValueError("regression num_classes must equal 1")
        if self.num_nodes is not None:
            if isinstance(self.num_nodes, bool) or not isinstance(
                self.num_nodes, int
            ):
                raise TypeError("num_nodes must be an integer or None")
            if self.num_nodes <= 0:
                raise ValueError("num_nodes must be positive")
        if not isinstance(self.allow_incomplete_class_vocabulary, bool):
            raise TypeError(
                "allow_incomplete_class_vocabulary must be boolean"
            )
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
        try:
            source_capabilities = frozenset(self.source_capabilities)
        except TypeError as error:
            raise TypeError(
                "source_capabilities must contain hashable capability tuples"
            ) from error
        object.__setattr__(
            self,
            "source_capabilities",
            source_capabilities,
        )
        if not isinstance(self.descriptor_only, bool):
            raise TypeError("descriptor_only must be boolean")
        if self.descriptor_only and not self.source_capabilities:
            raise ValueError(
                "descriptor-only selectors require source capabilities"
            )
        if not self.descriptor_only and self.source_capabilities:
            raise ValueError(
                "runtime dataset selectors may not declare source capabilities"
            )
        supported_source_capabilities = frozenset(
            {
                ("graph", "homogeneous", "cluster", "pyg"),
                ("heterogeneous", "heterogeneous", "cluster", "pyg"),
                ("heterogeneous", "heterogeneous", "neighbor", "pyg"),
            }
        )
        if not self.source_capabilities <= supported_source_capabilities:
            raise ValueError(
                "source_capabilities contains an unsupported "
                "domain/output/strategy/backend combination"
            )

    @property
    def task_contract(self) -> GraphTaskContract:
        """Return the coupled task contract declared by the selector."""
        return GraphTaskContract(
            task=self.task,
            task_level=self.task_level,
            learning_setting=self.learning_setting,
        )

    def supports_source(
        self,
        *,
        domain: str,
        output_kind: str,
        strategy: str,
        backend: str,
    ) -> bool:
        """Return whether this selector declares one exact source mode."""
        return (
            domain,
            output_kind,
            strategy,
            backend,
        ) in self.source_capabilities


def validate_classification_vocabulary(
    labels: Iterable[tuple[str, object, int]],
    *,
    selector: str,
    field: str,
    configured_num_classes: object,
    manifest_num_classes: int | None = None,
    allow_incomplete: bool = False,
) -> None:
    """Validate exact class-label tensors and the complete source vocabulary."""
    if isinstance(configured_num_classes, bool) or not isinstance(
        configured_num_classes, Integral
    ):
        raise TypeError(
            f"{selector}: full source field parameters.num_classes observed "
            f"dtype={type(configured_num_classes).__name__}; expected an "
            "integer class count"
        )
    num_classes = int(configured_num_classes)
    if num_classes < 2:
        raise ValueError(
            f"{selector}: full source field parameters.num_classes observed "
            f"range=[{num_classes}, {num_classes}]; expected at least 2"
        )
    if (
        manifest_num_classes is not None
        and num_classes != manifest_num_classes
    ):
        raise ValueError(
            f"{selector}: full source field parameters.num_classes observed "
            f"value={num_classes}; manifest expects "
            f"{manifest_num_classes}"
        )

    first_item: str | None = None
    seen: set[int] = set()
    item_count = 0
    for item, value, expected_size in labels:
        item_count += 1
        if first_item is None:
            first_item = item
        shape = tuple(value.shape) if isinstance(value, Tensor) else None
        dtype = value.dtype if isinstance(value, Tensor) else type(value).__name__
        context = (
            f"{selector}: {item} field {field} observed "
            f"shape={shape}, dtype={dtype}"
        )
        if not isinstance(value, Tensor):
            raise TypeError(
                f"{context}; expected rank-1 torch.long labels"
            )
        if value.dtype != torch.long:
            raise TypeError(
                f"{context}; expected dtype=torch.long"
            )
        if value.ndim != 1:
            raise ValueError(
                f"{context}; expected rank-1 labels"
            )
        if value.numel() != expected_size:
            raise ValueError(
                f"{context}; expected shape=({expected_size},)"
            )
        if value.numel() == 0:
            raise ValueError(
                f"{context}; expected at least one label in "
                f"range=[0, {num_classes})"
            )
        observed_min = int(value.min().item())
        observed_max = int(value.max().item())
        if observed_min < 0 or observed_max >= num_classes:
            raise ValueError(
                f"{context}, range=[{observed_min}, {observed_max}]; "
                f"expected range=[0, {num_classes})"
            )
        seen.update(int(class_id) for class_id in value.unique().tolist())

    if item_count == 0:
        raise ValueError(
            f"{selector}: full source field {field} observed shape=(0,); "
            "expected non-empty qualified labels"
        )
    missing = sorted(set(range(num_classes)) - seen)
    if missing and not allow_incomplete:
        raise ValueError(
            f"{selector}: full source {first_item} field {field} runtime "
            f"vocabulary={sorted(seen)} missing declared classes={missing}; "
            f"expected complete range=[0, {num_classes})"
        )


def qualify_heterogeneous_dataset(
    dataset: Mapping[str, Any],
) -> DatasetQualification:
    """Qualify heterogeneous supervision against the retained manifest."""
    if not isinstance(dataset, Mapping):
        raise TypeError("dataset must be a mapping")
    domain = _value_at_path(dataset, "loader.parameters.data_domain")
    if domain != "heterogeneous":
        raise ValueError(
            "dataset.loader.parameters.data_domain must be "
            f"'heterogeneous', got {domain!r}"
        )
    data_name = _value_at_path(dataset, "loader.parameters.data_name")
    selector = f"heterogeneous/{data_name}"
    qualification = DATASET_QUALIFICATION_MANIFEST.get(selector)
    if qualification is None:
        raise ValueError(
            f"{selector}: no retained heterogeneous qualification"
        )
    loader_family = _value_at_path(dataset, "loader._target_")
    if loader_family != qualification.loader_family:
        raise ValueError(
            f"{selector}: dataset.loader._target_={loader_family!r}; "
            f"manifest expects {qualification.loader_family!r}"
        )

    configured_classes = _value_at_path(
        dataset,
        "parameters.num_classes",
    )
    if (
        isinstance(configured_classes, bool)
        or not isinstance(configured_classes, Integral)
    ):
        raise TypeError(
            f"{selector}: dataset.parameters.num_classes observed "
            f"dtype={type(configured_classes).__name__}; manifest expects "
            f"integer {qualification.num_classes}"
        )
    if configured_classes != qualification.num_classes:
        raise ValueError(
            f"{selector}: dataset.parameters.num_classes="
            f"{configured_classes!r}; manifest expects "
            f"{qualification.num_classes}"
        )

    target_node_type = _value_at_path(
        dataset,
        "parameters.target_node_type",
    )
    if target_node_type != qualification.target_node_type:
        raise ValueError(
            f"{selector}: dataset.parameters.target_node_type="
            f"{target_node_type!r}; manifest expects "
            f"{qualification.target_node_type!r}"
        )
    return qualification


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
    if selector == "ParquetTypedGraph":
        return (
            "topobench.data.loaders.parquet.ParquetTypedGraphLoader"
        )
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
    num_classes: int | None = None,
    num_nodes: int | None = None,
    feature_transforms: frozenset[str] = frozenset(),
    edge_fields: frozenset[EdgeField] = frozenset(),
    extra_evidence: tuple[tuple[str, QualificationValue], ...] = (),
    descriptor_only: bool = False,
    source_capabilities: frozenset[tuple[str, str, str, str]] = frozenset(),
) -> GraphDatasetCapability:
    """Construct one manifest row with common path-level evidence."""
    class_count = (
        num_classes
        if num_classes is not None
        else (2 if task == "classification" else 1)
    )
    return GraphDatasetCapability(
        selector=selector,
        task=task,
        task_level=task_level,
        learning_setting=learning_setting,
        feature_policy=feature_policy,
        feature_width=feature_width,
        feature_transforms=feature_transforms,
        edge_fields=edge_fields,
        num_classes=class_count,
        num_nodes=num_nodes,
        descriptor_only=descriptor_only,
        source_capabilities=source_capabilities,
        qualification=(
            ("loader._target_", _loader_target(selector)),
            ("loader.parameters.data_name", data_name),
            ("parameters.task", task),
            ("parameters.task_level", task_level),
            ("parameters.feature_policy", feature_policy),
            ("parameters.num_classes", class_count),
            *(
                (("parameters.num_nodes", num_nodes),)
                if num_nodes is not None
                else ()
            ),
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
        num_classes=3,
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
        "ParquetTypedGraph",
        data_name="ParquetTypedGraph",
        task="classification",
        task_level="node",
        learning_setting="transductive",
        feature_policy="continuous",
        feature_width=2,
        descriptor_only=True,
        source_capabilities=frozenset(
            {
                ("graph", "homogeneous", "cluster", "pyg"),
                (
                    "heterogeneous",
                    "heterogeneous",
                    "cluster",
                    "pyg",
                ),
                (
                    "heterogeneous",
                    "heterogeneous",
                    "neighbor",
                    "pyg",
                ),
            }
        ),
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
        num_nodes=18,
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
        num_classes=5,
    ),
    _capability(
        "cocitation_citeseer",
        data_name="citeseer",
        task="classification",
        task_level="node",
        learning_setting="transductive",
        feature_policy="continuous",
        feature_width=3703,
        num_classes=6,
        num_nodes=3327,
    ),
    _capability(
        "cocitation_cora",
        data_name="Cora",
        task="classification",
        task_level="node",
        learning_setting="transductive",
        feature_policy="continuous",
        feature_width=1433,
        num_classes=7,
        num_nodes=2708,
    ),
    _capability(
        "cocitation_pubmed",
        data_name="PubMed",
        task="classification",
        task_level="node",
        learning_setting="transductive",
        feature_policy="continuous",
        feature_width=500,
        num_classes=3,
        num_nodes=19717,
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
        num_classes=10,
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
        num_classes=18,
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


def _descriptor_is_runnable_graph_source(
    capability: GraphDatasetCapability,
    dataset: Mapping[str, Any],
    domain: object,
) -> bool:
    """Admit one descriptor only through its exact configured source mode."""
    output_kind = _value_at_path(
        dataset,
        "loader.parameters.output_kind",
    )
    strategy = _value_at_path(
        dataset,
        "loader.parameters.partition.strategy",
    )
    backend = _value_at_path(
        dataset,
        "loader.parameters.partition.backend",
    )
    if (
        output_kind != "homogeneous"
        or not isinstance(domain, str)
        or not isinstance(strategy, str)
        or not isinstance(backend, str)
    ):
        return False
    return capability.supports_source(
        domain=domain,
        output_kind=output_kind,
        strategy=strategy,
        backend=backend,
    )


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
        if (
            not capability.descriptor_only
            or _descriptor_is_runnable_graph_source(
                capability,
                dataset,
                domain,
            )
        )
        if all(
            path == "parameters.num_classes"
            or _value_at_path(dataset, path) == expected
            for path, expected in capability.qualification
        )
    ]
    if len(matches) == 1:
        capability = matches[0]
        configured_classes = _value_at_path(
            dataset,
            "parameters.num_classes",
        )
        if isinstance(configured_classes, bool) or not isinstance(
            configured_classes,
            Integral,
        ):
            raise TypeError(
                f"{capability.selector}: full source field "
                "dataset.parameters.num_classes observed "
                f"dtype={type(configured_classes).__name__}; expected "
                "an integer"
            )
        if configured_classes != capability.num_classes:
            raise ValueError(
                f"{capability.selector}: full source field "
                "dataset.parameters.num_classes="
                f"{configured_classes!r}; manifest expects "
                f"{capability.num_classes}"
            )
        if capability.selector == "graphuniverse_transductive":
            generated_classes = _value_at_path(
                dataset,
                "loader.parameters.generation_parameters."
                "universe_parameters.K",
            )
            if generated_classes != configured_classes:
                raise ValueError(
                    f"{capability.selector}: dataset.parameters.num_classes="
                    f"{configured_classes!r} disagrees with "
                    "loader generation universe K="
                    f"{generated_classes!r}"
                )
        return capability

    observed_paths = (
        "loader.parameters.data_name",
        "parameters.task",
        "parameters.task_level",
        "parameters.feature_policy",
        "parameters.num_classes",
        "loader.parameters.output_kind",
        "loader.parameters.partition.strategy",
        "loader.parameters.partition.backend",
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
    "qualify_heterogeneous_dataset",
    "validate_classification_vocabulary",
]
