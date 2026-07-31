"""Immutable native graph model capabilities and composition validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from topobench.data.capabilities import (
    FeaturePolicy,
    GraphDatasetCapability,
    GraphTaskContract,
    qualify_graph_dataset,
)

EdgeMode = Literal["consume", "ignore", "reject"]
_EDGE_MODES = frozenset({"consume", "ignore", "reject"})
_GRAPH_NODE_ENCODER = "topobench.nn.encoders.GraphNodeFeatureEncoder"
_GNN_WRAPPER = "topobench.nn.wrappers.GNNWrapper"
_GRAPH_MLP_WRAPPER = "topobench.nn.wrappers.GraphMLPWrapper"
_NO_READOUT = "topobench.nn.readouts.NoReadOut"
_MLP_READOUT = "topobench.nn.readouts.MLPReadout"


@dataclass(frozen=True, slots=True)
class GraphModelCapability:
    """Static task, feature, edge, and component contract for one model."""

    selector: str
    tasks: frozenset[GraphTaskContract]
    feature_policies: frozenset[FeaturePolicy]
    edge_attr_mode: EdgeMode
    edge_weight_mode: EdgeMode
    backbone_target: str
    wrapper_target: str
    readout_target: str

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("tasks must not be empty")
        if not self.feature_policies:
            raise ValueError("feature_policies must not be empty")
        if self.edge_attr_mode not in _EDGE_MODES:
            raise ValueError("edge_attr_mode must be consume, ignore, or reject")
        if self.edge_weight_mode not in _EDGE_MODES:
            raise ValueError(
                "edge_weight_mode must be consume, ignore, or reject"
            )

    def supports(self, dataset: GraphDatasetCapability) -> bool:
        """Return whether this model accepts the complete dataset contract."""
        if dataset.task_contract not in self.tasks:
            return False
        if dataset.feature_policy not in self.feature_policies:
            return False
        for field in dataset.edge_fields:
            if getattr(self, f"{field}_mode") == "reject":
                return False
        return True


_GRAPH_CLASSIFICATION = GraphTaskContract(
    task="classification",
    task_level="graph",
    learning_setting="inductive",
)
_GRAPH_REGRESSION = GraphTaskContract(
    task="regression",
    task_level="graph",
    learning_setting="inductive",
)
_NODE_CLASSIFICATION = GraphTaskContract(
    task="classification",
    task_level="node",
    learning_setting="transductive",
)
_ALL_TASKS = frozenset(
    {_GRAPH_CLASSIFICATION, _GRAPH_REGRESSION, _NODE_CLASSIFICATION}
)
_NODE_TASKS = frozenset({_NODE_CLASSIFICATION})
_ALL_FEATURES = frozenset(
    {"continuous", "categorical_one_hot", "degree", "constant"}
)
_CONTINUOUS_FEATURES = frozenset({"continuous"})
_ROWS = (
    GraphModelCapability(
        selector="gcn",
        tasks=_ALL_TASKS,
        feature_policies=_ALL_FEATURES,
        edge_attr_mode="ignore",
        edge_weight_mode="consume",
        backbone_target="torch_geometric.nn.models.GCN",
        wrapper_target=_GNN_WRAPPER,
        readout_target=_NO_READOUT,
    ),
    GraphModelCapability(
        selector="gat",
        tasks=_ALL_TASKS,
        feature_policies=_ALL_FEATURES,
        edge_attr_mode="ignore",
        edge_weight_mode="reject",
        backbone_target="torch_geometric.nn.models.GAT",
        wrapper_target=_GNN_WRAPPER,
        readout_target=_NO_READOUT,
    ),
    GraphModelCapability(
        selector="gin",
        tasks=_ALL_TASKS,
        feature_policies=_ALL_FEATURES,
        edge_attr_mode="reject",
        edge_weight_mode="reject",
        backbone_target="torch_geometric.nn.models.GIN",
        wrapper_target=_GNN_WRAPPER,
        readout_target=_NO_READOUT,
    ),
    GraphModelCapability(
        selector="gps",
        tasks=_ALL_TASKS,
        feature_policies=_ALL_FEATURES,
        edge_attr_mode="ignore",
        edge_weight_mode="reject",
        backbone_target="topobench.nn.backbones.GPSEncoder",
        wrapper_target=_GNN_WRAPPER,
        readout_target=_MLP_READOUT,
    ),
    GraphModelCapability(
        selector="nsd",
        tasks=_ALL_TASKS,
        feature_policies=_ALL_FEATURES,
        edge_attr_mode="ignore",
        edge_weight_mode="ignore",
        backbone_target="topobench.nn.backbones.NSDEncoder",
        wrapper_target=_GNN_WRAPPER,
        readout_target=_MLP_READOUT,
    ),
    GraphModelCapability(
        selector="graph_mlp",
        tasks=_NODE_TASKS,
        feature_policies=_CONTINUOUS_FEATURES,
        edge_attr_mode="reject",
        edge_weight_mode="reject",
        backbone_target="topobench.nn.backbones.GraphMLP",
        wrapper_target=_GRAPH_MLP_WRAPPER,
        readout_target=_NO_READOUT,
    ),
    GraphModelCapability(
        selector="gcn_dgm",
        tasks=_NODE_TASKS,
        feature_policies=_CONTINUOUS_FEATURES,
        edge_attr_mode="reject",
        edge_weight_mode="reject",
        backbone_target="topobench.nn.backbones.GCNDGM",
        wrapper_target=_GNN_WRAPPER,
        readout_target=_NO_READOUT,
    ),
)
GRAPH_MODEL_CAPABILITIES: Mapping[str, GraphModelCapability] = MappingProxyType(
    {row.selector: row for row in _ROWS}
)


def _required_mapping(
    parent: Mapping[str, Any],
    key: str,
    *,
    path: str,
) -> Mapping[str, Any]:
    """Return one required child mapping with its full configuration path."""
    child_path = f"{path}.{key}"
    if key not in parent:
        raise ValueError(f"{child_path} is required")
    child = parent[key]
    if not isinstance(child, Mapping):
        raise TypeError(f"{child_path} must be a mapping")
    return child


def _required_value(
    parent: Mapping[str, Any],
    key: str,
    *,
    path: str,
) -> object:
    """Return one required scalar with its full configuration path."""
    child_path = f"{path}.{key}"
    if key not in parent:
        raise ValueError(f"{child_path} is required")
    return parent[key]


def _qualified_capability(
    dataset: Mapping[str, Any],
    model_selector: str,
) -> tuple[GraphDatasetCapability, GraphModelCapability]:
    """Validate dataset evidence against one declared model selector."""
    if not isinstance(dataset, Mapping):
        raise TypeError("dataset must be a mapping")
    if not isinstance(model_selector, str):
        raise TypeError("model.model_name must be a string")

    dataset_capability = qualify_graph_dataset(dataset)
    try:
        model_capability = GRAPH_MODEL_CAPABILITIES[model_selector]
    except KeyError as error:
        raise ValueError(
            f"model.model_name={model_selector!r} has no graph capability"
        ) from error

    if dataset_capability.task_contract not in model_capability.tasks:
        contract = dataset_capability.task_contract
        raise ValueError(
            "dataset.parameters.task, dataset.parameters.task_level, and "
            "dataset.split_params.learning_setting form an unsupported "
            f"contract ({contract.task!r}, {contract.task_level!r}, "
            f"{contract.learning_setting!r}) for "
            f"model.model_name={model_selector!r}"
        )
    if dataset_capability.feature_policy not in model_capability.feature_policies:
        raise ValueError(
            "dataset.parameters.feature_policy="
            f"{dataset_capability.feature_policy!r} is unsupported by "
            f"model.model_name={model_selector!r}"
        )
    for edge_field in sorted(dataset_capability.edge_fields):
        mode = getattr(model_capability, f"{edge_field}_mode")
        if mode == "reject":
            raise ValueError(
                f"dataset selector {dataset_capability.selector!r} declares "
                f"dataset.{edge_field}, but model.model_name={model_selector!r} "
                "declares reject"
            )
    return dataset_capability, model_capability


def _validated_pair(
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
) -> tuple[GraphDatasetCapability, GraphModelCapability]:
    """Validate a graph pair without resolving its edge-mode interpolations."""
    if not isinstance(dataset, Mapping):
        raise TypeError("dataset must be a mapping")
    if not isinstance(model, Mapping):
        raise TypeError("model must be a mapping")

    dataset_domain = _required_value(
        _required_mapping(dataset, "loader", path="dataset"),
        "parameters",
        path="dataset.loader",
    )
    if not isinstance(dataset_domain, Mapping):
        raise TypeError("dataset.loader.parameters must be a mapping")
    dataset_domain = _required_value(
        dataset_domain,
        "data_domain",
        path="dataset.loader.parameters",
    )
    model_domain = _required_value(model, "model_domain", path="model")
    if dataset_domain != model_domain:
        raise ValueError(
            "Cross-domain lifting is unsupported: "
            f"dataset={dataset_domain!r}, model={model_domain!r}"
        )
    if model_domain != "graph":
        raise ValueError(
            f"model.model_domain must be 'graph', got {model_domain!r}"
        )

    model_selector = _required_value(model, "model_name", path="model")
    dataset_capability, model_capability = _qualified_capability(
        dataset,
        model_selector,
    )
    encoder = _required_mapping(model, "feature_encoder", path="model")
    encoder_target = _required_value(
        encoder,
        "_target_",
        path="model.feature_encoder",
    )
    if encoder_target != _GRAPH_NODE_ENCODER:
        raise ValueError(
            "model.feature_encoder._target_ must be "
            f"{_GRAPH_NODE_ENCODER!r}, got {encoder_target!r}"
        )
    backbone = _required_mapping(model, "backbone", path="model")
    backbone_target = _required_value(
        backbone,
        "_target_",
        path="model.backbone",
    )
    if backbone_target != model_capability.backbone_target:
        raise ValueError(
            f"model.backbone._target_ must be "
            f"{model_capability.backbone_target!r}, got {backbone_target!r}"
        )
    wrapper = _required_mapping(model, "backbone_wrapper", path="model")
    wrapper_target = _required_value(
        wrapper,
        "_target_",
        path="model.backbone_wrapper",
    )
    if wrapper_target != model_capability.wrapper_target:
        raise ValueError(
            f"model.backbone_wrapper._target_ must be "
            f"{model_capability.wrapper_target!r}, got {wrapper_target!r}"
        )
    for mode_field in ("edge_attr_mode", "edge_weight_mode"):
        if mode_field not in wrapper.keys():
            raise ValueError(
                f"model.backbone_wrapper.{mode_field} is required"
            )
    readout = _required_mapping(model, "readout", path="model")
    readout_target = _required_value(readout, "_target_", path="model.readout")
    if readout_target != model_capability.readout_target:
        raise ValueError(
            f"model.readout._target_ must be "
            f"{model_capability.readout_target!r}, got {readout_target!r}"
        )
    return dataset_capability, model_capability


def validate_graph_composition(
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
) -> tuple[GraphDatasetCapability, GraphModelCapability]:
    """Validate an exact graph pair and its explicit resolved edge modes."""
    dataset_capability, model_capability = _validated_pair(dataset, model)
    wrapper = _required_mapping(model, "backbone_wrapper", path="model")
    for field in ("edge_attr", "edge_weight"):
        path = f"model.backbone_wrapper.{field}_mode"
        actual = _required_value(wrapper, f"{field}_mode", path="model.backbone_wrapper")
        expected = getattr(model_capability, f"{field}_mode")
        if actual != expected:
            raise ValueError(f"{path} must resolve to {expected!r}, got {actual!r}")
    return dataset_capability, model_capability


def compatible_graph_models(
    dataset: GraphDatasetCapability,
) -> tuple[GraphModelCapability, ...]:
    """Return every immutable model row accepting a dataset row."""
    if not isinstance(dataset, GraphDatasetCapability):
        raise TypeError("dataset must be a GraphDatasetCapability")
    return tuple(
        model
        for model in GRAPH_MODEL_CAPABILITIES.values()
        if model.supports(dataset)
    )


def validated_edge_attr_mode(
    dataset: Mapping[str, Any],
    model_selector: str,
) -> EdgeMode:
    """Resolve an edge-attribute mode after dataset/model qualification."""
    return _qualified_capability(dataset, model_selector)[1].edge_attr_mode


def validated_edge_weight_mode(
    dataset: Mapping[str, Any],
    model_selector: str,
) -> EdgeMode:
    """Resolve an edge-weight mode after dataset/model qualification."""
    return _qualified_capability(dataset, model_selector)[1].edge_weight_mode


__all__ = [
    "EdgeMode",
    "GRAPH_MODEL_CAPABILITIES",
    "GraphModelCapability",
    "compatible_graph_models",
    "validate_graph_composition",
    "validated_edge_attr_mode",
    "validated_edge_weight_mode",
]
