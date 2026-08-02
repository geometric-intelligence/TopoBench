"""Immutable native graph model capabilities and composition validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from topobench.data.capabilities import (
    FeaturePolicy,
    GRAPH_DATASET_MANIFEST,
    GraphDatasetCapability,
    GraphTaskContract,
    configured_graph_feature_width,
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
    requires_node_count: bool = False

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("tasks must not be empty")
        if not self.feature_policies:
            raise ValueError("feature_policies must not be empty")
        if self.edge_attr_mode not in _EDGE_MODES:
            raise ValueError(
                "edge_attr_mode must be consume, ignore, or reject"
            )
        if self.edge_weight_mode not in _EDGE_MODES:
            raise ValueError(
                "edge_weight_mode must be consume, ignore, or reject"
            )
        if not isinstance(self.requires_node_count, bool):
            raise TypeError("requires_node_count must be boolean")

    def supports(self, dataset: GraphDatasetCapability) -> bool:
        """Return whether this model accepts the complete dataset contract."""
        if dataset.task_contract not in self.tasks:
            return False
        if dataset.feature_policy not in self.feature_policies:
            return False
        if self.requires_node_count and dataset.num_nodes is None:
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
        requires_node_count=True,
    ),
)
GRAPH_MODEL_CAPABILITIES: Mapping[str, GraphModelCapability] = (
    MappingProxyType({row.selector: row for row in _ROWS})
)


def _configured_transforms(
    transforms: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    """Index composed transforms by their public constructor name."""
    if transforms is None:
        return {}
    if not isinstance(transforms, Mapping):
        raise TypeError("transforms must be a mapping or None")
    root_name = transforms.get("transform_name")
    if isinstance(root_name, str):
        return {root_name: transforms}
    declared: dict[str, Mapping[str, Any]] = {}
    for transform in transforms.values():
        if not isinstance(transform, Mapping):
            continue
        name = transform.get("transform_name")
        if isinstance(name, str):
            declared[name] = transform
    return declared


def validated_graph_feature_width(
    dataset: Mapping[str, Any],
    transforms: Mapping[str, Any] | None,
) -> int:
    """Validate one manifest-backed graph feature policy and output width."""
    capability = qualify_graph_dataset(dataset)
    configured_width = configured_graph_feature_width(dataset)
    if configured_width != capability.feature_width:
        raise ValueError(
            "dataset.parameters.num_features must declare qualified "
            f"feature width {capability.feature_width}, got "
            f"{configured_width}"
        )
    declared = _configured_transforms(transforms)
    if not declared and transforms is not None and "defaults" in transforms:
        return capability.feature_width
    known_feature_transforms = frozenset(
        name
        for row in GRAPH_DATASET_MANIFEST.values()
        for name in row.feature_transforms
    )
    actual = frozenset(declared) & known_feature_transforms
    feature_writers = frozenset(
        {"OneHotDegreeFeatures", "ConstantNodeFeatures"}
    )
    missing = capability.feature_transforms - actual
    wrong_writers = (actual & feature_writers) != (
        capability.feature_transforms & feature_writers
    )
    if missing or wrong_writers:
        raise ValueError(
            "dataset feature policy "
            f"(dataset.parameters.feature_policy={capability.feature_policy!r}) "
            "requires feature transforms "
            f"{sorted(capability.feature_transforms)!r}, got "
            f"{sorted(actual)!r}"
        )

    width = capability.feature_width
    if "OneHotDegreeFeatures" in declared:
        max_degree = declared["OneHotDegreeFeatures"].get("max_degree")
        if isinstance(max_degree, bool) or not isinstance(max_degree, int):
            raise TypeError(
                "transforms OneHotDegreeFeatures.max_degree must be an integer"
            )
        width = max_degree + 1
    elif "ConstantNodeFeatures" in declared:
        num_features = declared["ConstantNodeFeatures"].get("num_features")
        if isinstance(num_features, bool) or not isinstance(num_features, int):
            raise TypeError(
                "transforms ConstantNodeFeatures.num_features must be an "
                "integer"
            )
        width = num_features

    if width != capability.feature_width:
        raise ValueError(
            f"dataset feature width must resolve to {capability.feature_width}, "
            f"got {width}"
        )
    return width


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
    if (
        dataset_capability.feature_policy
        not in model_capability.feature_policies
    ):
        raise ValueError(
            "dataset.parameters.feature_policy="
            f"{dataset_capability.feature_policy!r} is unsupported by "
            f"model.model_name={model_selector!r}"
        )
    if (
        model_capability.requires_node_count
        and dataset_capability.num_nodes is None
    ):
        raise ValueError(
            f"model.model_name={model_selector!r} requires qualified "
            "dataset.parameters.num_nodes evidence"
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


def _expected_feature_width(
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
    capability: GraphDatasetCapability,
) -> int:
    """Resolve the authoritative composed width, including feature encodings."""
    get_root = getattr(model, "_get_root", None)
    if not callable(get_root):
        return capability.feature_width
    root = get_root()
    if not isinstance(root, Mapping) or "transforms" not in root:
        return capability.feature_width

    from topobench.utils.config_resolvers import infer_in_channels

    return infer_in_channels(dataset, root.get("transforms"))


def _positive_backbone_integer(
    backbone: Mapping[str, Any],
    key: str,
    *,
    minimum: int = 1,
) -> int:
    """Return one strict positive integer from a model backbone config."""
    value = _required_value(backbone, key, path="model.backbone")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"model.backbone.{key} must be an integer")
    if value < minimum:
        raise ValueError(
            f"model.backbone.{key} must be at least {minimum}, got {value}"
        )
    return value


def _validate_gcn_dgm_scale(
    dataset: GraphDatasetCapability,
    backbone: Mapping[str, Any],
) -> None:
    """Admit one exact GCN-DGM search before model construction or training."""
    node_count = dataset.num_nodes
    if node_count is None:
        raise ValueError(
            "model.model_name='gcn_dgm' requires qualified "
            "dataset.parameters.num_nodes evidence"
        )
    k = _positive_backbone_integer(backbone, "k", minimum=2)
    query_chunk_size = _positive_backbone_integer(
        backbone,
        "query_chunk_size",
    )
    feature_dim = _positive_backbone_integer(backbone, "hidden_channels")
    max_nodes = _positive_backbone_integer(backbone, "max_nodes")
    max_workspace_bytes = _positive_backbone_integer(
        backbone,
        "max_workspace_bytes",
    )
    if k >= node_count:
        raise ValueError(
            f"model.backbone.k={k} must be less than qualified "
            f"dataset node count {node_count}"
        )
    if max_nodes <= k:
        raise ValueError(
            f"model.backbone.max_nodes={max_nodes} must be greater than "
            f"model.backbone.k={k}"
        )
    if node_count > max_nodes:
        raise ValueError(
            f"qualified dataset node count {node_count} exceeds "
            f"model.backbone.max_nodes={max_nodes}"
        )
    if query_chunk_size > max_nodes:
        raise ValueError(
            "model.backbone.query_chunk_size must not exceed "
            "model.backbone.max_nodes"
        )

    from topobench.nn.backbones.graph.gcn_dgm import GCNDGM

    dataset_workspace = GCNDGM.estimate_workspace_bytes(
        node_count=node_count,
        query_chunk_size=query_chunk_size,
        k=k,
        feature_dim=feature_dim,
        element_size=8,
    )
    if dataset_workspace > max_workspace_bytes:
        raise ValueError(
            f"model.backbone.max_workspace_bytes={max_workspace_bytes} "
            f"cannot admit qualified dataset node count {node_count}; "
            f"requires {dataset_workspace} bytes"
        )
    declared_workspace = GCNDGM.estimate_workspace_bytes(
        node_count=max_nodes,
        query_chunk_size=query_chunk_size,
        k=k,
        feature_dim=feature_dim,
        element_size=8,
    )
    if declared_workspace > max_workspace_bytes:
        raise ValueError(
            f"model.backbone.max_workspace_bytes={max_workspace_bytes} "
            f"cannot admit model.backbone.max_nodes={max_nodes}; "
            f"requires {declared_workspace} bytes"
        )


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
    expected_in_channels = _expected_feature_width(
        dataset,
        model,
        dataset_capability,
    )
    in_channels = _required_value(
        encoder,
        "in_channels",
        path="model.feature_encoder",
    )
    if (
        isinstance(in_channels, bool)
        or not isinstance(in_channels, int)
        or in_channels <= 0
    ):
        raise ValueError(
            "model.feature_encoder.in_channels must resolve to a positive "
            "integer"
        )
    if in_channels != expected_in_channels:
        raise ValueError(
            "model.feature_encoder.in_channels must resolve to "
            f"{expected_in_channels}, got {in_channels}"
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
    if model_selector == "gcn_dgm":
        _validate_gcn_dgm_scale(dataset_capability, backbone)
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
        if mode_field not in wrapper:
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
        actual = _required_value(
            wrapper, f"{field}_mode", path="model.backbone_wrapper"
        )
        expected = getattr(model_capability, f"{field}_mode")
        if actual != expected:
            raise ValueError(
                f"{path} must resolve to {expected!r}, got {actual!r}"
            )
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
    "validated_graph_feature_width",
    "validated_edge_weight_mode",
]
