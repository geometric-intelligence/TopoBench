"""Centralized Hydra model construction from validated runtime metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from numbers import Integral
from typing import Any

import hydra
from lightning import LightningModule
from omegaconf import DictConfig, ListConfig, flag_override, open_dict
from omegaconf.nodes import AnyNode

from topobench.data import HeterogeneousDataSpec

_HETEROGENEOUS_DOMAIN = "heterogeneous"
_SAMPLING_MODES = {"full_batch", "neighbor"}
_RUNTIME_PLACEHOLDERS = (
    ("feature_encoder", "input_channels"),
    ("backbone", "metadata"),
    ("backbone_wrapper", "target_node_type"),
    ("readout", "target_node_type"),
    ("readout", "out_channels"),
    ("supervision_adapter", "target_node_type"),
    ("supervision_adapter", "mode"),
)


def _require_dict_config(
    value: object,
    *,
    path: str,
) -> DictConfig:
    """Require one mapping-shaped Hydra configuration node."""
    if not isinstance(value, DictConfig):
        raise TypeError(f"{path} must be a DictConfig")
    return value


def _required_child(
    parent: DictConfig,
    key: str,
    *,
    parent_path: str,
) -> Any:
    """Read one required configuration key with a path-rich error."""
    path = f"{parent_path}.{key}"
    if key not in parent:
        raise ValueError(f"{path} is required")
    return parent[key]


def _model_domain(cfg: DictConfig) -> str:
    """Return the optional model-domain discriminator."""
    model_cfg = _require_dict_config(
        _required_child(cfg, "model", parent_path="cfg"),
        path="cfg.model",
    )
    value = model_cfg.get("model_domain", "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError("model.model_domain must be a string")
    return value


def _require_runtime_placeholder(
    model_cfg: DictConfig,
    *,
    component_name: str,
    field_name: str,
) -> None:
    """Require an explicitly declared null field for runtime injection."""
    component_path = f"model.{component_name}"
    component = _require_dict_config(
        _required_child(
            model_cfg,
            component_name,
            parent_path="model",
        ),
        path=component_path,
    )
    field_path = f"{component_path}.{field_name}"
    if field_name not in component:
        raise ValueError(
            f"{field_path} must be explicitly declared null for "
            "runtime injection"
        )
    value = component[field_name]
    if value is not None:
        raise ValueError(
            f"{field_path} must be null for runtime injection; "
            f"received {value!r}"
        )


def _sampling_mode(runtime_cfg: DictConfig) -> str:
    """Return a validated heterogeneous loader mode."""
    dataset_cfg = _require_dict_config(
        _required_child(runtime_cfg, "dataset", parent_path="cfg"),
        path="cfg.dataset",
    )
    dataloader_cfg = _require_dict_config(
        _required_child(
            dataset_cfg,
            "dataloader_params",
            parent_path="cfg.dataset",
        ),
        path="cfg.dataset.dataloader_params",
    )
    mode = _required_child(
        dataloader_cfg,
        "mode",
        parent_path="cfg.dataset.dataloader_params",
    )
    if not isinstance(mode, str):
        raise TypeError("dataset.dataloader_params.mode must be a string")
    if mode not in _SAMPLING_MODES:
        raise ValueError(
            "dataset.dataloader_params.mode must be one of "
            f"{sorted(_SAMPLING_MODES)!r}; received {mode!r}"
        )
    return mode


def _fanout_depths(value: object) -> list[int]:
    """Collect non-empty global or relation-specific fanout depths.

    Fanout *values* are deliberately not interpreted here. Their positivity,
    scalar type, and relation-key agreement belong to the data module.
    """
    path = "dataset.dataloader_params.num_neighbors"
    if value is None:
        raise ValueError(f"{path} must be configured in neighbor mode")
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{path} must be a non-empty list or relation mapping")
    if isinstance(value, (DictConfig, Mapping)):
        if not value:
            raise ValueError(f"{path} relation mapping must not be empty")
        depths: list[int] = []
        for relation, relation_fanout in value.items():
            if isinstance(relation_fanout, (str, bytes)) or not isinstance(
                relation_fanout,
                (ListConfig, Sequence),
            ):
                raise TypeError(
                    f"{path}[{relation!r}] must be a non-empty list"
                )
            if not relation_fanout:
                raise ValueError(f"{path}[{relation!r}] must not be empty")
            depths.append(len(relation_fanout))
        return depths
    if not isinstance(value, (ListConfig, Sequence)):
        raise TypeError(f"{path} must be a non-empty list or relation mapping")
    if not value:
        raise ValueError(f"{path} must not be empty")
    return [len(value)]


def _positive_model_depth(model_cfg: DictConfig) -> int:
    """Return a non-boolean positive backbone depth."""
    backbone_cfg = _require_dict_config(
        _required_child(model_cfg, "backbone", parent_path="model"),
        path="model.backbone",
    )
    depth = _required_child(
        backbone_cfg,
        "num_layers",
        parent_path="model.backbone",
    )
    if isinstance(depth, bool) or not isinstance(depth, Integral):
        raise TypeError("model.backbone.num_layers must be a positive integer")
    normalized = int(depth)
    if normalized < 1:
        raise ValueError(
            "model.backbone.num_layers must be a positive integer"
        )
    return normalized


def _validate_sampling_depth(
    runtime_cfg: DictConfig,
    *,
    mode: str,
) -> None:
    """Require neighbor fanout depth to match message-passing depth."""
    if mode == "full_batch":
        return

    dataset_cfg = _require_dict_config(runtime_cfg.dataset, path="cfg.dataset")
    dataloader_cfg = _require_dict_config(
        dataset_cfg.dataloader_params,
        path="cfg.dataset.dataloader_params",
    )
    subgraph_type = _required_child(
        dataloader_cfg,
        "subgraph_type",
        parent_path="cfg.dataset.dataloader_params",
    )
    if subgraph_type != "directional":
        raise ValueError(
            "Heterogeneous neighbor mode supports only "
            "dataset.dataloader_params.subgraph_type='directional'"
        )
    fanout = _required_child(
        dataloader_cfg,
        "num_neighbors",
        parent_path="cfg.dataset.dataloader_params",
    )
    observed_depths = _fanout_depths(fanout)
    model_cfg = _require_dict_config(runtime_cfg.model, path="cfg.model")
    model_depth = _positive_model_depth(model_cfg)
    if any(depth != model_depth for depth in observed_depths):
        raise ValueError(
            "Neighbor sampling depth does not match message-passing depth: "
            f"model depth={model_depth}, "
            f"observed fanout depths={observed_depths}; remedy: make every "
            "num_neighbors list length equal model.backbone.num_layers"
        )


def _inject_runtime_metadata(
    model_cfg: DictConfig,
    *,
    data_spec: HeterogeneousDataSpec,
    mode: str,
) -> None:
    """Inject fresh validated values into the copied configuration root."""
    with open_dict(model_cfg.feature_encoder):
        model_cfg.feature_encoder.input_channels = (
            data_spec.input_channels_dict
        )
    # OmegaConf normally normalizes relation tuples into lists, but PyG's
    # public Metadata contract uses relation tuples. Store this validated
    # runtime-only object atomically so Hydra passes the exact native shape.
    with (
        open_dict(model_cfg.backbone),
        flag_override(model_cfg.backbone, "allow_objects", True),
    ):
        model_cfg.backbone.metadata = AnyNode(
            value=data_spec.pyg_metadata(),
            flags={"allow_objects": True},
        )
    with open_dict(model_cfg.backbone_wrapper):
        model_cfg.backbone_wrapper.target_node_type = (
            data_spec.target_node_type
        )
    with open_dict(model_cfg.readout):
        model_cfg.readout.target_node_type = data_spec.target_node_type
        model_cfg.readout.out_channels = data_spec.num_classes
    with open_dict(model_cfg.supervision_adapter):
        model_cfg.supervision_adapter.target_node_type = (
            data_spec.target_node_type
        )
        model_cfg.supervision_adapter.mode = mode


def _instantiate_from_root(runtime_cfg: DictConfig) -> LightningModule:
    """Instantiate a model while retaining copied-root interpolation context."""
    model = hydra.utils.instantiate(
        runtime_cfg.model,
        evaluator=runtime_cfg.evaluator,
        optimizer=runtime_cfg.optimizer,
        loss=runtime_cfg.loss,
    )
    if not isinstance(model, LightningModule):
        raise TypeError("cfg.model must instantiate a LightningModule")
    return model


def instantiate_model(
    cfg: DictConfig,
    *,
    data_spec: HeterogeneousDataSpec | None,
) -> LightningModule:
    """Instantiate a homogeneous or runtime-configured heterogeneous model.

    The heterogeneous path deep-copies the entire Hydra root before validation
    and injection. This preserves absolute interpolation context and guarantees
    that success and every error path leave the composed source untouched.
    """
    if not isinstance(cfg, DictConfig):
        raise TypeError("cfg must be a DictConfig")
    if data_spec is not None and not isinstance(
        data_spec,
        HeterogeneousDataSpec,
    ):
        raise TypeError("data_spec must be a HeterogeneousDataSpec or None")

    model_domain = _model_domain(cfg)
    if data_spec is None:
        if model_domain == _HETEROGENEOUS_DOMAIN:
            raise ValueError(
                "A heterogeneous model requires a validated data specification"
            )
        model = hydra.utils.instantiate(
            cfg.model,
            evaluator=cfg.evaluator,
            optimizer=cfg.optimizer,
            loss=cfg.loss,
        )
        if not isinstance(model, LightningModule):
            raise TypeError("cfg.model must instantiate a LightningModule")
        return model

    if model_domain != _HETEROGENEOUS_DOMAIN:
        raise ValueError(
            "A heterogeneous data specification requires "
            "model.model_domain='heterogeneous'"
        )

    runtime_cfg = deepcopy(cfg)
    model_cfg = _require_dict_config(runtime_cfg.model, path="cfg.model")
    mode = _sampling_mode(runtime_cfg)
    _validate_sampling_depth(runtime_cfg, mode=mode)
    for component_name, field_name in _RUNTIME_PLACEHOLDERS:
        _require_runtime_placeholder(
            model_cfg,
            component_name=component_name,
            field_name=field_name,
        )
    _inject_runtime_metadata(model_cfg, data_spec=data_spec, mode=mode)
    return _instantiate_from_root(runtime_cfg)


__all__ = ["instantiate_model"]
