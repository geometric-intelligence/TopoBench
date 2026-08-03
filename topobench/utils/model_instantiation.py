"""Centralized Hydra model construction from validated runtime metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import partial
from numbers import Integral
from typing import Any, cast

import hydra
from lightning import LightningModule
from omegaconf import DictConfig, ListConfig, OmegaConf

from topobench.data import HeterogeneousDataSpec
from topobench.nn.capabilities import CapabilityValidation
from topobench.utils.instantiators import (
    ExecutionProfileRecord,
    validate_execution_profile,
    validate_profile_capability,
)

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
    # DictConfig.__contains__ resolves interpolations; keys() is intentionally
    # used so validation cannot execute a resolver.
    if key not in parent.keys():  # noqa: SIM118
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
    # See _required_child: direct membership can execute the field resolver.
    if field_name not in component.keys():  # noqa: SIM118
        raise ValueError(
            f"{field_path} must be explicitly declared null for "
            "runtime injection"
        )
    if OmegaConf.is_interpolation(component, field_name):
        raise ValueError(
            f"{field_path} must be a literal null for runtime injection; "
            "interpolations are forbidden"
        )
    if OmegaConf.is_missing(component, field_name):
        raise ValueError(
            f"{field_path} must be a literal null for runtime injection; "
            "missing values are forbidden"
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


def _resolved_model_mapping(model_cfg: DictConfig) -> dict[str, Any]:
    """Resolve trusted static model config while it retains its copied root."""
    resolved = OmegaConf.to_container(
        model_cfg,
        resolve=True,
        throw_on_missing=True,
    )
    if not isinstance(resolved, dict):
        raise TypeError("model must resolve to a mapping")
    return cast(dict[str, Any], resolved)


def _pop_component_config(
    model_mapping: dict[str, Any],
    *,
    component_name: str,
    runtime_fields: tuple[str, ...],
) -> dict[str, Any]:
    """Detach one trusted component config and remove runtime-only fields."""
    component = model_mapping.pop(component_name, None)
    path = f"model.{component_name}"
    if not isinstance(component, dict):
        raise TypeError(f"{path} must resolve to a mapping")
    component = dict(component)
    for field_name in runtime_fields:
        field_path = f"{path}.{field_name}"
        if field_name not in component:
            raise ValueError(f"{field_path} is required")
        del component[field_name]
    component["_partial_"] = True
    return component


def _instantiate_factory(
    factory_cfg: dict[str, Any],
    *,
    path: str,
) -> Any:
    """Instantiate a callable factory from trusted, data-free static config."""
    factory = hydra.utils.instantiate(factory_cfg)
    if not callable(factory):
        raise TypeError(f"{path} must instantiate a callable factory")
    return factory


def _instantiate_static_dependency(
    runtime_cfg: DictConfig,
    *,
    name: str,
) -> object:
    """Instantiate one trusted dependency with copied-root interpolation."""
    dependency_cfg = _require_dict_config(
        _required_child(runtime_cfg, name, parent_path="cfg"),
        path=f"cfg.{name}",
    )
    return hydra.utils.instantiate(dependency_cfg)


def _instantiate_heterogeneous_model(
    runtime_cfg: DictConfig,
    *,
    data_spec: HeterogeneousDataSpec,
    mode: str,
) -> LightningModule:
    """Construct components without exposing runtime graph strings to Hydra."""
    model_cfg = _require_dict_config(runtime_cfg.model, path="cfg.model")
    model_mapping = _resolved_model_mapping(model_cfg)

    encoder_factory = _instantiate_factory(
        _pop_component_config(
            model_mapping,
            component_name="feature_encoder",
            runtime_fields=("input_channels",),
        ),
        path="model.feature_encoder",
    )
    backbone_factory = _instantiate_factory(
        _pop_component_config(
            model_mapping,
            component_name="backbone",
            runtime_fields=("metadata",),
        ),
        path="model.backbone",
    )
    wrapper_factory = _instantiate_factory(
        _pop_component_config(
            model_mapping,
            component_name="backbone_wrapper",
            runtime_fields=("target_node_type",),
        ),
        path="model.backbone_wrapper",
    )
    readout_factory = _instantiate_factory(
        _pop_component_config(
            model_mapping,
            component_name="readout",
            runtime_fields=("target_node_type", "out_channels"),
        ),
        path="model.readout",
    )
    supervision_factory = _instantiate_factory(
        _pop_component_config(
            model_mapping,
            component_name="supervision_adapter",
            runtime_fields=("target_node_type", "mode"),
        ),
        path="model.supervision_adapter",
    )

    # All values below come from the validated processed graph. They are bound
    # only through ordinary Python calls, never through OmegaConf or Hydra.
    feature_encoder = encoder_factory(
        input_channels=data_spec.input_channels_dict
    )
    backbone = backbone_factory(metadata=data_spec.pyg_metadata())
    backbone_wrapper = partial(
        wrapper_factory,
        target_node_type=data_spec.target_node_type,
    )
    readout = readout_factory(
        target_node_type=data_spec.target_node_type,
        out_channels=data_spec.num_classes,
    )
    supervision_adapter = supervision_factory(
        target_node_type=data_spec.target_node_type,
        mode=mode,
    )

    model_mapping["_partial_"] = True
    model_factory = _instantiate_factory(model_mapping, path="model")
    model = model_factory(
        feature_encoder=feature_encoder,
        backbone=backbone,
        backbone_wrapper=backbone_wrapper,
        readout=readout,
        supervision_adapter=supervision_adapter,
        evaluator=_instantiate_static_dependency(
            runtime_cfg,
            name="evaluator",
        ),
        optimizer=_instantiate_static_dependency(
            runtime_cfg,
            name="optimizer",
        ),
        loss=_instantiate_static_dependency(runtime_cfg, name="loss"),
    )
    if not isinstance(model, LightningModule):
        raise TypeError("cfg.model must instantiate a LightningModule")
    return model


def instantiate_model(
    cfg: DictConfig,
    *,
    data_spec: HeterogeneousDataSpec | None,
    capability_validation: CapabilityValidation | None = None,
    profile_record: ExecutionProfileRecord | None = None,
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

    current_profile_record = validate_execution_profile(cfg)
    if profile_record is None:
        profile_record = current_profile_record
    elif not isinstance(profile_record, ExecutionProfileRecord):
        raise TypeError("profile_record must be an ExecutionProfileRecord")
    elif profile_record != current_profile_record:
        raise ValueError("profile_record is stale for the current config")

    if capability_validation is None:
        validate_profile_capability(
            cfg,
            profile_record=profile_record,
        )
        raise ValueError(
            "model construction requires an observed runtime capability"
        )
    if not isinstance(capability_validation, CapabilityValidation):
        raise TypeError("capability_validation must be a CapabilityValidation")
    if capability_validation.observed is None:
        validate_profile_capability(
            cfg,
            profile_record=profile_record,
        )
        raise ValueError(
            "model construction requires an observed runtime capability"
        )
    capability_validation = validate_profile_capability(
        cfg,
        profile_record=profile_record,
        observed=capability_validation.observed,
    )
    if data_spec is not None:
        observed = capability_validation.observed
        if data_spec.input_channels != observed.feature_widths:
            raise ValueError(
                "data_spec.input_channels must match "
                f"observed.feature_widths={observed.feature_widths!r}, got "
                f"{data_spec.input_channels!r}"
            )
        if data_spec.num_classes != observed.num_classes:
            raise ValueError(
                "data_spec.num_classes must match "
                f"observed.num_classes={observed.num_classes!r}, got "
                f"{data_spec.num_classes!r}"
            )
        if data_spec.target_node_type != observed.target_node_type:
            raise ValueError(
                "data_spec.target_node_type must match "
                "observed.target_node_type="
                f"{observed.target_node_type!r}, got "
                f"{data_spec.target_node_type!r}"
            )

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
    return _instantiate_heterogeneous_model(
        runtime_cfg,
        data_spec=data_spec,
        mode=mode,
    )


__all__ = ["instantiate_model"]
