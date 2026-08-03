"""Utilities for logging hyperparameters."""

from collections.abc import Mapping, Sequence
from typing import Any

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from topobench.utils import pylogger
from topobench.utils.sanitization import is_sensitive_key

log = pylogger.RankedLogger(__name__, rank_zero_only=True)
REDACTED_VALUE = "<redacted>"


def _redact_container(value: object) -> object:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=False)

    if isinstance(value, Mapping):
        return {
            key: REDACTED_VALUE
            if is_sensitive_key(key)
            else _redact_container(item)
            for key, item in value.items()
        }

    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_redact_container(item) for item in value]

    return value


def redact_config(value: object, *, resolve: bool) -> object:
    """Return a plain logging-safe copy of a possibly nested configuration."""
    is_omegaconf_config = OmegaConf.is_config(value)
    if is_omegaconf_config:
        value = OmegaConf.to_container(value, resolve=False)

    redacted = _redact_container(value)
    if not (resolve and is_omegaconf_config):
        return redacted

    safe_config = OmegaConf.create(redacted)
    return OmegaConf.to_container(safe_config, resolve=True)


def redact_config_value(value: object, path: str) -> object:
    """Redact a config, then resolve only the selected value."""
    redacted = redact_config(value, resolve=False)
    if not isinstance(redacted, Mapping):
        raise TypeError("The redacted config root must be a mapping")

    safe_config = OmegaConf.create(redacted)
    selected = OmegaConf.select(safe_config, path, throw_on_missing=True)
    if OmegaConf.is_config(selected):
        return OmegaConf.to_container(selected, resolve=True)
    return selected


@rank_zero_only
def log_hyperparameters(object_dict: dict[str, Any]) -> None:
    r"""Control which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    Parameters
    ----------
    object_dict : dict[str, Any]
        A dictionary containing the following objects:
            - `"cfg"`: A DictConfig object containing the main config.
            - `"model"`: The Lightning model.
            - `"trainer"`: The Lightning trainer.
    """
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    cfg = redact_config(object_dict["cfg"], resolve=True)
    if not isinstance(cfg, Mapping):
        raise TypeError("The resolved logging config root must be a mapping")

    hparams = dict(cfg)

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
