"""Instantiators for callbacks and loggers."""

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from topobench.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    r"""Instantiate callbacks from config.

    Parameters
    ----------
    callbacks_cfg : DictConfig
        A DictConfig object containing callback configurations.

    Returns
    -------
    list[Callback]
        A list of instantiated callbacks.
    """
    callbacks: list[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for cb_conf in callbacks_cfg.values():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    r"""Instantiate loggers from config.

    Parameters
    ----------
    logger_cfg : DictConfig
        A DictConfig object containing logger configurations.

    Returns
    -------
    list[Logger]
        A list of instantiated loggers.
    """
    logger: list[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for lg_conf in logger_cfg.values():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger
