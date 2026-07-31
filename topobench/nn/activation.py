"""Shared activation-module construction."""

from __future__ import annotations

import torch

_ACTIVATIONS: dict[str, type[torch.nn.Module]] = {
    "relu": torch.nn.ReLU,
    "elu": torch.nn.ELU,
    "tanh": torch.nn.Tanh,
    "gelu": torch.nn.GELU,
    "id": torch.nn.Identity,
}


def make_activation(name: str) -> torch.nn.Module:
    """Create a fresh activation module from a supported exact name.

    Parameters
    ----------
    name : str
        One of ``"relu"``, ``"elu"``, ``"tanh"``, ``"gelu"``, or
        ``"id"``.

    Returns
    -------
    torch.nn.Module
        A new activation module instance.

    Raises
    ------
    TypeError
        If ``name`` is not a string.
    ValueError
        If ``name`` is not supported.
    """
    if not isinstance(name, str):
        raise TypeError("activation name must be a string")
    try:
        activation_type = _ACTIVATIONS[name]
    except KeyError as error:
        raise ValueError(f"Unsupported activation: {name!r}") from error
    return activation_type()


__all__ = ["make_activation"]
