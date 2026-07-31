"""Small serialization utilities without topology dependencies."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any


def ensure_serializable(obj: Any) -> Any:
    """Convert nested configuration containers into built-in containers."""
    if isinstance(obj, str | int | float | bool | type(None)):
        return obj
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = ensure_serializable(value)
        return obj
    if isinstance(obj, Mapping):
        return {key: ensure_serializable(value) for key, value in obj.items()}
    if isinstance(obj, Sequence):
        return [ensure_serializable(item) for item in obj]
    if isinstance(obj, set):
        return {ensure_serializable(item) for item in obj}
    return None


def make_hash(o: Any) -> int:
    """Return the existing stable 32-bit-range hash of an object."""
    digest = hashlib.sha1(str(o).encode()).hexdigest()
    return int(digest, 16) % 4294967295


__all__ = ["ensure_serializable", "make_hash"]
