"""Public contract for supported TopoBench domains."""

from typing import Final

SUPPORTED_DOMAINS: Final = ("graph", "heterogeneous", "hypergraph")


def require_supported_domain(domain: str) -> str:
    """Return a supported domain name or raise for an invalid value."""
    if not isinstance(domain, str):
        msg = "domain must be a string"
        raise TypeError(msg)
    if domain not in SUPPORTED_DOMAINS:
        msg = f"Unsupported domain {domain!r}; expected one of {SUPPORTED_DOMAINS}"
        raise ValueError(msg)
    return domain


__all__ = ["SUPPORTED_DOMAINS", "require_supported_domain"]
