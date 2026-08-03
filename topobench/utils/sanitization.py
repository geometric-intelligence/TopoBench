"""Shared credential-key recognition for observable runtime boundaries."""

from __future__ import annotations

import re

_KEY_SEPARATOR = re.compile(r"[^A-Za-z0-9]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?:authorization|authentication|bearer|"
    r"access[\s_-]*(?:token|key(?:[\s_-]*id)?)|"
    r"api[\s_-]*key|auth|client[\s_-]*secret|cookie|credentials?|oauth|"
    r"password|private[\s_-]*key|secret[\s_-]*key|session|"
    r"signing[\s_-]*key|token|secret)"
    r"(?:\s+|[\s_-]*[=:]\s*)[^/&?#\s]+",
    re.IGNORECASE,
)

_SENSITIVE_TOKENS = frozenset(
    {
        "auth",
        "authentication",
        "authorization",
        "bearer",
        "cookie",
        "credential",
        "oauth",
        "password",
        "secret",
        "session",
        "token",
    }
)
_SENSITIVE_KEY_PREFIXES = frozenset(
    {
        "access",
        "api",
        "auth",
        "oauth",
        "private",
        "secret",
        "session",
        "signing",
    }
)
_SENSITIVE_COMPACT_FRAGMENTS = (
    "accesskey",
    "apikey",
    "authkey",
    "clientsecret",
    "oauthkey",
    "privatekey",
    "secretkey",
    "accesstoken",
    "sessionkey",
    "authtoken",
    "bearercredential",
    "bearertoken",
    "refreshtoken",
    "sessiontoken",
    "signingkey",
    "signingsecret",
)
_SINGULAR_TOKENS = {
    "cookies": "cookie",
    "credentials": "credential",
    "keys": "key",
    "passwords": "password",
    "secrets": "secret",
    "sessions": "session",
    "tokens": "token",
}


def _normalized_tokens(key: str) -> tuple[str, ...]:
    separated = _CAMEL_BOUNDARY.sub("_", key)
    return tuple(
        _SINGULAR_TOKENS.get(token, token)
        for token in (
            part.casefold()
            for part in _KEY_SEPARATOR.sub("_", separated).split("_")
            if part
        )
    )


def is_sensitive_key(key: object) -> bool:
    """Return whether a mapping key conventionally carries credentials.

    Recognition is token- and credential-compound-based so ordinary words such
    as ``tokenizer``, ``passwordless``, and ``secretariat`` remain observable.
    """
    if not isinstance(key, str):
        return False

    tokens = _normalized_tokens(key)
    if not tokens:
        return False
    if _SENSITIVE_TOKENS.intersection(tokens):
        return True
    if any(
        left in _SENSITIVE_KEY_PREFIXES and right == "key"
        for left, right in zip(tokens, tokens[1:], strict=False)
    ):
        return True

    compact = "".join(tokens)
    sensitive_suffix = compact.endswith(
        ("credential", "password", "secret", "token")
    )
    return sensitive_suffix or any(
        fragment in compact for fragment in _SENSITIVE_COMPACT_FRAGMENTS
    )


def contains_sensitive_assignment(value: str) -> bool:
    """Return whether text embeds a conventional credential assignment."""
    return _SENSITIVE_ASSIGNMENT.search(value) is not None
