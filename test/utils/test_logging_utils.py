"""Behavioral security contracts for logging utilities."""

from __future__ import annotations

import logging
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf

from topobench.utils.logging_utils import log_hyperparameters, redact_config

_REDACTED = "<redacted>"
_CANARIES = {
    "credentials": "credentials-canary-1A",
    "authorization": "authorization-canary-2B",
    "auth": "auth-canary-3C",
    "cookie": "cookie-canary-4D",
    "session": "session-canary-5E",
    "access_key_id": "access-key-id-canary-6F",
    "access-key": "access-key-canary-7G",
    "private_key": "private-key-canary-8H",
    "private-key": "private-key-hyphen-canary-9I",
    "client_secret": "client-secret-canary-0J",
    "signing_key": "signing-key-canary-1K",
    "signing-key": "signing-key-hyphen-canary-2L",
    "api_key": "api-key-canary-3M",
    "token": "token-canary-4N",
    "password": "password-canary-5O",
    "secret": "secret-canary-6P",
}


class _Parameter:
    def __init__(self, count: int, *, requires_grad: bool) -> None:
        self._count = count
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._count


class _Model:
    def __init__(self) -> None:
        self._parameters = (
            _Parameter(7, requires_grad=True),
            _Parameter(3, requires_grad=True),
            _Parameter(5, requires_grad=False),
        )

    def parameters(self):
        return iter(self._parameters)


class _CapturingLogger:
    """Capture the concrete payload received through the logger boundary."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        self.payloads.append(deepcopy(params))


def _credential_config() -> DictConfig:
    return OmegaConf.create(
        {
            "runtime": {
                "credentials": _CANARIES["credentials"],
                "authorization": _CANARIES["authorization"],
                "auth": _CANARIES["auth"],
                "cookie": _CANARIES["cookie"],
                "session": _CANARIES["session"],
                "providerApi_KeyCredential": _CANARIES["api_key"],
                "SessionToKeNValue": _CANARIES["token"],
            },
            "cloud": {
                "access_key_id": _CANARIES["access_key_id"],
                "access-key": _CANARIES["access-key"],
            },
            "cryptography": {
                "private_key": _CANARIES["private_key"],
                "private-key": _CANARIES["private-key"],
                "client_secret": _CANARIES["client_secret"],
                "signing_key": _CANARIES["signing_key"],
                "signing-key": _CANARIES["signing-key"],
            },
            "stages": [
                {
                    "name": "download",
                    "DbPassWordCredential": _CANARIES["password"],
                },
                {
                    "name": "publish",
                    "metadata": {
                        "SigningSeCrEtMaterial": _CANARIES["secret"],
                    },
                },
            ],
            "references": {
                "cloud_reference": "${cloud.access_key_id}",
                "nested": [
                    {"signer_reference": "${cryptography.signing_key}"}
                ],
            },
            "ordinary": {
                "project": "safe-project",
                "retries": 2,
                "tokenizer": "wordpiece",
                "passwordless_mode": "webauthn",
                "secretariat": "visible-office",
                "public_key": "visible-public-material",
            },
        }
    )


def _unresolved_snapshot(cfg: DictConfig) -> dict[str, Any]:
    snapshot = OmegaConf.to_container(cfg, resolve=False)
    assert isinstance(snapshot, dict)
    return snapshot


def _assert_canaries_absent(value: object) -> None:
    rendered = repr(value)
    for canary in _CANARIES.values():
        assert canary not in rendered


def _assert_redacted_config(payload: dict[str, Any]) -> None:
    for key in (
        "credentials",
        "authorization",
        "auth",
        "cookie",
        "session",
        "providerApi_KeyCredential",
        "SessionToKeNValue",
    ):
        assert payload["runtime"][key] == _REDACTED
    for key in ("access_key_id", "access-key"):
        assert payload["cloud"][key] == _REDACTED
    for key in (
        "private_key",
        "private-key",
        "client_secret",
        "signing_key",
        "signing-key",
    ):
        assert payload["cryptography"][key] == _REDACTED
    assert payload["stages"][0]["DbPassWordCredential"] == _REDACTED
    assert (
        payload["stages"][1]["metadata"]["SigningSeCrEtMaterial"] == _REDACTED
    )
    assert payload["references"] == {
        "cloud_reference": _REDACTED,
        "nested": [{"signer_reference": _REDACTED}],
    }
    assert payload["ordinary"] == {
        "project": "safe-project",
        "retries": 2,
        "tokenizer": "wordpiece",
        "passwordless_mode": "webauthn",
        "secretariat": "visible-office",
        "public_key": "visible-public-material",
    }
    _assert_canaries_absent(payload)


def test_redact_config_handles_nested_mixed_case_keys_without_mutation() -> (
    None
):
    cfg = _credential_config()
    before = _unresolved_snapshot(cfg)

    redacted = redact_config(cfg, resolve=True)

    assert isinstance(redacted, dict)
    _assert_redacted_config(redacted)
    assert _unresolved_snapshot(cfg) == before
    assert before["references"]["cloud_reference"] == "${cloud.access_key_id}"


def test_log_hyperparameters_redacts_every_logger_payload() -> None:
    cfg = _credential_config()
    before = _unresolved_snapshot(cfg)
    loggers = [_CapturingLogger(), _CapturingLogger()]
    trainer = SimpleNamespace(logger=loggers[0], loggers=loggers)

    log_hyperparameters(
        {
            "cfg": cfg,
            "model": _Model(),
            "trainer": trainer,
        }
    )

    for logger in loggers:
        assert len(logger.payloads) == 1
        payload = logger.payloads[0]
        _assert_redacted_config(payload)
        assert payload["model/params/total"] == 15
        assert payload["model/params/trainable"] == 10
        assert payload["model/params/non_trainable"] == 5

    assert _unresolved_snapshot(cfg) == before
    assert before["references"]["cloud_reference"] == "${cloud.access_key_id}"


def test_log_hyperparameters_without_logger_warns_without_leaking(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _credential_config()
    before = _unresolved_snapshot(cfg)
    trainer = SimpleNamespace(logger=None, loggers=[])

    with caplog.at_level(
        logging.WARNING, logger="topobench.utils.logging_utils"
    ):
        result = log_hyperparameters(
            {
                "cfg": cfg,
                "model": _Model(),
                "trainer": trainer,
            }
        )

    assert result is None
    assert any(
        "Logger not found! Skipping hyperparameter logging..." in message
        for message in caplog.messages
    )
    _assert_canaries_absent(caplog.text)
    assert _unresolved_snapshot(cfg) == before
