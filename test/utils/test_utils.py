"""Tests for general utility functions."""

from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from topobench.utils import utils as utils_module
from topobench.utils.utils import extras, get_metric_value, task_wrapper


def test_get_metric_value() -> None:
    """A configured metric is converted to a scalar value."""
    metric_dict = {"accuracy": torch.tensor([90])}

    assert get_metric_value(metric_dict, "accuracy") == 90.0
    assert get_metric_value(metric_dict, None) is None

    with pytest.raises(Exception, match="Metric value not found"):
        get_metric_value(metric_dict, "some_metric")


def test_extras_redacts_credentials_from_all_config_tree_outputs(
    tmp_path, capsys
) -> None:
    """Config printing never exposes nested credential values."""
    canaries = (
        "api-key-canary-738ef",
        "token-canary-315ca",
        "password-canary-942bd",
        "secret-canary-671fa",
    )
    cfg = OmegaConf.create(
        {
            "paths": {"output_dir": str(tmp_path)},
            "extras": {
                "ignore_warnings": False,
                "enforce_tags": False,
                "print_config": True,
            },
            "service": {
                "safe_setting": "visible-public-value",
                "credentials": {
                    "Api_Key_primary": canaries[0],
                    "nested": {
                        "AUTH_TOKEN_backup": canaries[1],
                        "databasePassword": canaries[2],
                        "client_SECRET_value": canaries[3],
                    },
                },
                "credential_reference": "${service.credentials.Api_Key_primary}",
            },
        }
    )
    source_config = OmegaConf.to_container(cfg, resolve=False)

    extras(cfg)

    console_output = capsys.readouterr()
    console_text = console_output.out + console_output.err
    config_tree_text = (tmp_path / "config_tree.log").read_text(
        encoding="utf-8"
    )

    for output in (console_text, config_tree_text):
        assert "visible-public-value" in output
        assert "<redacted>" in output
        for canary in canaries:
            assert canary not in output

    assert OmegaConf.to_container(cfg, resolve=False) == source_config


def test_hydra_does_not_persist_raw_config_snapshots() -> None:
    """Generated run metadata must not bypass application redaction."""
    config_path = (
        Path(__file__).parents[2] / "configs" / "hydra" / "default.yaml"
    )

    hydra_config = OmegaConf.load(config_path)

    assert hydra_config.output_subdir is None


def test_task_wrapper_returns_task_results(monkeypatch, tmp_path) -> None:
    """The wrapper preserves successful task results."""
    monkeypatch.setattr(utils_module, "find_spec", lambda _name: None)
    cfg = DictConfig({"paths": {"output_dir": str(tmp_path)}})

    def task_func(*, cfg: DictConfig):
        assert cfg.paths.output_dir == str(tmp_path)
        return {"accuracy": torch.tensor([90])}, {"model": "model"}

    metric_dict, object_dict = task_wrapper(task_func)(cfg)

    assert metric_dict["accuracy"] == 90.0
    assert object_dict["model"] == "model"


def test_task_wrapper_redacts_credential_alias_in_output_dir_log(
    monkeypatch, caplog
) -> None:
    """The final output-directory message cannot resolve credential aliases."""
    canary = "output-dir-api-key-canary-247fa"
    cfg = OmegaConf.create(
        {
            "service": {"api_key": canary},
            "paths": {"output_dir": "${service.api_key}"},
            "unrelated": "${missing.value}",
        }
    )
    monkeypatch.setattr(utils_module, "find_spec", lambda _name: None)
    caplog.set_level("INFO", logger="topobench.utils.utils")

    task_wrapper(lambda *, cfg: ({}, {}))(cfg)

    def failing_task(*, cfg: DictConfig) -> None:
        raise RuntimeError("expected failure")

    with pytest.raises(RuntimeError, match="expected failure"):
        task_wrapper(failing_task)(cfg)

    assert canary not in caplog.text
    assert caplog.text.count("Output dir: <redacted>") == 2


def test_task_wrapper_reraises_task_exception(monkeypatch, tmp_path) -> None:
    """The wrapper does not swallow a task failure."""
    monkeypatch.setattr(utils_module, "find_spec", lambda _name: None)
    cfg = DictConfig({"paths": {"output_dir": str(tmp_path)}})

    def failing_task(*, cfg: DictConfig):
        raise RuntimeError("Test exception")

    with pytest.raises(RuntimeError, match="Test exception"):
        task_wrapper(failing_task)(cfg)
