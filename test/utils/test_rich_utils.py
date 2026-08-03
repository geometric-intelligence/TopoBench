"""Behavioral tests for rich_utils."""

from unittest.mock import patch

from omegaconf import DictConfig, OmegaConf

from topobench.utils.rich_utils import enforce_tags, print_config_tree


def test_print_config_tree_redacts_credentials_from_console_and_file(
    capsys, tmp_path
):
    """Credential values never reach either config-tree output."""
    canaries = {
        "credentials": "CANARY_CREDENTIALS_VALUE",
        "authorization": "CANARY_AUTHORIZATION_VALUE",
        "auth": "CANARY_AUTH_VALUE",
        "cookie": "CANARY_COOKIE_VALUE",
        "session": "CANARY_SESSION_VALUE",
        "access_key_id": "CANARY_ACCESS_KEY_ID_VALUE",
        "access-key": "CANARY_ACCESS_KEY_VALUE",
        "private_key": "CANARY_PRIVATE_KEY_VALUE",
        "private-key": "CANARY_PRIVATE_KEY_HYPHEN_VALUE",
        "client_secret": "CANARY_CLIENT_SECRET_VALUE",
        "signing_key": "CANARY_SIGNING_KEY_VALUE",
        "signing-key": "CANARY_SIGNING_KEY_HYPHEN_VALUE",
        "api_key": "CANARY_API_KEY_VALUE",
        "token": "CANARY_TOKEN_VALUE",
        "password": "CANARY_PASSWORD_VALUE",
        "secret": "CANARY_SECRET_VALUE",
    }
    cfg = DictConfig(
        {
            "dataset": {
                "name": "visible-dataset",
                "credentials": canaries["credentials"],
                "service_ApI_KeY": canaries["api_key"],
            },
            "model": {
                "name": "visible-model",
                "authorization": canaries["authorization"],
                "refresh_ToKeN": canaries["token"],
                "database_PaSsWoRd": canaries["password"],
            },
            "transforms": [
                {
                    "name": "visible-transform",
                    "auth": canaries["auth"],
                    "cookie": canaries["cookie"],
                }
            ],
            "callbacks": {
                "name": "visible-callback",
                "session": canaries["session"],
                "access_key_id": canaries["access_key_id"],
                "access-key": canaries["access-key"],
            },
            "logger": {
                "name": "visible-logger",
                "private_key": canaries["private_key"],
                "private-key": canaries["private-key"],
                "client_secret": canaries["client_secret"],
            },
            "trainer": {
                "max_epochs": 2,
                "signing_key": canaries["signing_key"],
                "signing-key": canaries["signing-key"],
                "SigningSeCrEtMaterial": canaries["secret"],
            },
            "paths": {"output_dir": str(tmp_path)},
            "extras": {
                "public_note": "safe-visible-value",
                "cloud_reference": "${callbacks.access_key_id}",
                "nested": [
                    {"signer_reference": "${trainer.signing_key}"}
                ],
                "tokenizer": "visible-tokenizer",
                "passwordless_mode": "visible-passwordless",
                "secretariat": "visible-secretariat",
                "public_key": "visible-public-key",
            },
            "task_name": "visible-task",
        }
    )
    source_before = OmegaConf.to_container(cfg, resolve=False)

    print_config_tree(cfg, resolve=True, save_to_file=True)

    captured = capsys.readouterr()
    console_output = captured.out + captured.err
    file_output = (tmp_path / "config_tree.log").read_text(encoding="utf-8")

    for output in (console_output, file_output):
        assert "<redacted>" in output
        assert "cloud_reference: <redacted>" in output
        assert "signer_reference: <redacted>" in output
        assert "visible-dataset" in output
        assert "visible-model" in output
        assert "safe-visible-value" in output
        assert "visible-tokenizer" in output
        assert "visible-passwordless" in output
        assert "visible-secretariat" in output
        assert "visible-public-key" in output
        assert "visible-task" in output
        for canary in canaries.values():
            assert canary not in output

    assert OmegaConf.to_container(cfg, resolve=False) == source_before
    assert OmegaConf.is_interpolation(cfg.extras, "cloud_reference")


@patch("topobench.utils.rich_utils.HydraConfig")
@patch(
    "topobench.utils.rich_utils.Prompt.ask",
    return_value="experiment, regression",
)
def test_enforce_tags_prompts_normalizes_and_saves(
    mock_prompt_ask, mock_hydra_config, tmp_path
):
    """Missing tags are collected, normalized, and persisted."""
    cfg = DictConfig({"paths": {"output_dir": str(tmp_path)}})
    mock_hydra_config.return_value.cfg.hydra.job = {}

    enforce_tags(cfg, save_to_file=True)

    assert cfg.tags == ["experiment", "regression"]
    saved_tags = (tmp_path / "tags.log").read_text(encoding="utf-8")
    assert "experiment" in saved_tags
    assert "regression" in saved_tags
    mock_prompt_ask.assert_called_once_with(
        "Enter a list of comma separated tags", default="dev"
    )


@patch("topobench.utils.rich_utils.Prompt.ask")
def test_enforce_tags_redacts_structured_credentials(
    mock_prompt_ask, capsys, tmp_path
) -> None:
    """Configured structured tags cannot bypass config redaction."""
    canary = "tag-api-key-canary-9d7bc"
    cfg = OmegaConf.create(
        {
            "paths": {"output_dir": str(tmp_path)},
            "service": {"api_key": canary},
            "tags": [
                {"api_key": canary},
                {"credential_alias": "${service.api_key}"},
                {"public_label": "visible-tag"},
            ],
            "unrelated": "${missing.value}",
        }
    )
    source_before = OmegaConf.to_container(cfg, resolve=False)

    enforce_tags(cfg, save_to_file=True)

    output = capsys.readouterr()
    saved_tags = (tmp_path / "tags.log").read_text(encoding="utf-8")
    assert canary not in output.out + output.err
    assert canary not in saved_tags
    assert "<redacted>" in saved_tags
    assert "visible-tag" in saved_tags
    assert OmegaConf.to_container(cfg, resolve=False) == source_before
    mock_prompt_ask.assert_not_called()
