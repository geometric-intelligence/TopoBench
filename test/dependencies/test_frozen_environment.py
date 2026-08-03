"""Static release policy for immutable dependencies and CI actions."""

import re
import tomllib
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CRITICAL_PINS = {
    "lightning": "2.4.0",
    "torch": "2.3.0",
    "torch-geometric": "2.8.0.post1",
}
_ACTION_PIN = re.compile(
    r"^\s*-?\s*uses:\s*[^\s@]+@[0-9a-f]{40}\s+#\s+\S+\s*$"
)
_EXPECTED_LOCAL_HOOKS = {
    "end-of-file-fixer": (
        "uv run --frozen --extra lint end-of-file-fixer"
    ),
    "trailing-whitespace": (
        "uv run --frozen --extra lint trailing-whitespace-fixer"
    ),
    "fix-byte-order-marker": (
        "uv run --frozen --extra lint fix-byte-order-marker"
    ),
    "check-case-conflict": (
        "uv run --frozen --extra lint check-case-conflict"
    ),
    "check-merge-conflict": (
        "uv run --frozen --extra lint check-merge-conflict"
    ),
    "check-ast": "uv run --frozen --extra lint check-ast",
    "check-json": "uv run --frozen --extra lint check-json",
    "check-yaml": "uv run --frozen --extra lint check-yaml",
    "check-symlinks": "uv run --frozen --extra lint check-symlinks",
    "mixed-line-ending": (
        "uv run --frozen --extra lint mixed-line-ending"
    ),
    "check-added-large-files": (
        "uv run --frozen --extra lint check-added-large-files"
    ),
    "requirements-txt-fixer": (
        "uv run --frozen --extra lint requirements-txt-fixer"
    ),
    "ruff-format": (
        "uv run --frozen --extra lint ruff format --force-exclude"
    ),
    "ruff": "uv run --frozen --extra lint ruff check --force-exclude",
    "numpydoc-validation": (
        "uv run --frozen --extra lint numpydoc lint"
    ),
}
_EXPECTED_LINT_PACKAGES = {
    "numpydoc",
    "pre-commit",
    "pre-commit-hooks",
    "ruff",
}


def test_critical_deserialization_dependencies_are_exactly_pinned() -> None:
    pyproject = tomllib.loads(
        (_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = pyproject["project"]["dependencies"]
    by_name = {
        requirement.split("==", maxsplit=1)[0].lower(): requirement
        for requirement in dependencies
        if "==" in requirement
    }

    assert {name: by_name.get(name) for name in _CRITICAL_PINS} == {
        name: f"{name}=={version}" for name, version in _CRITICAL_PINS.items()
    }
    lock = tomllib.loads(
        (_PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8")
    )
    assert "pre-commit" in {package["name"] for package in lock["package"]}


def test_lint_toolchain_is_declared_and_locked() -> None:
    pyproject = tomllib.loads(
        (_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    requirements = pyproject["project"]["optional-dependencies"]["lint"]
    lint_packages = {
        re.split(r"[<>=!~\[]", requirement, maxsplit=1)[0].lower()
        for requirement in requirements
    }
    assert lint_packages == _EXPECTED_LINT_PACKAGES

    lock = tomllib.loads(
        (_PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8")
    )
    locked_packages = {package["name"] for package in lock["package"]}
    assert _EXPECTED_LINT_PACKAGES <= locked_packages


def test_pre_commit_uses_only_the_frozen_local_toolchain() -> None:
    config = yaml.safe_load(
        (_PROJECT_ROOT / ".pre-commit-config.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["repos"] == [
        {
            "repo": "local",
            "hooks": config["repos"][0]["hooks"],
        }
    ]
    hooks = config["repos"][0]["hooks"]
    assert [hook["id"] for hook in hooks] == list(_EXPECTED_LOCAL_HOOKS)
    assert {hook["id"]: hook["entry"] for hook in hooks} == (
        _EXPECTED_LOCAL_HOOKS
    )
    assert all(hook["language"] == "system" for hook in hooks)
    assert all("additional_dependencies" not in hook for hook in hooks)


def test_numpydoc_keeps_the_narrow_validation_policy() -> None:
    pyproject = tomllib.loads(
        (_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert pyproject["tool"]["numpydoc_validation"] == {
        "checks": ["SS05", "PR02", "PR03"],
        "exclude": [
            r"\.undocumented_method$",
            r"\.__init__$",
            r"\.__repr__$",
        ],
    }


def test_environment_setup_only_syncs_the_committed_lock() -> None:
    script = (_PROJECT_ROOT / "uv_env_setup.sh").read_text(encoding="utf-8")

    assert re.search(r"uv sync\b[^\n]*--frozen", script)
    for forbidden in (
        "rm -f uv.lock",
        "uv lock",
        "uv pip install",
        'pyproject.toml"',
    ):
        assert forbidden not in script


def test_lint_workflow_runs_only_lock_backed_tools() -> None:
    lint_workflow = (
        _PROJECT_ROOT / ".github" / "workflows" / "lint.yml"
    ).read_text(encoding="utf-8")

    assert re.search(
        r"uses:\s*astral-sh/setup-uv@[0-9a-f]{40}\s+#\s+\S+",
        lint_workflow,
    )
    run_commands = re.findall(
        r"^\s+run:\s+(.+)$",
        lint_workflow,
        flags=re.MULTILINE,
    )
    assert run_commands == [
        "uv sync --frozen --all-extras",
        "uv run --frozen --extra lint ruff check",
        "uv run --frozen --extra lint ruff format --check",
        "uv sync --frozen --all-extras",
        "uv run --frozen --extra lint pre-commit run --all-files",
    ]
    assert all(
        command == "uv sync --frozen --all-extras"
        or command.startswith("uv run --frozen --extra lint ")
        for command in run_commands
    )
    assert "astral-sh/ruff-action@" not in lint_workflow
    assert "pre-commit/action@" not in lint_workflow


def test_every_github_action_is_commit_pinned_with_version_comment() -> None:
    workflow_dir = _PROJECT_ROOT / ".github" / "workflows"
    uses_lines = [
        line
        for workflow in workflow_dir.glob("*.yml")
        for line in workflow.read_text(encoding="utf-8").splitlines()
        if "uses:" in line
    ]

    assert uses_lines
    assert all(_ACTION_PIN.fullmatch(line) for line in uses_lines), uses_lines


def test_workflows_do_not_persist_checkout_credentials() -> None:
    workflow_dir = _PROJECT_ROOT / ".github" / "workflows"
    workflow_texts = [
        workflow.read_text(encoding="utf-8")
        for workflow in workflow_dir.glob("*.yml")
    ]

    checkout_steps = sum(
        text.count("uses: actions/checkout@") for text in workflow_texts
    )
    hardened_checkouts = sum(
        text.count("persist-credentials: false") for text in workflow_texts
    )
    assert checkout_steps > 0
    assert hardened_checkouts == checkout_steps
    assert all(
        not re.search(r"^permissions:", text, flags=re.MULTILINE)
        for text in workflow_texts
    )
