"""Composition and residue checks for current user-facing examples."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import hydra
import pytest
from hydra.core.global_hydra import GlobalHydra
from hydra.errors import HydraException

from test.architecture.test_domain_contract import FORBIDDEN_TOKENS
from topobench.utils.config_resolvers import register_all_resolvers

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CURRENT_DOC_ROOTS = (
    _PROJECT_ROOT / "README.md",
    _PROJECT_ROOT / "docs",
    _PROJECT_ROOT / "tutorials",
    _PROJECT_ROOT / "scripts",
)
_DOCUMENT_SUFFIXES = {".md", ".rst"}
_ARTIFACT_SUFFIXES = {".md", ".rst", ".ipynb", ".py", ".sh"}
_SHELL_BLOCK = re.compile(
    r"```(?:bash|console|shell|sh)\s*\n(?P<body>.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)
_FORBIDDEN_CURRENT_SURFACES = (
    *FORBIDDEN_TOKENS,
    "cell",
    "simplicial",
    "combinatorial",
    "lifting",
)


def _current_document_paths() -> list[Path]:
    paths = [_PROJECT_ROOT / "README.md"]
    paths.extend(
        path
        for path in (_PROJECT_ROOT / "docs").rglob("*")
        if path.is_file()
        and path.suffix.lower() in _DOCUMENT_SUFFIXES
        and "plans" not in path.relative_to(_PROJECT_ROOT / "docs").parts
    )
    return sorted(paths)


def _artifact_paths() -> list[Path]:
    paths = [_PROJECT_ROOT / "README.md"]
    for root in _CURRENT_DOC_ROOTS[1:]:
        if not root.exists():
            continue
        paths.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower()
            in (
                _DOCUMENT_SUFFIXES
                if root.name == "docs"
                else _ARTIFACT_SUFFIXES
            )
            and not (
                root.name == "docs" and "plans" in path.relative_to(root).parts
            )
        )
    return sorted(set(paths))


def _logical_shell_commands(block: str) -> list[str]:
    commands: list[str] = []
    pending = ""
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if line.startswith("$ "):
            line = line[2:].lstrip()
        if not line or line.startswith("#"):
            continue
        pending = f"{pending} {line}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        commands.append(pending)
        pending = ""
    if pending:
        commands.append(pending)
    return commands


def _topobench_overrides(command: str) -> list[str] | None:
    words = shlex.split(command)
    module_index = next(
        (
            index
            for index, word in enumerate(words)
            if word in {"topobench", "topobench.run"}
            and index > 0
            and words[index - 1] == "-m"
        ),
        None,
    )
    if module_index is None:
        module_index = next(
            (
                index
                for index, word in enumerate(words)
                if word == "topobench"
                and (index == 0 or words[index - 1] == "run")
            ),
            None,
        )
    if module_index is None:
        return None
    return [word for word in words[module_index + 1 :] if word != "-m"]


def _documented_commands() -> list[tuple[Path, str, list[str]]]:
    documented: list[tuple[Path, str, list[str]]] = []
    for path in _current_document_paths():
        text = path.read_text(encoding="utf-8")
        for match in _SHELL_BLOCK.finditer(text):
            for command in _logical_shell_commands(match.group("body")):
                overrides = _topobench_overrides(command)
                if overrides is not None:
                    documented.append((path, command, overrides))
    return documented


def _selector_path(kind: str, value: str) -> Path:
    return _PROJECT_ROOT / "configs" / kind / f"{value}.yaml"


def test_documented_hydra_commands_select_existing_configs_and_compose() -> (
    None
):
    """Every advertised invocation must be an executable strict-Hydra example."""
    documented = _documented_commands()
    assert documented, "No current TopoBench shell commands were documented"
    assert any(path.name == "README.md" for path, _, _ in documented)

    register_all_resolvers()
    for path, command, overrides in documented:
        selectors = {
            key: value
            for word in overrides
            if "=" in word
            for key, value in [word.split("=", maxsplit=1)]
            if key in {"dataset", "experiment", "model"}
        }
        for kind, value in selectors.items():
            assert _selector_path(kind, value).is_file(), (
                f"{path.relative_to(_PROJECT_ROOT)} advertises missing "
                f"{kind} selector {value!r}: {command}"
            )

        GlobalHydra.instance().clear()
        try:
            with hydra.initialize_config_dir(
                version_base="1.3",
                config_dir=str(_PROJECT_ROOT / "configs"),
                job_name="test_current_documentation",
            ):
                try:
                    hydra.compose(config_name="run.yaml", overrides=overrides)
                except HydraException as error:
                    pytest.fail(
                        f"{path.relative_to(_PROJECT_ROOT)} contains an invalid "
                        f"Hydra command: {command}\n{error}"
                    )
        finally:
            GlobalHydra.instance().clear()


def test_current_artifacts_do_not_advertise_removed_surfaces() -> None:
    """Current user-facing artifacts expose only the supported product surface."""
    violations: list[str] = []
    for path in _artifact_paths():
        text = path.read_text(encoding="utf-8", errors="ignore").casefold()
        violations.extend(
            f"{path.relative_to(_PROJECT_ROOT)} contains {token!r}"
            for token in _FORBIDDEN_CURRENT_SURFACES
            if token in text
        )
    assert not violations, "\n".join(violations)
