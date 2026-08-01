import re
import subprocess
import sys
import tomllib
from pathlib import Path

from test.architecture.test_domain_contract import FORBIDDEN_DEPENDENCIES


def _distribution_name(requirement: str) -> str:
    """Return the normalized distribution name from a requirement string."""
    name = re.split(r"[<>=!~;@\[]", requirement, maxsplit=1)[0]
    return name.strip().lower().replace("_", "-")


def test_forbidden_packages_are_not_direct_runtime_dependencies() -> None:
    project_root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads(
        (project_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    runtime_dependencies = {
        _distribution_name(requirement)
        for requirement in pyproject["project"]["dependencies"]
    }

    assert runtime_dependencies.isdisjoint(FORBIDDEN_DEPENDENCIES)


def test_supported_public_imports_do_not_require_forbidden_packages() -> None:
    project_root = Path(__file__).resolve().parents[2]
    probe = (
        "import runpy, sys; "
        "sys.path.insert(0, sys.argv[1]); "
        "runpy.run_module("
        "'test.architecture.verify_forbidden_imports', run_name='__main__')"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(project_root)],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
