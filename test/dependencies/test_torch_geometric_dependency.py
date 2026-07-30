from pathlib import Path

import tomllib


def test_torch_geometric_is_a_direct_runtime_dependency() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    dependencies = pyproject["project"]["dependencies"]

    assert "torch-geometric==2.8.0.post1" in dependencies
