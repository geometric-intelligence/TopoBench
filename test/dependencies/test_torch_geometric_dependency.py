import tomllib
from pathlib import Path


def test_torch_geometric_is_a_direct_runtime_dependency() -> None:
    project_root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads(
        (project_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = pyproject["project"]["dependencies"]

    assert "torch-geometric==2.8.0.post1" in dependencies
