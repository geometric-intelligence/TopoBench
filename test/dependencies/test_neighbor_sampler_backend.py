import subprocess
import sys
import textwrap
from pathlib import Path


def test_neighbor_loader_consumes_a_real_heterogeneous_batch() -> None:
    project_root = Path(__file__).resolve().parents[2]
    probe = textwrap.dedent(
        """
        import sys

        sys.path.insert(0, sys.argv[1])

        from torch_geometric.loader import NeighborLoader
        from torch_geometric.typing import WITH_TORCH_SPARSE

        from topobench.data.datasets import make_synthetic_heterogeneous_data

        assert WITH_TORCH_SPARSE, "torch-sparse sampling backend is unavailable"

        data = make_synthetic_heterogeneous_data()
        loader = NeighborLoader(
            data,
            num_neighbors=[2],
            input_nodes="paper",
            batch_size=4,
            shuffle=False,
        )
        batch = next(iter(loader))
        target_store = batch["paper"]

        assert hasattr(target_store, "batch_size")
        assert hasattr(target_store, "n_id")
        assert target_store.batch_size > 1
        assert target_store.n_id.numel() >= target_store.batch_size
        """
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(project_root)],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
