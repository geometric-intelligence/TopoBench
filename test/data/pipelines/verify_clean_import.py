"""Verify the pipeline package imports in a pristine Python subprocess."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[3]
EXPECTED_EXPORTS = (
    "AbstractDataPipeline,DataPipelineOutput,DefaultDataPipeline"
)


def main() -> None:
    """Run the clean-import contract outside a torch-loaded pytest process."""
    code = (
        "import topobench.data.pipelines as pipelines; "
        "print(','.join(pipelines.__all__))"
    )
    child_env = os.environ.copy()
    child_env["OMP_NUM_THREADS"] = "1"

    # This child detects package import cycles. Running this driver before
    # pytest isolates libomp bootstrap from the torch-initialized pytest parent.
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            env=child_env,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            "Pipeline package failed to import in a clean subprocess "
            f"(exit code {error.returncode}).\n"
            f"stderr:\n{error.stderr or '<empty>'}"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise SystemExit(
            "Pipeline package clean import exceeded 30 seconds.\n"
            f"stderr:\n{error.stderr or '<empty>'}"
        ) from error

    actual_exports = completed.stdout.strip()
    if actual_exports != EXPECTED_EXPORTS:
        raise SystemExit(
            "Pipeline package exported an unexpected public API.\n"
            f"expected: {EXPECTED_EXPORTS}\n"
            f"actual:   {actual_exports or '<empty>'}\n"
            f"stderr:\n{completed.stderr or '<empty>'}"
        )


if __name__ == "__main__":
    main()
