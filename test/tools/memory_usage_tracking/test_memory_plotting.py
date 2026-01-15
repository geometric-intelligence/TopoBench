"""Unit tests for memory usage tracking and plotting utilities."""

from __future__ import annotations

import builtins
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from tools.memory_usage_tracking.memory_plotting import (
    apply_ast_replacements,
    dataset_short,
    model_fs,
    monitor_script,
    plot_normalized_memory,
    plot_raw_time_memory,
)


class TestMemoryPlotting:
    """
    Test suite for memory usage tracking and plotting helpers.
    """

    def test_dataset_short(self):
        """
        Test dataset_short path splitting logic.

        Checks that the part after the first slash is returned when present.
        """
        assert dataset_short("graph/cora") == "cora"
        assert dataset_short("hypergraph/something") == "something"
        assert dataset_short("cora") == "cora"

    def test_model_fs(self):
        """
        Test model_fs path sanitization.

        Checks that slashes are replaced with double underscores.
        """
        assert model_fs("graph/gcn") == "graph__gcn"
        assert model_fs("deep/model/name") == "deep__model__name"
        assert model_fs("plainmodel") == "plainmodel"

    def test_apply_ast_replacements_injects_dataset_and_model(self, tmp_path: Path):
        """
        Test AST-based injection of DATASET and MODELS constants.

        Verifies that the returned code defines DATASET and MODELS as
        desired when executed.
        """
        original_content = (
            "DATASET = 'old_dataset'\n"
            "MODELS = ['old_model1', 'old_model2']\n"
        )
        dataset_to_inject = "graph/cora"
        model_to_inject = "graph/gcn"

        modified_code = apply_ast_replacements(
            original_content, dataset_to_inject, model_to_inject
        )

        # Execute the modified code and inspect the resulting globals.
        scope: dict[str, Any] = {}
        exec(modified_code, scope)  # noqa: S102 (exec used intentionally in test)

        assert scope["DATASET"] == dataset_to_inject
        assert scope["MODELS"] == [model_to_inject]

    def test_monitor_script_writes_csv_and_returns_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """
        Test monitor_script with mocked subprocess and psutil.

        Uses a fake pytest process and psutil.Process to avoid running
        real tests and to produce deterministic memory data.
        """
        # Create a minimal template script with DATASET and MODELS.
        template_path = tmp_path / "template.py"
        template_path.write_text(
            "DATASET = 'dummy'\n"
            "MODELS = ['dummy']\n"
            "def test_dummy():\n"
            "    assert True\n",
            encoding="utf-8",
        )

        # Fake child process returned by subprocess.Popen.
        class FakeProcess:
            """Fake process used to simulate pytest running."""

            def __init__(self):
                self.pid = 1234
                self._poll_calls = 0
                self.returncode = 0

            def poll(self):
                """Simulate process running then exiting."""
                self._poll_calls += 1
                # First call: still running, second call: finished.
                return None if self._poll_calls == 1 else self.returncode

            def wait(self):
                """Simulate waiting for process termination."""
                return self.returncode

        fake_proc = FakeProcess()

        # Fake psutil.Process wrapper.
        class FakePsutilProcess:
            """Fake psutil.Process to return deterministic memory info."""

            def __init__(self, pid: int):
                self.pid = pid

            def memory_info(self):
                """Return simple namespace with rss field."""
                return SimpleNamespace(rss=100 * 1024 * 1024)

        # Patch subprocess.Popen and psutil.Process.
        def fake_popen(args, cwd=None, env=None):
            return fake_proc

        import psutil as real_psutil
        import subprocess as real_subprocess

        monkeypatch.setattr(real_subprocess, "Popen", fake_popen)
        monkeypatch.setattr(real_psutil, "Process", FakePsutilProcess)

        # Output CSV path.
        output_csv = tmp_path / "mem.csv"

        # Call monitor_script.
        memory_data, return_code = monitor_script(
            script_path=str(template_path),
            dataset_to_inject="graph/cora",
            model_to_inject="graph/gcn",
            output_csv=str(output_csv),
            interval=0.001,
        )

        # Assertions on return values.
        assert return_code == 0
        assert isinstance(memory_data, list)
        assert len(memory_data) >= 1
        assert all(len(t) == 2 for t in memory_data)

        # Assertions on CSV content.
        assert output_csv.exists()
        df = pd.read_csv(output_csv)
        assert list(df.columns) == ["time_s", "memory_MB"]
        assert not df.empty

    def test_plot_normalized_memory_creates_file(self, tmp_path: Path):
        """
        Test normalized memory plotting from CSV.

        Writes a simple CSV and verifies that the plot file is created.
        """
        csv_path = tmp_path / "mem.csv"
        df = pd.DataFrame(
            {
                "time_s": [0.0, 0.5, 1.0],
                "memory_MB": [100.0, 120.0, 110.0],
            }
        )
        df.to_csv(csv_path, index=False)

        plot_path = tmp_path / "plot_norm.png"
        plot_normalized_memory(
            model_label="graph/gcn",
            csv_files=[str(csv_path)],
            labels=["cora"],
            plot_path=str(plot_path),
            colors=("blue",),
        )

        assert plot_path.exists()
        assert plot_path.stat().st_size > 0

    def test_plot_raw_time_memory_creates_file(self, tmp_path: Path):
        """
        Test raw time memory plotting from CSV.

        Writes a simple CSV and verifies that the plot file is created.
        """
        csv_path = tmp_path / "mem.csv"
        df = pd.DataFrame(
            {
                "time_s": [10.0, 10.5, 11.0],
                "memory_MB": [200.0, 210.0, 205.0],
            }
        )
        df.to_csv(csv_path, index=False)

        plot_path = tmp_path / "plot_raw.png"
        plot_raw_time_memory(
            model_label="graph/gcn",
            csv_files=[str(csv_path)],
            labels=["cora"],
            plot_path=str(plot_path),
            colors=("red",),
        )

        assert plot_path.exists()
        assert plot_path.stat().st_size > 0