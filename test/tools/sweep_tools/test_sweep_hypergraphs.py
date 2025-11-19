import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from matplotlib.figure import Figure

from tools.sweep_tools import sweep_hypergraphs as sh


class TestSweepHypergraphs(unittest.TestCase):
    """
    Unit tests for sweep_hypergraphs functionality.

    Each test validates correctness of the small, self-contained utilities
    used in the hyperedge coverage sweep pipeline.
    """

    def test_load_perm_to_global_missing(self):
        """
        Test that load_perm_to_global returns None when no mapping exists.
        """
        with tempfile.TemporaryDirectory() as tmp:
            handle = {"processed_dir": tmp}
            result = sh.load_perm_to_global(handle)
            self.assertIsNone(result)

    def test_load_perm_to_global_existing(self):
        """
        Test that load_perm_to_global loads and returns the perm array.
        """
        with tempfile.TemporaryDirectory() as tmp:
            processed_dir = Path(tmp)
            perm_dir = processed_dir / "perm_memmap"
            perm_dir.mkdir(parents=True, exist_ok=True)
            arr = np.array([2, 0, 1], dtype=np.int64)
            np.save(perm_dir / "perm_to_global.npy", arr)

            handle = {"processed_dir": str(processed_dir)}
            result = sh.load_perm_to_global(handle)
            self.assertIsInstance(result, torch.Tensor)
            self.assertTrue(torch.equal(result, torch.from_numpy(arr)))

    def test_resolve_true_global_ids_with_global_nid_and_perm(self):
        """
        Test resolving global node IDs when both global_nid and perm_to_global exist.
        """
        golden = MagicMock()
        golden.x_0 = torch.randn(3, 2)

        batch = MagicMock()
        batch.x = torch.randn(3, 2)
        batch.x_0 = torch.randn(3, 2)
        batch.global_nid = torch.tensor([2, 0, 1])

        perm_to_global = torch.tensor([10, 11, 12])

        result = sh.resolve_true_global_ids(golden, batch, perm_to_global)
        self.assertTrue(torch.equal(result, torch.tensor([12, 10, 11])))

    def test_resolve_true_global_ids_without_global_nid(self):
        """
        Test the fallback path where no global_nid attribute is present.
        """
        golden = MagicMock()
        batch = MagicMock()
        batch.x = torch.randn(4, 2)
        delattr(batch, "global_nid")

        result = sh.resolve_true_global_ids(golden, batch, None)
        self.assertTrue(torch.equal(result, torch.arange(4)))

    def test_hyperedges_from_incidence_hyperedges_dense(self):
        """
        Test hyperedge extraction from a dense incidence matrix.
        """
        class Data:
            pass

        data = Data()
        H = torch.tensor([
            [1, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ], dtype=torch.float32)
        data.incidence_hyperedges = H
        gids = torch.tensor([10, 11, 12, 13])

        hyperedges = sh.hyperedges_from_incidence_hyperedges(data, gids)
        self.assertEqual(hyperedges, {(10, 11, 12), (12, 13)})

    def test_hyperedges_from_incidence_hyperedges_sparse(self):
        """
        Test hyperedge extraction from a sparse incidence matrix.
        """
        class Data:
            pass

        data = Data()
        indices = torch.tensor([[0, 1, 2, 2, 3], [0, 0, 0, 1, 1]])
        values = torch.ones(indices.size(1))
        H_sparse = torch.sparse_coo_tensor(indices, values, size=(4, 2))
        data.incidence_hyperedges = H_sparse
        gids = torch.tensor([0, 1, 2, 3])

        hyperedges = sh.hyperedges_from_incidence_hyperedges(data, gids)
        self.assertEqual(hyperedges, {(0, 1, 2), (2, 3)})

    def test_compute_hyperedge_coverage_metrics_basic(self):
        """
        Test metrics for a case with one exact match and one partial match.
        """
        gold = [(0, 1, 2), (3, 4)]
        cand = [(0, 1, 2), (3,)]

        metrics = sh.compute_hyperedge_coverage_metrics(gold, cand)
        self.assertEqual(metrics["gold_n"], 2)
        self.assertEqual(metrics["cand_n"], 2)
        self.assertEqual(metrics["strict_match"], 1)
        self.assertEqual(metrics["partial_covered"], 1)
        self.assertAlmostEqual(metrics["strict_recall"], 0.5)
        self.assertAlmostEqual(metrics["partial_recall"], 0.5)

    def test_summarize_hyperedge_differences(self):
        """
        Test generating a readable summary of hyperedge differences.
        """
        gold = [(0, 1, 2), (3, 4)]
        cand = [(0, 1), (3, 4, 5)]

        summary = sh.summarize_hyperedge_differences(gold, cand)
        self.assertIn("match exactly", summary)
        self.assertIn("missing in the clustered pipeline", summary)
        self.assertIn("appear only in the clustered pipeline", summary)

    @patch("tools.sweep_tools.sweep_hypergraphs.build_golden_and_candidate_from_cfg")
    @patch("tools.sweep_tools.sweep_hypergraphs.load_perm_to_global")
    def test_run_single_experiment_basic(self, mock_load_perm, mock_build):
        """
        Test a minimal run_single_experiment evaluation path.
        """
        class Data:
            pass

        golden = Data()
        golden.x = torch.randn(3, 2)
        golden.incidence_hyperedges = torch.tensor([[1, 0], [1, 0], [0, 1]], dtype=torch.float32)

        batch = Data()
        batch.x = torch.randn(3, 2)
        batch.incidence_hyperedges = golden.incidence_hyperedges
        batch.global_nid = torch.tensor([0, 1, 2])

        handle = {"num_parts": 4}
        mock_build.return_value = (golden, [batch], handle)
        mock_load_perm.return_value = None

        cfg = MagicMock()
        metrics = sh.run_single_experiment(cfg, num_parts=4, bs=1)

        self.assertAlmostEqual(metrics["strict_recall"], 1.0)
        self.assertAlmostEqual(metrics["partial_recall"], 1.0)

    @patch("tools.sweep_tools.sweep_hypergraphs.run_single_experiment")
    @patch("tools.sweep_tools.sweep_hypergraphs.hydra.compose")
    @patch("tools.sweep_tools.sweep_hypergraphs.hydra.initialize")
    def test_run_sweep_calls_single_experiment(
        self, mock_init, mock_compose, mock_run_single
    ):
        """
        Test that run_sweep invokes run_single_experiment for each grid point.
        """
        mock_init.return_value.__enter__.return_value = None
        mock_init.return_value.__exit__.return_value = False
        mock_compose.return_value = MagicMock()
        mock_run_single.return_value = {
            "gold_n": 1,
            "cand_n": 1,
            "strict_match": 1,
            "partial_covered": 1,
            "strict_recall": 1.0,
            "partial_recall": 1.0,
        }

        with patch.object(sh, "run_sweep", wraps=sh.run_sweep):
            batch_sizes, num_parts_list, results = sh.run_sweep()
            self.assertIsInstance(batch_sizes, list)
            self.assertIsInstance(num_parts_list, list)
            self.assertIsInstance(results, dict)
            self.assertTrue(any(results.values()))

        self.assertTrue(mock_run_single.called)

    def test_build_metric_matrix(self):
        """
        Test forming the recall matrix with NaN entries for missing values.
        """
        batch_sizes = [1, 2]
        num_parts_list = [4, 8]
        results = {(1, 4): {"strict_recall": 0.5}, (2, 8): {"strict_recall": 1.0}}

        M = sh.build_metric_matrix(batch_sizes, num_parts_list, results, "strict_recall")
        self.assertEqual(M.shape, (2, 2))
        self.assertTrue(np.isnan(M[0, 1]))
        self.assertAlmostEqual(M[0, 0], 50.0)
        self.assertAlmostEqual(M[1, 1], 100.0)

    @patch.object(Figure, "savefig")
    def test_plot_two_heatmaps_runs_and_saves(self, mock_savefig):
        """
        Test that two-heatmap plot renders and triggers savefig().
        """
        batch_sizes = [1, 2]
        num_parts_list = [4, 8]
        strict_M = np.array([[50.0, np.nan], [75.0, 100.0]])
        partial_M = np.array([[60.0, np.nan], [80.0, 100.0]])

        sh.plot_two_heatmaps(batch_sizes, num_parts_list, strict_M, partial_M, save_prefix="test_cov")
        self.assertTrue(mock_savefig.called)
        args, kwargs = mock_savefig.call_args
        self.assertIn("test_cov_strict_vs_partial.png", args[0])

    @patch.object(Figure, "savefig")
    def test_plot_one_heatmap_runs_and_saves(self, mock_savefig):
        """
        Test plotting a single recall heatmap and saving the output.
        """
        batch_sizes = [1, 2]
        num_parts_list = [4, 8]
        strict_M = np.array([[50.0, 70.0], [80.0, 100.0]])

        fig, ax = sh.plot_one_heatmap(batch_sizes, num_parts_list, strict_M, save_prefix="test_one")
        self.assertIsInstance(fig, Figure)
        self.assertTrue(mock_savefig.called)
        args, kwargs = mock_savefig.call_args
        self.assertIn("test_one_strict.png", args[0])


if __name__ == "__main__":
    unittest.main()
