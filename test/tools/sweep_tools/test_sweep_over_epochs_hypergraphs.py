import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from matplotlib.figure import Figure

import tools.sweep_tools.sweep_over_epochs_hypergraphs as seh


class TestSweepOverEpochsHypergraph(unittest.TestCase):
    """
    Unit tests for sweep_over_epochs_hypergraph utilities.

    Covers permutation loading, ID resolution, hyperedge extraction,
    metrics, multi-epoch runs, sweeping, matrix building, and plotting.
    """

    def test_load_perm_to_global_missing(self):
        """
        Test load_perm_to_global returns None when file is missing.
        """
        with tempfile.TemporaryDirectory() as tmp:
            handle = {"processed_dir": tmp}
            result = seh.load_perm_to_global(handle)
            self.assertIsNone(result)

    def test_load_perm_to_global_existing(self):
        """
        Test load_perm_to_global loads a tensor when file exists.
        """
        with tempfile.TemporaryDirectory() as tmp:
            processed_dir = Path(tmp)
            perm_dir = processed_dir / "perm_memmap"
            perm_dir.mkdir(parents=True, exist_ok=True)
            arr = np.array([2, 0, 1], dtype=np.int64)
            np.save(perm_dir / "perm_to_global.npy", arr)

            handle = {"processed_dir": str(processed_dir)}
            result = seh.load_perm_to_global(handle)

            self.assertIsInstance(result, torch.Tensor)
            self.assertTrue(torch.equal(result, torch.from_numpy(arr)))

    def test_resolve_true_global_ids_with_global_nid_and_perm(self):
        """
        Test resolve_true_global_ids uses perm_to_global mapping.
        """
        golden = MagicMock()
        golden.x_0 = torch.randn(3, 2)

        batch = MagicMock()
        batch.x = torch.randn(3, 2)
        batch.x_0 = torch.randn(3, 2)
        batch.global_nid = torch.tensor([2, 0, 1])

        perm_to_global = torch.tensor([10, 11, 12])

        out = seh.resolve_true_global_ids(golden, batch, perm_to_global)
        self.assertTrue(torch.equal(out, torch.tensor([12, 10, 11])))

    def test_resolve_true_global_ids_without_global_nid(self):
        """
        Test resolve_true_global_ids falls back to arange for missing IDs.
        """
        golden = MagicMock()
        batch = MagicMock()
        batch.x = torch.randn(4, 2)
        if hasattr(batch, "global_nid"):
            delattr(batch, "global_nid")

        out = seh.resolve_true_global_ids(golden, batch, None)
        self.assertTrue(torch.equal(out, torch.arange(4)))

    def test_hyperedges_from_incidence_hyperedges_dense(self):
        """
        Test hyperedges_from_incidence_hyperedges with dense incidence.
        """
        class Data:
            pass

        data = Data()
        # 4 nodes, 2 hyperedges: {0,1,2}, {2,3}
        H = torch.tensor(
            [
                [1, 0],
                [1, 0],
                [1, 1],
                [0, 1],
            ],
            dtype=torch.float32,
        )
        data.incidence_hyperedges = H
        gids = torch.tensor([10, 11, 12, 13])

        hyperedges = seh.hyperedges_from_incidence_hyperedges(data, gids)
        self.assertEqual(
            hyperedges,
            {
                (10, 11, 12),
                (12, 13),
            },
        )

    def test_hyperedges_from_incidence_hyperedges_sparse(self):
        """
        Test hyperedges_from_incidence_hyperedges with sparse incidence.
        """
        class Data:
            pass

        data = Data()
        indices = torch.tensor([[0, 1, 2, 2, 3], [0, 0, 0, 1, 1]])
        values = torch.ones(indices.size(1))
        H_sparse = torch.sparse_coo_tensor(indices, values, size=(4, 2))
        data.incidence_hyperedges = H_sparse
        gids = torch.tensor([0, 1, 2, 3])

        hyperedges = seh.hyperedges_from_incidence_hyperedges(data, gids)
        self.assertEqual(hyperedges, {(0, 1, 2), (2, 3)})

    def test_compute_hyperedge_coverage_metrics_basic(self):
        """
        Test compute_hyperedge_coverage_metrics on simple overlap.
        """
        gold = [(0, 1, 2), (3, 4)]
        cand = [(0, 1, 2), (3,)]

        m = seh.compute_hyperedge_coverage_metrics(gold, cand)
        self.assertEqual(m["gold_n"], 2)
        self.assertEqual(m["cand_n"], 2)
        self.assertEqual(m["strict_match"], 1)
        self.assertEqual(m["partial_covered"], 1)
        self.assertAlmostEqual(m["strict_recall"], 0.5)
        self.assertAlmostEqual(m["partial_recall"], 0.5)

    def test_summarize_hyperedge_differences(self):
        """
        Test summarize_hyperedge_differences includes key phrases.
        """
        gold = [(0, 1, 2), (3, 4)]
        cand = [(0, 1), (3, 4, 5)]

        summary = seh.summarize_hyperedge_differences(gold, cand)
        self.assertIn("match exactly", summary)
        self.assertIn("missing in the clustered pipeline", summary)
        self.assertIn("appear only in the clustered pipeline", summary)

    @patch("tools.sweep_tools.sweep_over_epochs_hypergraphs.build_golden_and_loader_from_cfg")
    @patch("tools.sweep_tools.sweep_over_epochs_hypergraphs.load_perm_to_global")
    def test_run_single_experiment_multi_epoch_basic(self, mock_load_perm, mock_build):
        """
        Test run_single_experiment_multi_epoch aggregates metrics per epoch.
        """
        class Data:
            pass

        golden = Data()
        golden.x = torch.randn(3, 2)
        golden.incidence_hyperedges = torch.tensor(
            [
                [1, 0],
                [1, 0],
                [0, 1],
            ],
            dtype=torch.float32,
        )

        batch = Data()
        batch.x = torch.randn(3, 2)
        batch.incidence_hyperedges = golden.incidence_hyperedges
        batch.global_nid = torch.tensor([0, 1, 2])

        handle = {"num_parts": 4}
        loader = [batch]  # one batch per epoch

        mock_build.return_value = (golden, loader, handle)
        mock_load_perm.return_value = None

        cfg = MagicMock()
        metrics_list = seh.run_single_experiment_multi_epoch(
            cfg, num_parts=4, bs=1, num_epochs=3, accumulate=True
        )

        self.assertEqual(len(metrics_list), 3)
        for m in metrics_list:
            self.assertAlmostEqual(m["strict_recall"], 1.0)
            self.assertAlmostEqual(m["partial_recall"], 1.0)

    @patch("tools.sweep_tools.sweep_over_epochs_hypergraphs.run_single_experiment_multi_epoch")
    @patch("tools.sweep_tools.sweep_over_epochs_hypergraphs.hydra.compose")
    @patch("tools.sweep_tools.sweep_over_epochs_hypergraphs.hydra.initialize")
    def test_run_sweep_multi_epoch_calls_single(self, mock_init, mock_compose, mock_run_single):
        """
        Test run_sweep_multi_epoch integrates multi-epoch metrics.
        """
        mock_init.return_value.__enter__.return_value = None
        mock_init.return_value.__exit__.return_value = False
        mock_compose.return_value = MagicMock()
        mock_run_single.return_value = [
            {
                "gold_n": 1,
                "cand_n": 1,
                "strict_match": 1,
                "partial_covered": 1,
                "strict_recall": 1.0,
                "partial_recall": 1.0,
            }
        ]

        batch_sizes, num_parts_list, results_per_pair = seh.run_sweep_multi_epoch(
            num_epochs=2, accumulate=True
        )

        self.assertIsInstance(batch_sizes, list)
        self.assertIsInstance(num_parts_list, list)
        self.assertIsInstance(results_per_pair, dict)
        self.assertTrue(results_per_pair)  # some entries present
        self.assertTrue(mock_run_single.called)

    def test_build_metric_matrix_for_epoch(self):
        """
        Test build_metric_matrix_for_epoch builds percent matrix.
        """
        batch_sizes = [1, 2]
        num_parts_list = [4, 8]
        results_per_pair = {
            (1, 4): [
                {"strict_recall": 0.5},
                {"strict_recall": 0.75},
            ],
            (2, 8): [
                {"strict_recall": 1.0},
            ],
        }

        M1 = seh.build_metric_matrix_for_epoch(
            batch_sizes, num_parts_list, results_per_pair, epoch=1, metric_key="strict_recall"
        )
        self.assertEqual(M1.shape, (2, 2))
        self.assertAlmostEqual(M1[0, 0], 50.0)
        self.assertAlmostEqual(M1[1, 1], 100.0)

        M2 = seh.build_metric_matrix_for_epoch(
            batch_sizes, num_parts_list, results_per_pair, epoch=2, metric_key="strict_recall"
        )
        self.assertAlmostEqual(M2[0, 0], 75.0)
        self.assertTrue(np.isnan(M2[1, 1]))

    @patch.object(Figure, "savefig")
    def test_plot_summary_4panel_calls_savefig(self, mock_savefig):
        """
        Test plot_summary_4panel renders figure and saves to disk.
        """
        batch_sizes = [1, 2]
        num_parts_list = [4, 8]
        results_per_pair = {
            (1, 4): [
                {"strict_recall": 0.5},
                {"strict_recall": 0.75},
            ],
            (2, 8): [
                {"strict_recall": 0.4},
                {"strict_recall": 0.9},
            ],
        }

        seh.plot_summary_4panel(
            batch_sizes=batch_sizes,
            num_parts_list=num_parts_list,
            results_per_pair=results_per_pair,
            num_epochs=2,
            metric_key="strict_recall",
            target_bs=1,
            target_num_parts=4,
            save_prefix="cora_hg_epochs_test",
        )

        self.assertTrue(mock_savefig.called)
        args, kwargs = mock_savefig.call_args
        self.assertIn("cora_hg_epochs_test_strict_recall.png", args[0])


if __name__ == "__main__":
    unittest.main()
