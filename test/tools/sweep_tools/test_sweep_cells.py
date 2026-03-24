import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from matplotlib.figure import Figure

from tools.sweep_tools import sweep_cells as sc


class TestSweepCells(unittest.TestCase):
    """
    Unit tests for sweep_cells cell-complex utilities.

    Tests cover permutation handling, incidence helpers, metrics,
    sweep logic, and plotting.
    """

    def test_load_perm_to_global_missing(self):
        """
        Test load_perm_to_global returns None when mapping is absent.
        """
        with tempfile.TemporaryDirectory() as tmp:
            handle = {"processed_dir": tmp}
            result = sc.load_perm_to_global(handle)
            self.assertIsNone(result)

    def test_load_perm_to_global_existing(self):
        """
        Test load_perm_to_global loads a tensor from disk.
        """
        with tempfile.TemporaryDirectory() as tmp:
            processed_dir = Path(tmp)
            perm_dir = processed_dir / "perm_memmap"
            perm_dir.mkdir(parents=True, exist_ok=True)
            arr = np.array([2, 0, 1], dtype=np.int64)
            np.save(perm_dir / "perm_to_global.npy", arr)

            handle = {"processed_dir": str(processed_dir)}
            result = sc.load_perm_to_global(handle)

            self.assertIsInstance(result, torch.Tensor)
            self.assertTrue(torch.equal(result, torch.from_numpy(arr)))

    def test_resolve_true_global_ids_padded_identity(self):
        """
        Test resolve_true_global_ids returns identity for padded batches.
        """
        golden = MagicMock()
        batch = MagicMock()

        batch.x_0 = torch.randn(4, 3)
        batch.is_true_global_remapped = True
        batch.true_global_nid = torch.tensor([10, 11, 12, 13])

        result = sc.resolve_true_global_ids(golden, batch, None)
        self.assertTrue(torch.equal(result, torch.tensor([10, 11, 12, 13])))

    def test_resolve_true_global_ids_legacy_with_perm(self):
        """
        Test resolve_true_global_ids uses global_nid and perm_to_global.
        """
        golden = MagicMock()
        golden.x_0 = torch.randn(3, 2)

        batch = MagicMock()
        batch.x_0 = torch.randn(3, 2)
        batch.global_nid = torch.tensor([2, 0, 1])
        batch.is_true_global_remapped = False

        perm_to_global = torch.tensor([10, 11, 12])

        result = sc.resolve_true_global_ids(golden, batch, perm_to_global)
        self.assertTrue(torch.equal(result, torch.tensor([12, 10, 11])))

    def test_resolve_true_global_ids_no_global_nid(self):
        """
        Test resolve_true_global_ids falls back to local indices.
        """
        golden = MagicMock()
        batch = MagicMock()
        batch.x_0 = torch.randn(5, 2)
        delattr(batch, "global_nid")

        result = sc.resolve_true_global_ids(golden, batch, None)
        self.assertTrue(torch.equal(result, torch.arange(5)))

    def test_build_col_to_rows_mapping_dense(self):
        """
        Test _build_col_to_rows_mapping for a dense tensor.
        """
        M = torch.tensor(
            [
                [1, 0, 0],
                [0, 1, 1],
            ],
            dtype=torch.float32,
        )
        mapping = sc._build_col_to_rows_mapping(M, n_rows=2)
        self.assertEqual(mapping[0], {0})
        self.assertEqual(mapping[1], {1})
        self.assertEqual(mapping[2], {1})

    def test_build_col_to_rows_mapping_sparse(self):
        """
        Test _build_col_to_rows_mapping for a sparse COO tensor.
        """
        indices = torch.tensor([[0, 1, 1], [0, 1, 2]])
        values = torch.ones(indices.size(1))
        M = torch.sparse_coo_tensor(indices, values, size=(2, 3))

        mapping = sc._build_col_to_rows_mapping(M, n_rows=2)
        self.assertEqual(mapping[0], {0})
        self.assertEqual(mapping[1], {1})
        self.assertEqual(mapping[2], {1})

    def test_build_col_to_rows_mapping_invalid_shape(self):
        """
        Test _build_col_to_rows_mapping returns None on shape mismatch.
        """
        M = torch.ones(3, 3)
        mapping = sc._build_col_to_rows_mapping(M, n_rows=2)
        self.assertIsNone(mapping)

    def test_extract_1_cells_basic(self):
        """
        Test extract_1_cells builds sets of 0-cell IDs.
        """
        class Data:
            pass

        data = Data()
        data.x_0 = torch.randn(4, 1)
        data.incidence_1 = torch.tensor(
            [
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
            ],
            dtype=torch.float32,
        )
        gids = torch.tensor([10, 11, 12, 13])

        cells_list, cells_set = sc.extract_1_cells(data, gids)
        self.assertEqual(len(cells_list), 2)
        self.assertIn(frozenset({10, 11}), cells_set)
        self.assertIn(frozenset({12, 13}), cells_set)

    def test_extract_2_cells_case_a_0_to_2(self):
        """
        Test extract_2_cells when incidence_2 is 0→2.
        """
        class Data:
            pass

        data = Data()
        data.x_0 = torch.randn(4, 1)
        data.incidence_1 = torch.tensor(
            [
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
            ],
            dtype=torch.float32,
        )
        data.incidence_2 = torch.tensor(
            [
                [1, 0],
                [1, 0],
                [1, 1],
                [0, 1],
            ],
            dtype=torch.float32,
        )
        gids = torch.tensor([10, 11, 12, 13])
        cells_1_list, _ = sc.extract_1_cells(data, gids)
        cells_2 = sc.extract_2_cells(data, gids, cells_1_list)

        self.assertIn(frozenset({10, 11, 12}), cells_2)
        self.assertIn(frozenset({12, 13}), cells_2)

    def test_extract_2_cells_case_b_1_to_2(self):
        """
        Test extract_2_cells when incidence_2 is 1→2.
        """
        class Data:
            pass

        data = Data()
        data.x_0 = torch.randn(4, 1)
        data.incidence_1 = torch.tensor(
            [
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
            ],
            dtype=torch.float32,
        )
        data.incidence_2 = torch.tensor(
            [
                [1],
                [1],
            ],
            dtype=torch.float32,
        )
        gids = torch.tensor([0, 1, 2, 3])
        cells_1_list, _ = sc.extract_1_cells(data, gids)
        cells_2 = sc.extract_2_cells(data, gids, cells_1_list)

        self.assertEqual(cells_2, {frozenset({0, 1, 2, 3})})

    def test_compute_structure_coverage_metrics_subset(self):
        """
        Test structure coverage metrics in subset mode.
        """
        gold = [frozenset({1, 2, 3}), frozenset({4, 5})]
        cand = [frozenset({1, 2, 3}), frozenset({4})]

        m = sc.compute_structure_coverage_metrics(gold, cand)
        self.assertEqual(m["gold_n"], 2)
        self.assertEqual(m["strict_match"], 1)
        # both gold structures are partially covered
        self.assertEqual(m["partial_covered"], 2)
        self.assertAlmostEqual(m["strict_recall"], 0.5)
        self.assertAlmostEqual(m["partial_recall"], 1.0)

    def test_compute_structure_coverage_metrics_jaccard(self):
        """
        Test structure coverage metrics in Jaccard mode.
        """
        gold = [frozenset({1, 2, 3})]
        cand = [frozenset({1, 2})]

        m = sc.compute_structure_coverage_metrics(
            gold, cand, mode="jaccard", jaccard_thresh=0.4
        )
        self.assertEqual(m["gold_n"], 1)
        self.assertEqual(m["cand_n"], 1)

    @patch("tools.sweep_tools.sweep_cells.load_perm_to_global")
    @patch("tools.sweep_tools.sweep_cells.build_golden_and_candidate_from_cfg")
    def test_run_single_experiment_basic(self, mock_build, mock_load_perm):
        """
        Test a minimal run_single_experiment call for 1- and 2-cells.
        """
        class Data:
            pass

        golden = Data()
        golden.x_0 = torch.randn(4, 1)
        golden.incidence_1 = torch.tensor(
            [
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
            ],
            dtype=torch.float32,
        )
        golden.incidence_2 = torch.tensor(
            [
                [1],
                [1],
                [1],
                [1],
            ],
            dtype=torch.float32,
        )

        batch = Data()
        batch.x_0 = golden.x_0.clone()
        batch.incidence_1 = golden.incidence_1
        batch.incidence_2 = golden.incidence_2
        batch.is_true_global_remapped = True
        batch.true_global_nid = torch.arange(4)

        handle = {"num_parts": 2}
        mock_build.return_value = (golden, [batch], handle)
        mock_load_perm.return_value = None

        cfg = MagicMock()
        results = sc.run_single_experiment(cfg, num_parts=2, bs=1)

        self.assertIn(1, results)
        self.assertAlmostEqual(results[1]["strict_recall"], 1.0)
        self.assertIn(2, results)
        self.assertAlmostEqual(results[2]["strict_recall"], 1.0)

    @patch("tools.sweep_tools.sweep_cells.run_single_experiment")
    @patch("tools.sweep_tools.sweep_cells.hydra.compose")
    @patch("tools.sweep_tools.sweep_cells.hydra.initialize")
    def test_run_sweep_calls_single_experiment(
        self, mock_init, mock_compose, mock_run_single
    ):
        """
        Test run_sweep integrates run_single_experiment over a grid.
        """
        mock_init.return_value.__enter__.return_value = None
        mock_init.return_value.__exit__.return_value = False
        mock_compose.return_value = MagicMock()
        mock_run_single.return_value = {
            1: {
                "gold_n": 1,
                "cand_n": 1,
                "strict_match": 1,
                "partial_covered": 1,
                "strict_recall": 1.0,
                "partial_recall": 1.0,
            }
        }

        batch_sizes, num_parts_list, results_by_dim = sc.run_sweep()
        self.assertIsInstance(batch_sizes, list)
        self.assertIsInstance(num_parts_list, list)
        self.assertIsInstance(results_by_dim, dict)
        self.assertIn(1, results_by_dim)
        self.assertTrue(results_by_dim[1])

    def test_build_metric_matrix(self):
        """
        Test build_metric_matrix returns a dense percent matrix.
        """
        batch_sizes = [1, 2]
        num_parts_list = [4, 8]
        results_dim = {
            (1, 4): {"strict_recall": 0.5},
            (2, 8): {"strict_recall": 1.0},
        }

        M = sc.build_metric_matrix(batch_sizes, num_parts_list, results_dim, "strict_recall")
        self.assertEqual(M.shape, (2, 2))
        self.assertAlmostEqual(M[0, 0], 50.0)
        self.assertTrue(np.isnan(M[0, 1]))
        self.assertAlmostEqual(M[1, 1], 100.0)

    @patch.object(Figure, "savefig")
    def test_plot_two_heatmaps_calls_savefig(self, mock_savefig):
        """
        Test plot_two_heatmaps renders and saves a figure.
        """
        batch_sizes = [1, 2]
        num_parts_list = [4, 8]
        strict_M = np.array([[50.0, np.nan], [75.0, 100.0]])
        partial_M = np.array([[60.0, np.nan], [80.0, 100.0]])

        sc.plot_two_heatmaps(
            batch_sizes,
            num_parts_list,
            strict_M,
            partial_M,
            save_prefix="cells_test",
            title_prefix="Dim 1",
        )
        self.assertTrue(mock_savefig.called)
        args, kwargs = mock_savefig.call_args
        self.assertIn("cells_test_strict_vs_partial.png", args[0])

    @patch.object(Figure, "savefig")
    def test_plot_recall_by_dim_calls_savefig(self, mock_savefig):
        """
        Test plot_recall_by_dim renders combined recall plots.
        """
        batch_sizes = [1, 2]
        num_parts_list = [4, 8]
        M1 = np.array([[50.0, 60.0], [70.0, 80.0]])
        M2 = np.array([[10.0, 20.0], [30.0, 40.0]])

        sc.plot_recall_by_dim(
            batch_sizes,
            num_parts_list,
            M_dim1=M1,
            M_dim2=M2,
            save_prefix="cells_recall_test",
            title_prefix="Recall test",
        )
        self.assertTrue(mock_savefig.called)
        args, kwargs = mock_savefig.call_args
        self.assertIn("cells_recall_test.png", args[0])


if __name__ == "__main__":
    unittest.main()
