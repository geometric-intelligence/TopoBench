import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch_geometric
from matplotlib.figure import Figure

from tools.sweep_tools import sweep_over_epochs_cells as sec


class TestSweepOverEpochsCells(unittest.TestCase):
    """
    Unit tests for sweep_over_epochs_cells utilities.

    Covers permutation loading, ID resolution, incidence handling,
    metrics, batch processing, multi-epoch runs, sweeping, and plotting.
    """

    def test_load_perm_to_global_missing(self):
        """
        Test load_perm_to_global returns None if mapping is missing.
        """
        with tempfile.TemporaryDirectory() as tmp:
            handle = {"processed_dir": tmp}
            result = sec.load_perm_to_global(handle)
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
            result = sec.load_perm_to_global(handle)

            self.assertIsInstance(result, torch.Tensor)
            self.assertTrue(torch.equal(result, torch.from_numpy(arr)))

    def test_resolve_true_global_ids_padded_identity(self):
        """
        Test resolve_true_global_ids uses true_global_nid for padded batches.
        """
        golden = MagicMock()
        batch = MagicMock()
        batch.x_0 = torch.randn(4, 2)
        batch.is_true_global_remapped = True
        batch.true_global_nid = torch.tensor([10, 11, 12, 13])

        out = sec.resolve_true_global_ids(golden, batch, None)
        self.assertTrue(torch.equal(out, torch.tensor([10, 11, 12, 13])))

    def test_resolve_true_global_ids_legacy_with_perm(self):
        """
        Test resolve_true_global_ids maps via global_nid and perm_to_global.
        """
        golden = MagicMock()
        golden.x_0 = torch.randn(3, 2)

        batch = MagicMock()
        batch.x_0 = torch.randn(3, 2)
        batch.global_nid = torch.tensor([2, 0, 1])
        batch.is_true_global_remapped = False

        perm_to_global = torch.tensor([10, 11, 12])

        out = sec.resolve_true_global_ids(golden, batch, perm_to_global)
        self.assertTrue(torch.equal(out, torch.tensor([12, 10, 11])))

    def test_resolve_true_global_ids_no_global_nid(self):
        """
        Test resolve_true_global_ids falls back to local indices.
        """
        golden = MagicMock()
        batch = MagicMock()
        batch.x_0 = torch.randn(5, 2)
        delattr(batch, "global_nid")

        out = sec.resolve_true_global_ids(golden, batch, None)
        self.assertTrue(torch.equal(out, torch.arange(5)))

    def test_build_col_to_rows_mapping_dense(self):
        """
        Test _build_col_to_rows_mapping on a dense tensor.
        """
        M = torch.tensor(
            [
                [1, 0, 0],
                [0, 1, 1],
            ],
            dtype=torch.float32,
        )
        mapping = sec._build_col_to_rows_mapping(M, n_rows=2)
        self.assertEqual(mapping[0], {0})
        self.assertEqual(mapping[1], {1})
        self.assertEqual(mapping[2], {1})

    def test_build_col_to_rows_mapping_sparse(self):
        """
        Test _build_col_to_rows_mapping on a sparse COO tensor.
        """
        indices = torch.tensor([[0, 1, 1], [0, 1, 2]])
        values = torch.ones(indices.size(1))
        M = torch.sparse_coo_tensor(indices, values, size=(2, 3))

        mapping = sec._build_col_to_rows_mapping(M, n_rows=2)
        self.assertEqual(mapping[0], {0})
        self.assertEqual(mapping[1], {1})
        self.assertEqual(mapping[2], {1})

    def test_build_col_to_rows_mapping_invalid_shape(self):
        """
        Test _build_col_to_rows_mapping returns None on shape mismatch.
        """
        M = torch.ones(3, 3)
        mapping = sec._build_col_to_rows_mapping(M, n_rows=2)
        self.assertIsNone(mapping)

    def test_extract_1_cells_basic(self):
        """
        Test extract_1_cells constructs 1-cells as 0-cell sets.
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

        cells_list, cells_set = sec.extract_1_cells(data, gids)
        self.assertEqual(len(cells_list), 2)
        self.assertIn(frozenset({10, 11}), cells_set)
        self.assertIn(frozenset({12, 13}), cells_set)

    def test_extract_2_cells_case_a_0_to_2(self):
        """
        Test extract_2_cells for 0→2 incidence.
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
        cells_1_list, _ = sec.extract_1_cells(data, gids)
        cells_2 = sec.extract_2_cells(data, gids, cells_1_list)

        self.assertIn(frozenset({10, 11, 12}), cells_2)
        self.assertIn(frozenset({12, 13}), cells_2)

    def test_extract_2_cells_case_b_1_to_2(self):
        """
        Test extract_2_cells for 1→2 incidence with union over 1-cells.
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
        data.incidence_2 = torch.tensor([[1], [1]], dtype=torch.float32)

        gids = torch.tensor([0, 1, 2, 3])
        cells_1_list, _ = sec.extract_1_cells(data, gids)
        cells_2 = sec.extract_2_cells(data, gids, cells_1_list)

        self.assertEqual(cells_2, {frozenset({0, 1, 2, 3})})

    def test_compute_structure_coverage_metrics_subset(self):
        """
        Test coverage metrics in subset mode.
        """
        gold = [frozenset({1, 2, 3}), frozenset({4, 5})]
        cand = [frozenset({1, 2, 3}), frozenset({4})]

        m = sec.compute_structure_coverage_metrics(gold, cand)
        self.assertEqual(m["gold_n"], 2)
        self.assertEqual(m["strict_match"], 1)
        self.assertEqual(m["partial_covered"], 2)
        self.assertAlmostEqual(m["strict_recall"], 0.5)
        self.assertAlmostEqual(m["partial_recall"], 1.0)

    def test_process_raw_batch_to_lifted_basic(self):
        """
        Test process_raw_batch_to_lifted remaps, pads, and flags metadata.
        """
        raw_batch = torch_geometric.data.Data()
        raw_batch.num_nodes = 3
        raw_batch.edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        raw_batch.x_0 = torch.randn(3, 2)
        raw_batch.global_nid = torch.tensor([2, 0, 4])

        perm_to_global = None
        target_n0 = 5

        def pad_to_global(val, sorted_gids):
            if val.size(0) == target_n0:
                return val
            out = torch.zeros((target_n0,) + val.shape[1:], dtype=val.dtype)
            out[sorted_gids] = val
            return out

        def post_batch_transform(graph):
            graph.lifted_flag = True
            return graph

        lifted = sec.process_raw_batch_to_lifted(
            raw_batch,
            perm_to_global=perm_to_global,
            post_batch_transform=post_batch_transform,
            target_n0=target_n0,
            pad_to_global_fn=pad_to_global,
        )

        self.assertEqual(lifted.num_nodes, target_n0)
        self.assertTrue(hasattr(lifted, "lifted_flag"))
        self.assertTrue(torch.equal(lifted.global_nid, torch.arange(target_n0)))
        self.assertTrue(hasattr(lifted, "present_mask_0"))
        self.assertEqual(lifted.present_mask_0.dtype, torch.bool)

    @patch("tools.sweep_tools.sweep_over_epochs_cells.process_raw_batch_to_lifted")
    @patch("tools.sweep_tools.sweep_over_epochs_cells.build_golden_and_loader_from_cfg_cells")
    def test_run_single_experiment_multi_epoch_cells_basic(self, mock_build, mock_process):
        """
        Test run_single_experiment_multi_epoch_cells aggregates metrics over epochs.
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

        lifted = Data()
        lifted.x_0 = golden.x_0.clone()
        lifted.incidence_1 = golden.incidence_1
        lifted.incidence_2 = golden.incidence_2
        lifted.is_true_global_remapped = True
        lifted.true_global_nid = torch.arange(4)

        loader = [MagicMock()]  # one raw batch per epoch
        handle = {"num_parts": 2}
        perm_to_global = None
        post_batch_transform = None
        target_n0 = 4

        def pad_to_global(val, sorted_gids):
            return val

        mock_build.return_value = (
            golden,
            loader,
            handle,
            perm_to_global,
            post_batch_transform,
            target_n0,
            pad_to_global,
        )
        mock_process.return_value = lifted

        cfg = MagicMock()
        results = sec.run_single_experiment_multi_epoch_cells(
            cfg, num_parts=2, bs=1, num_epochs=3, accumulate=True
        )

        self.assertIn(1, results)
        self.assertEqual(len(results[1]), 3)
        for m in results[1]:
            self.assertAlmostEqual(m["strict_recall"], 1.0)

        self.assertIn(2, results)
        self.assertEqual(len(results[2]), 3)
        for m in results[2]:
            self.assertAlmostEqual(m["strict_recall"], 1.0)

    @patch("tools.sweep_tools.sweep_over_epochs_cells.run_single_experiment_multi_epoch_cells")
    @patch("tools.sweep_tools.sweep_over_epochs_cells.hydra.compose")
    @patch("tools.sweep_tools.sweep_over_epochs_cells.hydra.initialize")
    def test_run_sweep_cells_multi_epoch_calls_single(
        self, mock_init, mock_compose, mock_run_single
    ):
        """
        Test run_sweep_cells_multi_epoch integrates single-experiment metrics.
        """
        mock_init.return_value.__enter__.return_value = None
        mock_init.return_value.__exit__.return_value = False
        mock_compose.return_value = MagicMock()
        mock_run_single.return_value = {
            1: [
                {
                    "gold_n": 1,
                    "cand_n": 1,
                    "strict_match": 1,
                    "partial_covered": 1,
                    "strict_recall": 1.0,
                    "partial_recall": 1.0,
                }
            ]
        }

        batch_sizes, num_parts_list, results_by_dim_epochs = sec.run_sweep_cells_multi_epoch(
            num_epochs=2, accumulate=True
        )

        self.assertIsInstance(batch_sizes, list)
        self.assertIsInstance(num_parts_list, list)
        self.assertIsInstance(results_by_dim_epochs, dict)
        self.assertIn(1, results_by_dim_epochs)
        self.assertTrue(results_by_dim_epochs[1])
        self.assertTrue(mock_run_single.called)

    def test_build_metric_matrix_for_epoch(self):
        """
        Test build_metric_matrix_for_epoch builds a percent matrix.
        """
        batch_sizes = [1, 2]
        num_parts_list = [4, 8]
        results_per_pair = {
            (1, 4): [{"strict_recall": 0.5}, {"strict_recall": 0.75}],
            (2, 8): [{"strict_recall": 1.0}],
        }

        M1 = sec.build_metric_matrix_for_epoch(
            batch_sizes, num_parts_list, results_per_pair, epoch=1, metric_key="strict_recall"
        )
        self.assertEqual(M1.shape, (2, 2))
        self.assertAlmostEqual(M1[0, 0], 50.0)
        self.assertAlmostEqual(M1[1, 1], 100.0)

        M2 = sec.build_metric_matrix_for_epoch(
            batch_sizes, num_parts_list, results_per_pair, epoch=2, metric_key="strict_recall"
        )
        self.assertAlmostEqual(M2[0, 0], 75.0)
        self.assertTrue(np.isnan(M2[1, 1]))

    @patch.object(Figure, "savefig")
    def test_plot_summary_4panel_dim_calls_savefig(self, mock_savefig):
        """
        Test plot_summary_4panel_dim renders and saves the summary figure.
        """
        batch_sizes = [1, 2]
        num_parts_list = [4, 8]
        results_per_pair_dim = {
            (1, 4): [
                {"strict_recall": 0.5},
                {"strict_recall": 0.75},
            ],
            (2, 8): [
                {"strict_recall": 0.4},
                {"strict_recall": 0.9},
            ],
        }

        sec.plot_summary_4panel_dim(
            batch_sizes=batch_sizes,
            num_parts_list=num_parts_list,
            results_per_pair_dim=results_per_pair_dim,
            num_epochs=2,
            metric_key="strict_recall",
            target_bs=1,
            target_num_parts=4,
            save_prefix="cells_dim1_epochs_test",
            dim_label="Dim 1 (test)",
        )

        self.assertTrue(mock_savefig.called)
        args, kwargs = mock_savefig.call_args
        self.assertIn("cells_dim1_epochs_test_strict_recall.png", args[0])


if __name__ == "__main__":
    unittest.main()
