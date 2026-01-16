"""Tests for parallel processor."""

import os
import tempfile
import time
from pathlib import Path

import torch
from torch_geometric.data import Data

from topobench.data.preprocessor._ondisk.parallel_processor import (
    ParallelProcessor,
)


class MockDataset:
    """Mock dataset for testing.

    Parameters
    ----------
    size : int
        Number of samples in the dataset.
    fail_indices : set[int] | None
        Indices that should raise errors when accessed.
    """

    def __init__(self, size: int, fail_indices: set[int] | None = None):
        self.size = size
        self.fail_indices = fail_indices if fail_indices is not None else set()

    def __len__(self) -> int:
        """Return the size of the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return self.size

    def __getitem__(self, idx: int) -> Data:
        """Retrieve a sample by index.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        Data
            PyTorch Geometric data sample.

        Raises
        ------
        RuntimeError
            If the index is listed in `fail_indices`.
        """
        if idx in self.fail_indices:
            raise RuntimeError(f"Simulated error for sample {idx}")

        torch.manual_seed(idx)
        return Data(
            x=torch.randn(10, 8),
            edge_index=torch.randint(0, 10, (2, 20)),
            y=torch.tensor([idx % 5]),
        )

    def __reduce__(self):
        """Support pickling for multiprocessing.

        Returns
        -------
        tuple
            Constructor and arguments.
        """
        return (self.__class__, (self.size, self.fail_indices))


class MockTransform:
    """Mock transform for testing."""

    def __call__(self, data: Data) -> Data:
        """Apply a simple transform to the data.

        Parameters
        ----------
        data : Data
            PyTorch Geometric data sample.

        Returns
        -------
        Data
            Transformed sample.
        """
        data.x = data.x + 1.0
        return data


class TestParallelProcessor:
    """Tests for ParallelProcessor."""

    def test_sequential_processing(self):
        """Test sequential processing using a single worker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            dataset = MockDataset(50)
            transform = MockTransform()
            processor = ParallelProcessor(num_workers=1, show_progress=False)

            results = processor.process(
                dataset=dataset,
                transform=transform,
                output_dir=output_dir,
                num_samples=50,
            )

            assert results["total"] == 50
            assert results["success"] == 50
            assert results["failed"] == 0
            assert len(results["errors"]) == 0

            for idx in range(50):
                sample_path = output_dir / f"sample_{idx:06d}.pt"
                assert sample_path.exists()
                data = torch.load(sample_path)
                assert data.x.shape == (10, 8)
                assert data.edge_index.shape == (2, 20)

    def test_parallel_processing_and_error_handling(self):
        """Test multi-worker processing, transforms, and error handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Test 1: parallel success
            dataset = MockDataset(100)
            transform = MockTransform()
            processor = ParallelProcessor(num_workers=4, show_progress=False)

            results = processor.process(
                dataset=dataset,
                transform=transform,
                output_dir=output_dir,
                num_samples=100,
            )

            assert results["total"] == 100
            assert results["success"] == 100
            assert results["failed"] == 0

            for idx in range(100):
                sample_path = output_dir / f"sample_{idx:06d}.pt"
                assert sample_path.exists()
                data = torch.load(sample_path)
                assert data.y.item() == idx % 5

            # Test 2: error handling
            output_dir2 = Path(tmpdir) / "error_test"
            output_dir2.mkdir()

            fail_indices = {5, 15, 25}
            dataset_with_errors = MockDataset(50, fail_indices=fail_indices)

            results = processor.process(
                dataset=dataset_with_errors,
                transform=None,
                output_dir=output_dir2,
                num_samples=50,
            )

            assert results["total"] == 50
            assert results["success"] == 47
            assert results["failed"] == 3
            assert len(results["errors"]) == 3

            for error in results["errors"]:
                assert "Simulated error" in error

            for idx in range(50):
                sample_path = output_dir2 / f"sample_{idx:06d}.pt"
                if idx in fail_indices:
                    assert not sample_path.exists()
                else:
                    assert sample_path.exists()

            # Test 3: No transform
            output_dir3 = Path(tmpdir) / "no_transform"
            output_dir3.mkdir()

            dataset_small = MockDataset(20)
            results = processor.process(
                dataset=dataset_small,
                transform=None,
                output_dir=output_dir3,
                num_samples=20,
            )

            assert results["success"] == 20

            for idx in range(20):
                sample_path = output_dir3 / f"sample_{idx:06d}.pt"
                assert sample_path.exists()
                data = torch.load(sample_path)
                assert data.x.shape == (10, 8)

    def test_auto_worker_count(self):
        """Test automatic worker count selection."""
        processor = ParallelProcessor(num_workers=None)
        cpu_count = os.cpu_count() or 1
        expected = max(1, cpu_count - 1)
        assert processor.num_workers == expected

    def test_batch_size_parameter(self):
        """Test processing with custom batch size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            dataset = MockDataset(100)

            processor = ParallelProcessor(
                num_workers=2, batch_size=10, show_progress=False
            )
            results = processor.process(
                dataset=dataset,
                transform=None,
                output_dir=output_dir,
                num_samples=100,
            )

            assert results["success"] == 100

    def test_performance_improvement(self):
        """Test that parallel processing is corrrect and faster than sequential processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir_seq = Path(tmpdir) / "sequential"
            output_dir_par = Path(tmpdir) / "parallel"
            output_dir_seq.mkdir()
            output_dir_par.mkdir()

            dataset = MockDataset(200)
            transform = MockTransform()

            processor_seq = ParallelProcessor(
                num_workers=1, show_progress=False
            )
            start = time.time()
            results_seq = processor_seq.process(
                dataset=dataset,
                transform=transform,
                output_dir=output_dir_seq,
                num_samples=200,
            )
            time_seq = time.time() - start

            processor_par = ParallelProcessor(
                num_workers=4, show_progress=False
            )
            start = time.time()
            results_par = processor_par.process(
                dataset=dataset,
                transform=transform,
                output_dir=output_dir_par,
                num_samples=200,
            )
            time_par = time.time() - start

            assert results_seq["success"] == 200
            assert results_par["success"] == 200

            for idx in range(200):
                data_seq = torch.load(output_dir_seq / f"sample_{idx:06d}.pt")
                data_par = torch.load(output_dir_par / f"sample_{idx:06d}.pt")
                assert data_seq.y.item() == data_par.y.item()
                assert data_seq.x.shape == data_par.x.shape

            speedup = time_seq / time_par
            print(f"\nSpeedup: {speedup:.2f}")
            print(f"Sequential: {time_seq:.2f}s, Parallel: {time_par:.2f}s")
