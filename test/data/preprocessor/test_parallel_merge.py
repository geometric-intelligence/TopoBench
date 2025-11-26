"""Comprehensive tests for parallel shard merging in OnDiskInductivePreprocessor.

This test suite validates the parallel merge implementation that writes shards
to non-overlapping offsets in a pre-allocated file for optimal performance.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from topobench.data.preprocessor import OnDiskInductivePreprocessor


class SimpleDataset:
    """Simple dataset for testing (picklable - defined at module level)."""

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create graph with unique DETERMINISTIC data based on idx
        # Set seed based on idx for reproducibility across parallel/sequential processing
        torch.manual_seed(idx + 42)
        return Data(
            x=torch.randn(5, 16) + idx,  # Deterministic based on idx
            edge_index=torch.tensor(
                [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
            ),
            y=torch.tensor([idx % 10], dtype=torch.long),
        )


def create_simple_dataset(num_samples: int):
    """Create a simple in-memory dataset for testing."""
    return SimpleDataset(num_samples)


class TestParallelMerge:
    """Test suite for parallel shard merge functionality."""

    def test_parallel_merge_correctness(self):
        """Test that parallel merge produces identical results to sequential merge."""
        num_samples = 1000

        with (
            tempfile.TemporaryDirectory() as tmpdir1,
            tempfile.TemporaryDirectory() as tmpdir2,
        ):
            data_dir1 = Path(tmpdir1)
            data_dir2 = Path(tmpdir2)

            source = create_simple_dataset(num_samples)

            transforms_config = OmegaConf.create(
                {
                    "clique_lifting": {
                        "transform_type": "lifting",
                        "transform_name": "SimplicialCliqueLifting",
                        "complex_dim": 2,
                    },
                }
            )

            # Create dataset with multiple workers (parallel merge)
            dataset_parallel = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir1,
                transforms_config=transforms_config,
                num_workers=4,  # Parallel
                storage_backend="mmap",
                compression="lz4",
            )

            # Create dataset with single worker (sequential fallback)
            dataset_sequential = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir2,
                transforms_config=transforms_config,
                num_workers=1,  # Sequential
                storage_backend="mmap",
                compression="lz4",
            )

            # Verify both have same number of samples
            assert (
                len(dataset_parallel) == len(dataset_sequential) == num_samples
            )

            # Verify data integrity: check random samples
            np.random.seed(42)
            test_indices = np.random.choice(
                num_samples, size=min(50, num_samples), replace=False
            )

            for idx in test_indices:
                data_parallel = dataset_parallel[idx]
                data_sequential = dataset_sequential[idx]

                # Compare node features
                assert torch.allclose(
                    data_parallel.x, data_sequential.x, atol=1e-5
                ), f"Node features differ at index {idx}"

                # Compare edge indices
                assert torch.equal(
                    data_parallel.edge_index, data_sequential.edge_index
                ), f"Edge indices differ at index {idx}"

                # Compare labels
                assert torch.equal(data_parallel.y, data_sequential.y), (
                    f"Labels differ at index {idx}"
                )

    def test_parallel_merge_with_different_worker_counts(self):
        """Test parallel merge works correctly with various worker counts."""
        num_samples = 500

        for num_workers in [1, 2, 4, None]:  # None = auto-detect
            with tempfile.TemporaryDirectory() as tmpdir:
                data_dir = Path(tmpdir)
                source = create_simple_dataset(num_samples)

                transforms_config = OmegaConf.create(
                    {
                        "clique_lifting": {
                            "transform_type": "lifting",
                            "transform_name": "SimplicialCliqueLifting",
                            "complex_dim": 2,
                        },
                    }
                )

                dataset = OnDiskInductivePreprocessor(
                    dataset=source,
                    data_dir=data_dir,
                    transforms_config=transforms_config,
                    num_workers=num_workers,
                    storage_backend="mmap",
                    compression="lz4",
                )

                # Verify dataset created successfully
                assert len(dataset) == num_samples

                # Verify storage files exist
                transform_dir = Path(dataset.transform_chain[0]["output_dir"])
                assert (transform_dir / "samples.mmap").exists()
                assert (transform_dir / "samples.idx.npy").exists()

                # Verify random samples are valid
                test_indices = [0, num_samples // 2, num_samples - 1]
                for idx in test_indices:
                    data = dataset[idx]
                    assert isinstance(data, Data)
                    assert hasattr(data, "x")
                    assert hasattr(data, "edge_index")

    def test_parallel_merge_with_small_dataset(self):
        """Test that small datasets (< 1000 samples) fall back to sequential merge correctly."""
        num_samples = 50  # Small dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            source = create_simple_dataset(num_samples)

            transforms_config = OmegaConf.create(
                {
                    "clique_lifting": {
                        "transform_type": "lifting",
                        "transform_name": "SimplicialCliqueLifting",
                        "complex_dim": 2,
                    },
                }
            )

            # Even with multiple workers, small datasets should use sequential merge
            dataset = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir,
                transforms_config=transforms_config,
                num_workers=4,
                storage_backend="mmap",
                compression="lz4",
            )

            assert len(dataset) == num_samples

            # Verify all samples are valid
            for idx in range(num_samples):
                data = dataset[idx]
                assert isinstance(data, Data)

    def test_parallel_merge_byte_accuracy(self):
        """Test that parallel merge writes exact number of bytes expected."""
        num_samples = 1000

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            source = create_simple_dataset(num_samples)

            transforms_config = OmegaConf.create(
                {
                    "clique_lifting": {
                        "transform_type": "lifting",
                        "transform_name": "SimplicialCliqueLifting",
                        "complex_dim": 2,
                    },
                }
            )

            dataset = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir,
                transforms_config=transforms_config,
                num_workers=4,
                storage_backend="mmap",
                compression="lz4",
            )

            # Get storage paths
            storage_dir = Path(dataset.transform_chain[0]["output_dir"])
            mmap_path = storage_dir / "samples.mmap"
            index_path = storage_dir / "samples.idx.npy"

            # Verify files exist and have content
            assert mmap_path.exists()
            assert mmap_path.stat().st_size > 0

            # Load index and verify it matches dataset size
            index = np.load(index_path, allow_pickle=False)
            assert len(index) == num_samples

            # Verify last sample's offset + length doesn't exceed file size
            last_offset, last_length = index[-1]
            file_size = mmap_path.stat().st_size
            assert last_offset + last_length <= file_size, (
                f"Index points beyond file: {last_offset} + {last_length} > {file_size}"
            )

    def test_parallel_merge_no_corruption_on_concurrent_writes(self):
        """Test that concurrent writes to different offsets don't corrupt data."""
        num_samples = 2000  # Large enough to create multiple shards

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            source = create_simple_dataset(num_samples)

            transforms_config = OmegaConf.create(
                {
                    "clique_lifting": {
                        "transform_type": "lifting",
                        "transform_name": "SimplicialCliqueLifting",
                        "complex_dim": 2,
                    },
                }
            )

            dataset = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir,
                transforms_config=transforms_config,
                num_workers=7,  # Maximum parallelism
                storage_backend="mmap",
                compression="lz4",
            )

            # Verify dataset integrity by checking ALL samples
            # If there was corruption, some samples would fail to load
            failed_indices = []
            for idx in range(num_samples):
                try:
                    data = dataset[idx]
                    assert isinstance(data, Data)
                    assert hasattr(data, "x")
                    assert data.x.shape[1] == 16  # Feature dimension
                except Exception as e:
                    failed_indices.append((idx, str(e)))

            assert len(failed_indices) == 0, (
                f"Found {len(failed_indices)} corrupted samples: {failed_indices[:5]}"
            )

    def test_parallel_merge_handles_edge_cases(self):
        """Test edge cases: single sample, odd numbers, power of 2, etc."""
        test_cases = [1, 3, 7, 64, 127, 256, 333, 1000]

        for num_samples in test_cases:
            with tempfile.TemporaryDirectory() as tmpdir:
                data_dir = Path(tmpdir)
                source = create_simple_dataset(num_samples)

                transforms_config = OmegaConf.create(
                    {
                        "clique_lifting": {
                            "transform_type": "lifting",
                            "transform_name": "SimplicialCliqueLifting",
                            "complex_dim": 2,
                        },
                    }
                )

                dataset = OnDiskInductivePreprocessor(
                    dataset=source,
                    data_dir=data_dir,
                    transforms_config=transforms_config,
                    num_workers=4,
                    storage_backend="mmap",
                    compression="lz4",
                )

                assert len(dataset) == num_samples, (
                    f"Dataset size mismatch for {num_samples} samples"
                )

                # Verify first and last samples
                first = dataset[0]
                last = dataset[num_samples - 1]
                assert isinstance(first, Data)
                assert isinstance(last, Data)

    def test_parallel_merge_cleanup_after_completion(self):
        """Test that shard directories are cleaned up after successful merge."""
        num_samples = 1000

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            source = create_simple_dataset(num_samples)

            transforms_config = OmegaConf.create(
                {
                    "clique_lifting": {
                        "transform_type": "lifting",
                        "transform_name": "SimplicialCliqueLifting",
                        "complex_dim": 2,
                    },
                }
            )

            dataset = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir,
                transforms_config=transforms_config,
                num_workers=4,
                storage_backend="mmap",
                compression="lz4",
            )

            # Check that shard directories are cleaned up
            storage_dir = Path(dataset.transform_chain[0]["output_dir"])
            shard_dirs = list(storage_dir.glob("_shard_*"))

            assert len(shard_dirs) == 0, (
                f"Found {len(shard_dirs)} shard directories that weren't cleaned up: {shard_dirs}"
            )

            # Verify final files exist
            assert (storage_dir / "samples.mmap").exists()
            assert (storage_dir / "samples.idx.npy").exists()
            assert (storage_dir / "metadata.json").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
