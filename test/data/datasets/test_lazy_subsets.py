"""Tests for lazy splits."""

import os
import tempfile

import psutil
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from topobench.data.datasets import GeneratedInductiveDataset
from topobench.data.datasets._lazy import LazySubset


class SimpleGeneratedDataset(GeneratedInductiveDataset):
    """Simple generated dataset for testing."""

    def __init__(self, root, num_samples=100):
        super().__init__(root, num_samples, seed=42, cache_samples=True)

    def _generate_sample(self, idx, rng):
        """Generate test sample."""
        x = torch.randn(10, 8, generator=rng)
        edge_index = torch.randint(0, 10, (2, 20), generator=rng)
        y = torch.tensor([idx % 5])
        return Data(x=x, edge_index=edge_index, y=y)


class TestLazySubset:
    """Test LazySubset functionality."""

    def test_basic_operations(self):
        """Test basic LazySubset operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = SimpleGeneratedDataset(tmpdir, num_samples=100)
            indices = [0, 5, 10, 15, 20]
            subset = LazySubset(dataset, indices)

            # Test length
            assert len(subset) == 5

            # Test item access
            assert subset[0] is not None
            assert subset[2] is not None

            # Test tensor indices
            tensor_subset = LazySubset(dataset, torch.tensor(indices))
            assert len(tensor_subset) == 5

            # Test repr
            assert "LazySubset" in repr(subset)
            assert "size=5" in repr(subset)

            # Test bounds checking
            with pytest.raises(IndexError):
                _ = subset[10]

            with pytest.raises(IndexError):
                _ = subset[-1]

    def test_correctness(self):
        """Test that LazySubset returns correct samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = SimpleGeneratedDataset(tmpdir, num_samples=100)
            indices = [5, 10, 15, 20, 25]
            subset = LazySubset(dataset, indices)

            # Verify subset[i] returns dataset[indices[i]]
            for subset_idx, dataset_idx in enumerate(indices):
                subset_sample = subset[subset_idx]
                dataset_sample = dataset[dataset_idx]

                assert type(subset_sample) is type(dataset_sample)
                assert hasattr(subset_sample, "x")
                assert hasattr(subset_sample, "edge_index")

    def test_memory_efficiency(self):
        """Test that lazy splits use O(1) memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = SimpleGeneratedDataset(tmpdir, num_samples=10000)

            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            # Create 100 splits - should use minimal memory
            _splits = [
                LazySubset(dataset, list(range(i * 100, (i + 1) * 100)))
                for i in range(100)
            ]

            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_growth = mem_after - mem_before

            # Memory growth should be minimal (< 10 MB for 100 splits)
            assert mem_growth < 10, (
                f"Memory growth too high: {mem_growth:.1f} MB"
            )

    def test_dataloader_integration(self):
        """Test LazySubset works with PyTorch DataLoader."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = SimpleGeneratedDataset(tmpdir, num_samples=100)
            subset = LazySubset(dataset, list(range(20)))

            loader = DataLoader(subset, batch_size=4, shuffle=True)

            # Verify iteration works
            batch_count = sum(1 for batch in loader if batch is not None)
            assert batch_count == 5  # 20 samples / 4 batch_size
