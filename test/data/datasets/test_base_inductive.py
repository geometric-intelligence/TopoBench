"""Tests for high-performance base dataset classes.

This module tests the core base classes:
- BaseOnDiskInductiveDataset
- FileBasedInductiveDataset
- GeneratedInductiveDataset

For adapter tests, see test_inductive_ondisk_adapters.py
"""

import pickle
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from topobench.data.datasets import (
    FileBasedInductiveDataset,
    GeneratedInductiveDataset,
)


def _torch_load_compat(path, **kwargs):
    """Backwards-compatible torch.load wrapper for tests."""
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


class SimpleGeneratedDataset(GeneratedInductiveDataset):
    """Simple generated dataset for testing.

    Parameters
    ----------
    root : str
        Root directory.
    num_samples : int
        Number of samples.
    seed : int
        Random seed.
    cache_samples : bool
        Enable caching.
    """

    def __init__(self, root, num_samples=100, seed=42, cache_samples=True):
        super().__init__(
            root, num_samples, seed=seed, cache_samples=cache_samples
        )

    def _generate_sample(self, idx, rng):
        """Generate test sample.

        Parameters
        ----------
        idx : int
            Sample index.
        rng : torch.Generator
            Random number generator.

        Returns
        -------
        Data
            Generated sample.
        """
        x = torch.randn(10, 8, generator=rng)
        edge_index = torch.randint(0, 10, (2, 20), generator=rng)
        y = torch.tensor([idx % 5])
        return Data(x=x, edge_index=edge_index, y=y)


class SimpleFileDataset(FileBasedInductiveDataset):
    """Simple file-based dataset for testing."""

    def _load_file(self, file_path):
        """Load file for testing.

        Parameters
        ----------
        file_path : Path
            File path.

        Returns
        -------
        Data
            Loaded data.
        """
        return _torch_load_compat(file_path)


class TestGeneratedInductiveDataset:
    """Test suite for base inductive dataset classes."""

    def test_generated_dataset_basic(self):
        """Test basic functionality of GeneratedInductiveDataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = SimpleGeneratedDataset(tmpdir, num_samples=50)

            # Check length
            assert len(dataset) == 50

            # Check sample loading
            sample = dataset[0]
            assert isinstance(sample, Data)
            assert sample.x.shape == (10, 8)
            assert sample.edge_index.shape[0] == 2
            assert sample.y.item() == 0

            # Check determinism
            sample1 = dataset[5]
            sample2 = dataset[5]
            assert torch.equal(sample1.x, sample2.x)
            assert torch.equal(sample1.edge_index, sample2.edge_index)

    def test_generated_dataset_caching(self):
        """Test that caching works for generated datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = SimpleGeneratedDataset(tmpdir, num_samples=10)

            # First access generates and caches
            _ = dataset[0]
            cache_file = Path(tmpdir) / ".sample_cache" / "sample_000000.pt"
            assert cache_file.exists()

            # Second access loads from cache
            sample = dataset[0]
            assert isinstance(sample, Data)

    def test_generated_dataset_pickling(self):
        """Test that dataset can be pickled (critical for multiprocessing)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = SimpleGeneratedDataset(tmpdir, num_samples=20)

            # Pickle and unpickle
            pickled = pickle.dumps(dataset)
            dataset2 = pickle.loads(pickled)

            # Verify functionality after unpickling
            assert len(dataset2) == 20
            sample = dataset2[0]
            assert isinstance(sample, Data)

            # Verify pickle size is small (< 10KB for optimal parallel performance)
            pickle_size = sys.getsizeof(pickled)
            assert pickle_size < 10000, (
                f"Pickle size {pickle_size} too large (should be < 10KB)"
            )

    def test_deterministic_generation(self):
        """Test that generation is deterministic across dataset instances."""
        with (
            tempfile.TemporaryDirectory() as tmpdir1,
            tempfile.TemporaryDirectory() as tmpdir2,
        ):
            dataset1 = SimpleGeneratedDataset(tmpdir1, num_samples=20)
            dataset2 = SimpleGeneratedDataset(tmpdir2, num_samples=20)

            # Same index should generate same data
            for idx in [0, 5, 10, 15]:
                sample1 = dataset1[idx]
                sample2 = dataset2[idx]

                assert torch.equal(sample1.x, sample2.x)
                assert torch.equal(sample1.edge_index, sample2.edge_index)
                assert sample1.y.item() == sample2.y.item()


class TestFileBasedInductiveDataset:
    """Test suite for FileBasedInductiveDataset."""

    def test_file_based_dataset(self):
        """Test FileBasedInductiveDataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create some test files
            for i in range(10):
                data = Data(
                    x=torch.randn(5, 3),
                    edge_index=torch.tensor([[0, 1], [1, 0]]),
                    y=torch.tensor([i]),
                )
                torch.save(data, tmpdir / f"graph_{i:03d}.pt")

            # Load dataset
            dataset = SimpleFileDataset(tmpdir)

            # Check length
            assert len(dataset) == 10

            # Check sample loading
            sample = dataset[0]
            assert isinstance(sample, Data)
            assert sample.x.shape == (5, 3)

    def test_file_based_dataset_custom_pattern(self):
        """Test FileBasedInductiveDataset with custom file pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create files with custom extension
            for i in range(5):
                data = Data(x=torch.randn(3, 2), y=torch.tensor([i]))
                torch.save(data, tmpdir / f"molecule_{i}.data")

            # Load with custom pattern
            class CustomFileDataset(FileBasedInductiveDataset):
                def __init__(self, root):
                    super().__init__(root, file_pattern="*.data")

                def _load_file(self, file_path):
                    return _torch_load_compat(file_path)

            dataset = CustomFileDataset(tmpdir)
            assert len(dataset) == 5


class TestLightweightProperties:
    """Test that datasets maintain lightweight properties for parallel processing."""

    def test_pickle_size_generated_dataset(self):
        """Test that generated dataset has small pickle size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = SimpleGeneratedDataset(tmpdir, num_samples=100)

            pickled = pickle.dumps(dataset)
            size_kb = sys.getsizeof(pickled) / 1024

            assert size_kb < 10, (
                f"Generated dataset pickle ({size_kb:.2f}KB) exceeds 10KB limit"
            )

    def test_pickle_size_file_dataset(self):
        """Test that file-based dataset has small pickle size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            for i in range(50):
                torch.save(
                    Data(x=torch.randn(5, 3), y=torch.tensor([i])),
                    tmpdir / f"graph_{i}.pt",
                )

            dataset = SimpleFileDataset(tmpdir)

            pickled = pickle.dumps(dataset)
            size_kb = sys.getsizeof(pickled) / 1024

            assert size_kb < 10, (
                f"File-based dataset pickle ({size_kb:.2f}KB) exceeds 10KB limit"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
