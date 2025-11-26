"""Tests for PyG dataset adapters with performance comparison.

This module tests the adapter system that converts existing PyG datasets
(InMemoryDataset, TUDataset, etc.) into optimized on-disk format.

Key tests:
- Adapter basic functionality
- Performance comparison: adapted vs non-adapted
- Lightweight pickling after caching
- Integration with OnDiskInductivePreprocessor
"""

import pickle
import sys
import tempfile
import time
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data, InMemoryDataset

from topobench.data.datasets import (
    PyGDatasetAdapter,
    adapt_dataset,
)
from topobench.data.preprocessor import OnDiskInductivePreprocessor


class MockInMemoryDataset(InMemoryDataset):
    """Mock InMemoryDataset for testing (simulates PyG's TUDataset pattern).

    Parameters
    ----------
    root : str
        Root directory.
    num_samples : int
        Number of samples.
    """

    def __init__(self, root, num_samples=100):
        self.num_samples = num_samples
        super().__init__(root)
        self.data, self.slices = self._generate_data()

    def _generate_data(self):
        """Generate data in InMemoryDataset format.

        Returns
        -------
        tuple
            Data and slices.
        """
        data_list = []
        for i in range(self.num_samples):
            torch.manual_seed(i)
            data_list.append(
                Data(
                    x=torch.randn(10, 8),
                    edge_index=torch.randint(0, 10, (2, 20)),
                    y=torch.tensor([i % 5]),
                )
            )
        return self.collate(data_list)

    @property
    def raw_file_names(self):
        """Raw file names (empty for mock).

        Returns
        -------
        list
            Empty list.
        """
        return []

    @property
    def processed_file_names(self):
        """Processed file names.

        Returns
        -------
        list
            List of processed files.
        """
        return ["data.pt"]

    def download(self):
        """Download (no-op for mock)."""

    def process(self):
        """Do nothing (no-op for mock dataset)."""


class SimpleMockDataset:
    """Simple mock dataset (not InMemoryDataset) for basic adapter tests.

    Parameters
    ----------
    num_samples : int
        Number of samples.
    """

    def __init__(self, num_samples=10):
        self.num_samples = num_samples

    def __len__(self):
        """Return length of dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return self.num_samples

    def __getitem__(self, idx):
        """Get sample by index.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Data
            Sample data.
        """
        torch.manual_seed(idx)
        return Data(
            x=torch.randn(5, 3),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            y=torch.tensor([idx]),
        )


class TestPyGDatasetAdapter:
    """Test suite for PyG dataset adapter."""

    def test_adapter_basic(self):
        """Test basic adapter functionality with simple mock dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dataset = SimpleMockDataset(num_samples=10)
            adapter = PyGDatasetAdapter(source_dataset, root=tmpdir)

            # Check length
            assert len(adapter) == 10

            # Check sample loading
            sample = adapter[0]
            assert isinstance(sample, Data)
            assert sample.x.shape == (5, 3)

    def test_adapter_with_inmemory_dataset(self):
        """Test adapter with actual InMemoryDataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create InMemoryDataset
            source = MockInMemoryDataset(tmpdir + "/source", num_samples=20)

            # Adapt it
            adapter = PyGDatasetAdapter(source, root=tmpdir + "/adapted")

            # Check functionality
            assert len(adapter) == 20
            sample = adapter[0]
            assert isinstance(sample, Data)
            assert sample.x.shape == (10, 8)

    def test_adapter_caching(self):
        """Test that adapter caches samples correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = SimpleMockDataset(num_samples=5)
            adapted = adapt_dataset(source, root=tmpdir)

            # Check cache files exist
            cache_dir = Path(tmpdir) / ".sample_cache"
            assert cache_dir.exists()
            assert len(adapted) == 5

            # All samples should be cached after adaptation
            cached_files = list(cache_dir.glob("sample_*.pt"))
            assert len(cached_files) == 5

    def test_adapter_lightweight_pickling(self):
        """Test that adapter is lightweight for multiprocessing after caching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = SimpleMockDataset(num_samples=10)
            adapter = adapt_dataset(source, root=tmpdir)

            pickled = pickle.dumps(adapter)
            size_kb = sys.getsizeof(pickled) / 1024

            assert size_kb < 50, (
                f"Adapter pickle size ({size_kb:.2f}KB) exceeds 50KB limit"
            )

    def test_adapt_dataset_convenience(self):
        """Test convenience function for adapting datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = SimpleMockDataset(num_samples=5)

            # Use convenience function
            adapter = adapt_dataset(source, root=tmpdir)

            # Check it works
            assert len(adapter) == 5
            sample = adapter[0]
            assert isinstance(sample, Data)


class TestAdapterPerformanceComparison:
    """Performance comparison tests: adapted vs non-adapted datasets."""

    def test_pickle_size_comparison_mock(self):
        """Compare pickle size: InMemoryDataset vs Adapted (mock dataset)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = 50  # Small for CI/CD

            inmemory = MockInMemoryDataset(
                tmpdir + "/source", num_samples=num_samples
            )
            adapted = adapt_dataset(inmemory, root=tmpdir + "/adapted")

            inmemory_pickled = pickle.dumps(inmemory)
            adapted_pickled = pickle.dumps(adapted)

            inmemory_size_kb = sys.getsizeof(inmemory_pickled) / 1024
            adapted_size_kb = sys.getsizeof(adapted_pickled) / 1024

            assert adapted_size_kb < inmemory_size_kb / 2, (
                f"Adapted ({adapted_size_kb:.1f}KB) should be <2× smaller than InMemory ({inmemory_size_kb:.1f}KB)"
            )

    def test_pickle_size_comparison_real_dataset(self):
        """Compare pickle size with real TU dataset."""
        pytest.importorskip("torch_geometric.datasets")
        from torch_geometric.datasets import TUDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Use MUTAG (small dataset with only 188 graphs)
            tu_dataset = TUDataset(root=str(tmpdir / "tu"), name="MUTAG")
            adapted = adapt_dataset(tu_dataset, root=tmpdir / "adapted")

            tu_pickled = pickle.dumps(tu_dataset)
            adapted_pickled = pickle.dumps(adapted)

            tu_size_kb = sys.getsizeof(tu_pickled) / 1024
            adapted_size_kb = sys.getsizeof(adapted_pickled) / 1024

            assert adapted_size_kb < tu_size_kb / 2, (
                f"Adapted ({adapted_size_kb:.1f}KB) should be <2× smaller than TUDataset ({tu_size_kb:.1f}KB)"
            )

    def test_parallel_preprocessing_comparison(self):
        """Compare adapted vs non-adapted parallel preprocessing.

        Verifies correctness and reports speedup (informational, not enforced).
        Speedup varies by system; lightweight pickle benefit is consistent.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            num_samples = 100

            inmemory = MockInMemoryDataset(
                str(tmpdir / "source"), num_samples=num_samples
            )

            # Non-adapted: heavy pickling
            start = time.time()
            preprocessor_heavy = OnDiskInductivePreprocessor(
                dataset=inmemory,
                data_dir=str(tmpdir / "processed_heavy"),
                transforms_config=None,
                num_workers=2,
                force_reload=True,
            )
            time_heavy = time.time() - start

            # Adapted: lightweight pickling
            adapted = adapt_dataset(
                inmemory,
                root=str(tmpdir / "adapted"),
                extraction_workers=2,
                verbose=False,
            )

            start = time.time()
            preprocessor_light = OnDiskInductivePreprocessor(
                dataset=adapted,
                data_dir=str(tmpdir / "processed_light"),
                transforms_config=None,
                num_workers=2,
                force_reload=True,
            )
            time_light = time.time() - start

            # Verify correctness
            assert (
                len(preprocessor_heavy)
                == len(preprocessor_light)
                == num_samples
            )

            # Report speedup (informational)
            speedup = time_heavy / time_light
            print(
                f"\nParallel preprocessing: {speedup:.2f}× speedup (adapted vs non-adapted)"
            )

    def test_memory_efficiency(self):
        """Test that adapted dataset enables O(1) memory preprocessing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = 50  # Small for CI/CD

            inmemory = MockInMemoryDataset(
                tmpdir + "/source", num_samples=num_samples
            )
            adapted = adapt_dataset(inmemory, root=tmpdir + "/adapted")

            inmemory_attrs_size = sum(
                sys.getsizeof(v) for v in vars(inmemory).values()
            )
            adapted_attrs_size = sum(
                sys.getsizeof(v) for v in vars(adapted).values()
            )

            assert adapted_attrs_size < inmemory_attrs_size, (
                "Adapted dataset should have smaller memory footprint"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
