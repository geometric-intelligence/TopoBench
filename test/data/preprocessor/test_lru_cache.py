"""Tests for in-memory LRU cache in OnDiskInductivePreprocessor.

Tests verify:
1. Cache functionality (hit/miss behavior)
2. LRU eviction policy
3. Cache statistics tracking
4. Performance improvements
5. Backward compatibility (cache disabled)
"""

import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from topobench.data.datasets import GeneratedInductiveDataset
from topobench.data.preprocessor.ondisk_inductive import (
    OnDiskInductivePreprocessor,
)


class SimpleGraphDataset(GeneratedInductiveDataset):
    """Simple synthetic dataset for cache testing."""

    def __init__(self, root, num_samples=100, seed=42, cache_samples=False):
        super().__init__(root, num_samples, seed, cache_samples)

    def _generate_sample(self, idx, rng):
        """Generate deterministic graph sample."""
        x = torch.randn(10, 4, generator=rng)
        edge_index = torch.randint(0, 10, (2, 20), generator=rng)
        y = torch.tensor([idx % 5])
        return Data(x=x, edge_index=edge_index, y=y)

    def _get_pickle_args(self) -> tuple:
        """Return arguments for pickling."""
        return (str(self.root), self.num_samples, self.seed, self.cache_samples)


class TestLRUCacheBasicFunctionality:
    """Test basic cache operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def simple_dataset(self, temp_dir):
        """Create simple dataset."""
        dataset_dir = temp_dir / "source"
        return SimpleGraphDataset(dataset_dir, num_samples=50)

    def test_cache_enabled_by_default(self, temp_dir, simple_dataset):
        """Test cache is enabled by default with cache_size=100."""
        preprocessor = OnDiskInductivePreprocessor(
            dataset=simple_dataset,
            data_dir=temp_dir / "processed",
            transforms_config=None,
        )

        assert preprocessor.cache_size == 100
        stats = preprocessor.get_cache_stats()
        assert stats["enabled"] is True
        assert stats["capacity"] == 100

    def test_cache_disabled(self, temp_dir, simple_dataset):
        """Test cache can be disabled with cache_size=0."""
        preprocessor = OnDiskInductivePreprocessor(
            dataset=simple_dataset,
            data_dir=temp_dir / "processed",
            transforms_config=None,
            cache_size=0,
        )

        # Access samples
        _ = preprocessor[0]
        _ = preprocessor[0]  # Second access

        stats = preprocessor.get_cache_stats()
        assert stats["enabled"] is False
        assert stats["hits"] == 0  # No cache hits since disabled
        assert stats["misses"] == 2  # Both were misses

    def test_cache_hit_on_repeated_access(self, temp_dir, simple_dataset):
        """Test cache hit when accessing same sample twice."""
        preprocessor = OnDiskInductivePreprocessor(
            dataset=simple_dataset,
            data_dir=temp_dir / "processed",
            transforms_config=None,
            cache_size=10,
        )

        # First access - cache miss
        sample1 = preprocessor[5]
        stats1 = preprocessor.get_cache_stats()
        assert stats1["hits"] == 0
        assert stats1["misses"] == 1
        assert stats1["size"] == 1

        # Second access - cache hit
        sample2 = preprocessor[5]
        stats2 = preprocessor.get_cache_stats()
        assert stats2["hits"] == 1
        assert stats2["misses"] == 1
        assert stats2["size"] == 1

        # Verify samples are identical
        assert torch.equal(sample1.x, sample2.x)
        assert torch.equal(sample1.edge_index, sample2.edge_index)

    def test_cache_hit_rate_calculation(self, temp_dir, simple_dataset):
        """Test cache hit rate is calculated correctly."""
        preprocessor = OnDiskInductivePreprocessor(
            dataset=simple_dataset,
            data_dir=temp_dir / "processed",
            transforms_config=None,
            cache_size=10,
        )

        # Access pattern: [0, 1, 2, 0, 1, 2, 0, 1, 2]
        # First 3: misses, next 6: hits
        for _ in range(3):
            for idx in [0, 1, 2]:
                _ = preprocessor[idx]

        stats = preprocessor.get_cache_stats()
        assert stats["hits"] == 6
        assert stats["misses"] == 3
        assert stats["total_accesses"] == 9
        assert stats["hit_rate"] == pytest.approx(6 / 9, rel=1e-5)


class TestLRUEvictionPolicy:
    """Test LRU eviction behavior."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def simple_dataset(self, temp_dir):
        """Create simple dataset."""
        dataset_dir = temp_dir / "source"
        return SimpleGraphDataset(dataset_dir, num_samples=50)

    def test_lru_eviction_on_cache_full(self, temp_dir, simple_dataset):
        """Test oldest sample is evicted when cache is full."""
        preprocessor = OnDiskInductivePreprocessor(
            dataset=simple_dataset,
            data_dir=temp_dir / "processed",
            transforms_config=None,
            cache_size=3,  # Small cache for testing
        )

        # Fill cache: [0, 1, 2]
        _ = preprocessor[0]
        _ = preprocessor[1]
        _ = preprocessor[2]
        assert preprocessor.get_cache_stats()["size"] == 3

        # Add new sample: [1, 2, 3] (0 evicted)
        _ = preprocessor[3]
        stats = preprocessor.get_cache_stats()
        assert stats["size"] == 3
        assert 0 not in preprocessor._cache
        assert 1 in preprocessor._cache
        assert 2 in preprocessor._cache
        assert 3 in preprocessor._cache

    def test_lru_update_on_access(self, temp_dir, simple_dataset):
        """Test accessing a sample updates its recency."""
        preprocessor = OnDiskInductivePreprocessor(
            dataset=simple_dataset,
            data_dir=temp_dir / "processed",
            transforms_config=None,
            cache_size=3,
        )

        # Fill cache: [0, 1, 2]
        _ = preprocessor[0]
        _ = preprocessor[1]
        _ = preprocessor[2]

        # Access 0 again (makes it most recent): [1, 2, 0]
        _ = preprocessor[0]

        # Add new sample: [2, 0, 3] (1 evicted, not 0)
        _ = preprocessor[3]

        assert 0 in preprocessor._cache  # Still in cache
        assert 1 not in preprocessor._cache  # Evicted
        assert 2 in preprocessor._cache
        assert 3 in preprocessor._cache

    def test_lru_maintains_order(self, temp_dir, simple_dataset):
        """Test LRU order is maintained correctly."""
        preprocessor = OnDiskInductivePreprocessor(
            dataset=simple_dataset,
            data_dir=temp_dir / "processed",
            transforms_config=None,
            cache_size=5,
        )

        # Access pattern
        access_sequence = [0, 1, 2, 3, 4, 0, 5]  # 1 will be evicted
        for idx in access_sequence:
            _ = preprocessor[idx]

        # Cache should be: [2, 3, 4, 0, 5] (1 evicted)
        assert 1 not in preprocessor._cache
        assert all(
            idx in preprocessor._cache for idx in [0, 2, 3, 4, 5]
        )


class TestCacheStatistics:
    """Test cache statistics tracking."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def simple_dataset(self, temp_dir):
        """Create simple dataset."""
        dataset_dir = temp_dir / "source"
        return SimpleGraphDataset(dataset_dir, num_samples=50)

    def test_initial_stats(self, temp_dir, simple_dataset):
        """Test initial cache statistics are correct."""
        preprocessor = OnDiskInductivePreprocessor(
            dataset=simple_dataset,
            data_dir=temp_dir / "processed",
            transforms_config=None,
            cache_size=10,
        )

        stats = preprocessor.get_cache_stats()
        assert stats["enabled"] is True
        assert stats["size"] == 0
        assert stats["capacity"] == 10
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["total_accesses"] == 0

    def test_clear_cache_resets_stats(self, temp_dir, simple_dataset):
        """Test clear_cache() resets statistics."""
        preprocessor = OnDiskInductivePreprocessor(
            dataset=simple_dataset,
            data_dir=temp_dir / "processed",
            transforms_config=None,
            cache_size=10,
        )

        # Access some samples
        _ = preprocessor[0]
        _ = preprocessor[0]
        _ = preprocessor[1]

        stats_before = preprocessor.get_cache_stats()
        assert stats_before["size"] == 2
        assert stats_before["hits"] == 1
        assert stats_before["misses"] == 2

        # Clear cache
        preprocessor.clear_cache()

        stats_after = preprocessor.get_cache_stats()
        assert stats_after["size"] == 0
        assert stats_after["hits"] == 0
        assert stats_after["misses"] == 0
        assert stats_after["hit_rate"] == 0.0

    def test_stats_after_many_accesses(self, temp_dir, simple_dataset):
        """Test statistics remain accurate after many accesses."""
        preprocessor = OnDiskInductivePreprocessor(
            dataset=simple_dataset,
            data_dir=temp_dir / "processed",
            transforms_config=None,
            cache_size=10,
        )

        # Simulate training-like access pattern
        num_epochs = 3
        samples_per_epoch = 20

        for _ in range(num_epochs):
            for idx in range(samples_per_epoch):
                _ = preprocessor[idx % 10]  # Only access first 10 samples

        stats = preprocessor.get_cache_stats()

        # First epoch: 10 misses
        # Second epoch: 10 hits
        # Third epoch: 10 hits
        expected_total = num_epochs * samples_per_epoch
        assert stats["total_accesses"] == expected_total
        assert stats["hits"] + stats["misses"] == expected_total


class TestCachePerformance:
    """Test cache improves performance."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def simple_dataset(self, temp_dir):
        """Create simple dataset."""
        dataset_dir = temp_dir / "source"
        return SimpleGraphDataset(dataset_dir, num_samples=50)

    def test_cache_reduces_disk_io(self, temp_dir, simple_dataset):
        """Test cache actually reduces disk I/O operations."""
        import time

        preprocessor = OnDiskInductivePreprocessor(
            dataset=simple_dataset,
            data_dir=temp_dir / "processed",
            transforms_config=None,
            cache_size=10,
        )

        # Warm up cache
        _ = preprocessor[0]

        # Measure cached access time
        start_cached = time.perf_counter()
        for _ in range(100):
            _ = preprocessor[0]  # Cache hit
        time_cached = time.perf_counter() - start_cached

        # Clear cache and measure cold access time
        preprocessor.clear_cache()
        start_cold = time.perf_counter()
        for _ in range(100):
            _ = preprocessor[0]
            preprocessor.clear_cache()  # Force disk read each time
        time_cold = time.perf_counter() - start_cold

        # Cache should be significantly faster (at least 10×)
        speedup = time_cold / time_cached
        assert speedup > 10, f"Expected >10× speedup, got {speedup:.1f}×"

    def test_realistic_training_pattern(self, temp_dir, simple_dataset):
        """Test cache with realistic training access pattern."""
        preprocessor = OnDiskInductivePreprocessor(
            dataset=simple_dataset,
            data_dir=temp_dir / "processed",
            transforms_config=None,
            cache_size=20,  # Cache enough for 20 samples
        )

        # Simulate 3 epochs of training accessing first 20 samples repeatedly
        # (common pattern where we iterate over training set multiple times)
        batch_size = 8
        samples_per_epoch = 20  # Access first 20 samples (fits in cache)
        num_epochs = 3

        for _epoch in range(num_epochs):
            for batch_start in range(0, samples_per_epoch, batch_size):
                # Access batch
                for idx in range(
                    batch_start, min(batch_start + batch_size, samples_per_epoch)
                ):
                    _ = preprocessor[idx]

        stats = preprocessor.get_cache_stats()

        # First epoch: 20 misses (cold start)
        # Epochs 2-3: 40 hits (all in cache)
        # Total: 40 hits / 60 accesses = 66.7% hit rate
        assert stats["hit_rate"] > 0.6, (
            f"Expected >60% hit rate with caching, "
            f"got {stats['hit_rate']:.1%} (hits={stats['hits']}, misses={stats['misses']})"
        )


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def simple_dataset(self, temp_dir):
        """Create simple dataset."""
        dataset_dir = temp_dir / "source"
        return SimpleGraphDataset(dataset_dir, num_samples=20)

    def test_old_code_works_without_cache_param(self, temp_dir, simple_dataset):
        """Test code without cache_size parameter still works."""
        # Old code pattern (no cache_size specified)
        preprocessor = OnDiskInductivePreprocessor(
            dataset=simple_dataset,
            data_dir=temp_dir / "processed",
            transforms_config=None,
        )

        # Should work fine with default cache_size=100
        sample = preprocessor[0]
        assert sample is not None
        assert hasattr(sample, "x")
        assert hasattr(sample, "edge_index")

    def test_cache_transparent_to_user(self, temp_dir, simple_dataset):
        """Test cache is transparent - user gets same results."""
        # Create two preprocessors: one with cache, one without
        preprocessor_cached = OnDiskInductivePreprocessor(
            dataset=simple_dataset,
            data_dir=temp_dir / "processed_cached",
            transforms_config=None,
            cache_size=10,
        )

        preprocessor_nocache = OnDiskInductivePreprocessor(
            dataset=simple_dataset,
            data_dir=temp_dir / "processed_nocache",
            transforms_config=None,
            cache_size=0,
        )

        # Access same samples from both
        for idx in [0, 5, 10, 5, 0, 10]:  # Mix of first access and repeats
            sample_cached = preprocessor_cached[idx]
            sample_nocache = preprocessor_nocache[idx]

            # Results should be identical
            assert torch.equal(sample_cached.x, sample_nocache.x)
            assert torch.equal(
                sample_cached.edge_index, sample_nocache.edge_index
            )
            assert torch.equal(sample_cached.y, sample_nocache.y)
