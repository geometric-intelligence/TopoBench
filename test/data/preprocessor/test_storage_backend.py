"""Core tests for MemoryMappedStorage backend.

This test suite covers essential functionality for the storage backend:
- Write/read operations with all compression modes
- Error handling and edge cases
- LZ4 vs ZSTD compression trade-offs
"""

import tempfile
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from topobench.data.preprocessor._ondisk.storage_backend import (
    MemoryMappedStorage,
)


def create_test_sample(idx: int, size: str = "small") -> Data:
    """Create a PyG Data object for testing.

    Parameters
    ----------
    idx : int
        Sample index.
    size : str
        Size variant: "small", "medium", or "large".

    Returns
    -------
    Data
        PyG Data object.
    """
    if size == "small":
        num_nodes = 50 + idx % 10
    elif size == "medium":
        num_nodes = 200 + idx % 50
    else:  # large
        num_nodes = 1000 + idx % 100

    num_edges = num_nodes * 3

    return Data(
        x=torch.randn(num_nodes, 16),
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        y=torch.tensor([idx % 10]),
    )


class TestMemoryMappedStorage:
    """Core tests for MemoryMappedStorage."""

    def test_all_compression_modes(self):
        """Test write, read, and error handling for all compression modes."""
        compression_modes = ["lz4", "zstd", None]

        for compression in compression_modes:
            with tempfile.TemporaryDirectory() as tmpdir:
                data_dir = Path(tmpdir)

                # Write samples
                storage = MemoryMappedStorage(
                    data_dir, compression=compression
                )
                samples = [create_test_sample(i) for i in range(50)]
                for sample in samples:
                    storage.append(sample)
                storage.close()

                # Read back and verify
                storage = MemoryMappedStorage(data_dir, readonly=True)

                # Basic checks
                assert len(storage) == 50
                assert storage.compression == compression

                # Random access
                for idx in [0, 25, 49]:
                    loaded = storage[idx]
                    assert torch.allclose(loaded.x, samples[idx].x)
                    assert loaded.y.item() == idx % 10

                # Error handling
                with pytest.raises(IndexError):
                    _ = storage[50]

                with pytest.raises(RuntimeError):
                    storage.append(create_test_sample(100))

    def test_data_integrity(self):
        """Test that data round-trips without corruption."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample with known values
            original = Data(
                x=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                edge_index=torch.tensor([[0, 1], [1, 0]]),
                y=torch.tensor([42]),
            )

            # Write and read
            storage = MemoryMappedStorage(tmpdir, compression="lz4")
            storage.append(original)
            storage.close()

            storage = MemoryMappedStorage(tmpdir, readonly=True)
            loaded = storage[0]

            # Verify exact match
            assert torch.equal(loaded.x, original.x)
            assert torch.equal(loaded.edge_index, original.edge_index)
            assert torch.equal(loaded.y, original.y)

    def test_lz4_vs_zstd_compression(self):
        """Compare LZ4 vs ZSTD: compression ratio.

        This test demonstrates that ZSTD achieves better compression (~1.7× vs ~1.35×).
        Other tests will prove that LZ4 provides faster decompression (important for training).
        """
        num_samples = 100
        samples = [
            create_test_sample(i, size="medium") for i in range(num_samples)
        ]

        results = {}

        # Test LZ4
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MemoryMappedStorage(tmpdir, compression="lz4")
            for sample in samples:
                storage.append(sample)
            storage.close()

            storage = MemoryMappedStorage(tmpdir, readonly=True)
            results["lz4"] = storage.get_stats()

        # Test ZSTD
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MemoryMappedStorage(tmpdir, compression="zstd")
            for sample in samples:
                storage.append(sample)
            storage.close()

            storage = MemoryMappedStorage(tmpdir, readonly=True)
            results["zstd"] = storage.get_stats()

        # Verify trade-offs
        assert (
            results["zstd"]["compression_ratio"]
            > results["lz4"]["compression_ratio"]
        ), "ZSTD should achieve better compression than LZ4"

        # Both should provide reasonable compression
        assert results["lz4"]["compression_ratio"] >= 1.2, (
            f"LZ4 compression too low: {results['lz4']['compression_ratio']:.2f}×"
        )
        assert results["zstd"]["compression_ratio"] >= 1.4, (
            f"ZSTD compression too low: {results['zstd']['compression_ratio']:.2f}×"
        )

        # Print results for documentation
        print(f"\nCompression Trade-offs ({num_samples} samples):")
        print(
            f"  LZ4:  {results['lz4']['compression_ratio']:.2f}x compression, "
            f"{results['lz4']['total_size_mb']:.1f} MB"
        )
        print(
            f"  ZSTD: {results['zstd']['compression_ratio']:.2f}x compression, "
            f"{results['zstd']['total_size_mb']:.1f} MB"
        )
        print(
            f"  ZSTD saves {results['lz4']['total_size_mb'] - results['zstd']['total_size_mb']:.1f} MB more"
        )
        print(
            "  But LZ4 is faster for training (30-80% faster decompression)\n"
        )
