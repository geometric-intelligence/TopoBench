"""High-performance storage backend for on-disk preprocessing."""

import json
import pickle
from pathlib import Path
from typing import Any

import lz4.frame
import numpy as np
import zstandard as zstd


class MemoryMappedStorage:
    """High-performance memory-mapped storage with compression.

    Uses memory-mapping and compression for efficient on-disk preprocessing:
    - Single memory-mapped file (zero-copy: direct access without copying to RAM)
    - Separate index file (O(1) random access)
    - LZ4 compression (default, ~1.35× ratio, fast decompression)
    - ZSTD compression (optional, ~1.7× ratio, slower decompression)

    Storage Layout
    --------------
    data_dir/samples.mmap    -- All samples (compressed, contiguous)
    data_dir/samples.idx     -- Index: [(offset, length), ...] per sample
    data_dir/metadata.json   -- Storage metadata

    Performance
    -----------
    Random access: O(1) via index lookup
    Memory usage: O(1) - only index in RAM (~16 bytes per sample)

    Parameters
    ----------
    data_dir : Path
        Directory for storage files.
    compression : str, optional
        Compression algorithm: "lz4", "zstd", or None (default: "lz4").
    readonly : bool, optional
        If True, open in read-only mode (default: False).

    Examples
    --------
    >>> storage = MemoryMappedStorage(Path("./processed"), compression="lz4")
    >>>
    >>> # Write samples
    >>> for sample in samples:
    ...     storage.append(sample)
    >>> storage.close()
    >>>
    >>> # Read samples
    >>> storage = MemoryMappedStorage(Path("./processed"), readonly=True)
    >>> sample = storage[42]  # O(1) access
    >>> stats = storage.get_stats()
    """

    def __init__(
        self,
        data_dir: Path,
        compression: str | None = "lz4",
        readonly: bool = False,
    ) -> None:
        """Initialize memory-mapped storage.

        Parameters
        ----------
        data_dir : Path
            Directory for storage files.
        compression : str, optional
            Compression algorithm (default: "lz4").
        readonly : bool, optional
            Open in read-only mode (default: False).
        """
        self.data_dir = Path(data_dir)
        self.compression = compression
        self.readonly = readonly

        # Storage file paths
        self.mmap_path = self.data_dir / "samples.mmap"
        self.index_path = self.data_dir / "samples.idx"
        self.metadata_path = self.data_dir / "metadata.json"

        # Initialize storage
        if readonly:
            self._load_existing()
        else:
            self._initialize_new()

    def _initialize_new(self) -> None:
        """Initialize new storage for writing."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize zstd decompressor if needed (reused for efficiency)
        self._zstd_dctx = (
            zstd.ZstdDecompressor() if self.compression == "zstd" else None
        )

        # Open mmap file in append mode
        self.mmap_file = self.mmap_path.open("ab")

        # Initialize index (list of (offset, length) tuples)
        self.index = []
        self.current_offset = 0

        # Track statistics
        self.total_uncompressed_bytes = 0
        self.total_compressed_bytes = 0

    def _load_existing(self) -> None:
        """Load existing storage for reading."""
        if not self.mmap_path.exists():
            raise FileNotFoundError(
                f"Storage file not found: {self.mmap_path}. "
                f"Cannot open in readonly mode."
            )

        # Load metadata first to get actual compression type
        self.metadata = self._load_metadata()

        # Override compression with actual value from metadata
        # (in case user specified wrong compression on open)
        # Check if key exists, not if value is not None (compression can be None)
        if "compression" in self.metadata:
            self.compression = self.metadata["compression"]

        # Initialize zstd decompressor after compression is determined (reused for efficiency)
        self._zstd_dctx = (
            zstd.ZstdDecompressor() if self.compression == "zstd" else None
        )

        # Load index
        self.index = self._load_index()

        # Memory-map the data file for zero-copy reads
        self.mmap_array = np.memmap(self.mmap_path, dtype=np.uint8, mode="r")

    def __len__(self) -> int:
        """Return number of samples (O(1)).

        Returns
        -------
        int
            Number of samples stored.
        """
        return len(self.index)

    def __getitem__(self, idx: int) -> Any:
        """Load sample via memory-mapped access (O(1)).

        Steps: index lookup → mmap read (zero-copy: direct access without copying) →
        decompress → deserialize.

        Parameters
        ----------
        idx : int
            Sample index (0-indexed).

        Returns
        -------
        Any
            Loaded sample data.

        Raises
        ------
        IndexError
            If index out of range.
        """
        if idx < 0 or idx >= len(self.index):
            raise IndexError(
                f"Index {idx} out of range for storage with "
                f"{len(self.index)} samples"
            )

        # Step 1: Index lookup (O(1))
        offset, length = self.index[idx]

        # Step 2: Memory-mapped read (zero-copy)
        compressed_data = bytes(self.mmap_array[offset : offset + length])

        # Step 3: Decompress if needed
        if self.compression == "lz4":
            serialized_data = lz4.frame.decompress(compressed_data)
        elif self.compression == "zstd":
            serialized_data = self._zstd_dctx.decompress(compressed_data)
        else:
            serialized_data = compressed_data

        # Step 4: Deserialize
        data = pickle.loads(serialized_data)

        return data

    def append(self, data: Any) -> None:
        """Append sample to storage (O(1) amortized).

        Compression and serialization happen here, ensuring fast
        preprocessing with minimal disk usage.

        Parameters
        ----------
        data : Any
            Sample to store.
        """
        if self.readonly:
            raise RuntimeError("Cannot append to readonly storage")

        # Step 1: Serialize
        serialized_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        uncompressed_size = len(serialized_data)

        # Step 2: Compress if enabled
        if self.compression == "lz4":
            compressed_data = lz4.frame.compress(
                serialized_data,
                compression_level=0,  # Fast compression
            )
        elif self.compression == "zstd":
            cctx = zstd.ZstdCompressor(level=3)  # Balanced speed/ratio
            compressed_data = cctx.compress(serialized_data)
        else:
            compressed_data = serialized_data

        compressed_size = len(compressed_data)

        # Step 3: Write to mmap file
        self.mmap_file.write(compressed_data)

        # Step 4: Update index
        self.index.append((self.current_offset, compressed_size))
        self.current_offset += compressed_size

        # Step 5: Update statistics
        self.total_uncompressed_bytes += uncompressed_size
        self.total_compressed_bytes += compressed_size

    def close(self) -> None:
        """Flush writes and close storage.

        Saves index and metadata to disk for future access.
        """
        if not self.readonly:
            # Close mmap file
            self.mmap_file.close()

            # Save index
            self._save_index()

            # Save metadata
            self._save_metadata()

    def _save_index(self) -> None:
        """Save index to disk for O(1) lookups."""
        # Convert to numpy array for efficient storage
        index_array = np.array(self.index, dtype=np.int64)
        # Use allow_pickle=False for security and performance
        np.save(self.index_path, index_array, allow_pickle=False)

    def _load_index(self) -> list:
        """Load index from disk.

        Returns
        -------
        list
            List of (offset, length) tuples.
        """
        # Add .npy extension if not present
        index_path = self.index_path
        if not index_path.exists() and not str(index_path).endswith(".npy"):
            index_path = Path(str(index_path) + ".npy")

        index_array = np.load(index_path, allow_pickle=False)
        return [tuple(row) for row in index_array]

    def _save_metadata(self) -> None:
        """Save storage metadata."""
        metadata = {
            "compression": self.compression,
            "num_samples": len(self.index),
            "total_uncompressed_bytes": self.total_uncompressed_bytes,
            "total_compressed_bytes": self.total_compressed_bytes,
            "compression_ratio": (
                self.total_uncompressed_bytes / self.total_compressed_bytes
                if self.total_compressed_bytes > 0
                else 1.0
            ),
        }

        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _load_metadata(self) -> dict:
        """Load storage metadata.

        Returns
        -------
        dict
            Metadata dictionary.
        """
        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                return json.load(f)
        return {}

    def get_stats(self) -> dict:
        """Get storage statistics.

        Returns
        -------
        dict
            Statistics including:
            - num_samples: Number of samples stored
            - total_size_mb: Total disk usage in MB
            - compression_ratio: Uncompressed / compressed ratio
            - avg_sample_size_kb: Average compressed sample size
        """
        if self.readonly:
            metadata = self.metadata
        else:
            metadata = {
                "num_samples": len(self.index),
                "total_compressed_bytes": self.total_compressed_bytes,
                "total_uncompressed_bytes": self.total_uncompressed_bytes,
            }

        num_samples = metadata.get("num_samples", len(self.index))
        compressed_bytes = metadata.get("total_compressed_bytes", 0)
        uncompressed_bytes = metadata.get("total_uncompressed_bytes", 0)

        return {
            "compression": self.compression,
            "num_samples": num_samples,
            "total_size_mb": compressed_bytes / (1024 * 1024),
            "compression_ratio": (
                uncompressed_bytes / compressed_bytes
                if compressed_bytes > 0
                else 1.0
            ),
            "avg_sample_size_kb": (
                compressed_bytes / num_samples / 1024 if num_samples > 0 else 0
            ),
        }
