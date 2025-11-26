"""On-disk preprocessor for inductive learning with large datasets.

This module provides a memory-efficient preprocessing implementation that
processes samples sequentially and stores them on disk, enabling training on
datasets larger than available RAM. This preprocessor applies transforms (e.g.,
lifting operations from graphs to simplicial complexes) one sample at a time to
maintain constant memory usage.
"""

import contextlib
import json
import os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch_geometric
from omegaconf import DictConfig
from torch.utils.data import Dataset

from topobench.data.preprocessor._ondisk.parallel_processor import (
    ParallelProcessor,
)
from topobench.data.preprocessor._ondisk.storage_backend import (
    MemoryMappedStorage,
)
from topobench.data.preprocessor._ondisk.transform_pipeline import (
    TransformPipeline,
)
from topobench.data.utils import (
    ensure_serializable,
    load_inductive_splits,
    make_hash,
)
from topobench.dataloader import DataloadDataset
from topobench.transforms.data_transform import DataTransform


def _torch_load_compat(path, **kwargs):
    """Backwards-compatible torch.load wrapper.

    PyTorch 2.6+ requires weights_only=False for PyG Data objects.
    Older versions don't have the weights_only parameter.
    """
    try:
        # Try with weights_only (PyTorch 2.6+)
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        # Fall back for older PyTorch versions
        return torch.load(path, **kwargs)


def _convert_shard_to_mmap(
    start_idx: int,
    end_idx: int,
    processed_dir: Path,
    shard_id: int,
    compression: str | None,
) -> dict[str, Any]:
    """Convert a shard of samples to a temporary mmap file.

    This function runs in a worker process to parallelize mmap conversion.

    Parameters
    ----------
    start_idx : int
        Starting sample index (inclusive).
    end_idx : int
        Ending sample index (exclusive).
    processed_dir : Path
        Directory containing individual .pt files.
    shard_id : int
        Shard identifier for temporary file naming.
    compression : str | None
        Compression algorithm ("lz4", "zstd", or None).

    Returns
    -------
    dict
        Statistics: num_samples, success_count, error_count.
    """
    # Create shard-specific storage
    shard_dir = processed_dir / f"_shard_{shard_id}"
    shard_dir.mkdir(exist_ok=True)

    storage = MemoryMappedStorage(
        data_dir=shard_dir,
        compression=compression,
        readonly=False,
    )

    success_count = 0
    error_count = 0
    files_to_delete = []  # Batch deletions for better I/O performance

    for idx in range(start_idx, end_idx):
        sample_path = processed_dir / f"sample_{idx:06d}.pt"

        if sample_path.exists():
            try:
                data = _torch_load_compat(sample_path)
                storage.append(data)
                # Mark for deletion (batch delete later)
                files_to_delete.append(sample_path)
                success_count += 1
            except Exception as e:
                error_count += 1
                # Log conversion failure for this sample
                print(
                    f"[OnDiskInductivePreprocessor] Failed to convert sample {idx}: {e}"
                )
        else:
            error_count += 1

    # Batch delete files for better I/O performance
    for file_path in files_to_delete:
        with contextlib.suppress(OSError):
            file_path.unlink()  # File might have been deleted already

    # Close storage to flush writes
    storage.close()

    return {
        "shard_id": shard_id,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "num_samples": end_idx - start_idx,
        "success": success_count,
        "errors": error_count,
    }


def _write_shard_to_offset(
    processed_dir: Path,
    shard_id: int,
    final_mmap_path: Path,
    offset: int,
    expected_size: int,
) -> dict[str, Any]:
    """Write shard data to specific offset in pre-allocated final file (parallel-safe).

    This function is designed for parallel execution - multiple workers can write
    to different offsets of the same file simultaneously without data corruption.

    Parameters
    ----------
    processed_dir : Path
        Directory containing shard subdirectories.
    shard_id : int
        Shard identifier.
    final_mmap_path : Path
        Path to final pre-allocated mmap file.
    offset : int
        Byte offset where this shard's data should be written.
    expected_size : int
        Expected size of shard data (for validation).

    Returns
    -------
    dict
        Statistics: shard_id, bytes_written, success status.

    Raises
    ------
    RuntimeError
        If shard size doesn't match expected size or write fails.
    """
    shard_dir = processed_dir / f"_shard_{shard_id}"
    shard_mmap_path = shard_dir / "samples.mmap"

    # Validate shard exists and has correct size
    if not shard_mmap_path.exists():
        raise RuntimeError(
            f"Shard {shard_id} mmap file not found: {shard_mmap_path}"
        )

    actual_size = shard_mmap_path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Shard {shard_id} size mismatch: expected {expected_size} bytes, "
            f"got {actual_size} bytes"
        )

    bytes_written = 0

    try:
        # Open final file for writing at specific offset (r+b = read+write binary)
        with (
            open(final_mmap_path, "r+b") as final_file,
            open(shard_mmap_path, "rb") as shard_file,
        ):
            # Use pwrite for atomic writes if available (Linux)
            if hasattr(os, "pwrite"):
                # Read entire shard (sizes are typically <100MB per shard)
                shard_data = shard_file.read()

                # Atomic write to specific offset
                written = os.pwrite(final_file.fileno(), shard_data, offset)
                bytes_written = written

                if written != expected_size:
                    raise RuntimeError(
                        f"Incomplete write for shard {shard_id}: "
                        f"wrote {written}/{expected_size} bytes"
                    )
            else:
                # Fallback: seek + buffered write (still parallel-safe for non-overlapping offsets)
                final_file.seek(offset)

                # Use large buffer for efficiency (4MB chunks)
                chunk_size = 4 * 1024 * 1024
                while True:
                    chunk = shard_file.read(chunk_size)
                    if not chunk:
                        break
                    written_chunk = final_file.write(chunk)
                    bytes_written += written_chunk

                if bytes_written != expected_size:
                    raise RuntimeError(
                        f"Incomplete write for shard {shard_id}: "
                        f"wrote {bytes_written}/{expected_size} bytes"
                    )

        return {
            "shard_id": shard_id,
            "bytes_written": bytes_written,
            "success": True,
        }

    except Exception as e:
        return {
            "shard_id": shard_id,
            "bytes_written": bytes_written,
            "success": False,
            "error": str(e),
        }


class _CachedTransformDataset(Dataset):
    """Dataset that loads from cached transform output (picklable for multiprocessing).

    This class is defined at module level to enable pickling for parallel processing.
    It wraps a cached transform directory and provides indexed access to samples.

    Parameters
    ----------
    cache_dir : Path
        Directory containing cached samples.
    num_samples : int
        Number of samples in cache.
    storage_backend : str
        Storage backend ("files" or "mmap").
    compression : str
        Compression algorithm (for mmap).
    """

    def __init__(
        self,
        cache_dir: Path,
        num_samples: int,
        storage_backend: str,
        compression: str,
    ):
        """Initialize cached dataset."""
        self.cache_dir = Path(cache_dir)
        self.num_samples = num_samples
        self.storage_backend = storage_backend
        self.compression = compression

        # Load storage if mmap
        if storage_backend == "mmap":
            try:
                self._storage = MemoryMappedStorage(
                    data_dir=self.cache_dir,
                    compression=compression,
                    readonly=True,
                )
            except FileNotFoundError:
                self._storage = None
        else:
            self._storage = None

    def __len__(self):
        """Return number of samples.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, idx):
        """Load sample from cache.

        Parameters
        ----------
        idx : int
            Index of the sample to load.

        Returns
        -------
        torch_geometric.data.Data
            The loaded sample data.
        """
        if self._storage is not None:
            return self._storage[idx]
        else:
            # Load from file
            sample_path = self.cache_dir / f"sample_{idx:06d}.pt"
            return _torch_load_compat(sample_path)

    def __reduce__(self):
        """Support pickling for multiprocessing.

        Returns
        -------
        tuple
            Tuple containing reconstruction function and serializable arguments.
        """
        return (
            _reconstruct_cached_transform_dataset,
            (
                str(self.cache_dir),
                self.num_samples,
                self.storage_backend,
                self.compression,
            ),
        )


def _reconstruct_cached_transform_dataset(
    cache_dir: str, num_samples: int, storage_backend: str, compression: str
):
    """Reconstruct _CachedTransformDataset from pickle (helper for __reduce__).

    Parameters
    ----------
    cache_dir : str
        Directory containing cached samples.
    num_samples : int
        Number of samples.
    storage_backend : str
        Storage backend.
    compression : str
        Compression algorithm.

    Returns
    -------
    _CachedTransformDataset
        Reconstructed dataset instance.
    """
    return _CachedTransformDataset(
        Path(cache_dir), num_samples, storage_backend, compression
    )


class OnDiskInductivePreprocessor(Dataset):
    """Disk-backed preprocessor for large-scale inductive learning.

    Processes samples one-at-a-time and stores results on disk to maintain O(1)
    memory usage regardless of dataset size. Supports DAG-based transform caching,
    parallel multi-worker processing, and flexible storage backends.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset or torch.utils.data.Dataset
        Source dataset to process. Memory usage is O(1) regardless of dataset type.
    data_dir : str or Path
        Root directory for storing processed samples.
    transforms_config : DictConfig, optional
        Transform configuration. If None, data is used as-is (default: None).
    force_reload : bool, optional
        If True, reprocess all samples even if cache exists (default: False).
    num_workers : int, optional
        Number of parallel workers (default: None = cpu_count-1).
        Set to 1 for sequential processing.
    batch_size : int, optional
        Batch size for parallel processing (default: 32).
    cache_size : int, optional
        Number of samples to cache in memory (default: 100).
        Set to 0 to disable. LRU eviction policy.
    storage_backend : str, optional
        Storage backend: "mmap" (compressed, default) or "files" (faster parallel).

        **Trade-off:**
        - "files" + many workers: 3-6× faster preprocessing, 4-7× larger disk usage
        - "mmap" + 1 worker: 4-7× smaller storage, slower preprocessing

    compression : str, optional
        Compression for mmap: None, "lz4" (default, fast), or "zstd" (better ratio).
        Only applies when storage_backend="mmap".
    transform_tier : str, optional
        Transform classification mode: "all_heavy" (default), "auto",
        "all_light", or "manual". Default: "all_heavy".
    tier_override : dict, optional
        Manual transform classification. Maps class names to "heavy" or "light".
        Only used when transform_tier="manual". Default: None.
    **kwargs : dict
        Additional arguments passed to parent Dataset class.

    Attributes
    ----------
    processed_dir : Path
        Directory containing processed sample files.
    num_samples : int
        Total number of samples in the dataset.
    transforms_parameters : dict
        Serialized transform parameters for cache validation.

    Examples
    --------
    >>> from torch_geometric.datasets import TUDataset
    >>> from omegaconf import DictConfig
    >>>
    >>> # Load source dataset
    >>> source = TUDataset(root='/tmp/data', name='ENZYMES')
    >>>
    >>> # Configure lifting transform
    >>> config = DictConfig({
    ...     'transform_name': 'liftings.graph2simplicial',
    ...     'complex_dim': 2
    ... })
    >>>
    >>> # Create preprocessor with parallel processing
    >>> dataset = OnDiskInductivePreprocessor(
    ...     dataset=source,
    ...     data_dir='/tmp/enzymes_processed',
    ...     transforms_config=config,
    ...     storage_backend="files",  # Fast for development
    ...     num_workers=4
    ... )
    >>>
    >>> # Use with dataloader for training
    >>> from topobench.dataloader import TBDataloader
    >>> train_ds, val_ds, test_ds = dataset.load_dataset_splits(split_params)
    >>> datamodule = TBDataloader(
    ...     dataset_train=train_ds,
    ...     dataset_val=val_ds,
    ...     dataset_test=test_ds,
    ...     batch_size=32
    ... )
    >>> # Train with PyTorch Lightning
    >>> trainer.fit(model, datamodule)
    """

    def __init__(
        self,
        dataset: torch_geometric.data.Dataset | torch.utils.data.Dataset,
        data_dir: str | Path,
        transforms_config: DictConfig | None = None,
        force_reload: bool = False,
        num_workers: int | None = None,
        batch_size: int | None = 32,
        cache_size: int = 100,
        storage_backend: str = "mmap",
        compression: str | None = "lz4",
        transform_tier: str = "all_heavy",
        tier_override: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OnDiskInductiveDataset.

        Parameters
        ----------
        dataset : torch_geometric.data.Dataset or torch.utils.data.Dataset
            Source dataset to process.
        data_dir : str or Path
            Root directory for storing processed samples.
        transforms_config : DictConfig, optional
            Configuration parameters for transforms (default: None).
        force_reload : bool, optional
            If True, reprocess all samples even if cache exists (default: False).
        num_workers : int, optional
            Number of parallel workers (default: None = auto-detect).
        batch_size : int, optional
            Batch size for parallel processing (default: 32).
        cache_size : int, optional
            Number of samples to cache in memory (default: 100).
        storage_backend : str, optional
            Storage backend: "mmap" or "files" (default: "mmap").
        compression : str, optional
            Compression: "lz4", "zstd", or None (default: "lz4").
        transform_tier : str, optional
            Transform classification mode (default: "all_heavy").
            - "all_heavy": All transforms processed offline
            - "auto": Automatic classification into heavy/light
            - "all_light": All transforms applied at runtime
            - "manual": Use tier_override for classification
        tier_override : dict, optional
            Manual transform classification overrides (default: None).
            Maps transform class names to "heavy" or "light".
        **kwargs : dict
            Additional arguments passed to parent Dataset class.
        """
        super().__init__()
        self.dataset = dataset
        self.data_dir = Path(data_dir)
        self.transforms_config = transforms_config
        self.force_reload = force_reload
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.storage_backend = storage_backend
        self.compression = compression
        self.transform_tier = transform_tier
        self.tier_override = tier_override

        # Initialize in-memory LRU cache for training speedup (1.2-1.3×)
        # OrderedDict provides O(1) access, insertion, and deletion
        self._cache: OrderedDict[int, torch_geometric.data.Data] = (
            OrderedDict()
        )
        self._cache_hits = 0
        self._cache_misses = 0

        # Storage will be initialized after processed_dir is set
        self._storage: MemoryMappedStorage | None = None

        # Store split_idx if available (for datasets with fixed splits)
        if hasattr(dataset, "split_idx"):
            self.split_idx = dataset.split_idx

        # Initialize transform and processed directory
        if transforms_config is not None:
            self.pre_transform = self._instantiate_pre_transform(
                transforms_config
            )
            # Create two-tier transform pipeline
            self._create_transform_pipeline()
            self._set_processed_data_dir(transforms_config)
        else:
            # No transforms - use data as-is
            self.pre_transform = None
            self.transform_pipeline = None
            self.processed_dir = self.data_dir / "no_transforms"

        # Ensure processed directory exists
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Load or create metadata (use dataset_metadata to avoid conflict with storage)
        self.metadata_path = self.processed_dir / "dataset_metadata.json"
        if self._should_process():
            self._process_samples()
        else:
            self._load_metadata()

        # Initialize storage backend for reading
        if self.storage_backend == "mmap":
            self._init_storage_backend()

    def __repr__(self) -> str:
        """Return string representation of dataset.

        Returns
        -------
        str
            String describing the dataset and its size.
        """
        return (
            f"{self.__class__.__name__}("
            f"num_samples={self.num_samples}, "
            f"processed_dir={self.processed_dir})"
        )

    def __len__(self) -> int:
        """Return number of samples in dataset.

        Returns
        -------
        int
            Total number of samples.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> torch_geometric.data.Data:
        """Load sample from disk with LRU caching.

        Fast path (cache hit, ~0.01 ms):
        1. Check cache → return immediately

        Slow path (cache miss, ~15 ms):
        1. Load from disk (torch.load)
        2. Add to cache (if cache enabled)
        3. Evict oldest if cache full (LRU policy)

        Parameters
        ----------
        idx : int
            Sample index (0-indexed).

        Returns
        -------
        torch_geometric.data.Data
            Loaded data sample.

        Raises
        ------
        IndexError
            If index is out of range.
        FileNotFoundError
            If sample file doesn't exist on disk.
        """
        # Support negative indexing like Python lists
        if idx < 0:
            idx = self.num_samples + idx

        if idx < 0 or idx >= self.num_samples:
            raise IndexError(
                f"Index {idx} out of range for dataset of size "
                f"{self.num_samples}"
            )

        # Fast path: Check cache first (O(1))
        if self.cache_size > 0 and idx in self._cache:
            self._cache_hits += 1
            # Move to end (most recently used)
            self._cache.move_to_end(idx)
            return self._cache[idx]

        # Slow path: Load from storage
        self._cache_misses += 1

        # Load based on storage backend
        if self.storage_backend == "mmap" and self._storage is not None:
            # Fast: Memory-mapped storage with zero-copy reads
            data = self._storage[idx]
        else:
            # Fallback: File-based storage
            sample_path = self._get_sample_path(idx)

            if not sample_path.exists():
                raise FileNotFoundError(
                    f"Sample file not found: {sample_path}. "
                    f"Dataset may be corrupted. Try force_reload=True."
                )

            # Load sample from disk
            # PyTorch 2.6+ requires weights_only=False for PyG Data objects
            data = _torch_load_compat(sample_path)

        # Apply light transforms at runtime (two-tier system)
        if (
            hasattr(self, "transform_pipeline")
            and self.transform_pipeline is not None
            and self.transform_pipeline.light_compose is not None
        ):
            data = self.transform_pipeline.apply_light(data)

        # Add to cache if enabled
        if self.cache_size > 0:
            self._cache[idx] = data
            self._cache.move_to_end(idx)  # Mark as most recently used

            # Evict oldest if cache full (LRU policy)
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)  # Remove oldest (FIFO)

        return data

    def _should_process(self) -> bool:
        """Check if dataset needs processing (DAG-aware).

        For DAG-based caching:
        - Check if final transform output exists and is valid
        - If transform chain exists, use incremental checking
        - Returns True if any transform in chain needs processing

        Returns
        -------
        bool
            True if processing is needed, False if all transforms cached.
        """
        if self.force_reload:
            return True

        # DAG-aware: Check if any transform in chain needs processing
        if hasattr(self, "transform_chain") and self.transform_chain:
            # Check each transform in chain
            for entry in self.transform_chain:
                if not entry["cached"]:
                    # At least one transform not cached, need processing
                    return True

            # All transforms cached, check final output validity
            if not self.metadata_path.exists():
                return True

            try:
                with open(self.metadata_path) as f:
                    metadata = json.load(f)

                # Verify transform chain matches
                saved_chain = metadata.get("transform_chain", [])
                if len(saved_chain) != len(self.transform_chain):
                    return True

                # Check each transform hash matches
                for saved, current in zip(
                    saved_chain, self.transform_chain, strict=True
                ):
                    if saved.get("hash") != current["hash"]:
                        return True

                return False
            except (json.JSONDecodeError, KeyError, FileNotFoundError):
                return True

        # No transform chain
        if not self.metadata_path.exists():
            return True

        # Verify storage files exist based on backend
        try:
            with open(self.metadata_path) as f:
                metadata = json.load(f)
            num_samples = metadata.get("num_samples", 0)

            if self.storage_backend == "mmap":
                # Check for memory-mapped storage files
                mmap_path = self.processed_dir / "samples.mmap"
                idx_path = self.processed_dir / "samples.idx.npy"
                storage_metadata_path = self.processed_dir / "metadata.json"

                if not (
                    mmap_path.exists()
                    and idx_path.exists()
                    and storage_metadata_path.exists()
                ):
                    return True
            else:
                # Check for individual sample files
                for idx in range(num_samples):
                    sample_path = self._get_sample_path(idx)
                    if not sample_path.exists():
                        return True

            # Verify transform parameters match (if applicable)
            if self.transforms_config is not None:
                saved_params = metadata.get("transforms_parameters", {})
                if saved_params != self.transforms_parameters:
                    return True

            return False

        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            return True

    def _process_samples(self) -> None:
        """Apply DAG-based incremental caching to process samples.

        Only process uncached transforms, reuse cached ones!
        """
        # DAG-aware incremental processing
        if hasattr(self, "transform_chain") and self.transform_chain:
            self._process_samples_incremental()
        else:
            # Legacy: Full processing (no transform chain)
            self._process_samples_full()

    def _process_samples_incremental(self) -> None:
        """Apply incremental processing using transform chain cache.

        Only process uncached transforms.
        """
        # Find which transforms need processing
        uncached_indices = [
            i
            for i, entry in enumerate(self.transform_chain)
            if not entry["cached"]
        ]

        if not uncached_indices:
            # All cached! Just load metadata
            self._load_metadata()
            return

        # Process each uncached transform individually to maintain DAG structure
        # Each transform writes to its own directory
        dag = self.transform_pipeline.get_dag()

        for transform_idx in uncached_indices:
            chain_entry = self.transform_chain[transform_idx]
            transform_id = chain_entry["transform_id"]
            output_dir = Path(chain_entry["output_dir"])

            # Determine source for this transform
            if transform_idx == 0:
                # First transform: use original dataset
                source_dataset = self.dataset
            else:
                # Load from previous transform's output
                prev_entry = self.transform_chain[transform_idx - 1]
                prev_dir = Path(prev_entry["output_dir"])
                source_dataset = self._create_cached_dataset(prev_dir)

            # Get single transform (not composed)
            node = dag.nodes[transform_id]
            source_transform = node.transform

            # Save original processed_dir and metadata_path, temporarily set to this transform's output
            original_processed_dir = self.processed_dir
            original_metadata_path = self.metadata_path
            self.processed_dir = output_dir
            self.metadata_path = output_dir / "dataset_metadata.json"

            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Process this transform only
                self._process_samples_full(
                    source_dataset=source_dataset,
                    source_transform=source_transform,
                )
            finally:
                # Restore original processed_dir and metadata_path
                self.processed_dir = original_processed_dir
                self.metadata_path = original_metadata_path

    def _create_cached_dataset(self, cached_dir: Path) -> Dataset:
        """Create dataset that loads from cached transform output.

        Parameters
        ----------
        cached_dir : Path
            Directory containing cached samples.

        Returns
        -------
        Dataset
            Dataset that loads from cache.
        """
        # Load metadata to get num_samples
        metadata_path = cached_dir / "dataset_metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        num_samples = metadata["num_samples"]

        return _CachedTransformDataset(
            cached_dir, num_samples, self.storage_backend, self.compression
        )

    def _create_partial_transform(
        self, start_idx: int
    ) -> torch_geometric.transforms.Compose:
        """Create transform composed of only uncached transforms.

        Parameters
        ----------
        start_idx : int
            Index of first uncached transform in chain.

        Returns
        -------
        Compose
            Composed transform of uncached transforms.
        """
        # Get uncached transforms from pipeline
        dag = self.transform_pipeline.get_dag()
        uncached_transform_ids = [
            self.transform_chain[i]["transform_id"]
            for i in range(start_idx, len(self.transform_chain))
        ]

        # Extract transform objects
        transforms = [
            dag.nodes[tid].transform for tid in uncached_transform_ids
        ]

        if not transforms:
            return None

        return torch_geometric.transforms.Compose(transforms)

    def _process_samples_full(
        self,
        source_dataset: Dataset | None = None,
        source_transform: torch_geometric.transforms.Compose | None = None,
    ) -> None:
        """Full processing (used by both legacy and incremental paths).

        Parameters
        ----------
        source_dataset : Dataset, optional
            Source dataset to process (default: self.dataset).
        source_transform : Compose, optional
            Transform to apply (default: self.pre_transform).
        """
        if source_dataset is None:
            source_dataset = self.dataset
        if source_transform is None:
            source_transform = self.pre_transform

        import time as time_module

        processing_start = time_module.time()

        # High-level progress message
        print(
            f"[OnDiskInductivePreprocessor] Processing {len(source_dataset)} "
            f"samples into {self.processed_dir}"
        )

        # Clear existing files if force_reload
        if self.force_reload:
            self._clear_processed_files()

        # Process using parallel processor
        processor = ParallelProcessor(
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            show_progress=True,
        )

        # TIMING: Transform application
        transform_start = time_module.time()

        # Process samples (parallel or sequential)
        results = processor.process(
            dataset=source_dataset,
            transform=source_transform,
            output_dir=self.processed_dir,
            num_samples=len(source_dataset),
        )

        transform_time = time_module.time() - transform_start

        # DEBUG: Print processing results
        print(f"[OnDiskInductivePreprocessor] Processing results: {results}")
        if results["failed"] > 0:
            print(
                f"[OnDiskInductivePreprocessor] Errors: {results['errors'][:3]}"
            )  # Show first 3 errors

        # Save metadata
        self.num_samples = len(source_dataset)
        self._save_metadata()

        # Convert to memory-mapped storage if requested (only if samples succeeded)
        if self.storage_backend == "mmap" and results["success"] > 0:
            self._convert_to_mmap_storage()

        total_processing_time = time_module.time() - processing_start

        # Summary timing information
        print(
            f"[OnDiskInductivePreprocessor] Transform processing time: "
            f"{transform_time:.2f}s "
            f"({self.num_samples / max(transform_time, 1e-6):.1f} samples/s)"
        )
        print(
            f"[OnDiskInductivePreprocessor] Total preprocessing time "
            f"(transforms + storage): {total_processing_time:.2f}s"
        )

    def _instantiate_pre_transform(
        self, transforms_config: DictConfig
    ) -> torch_geometric.transforms.Compose:
        """Instantiate transform from configuration.

        Parameters
        ----------
        transforms_config : DictConfig
            Transform configuration parameters.

        Returns
        -------
        torch_geometric.transforms.Compose
            Composed transform object.
        """
        # Handle nested liftings config
        if transforms_config.keys() == {"liftings"}:
            transforms_config = transforms_config.liftings

        # Check if single or multiple transforms
        if "transform_name" in transforms_config:
            # Single transform
            pre_transforms_dict = {
                transforms_config.transform_name: DataTransform(
                    **transforms_config
                )
            }
        else:
            # Multiple transforms
            pre_transforms_dict = {
                key: DataTransform(**value)
                for key, value in transforms_config.items()
            }

        # Store parameters for caching
        transforms_parameters = {
            transform_name: transform.parameters
            for transform_name, transform in pre_transforms_dict.items()
        }
        self.transforms_parameters = ensure_serializable(transforms_parameters)

        # Return composed transform
        return torch_geometric.transforms.Compose(
            list(pre_transforms_dict.values())
        )

    def _create_transform_pipeline(self) -> None:
        """Create two-tier transform pipeline from pre_transform.

        Separates transforms into heavy (offline) and light (runtime) tiers
        based on transform_tier setting. Updates self.pre_transform to use
        only heavy transforms for preprocessing.
        """
        if self.pre_transform is None:
            self.transform_pipeline = None
            return

        # Extract transforms from Compose object
        if hasattr(self.pre_transform, "transforms"):
            transforms = self.pre_transform.transforms
        else:
            # Single transform, wrap in list
            transforms = [self.pre_transform]

        # Create pipeline with tier classification
        self.transform_pipeline = TransformPipeline(
            transforms=transforms,
            transform_tier=self.transform_tier,
            tier_override=self.tier_override,
        )

        # Update pre_transform to use only heavy transforms
        # This ensures preprocessing only applies heavy transforms
        if self.transform_pipeline.heavy_compose is not None:
            self.pre_transform = self.transform_pipeline.heavy_compose
        else:
            # No heavy transforms, set to None
            self.pre_transform = None

    def _set_processed_data_dir(self, transforms_config: DictConfig) -> None:
        """Set processed data directory based on transform chain (DAG-aware).

        Creates per-transform directories enabling incremental caching:
        - Each transform gets its own directory with unique hash
        - Adding transforms reuses existing cached transforms
        - Only changed/new transforms are recomputed

        Directory structure:
            data_dir/
                transform_chain/
                    {transform_name_0}/
                        {hash_0}/
                            samples.mmap
                    {transform_name_1}/
                        {hash_1}/
                            samples.mmap

        Parameters
        ----------
        transforms_config : DictConfig
            Transform configuration parameters.
        """
        # Resolve transform chain from DAG
        if hasattr(self, "transform_pipeline") and self.transform_pipeline:
            self._resolve_transform_chain()
        else:
            # Fallback: No pipeline, use legacy behavior
            repo_name = "_".join(list(transforms_config.keys()))
            params_hash = make_hash(self.transforms_parameters)
            self.processed_dir = self.data_dir / repo_name / f"{params_hash}"
            self.transform_chain = None

    def _resolve_transform_chain(self) -> None:
        """Resolve transform chain from DAG with incremental cache checking.

        This is the key method for DAG-based caching. It:
        1. Iterates through transforms in execution order
        2. Checks which transforms are already cached
        3. Determines which transforms need processing
        4. Sets final output directory (processed_dir)
        """
        dag = self.transform_pipeline.get_dag()

        # Build transform chain metadata
        chain = []
        for transform_id in dag.execution_order:
            node = dag.nodes[transform_id]

            # Only track heavy transforms (light transforms are runtime)
            if node.tier != "heavy":
                continue

            # Use transform_id + hash for cache directory to avoid collisions:
            # - transform_id: Distinguishes duplicate transforms at different positions
            # - hash: Distinguishes same transform with different parameters
            # Format: {transform_id}_{hash} (e.g., DataTransform_0_b13b3327)
            transform_class = node.transform.__class__.__name__
            transform_hash = node.hash_value
            transform_dir = (
                self.data_dir
                / "transform_chain"
                / f"{transform_id}_{transform_hash}"
            )

            # Check if this transform is cached
            is_cached = self._check_transform_cached(transform_dir)

            chain_entry = {
                "transform_id": transform_id,
                "transform_class": transform_class,
                "hash": transform_hash,
                "dependencies": node.dependencies,
                "output_dir": str(transform_dir),
                "cached": is_cached,
            }
            chain.append(chain_entry)

        # Store chain for metadata and processing
        self.transform_chain = chain

        # Set processed_dir to final transform output
        if chain:
            self.processed_dir = Path(chain[-1]["output_dir"])
        else:
            # No heavy transforms, use default
            self.processed_dir = self.data_dir / "no_heavy_transforms"

    def _check_transform_cached(self, transform_dir: Path) -> bool:
        """Check if a transform's output is cached.

        Parameters
        ----------
        transform_dir : Path
            Directory where transform output should be stored.

        Returns
        -------
        bool
            True if cached (all required files exist), False otherwise.
        """
        # Force reload: Treat all transforms as uncached
        if self.force_reload:
            return False

        if not transform_dir.exists():
            return False

        # Check metadata exists
        metadata_path = transform_dir / "dataset_metadata.json"
        if not metadata_path.exists():
            return False

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            num_samples = metadata.get("num_samples", 0)

            # Check storage files exist
            if self.storage_backend == "mmap":
                mmap_path = transform_dir / "samples.mmap"
                idx_path = transform_dir / "samples.idx.npy"
                storage_metadata_path = transform_dir / "metadata.json"

                if not (
                    mmap_path.exists()
                    and idx_path.exists()
                    and storage_metadata_path.exists()
                ):
                    return False
            else:
                # Check sample files exist
                for idx in range(
                    min(10, num_samples)
                ):  # Sample check first 10
                    sample_path = transform_dir / f"sample_{idx:06d}.pt"
                    if not sample_path.exists():
                        return False

            return True
        except (json.JSONDecodeError, KeyError, OSError):
            return False

    def _get_sample_path(self, idx: int) -> Path:
        """Get file path for sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Path
            Path to sample file on disk.
        """
        return self.processed_dir / f"sample_{idx:06d}.pt"

    def _clear_processed_files(self) -> None:
        """Clear all processed sample files."""
        for sample_file in self.processed_dir.glob("sample_*.pt"):
            sample_file.unlink()

        if self.metadata_path.exists():
            self.metadata_path.unlink()

    def _save_metadata(self) -> None:
        """Save dataset metadata to disk with DAG transform chain info."""
        metadata = {
            "num_samples": self.num_samples,
            "source_dataset": str(type(self.dataset).__name__),
            "processed_dir": str(self.processed_dir),
        }

        # Add DAG transform chain (enables incremental caching)
        if hasattr(self, "transform_chain") and self.transform_chain:
            metadata["transform_chain"] = self.transform_chain
            metadata["final_output"] = str(self.processed_dir)

        # Add transform tier information
        if hasattr(self, "transform_pipeline") and self.transform_pipeline:
            summary = self.transform_pipeline.get_summary()
            metadata["transform_tier"] = self.transform_tier
            metadata["heavy_transforms"] = summary["heavy_names"]
            metadata["light_transforms"] = summary["light_names"]

        # Always use self.transforms_parameters for consistency
        # This ensures cache validation uses the same keys as instantiation
        if self.transforms_config is not None:
            metadata["transforms_parameters"] = self.transforms_parameters

        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _load_metadata(self) -> None:
        """Load dataset metadata from disk.

        Raises
        ------
        FileNotFoundError
            If metadata file doesn't exist.
        json.JSONDecodeError
            If metadata file is corrupted.
        """
        with open(self.metadata_path) as f:
            metadata = json.load(f)
        self.num_samples = metadata["num_samples"]

    def load_dataset_splits(
        self, split_params: DictConfig
    ) -> tuple[
        DataloadDataset, DataloadDataset | None, DataloadDataset | None
    ]:
        """Load dataset splits for training/validation/testing.

        This method creates DataloadDataset wrappers for use with TopoBench's
        TBDataloader. For inductive learning, separate datasets are created
        for each split.

        Parameters
        ----------
        split_params : DictConfig
            Parameters for splitting the dataset. Must include:
            - learning_setting: 'inductive' (required)
            - Additional split configuration (e.g., train_ratio, val_ratio)

        Returns
        -------
        tuple
            A tuple containing (train_dataset, val_dataset, test_dataset).
            Val and test may be None depending on split_params.

        Raises
        ------
        ValueError
            If learning_setting is not 'inductive' or is missing.
        """
        if not split_params.get("learning_setting", False):
            raise ValueError("No learning setting specified in split_params")

        if split_params.learning_setting != "inductive":
            raise ValueError(
                f"OnDiskInductiveDataset only supports inductive learning. "
                f"Got: {split_params.learning_setting}"
            )

        # Use lazy splits to maintain O(1) memory per split.
        # Labels are accumulated (O(n)) but are typically small compared to graph data.
        return load_inductive_splits(self, split_params, use_lazy=True)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics.

        Returns
        -------
        dict
            Cache statistics including:
            - enabled: Whether cache is enabled
            - size: Current number of cached samples
            - capacity: Maximum cache size
            - hits: Number of cache hits
            - misses: Number of cache misses
            - hit_rate: Cache hit rate (0-1)
            - total_accesses: Total number of __getitem__ calls

        Examples
        --------
        >>> dataset = OnDiskInductivePreprocessor(..., cache_size=100)
        >>> # Train for a few epochs
        >>> stats = dataset.get_cache_stats()
        >>> print(f"Cache hit rate: {stats['hit_rate']:.1%}")
        Cache hit rate: 68.5%
        """
        total_accesses = self._cache_hits + self._cache_misses
        hit_rate = (
            self._cache_hits / total_accesses if total_accesses > 0 else 0.0
        )

        return {
            "enabled": self.cache_size > 0,
            "size": len(self._cache),
            "capacity": self.cache_size,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "total_accesses": total_accesses,
        }

    def clear_cache(self) -> None:
        """Clear the in-memory cache and reset statistics.

        Useful for:
        - Freeing memory after training
        - Resetting statistics between experiments
        - Forcing cold reads for benchmarking

        Examples
        --------
        >>> dataset.clear_cache()  # Free ~50 MB of memory
        >>> stats = dataset.get_cache_stats()
        >>> assert stats['size'] == 0
        >>> assert stats['hits'] == 0
        """
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def _init_storage_backend(self) -> None:
        """Initialize memory-mapped storage backend for reading.

        Opens existing storage in readonly mode for fast sample access.
        """
        try:
            self._storage = MemoryMappedStorage(
                data_dir=self.processed_dir,
                compression=self.compression,
                readonly=True,
            )
        except FileNotFoundError:
            # Storage files don't exist, will use file-based fallback
            self._storage = None

    def _convert_to_mmap_storage(self) -> None:
        """Convert individual .pt files to memory-mapped storage using parallel workers.

        This consolidates individual sample files into a single mmap file
        with compression.
        """
        import time as time_module

        conversion_start = time_module.time()
        print(
            "[OnDiskInductivePreprocessor] Converting samples to memory-mapped storage (parallel mode)"
        )

        # Determine number of workers (use same as preprocessing)
        if self.num_workers is None:
            # Auto-detect optimal worker count (same as ParallelProcessor)
            cpu_count = os.cpu_count() or 1
            num_workers = max(1, cpu_count - 1)
        else:
            num_workers = max(1, self.num_workers)

        if num_workers == 1 or self.num_samples < 1000:
            # Use sequential for small datasets or single worker
            self._convert_to_mmap_storage_sequential()
            return

        # Divide samples into shards
        shard_size = (self.num_samples + num_workers - 1) // num_workers
        shards = []

        for shard_id in range(num_workers):
            start_idx = shard_id * shard_size
            end_idx = min(start_idx + shard_size, self.num_samples)

            if start_idx < self.num_samples:
                shards.append((start_idx, end_idx, shard_id))

        # TIMING: Shard creation
        shard_start = time_module.time()

        # Process shards in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _convert_shard_to_mmap,
                    start_idx,
                    end_idx,
                    self.processed_dir,
                    shard_id,
                    self.compression,
                ): shard_id
                for start_idx, end_idx, shard_id in shards
            }

            # Wait for all shards to complete
            results = []
            total_errors = 0
            for future in as_completed(futures):
                shard_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    total_errors += result["errors"]
                except Exception as e:
                    raise RuntimeError(
                        f"Shard {shard_id} conversion failed: {e}"
                    ) from e

        # Check if any samples failed
        total_samples = sum(r["num_samples"] for r in results)
        if total_errors > 0:
            failure_rate = (
                total_errors / total_samples if total_samples > 0 else 0
            )
            raise RuntimeError(
                f"Sample conversion failed during mmap storage creation: "
                f"{total_errors}/{total_samples} samples failed ({failure_rate * 100:.2f}%).\n"
                f"All samples must convert successfully to maintain dataset integrity.\n"
                f"Possible causes: corrupted .pt files, disk I/O errors, insufficient disk space.\n"
                f"Check the error messages above for details."
            )

        shard_time = time_module.time() - shard_start
        print(
            f"[OnDiskInductivePreprocessor] Shard creation completed in "
            f"{shard_time:.2f}s "
            f"({self.num_samples / max(shard_time, 1e-6):.1f} samples/s)"
        )

        # TIMING: Merge phase
        merge_start = time_module.time()

        # Merge shards into final mmap file
        self._merge_shards(len(shards))

        merge_time = time_module.time() - merge_start
        print(
            f"[OnDiskInductivePreprocessor] Shard merge completed in {merge_time:.2f}s"
        )

        # TIMING: Cleanup phase
        cleanup_start = time_module.time()

        # Clean up shard directories
        for shard_id in range(len(shards)):
            shard_dir = self.processed_dir / f"_shard_{shard_id}"
            if shard_dir.exists():
                # Remove shard files
                for f in shard_dir.iterdir():
                    f.unlink()
                shard_dir.rmdir()

        cleanup_time = time_module.time() - cleanup_start
        total_conversion_time = time_module.time() - conversion_start

        print(
            f"[OnDiskInductivePreprocessor] Cleanup completed in {cleanup_time:.2f}s"
        )
        print(
            f"[OnDiskInductivePreprocessor] Total mmap conversion time: "
            f"{total_conversion_time:.2f}s"
        )

    def _convert_to_mmap_storage_sequential(self) -> None:
        """Sequential fallback for mmap conversion (small datasets or single worker)."""
        # Create new storage in write mode
        self._storage = MemoryMappedStorage(
            data_dir=self.processed_dir,
            compression=self.compression,
            readonly=False,
        )

        # Read all samples from individual files and write to mmap
        for idx in range(self.num_samples):
            sample_path = self._get_sample_path(idx)
            if sample_path.exists():
                data = _torch_load_compat(sample_path)
                self._storage.append(data)
                # Delete individual file to save space
                sample_path.unlink()

        # Close storage to flush writes and save index
        self._storage.close()

        # Reopen in readonly mode for subsequent reads
        self._storage = MemoryMappedStorage(
            data_dir=self.processed_dir,
            compression=self.compression,
            readonly=True,
        )

    def _merge_shards(self, num_shards: int) -> None:
        """Merge shard mmap files into final consolidated mmap file using PARALLEL writes.

        Uses parallel workers to write shards to non-overlapping offsets in a pre-allocated
        file for optimal performance (4-8× faster than sequential merge on multi-core systems).

        Parameters
        ----------
        num_shards : int
            Number of shards to merge.
        """
        final_mmap_path = self.processed_dir / "samples.mmap"
        final_index_path = self.processed_dir / "samples.idx.npy"
        final_metadata_path = self.processed_dir / "metadata.json"

        import time as time_module

        # TIMING: Load shard metadata
        load_start = time_module.time()

        # Pre-load all shard metadata and indices for vectorized operations
        shard_indices = []
        shard_sizes = []
        shard_metadata_list = []
        total_samples = 0

        for shard_id in range(num_shards):
            shard_dir = self.processed_dir / f"_shard_{shard_id}"
            shard_mmap_path = shard_dir / "samples.mmap"
            shard_index_path = shard_dir / "samples.idx.npy"
            shard_metadata_path = shard_dir / "metadata.json"

            # Load shard index and metadata
            shard_index = np.load(shard_index_path, allow_pickle=False)
            with open(shard_metadata_path) as f:
                shard_metadata = json.load(f)

            shard_indices.append(shard_index)
            shard_sizes.append(shard_mmap_path.stat().st_size)
            shard_metadata_list.append(shard_metadata)
            total_samples += len(shard_index)

        load_time = time_module.time() - load_start
        print(
            f"[OnDiskInductivePreprocessor] Loaded shard metadata in {load_time:.2f}s"
        )

        # Vectorized: Compute cumulative offsets for all shards
        cumulative_offsets = np.concatenate(([0], np.cumsum(shard_sizes[:-1])))
        total_size = sum(shard_sizes)

        # Pre-allocation
        prealloc_start = time_module.time()

        # Pre-allocate final file with total size
        # This enables parallel writes to different offsets
        with open(final_mmap_path, "wb") as f:
            # Sparse file allocation (instant on most filesystems)
            f.seek(total_size - 1)
            f.write(b"\0")

        prealloc_time = time_module.time() - prealloc_start
        print(
            f"[OnDiskInductivePreprocessor] Pre-allocated {total_size / (1024 * 1024):.1f} MB "
            f"for final storage in {prealloc_time:.2f}s"
        )

        # Parallel write
        write_start = time_module.time()

        # Write shards to pre-allocated file using multiple workers
        # Each worker writes to its designated offset (no overlap = no corruption!)
        if num_shards == 1 or self.num_workers == 1:
            # Single shard or single worker: skip parallel overhead
            result = _write_shard_to_offset(
                self.processed_dir,
                0,
                final_mmap_path,
                int(cumulative_offsets[0]),
                shard_sizes[0],
            )
            if not result["success"]:
                raise RuntimeError(
                    f"Shard merge failed: {result.get('error', 'Unknown error')}"
                )
        else:
            # Determine number of workers (use same as preprocessing)
            if self.num_workers is None:
                num_workers = max(1, (os.cpu_count() or 1) - 1)
            else:
                num_workers = max(1, self.num_workers)

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {}
                for shard_id in range(num_shards):
                    future = executor.submit(
                        _write_shard_to_offset,
                        self.processed_dir,
                        shard_id,
                        final_mmap_path,
                        int(
                            cumulative_offsets[shard_id]
                        ),  # Each shard has unique offset
                        shard_sizes[shard_id],
                    )
                    futures[future] = shard_id

                # Wait for all writes to complete
                total_written = 0
                for future in as_completed(futures):
                    shard_id = futures[future]
                    try:
                        result = future.result()
                        if not result["success"]:
                            raise RuntimeError(
                                f"Shard {shard_id} merge failed: {result.get('error', 'Unknown error')}"
                            )
                        total_written += result["bytes_written"]
                    except Exception as e:
                        raise RuntimeError(
                            f"Parallel shard merge failed for shard {shard_id}: {e}"
                        ) from e

                # Validate total bytes written
                if total_written != total_size:
                    raise RuntimeError(
                        f"Incomplete merge: wrote {total_written}/{total_size} bytes"
                    )

        write_time = time_module.time() - write_start
        print(
            f"[OnDiskInductivePreprocessor] Parallel write completed in {write_time:.2f}s"
        )

        # Vectorized: Adjust all indices at once using NumPy broadcasting
        final_index = np.empty((total_samples, 2), dtype=np.int64)
        current_pos = 0
        for _shard_id, (shard_index, offset) in enumerate(
            zip(shard_indices, cumulative_offsets, strict=True)
        ):
            num_shard_samples = len(shard_index)
            # Vectorized offset adjustment
            final_index[current_pos : current_pos + num_shard_samples, 0] = (
                shard_index[:, 0] + offset
            )
            final_index[current_pos : current_pos + num_shard_samples, 1] = (
                shard_index[:, 1]
            )
            current_pos += num_shard_samples

        # Save final index
        np.save(final_index_path, final_index, allow_pickle=False)

        index_time = time_module.time() - load_start
        print(
            f"[OnDiskInductivePreprocessor] Index computation completed in {index_time:.2f}s"
        )

        # Vectorized: Sum stats using NumPy
        total_uncompressed = sum(
            m.get("total_uncompressed_bytes", 0) for m in shard_metadata_list
        )
        total_compressed = sum(
            m.get("total_compressed_bytes", 0) for m in shard_metadata_list
        )

        # Save final metadata as JSON
        metadata = {
            "compression": self.compression,
            "num_samples": len(final_index),
            "total_uncompressed_bytes": total_uncompressed,
            "total_compressed_bytes": total_compressed,
            "compression_ratio": (
                total_uncompressed / total_compressed
                if total_compressed > 0
                else 1.0
            ),
        }

        with open(final_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Reopen in readonly mode for subsequent reads
        self._storage = MemoryMappedStorage(
            data_dir=self.processed_dir,
            compression=self.compression,
            readonly=True,
        )
