"""Factory for creating appropriate preprocessor based on dataset and mode."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

    import torch
    import torch_geometric
    from omegaconf import DictConfig

    from .ondisk_inductive import OnDiskInductivePreprocessor
    from .preprocessor import PreProcessor

    PreprocessorType = PreProcessor | OnDiskInductivePreprocessor


def _is_transductive(
    dataset: torch_geometric.data.Dataset | torch.utils.data.Dataset,
) -> bool:
    """Determine if dataset is for transductive learning.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset or torch.utils.data.Dataset
        Dataset to check.

    Returns
    -------
    bool
        True if transductive (single graph), False if inductive (multiple graphs).
    """
    return len(dataset) == 1


def _estimate_memory_requirement(
    dataset: torch_geometric.data.Dataset | torch.utils.data.Dataset,
    complex_dim: int = 2,
) -> float:
    """Estimate memory requirement for in-memory preprocessing in GB.

    This is a rough estimate based on graph statistics and target complex dimension.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset or torch.utils.data.Dataset
        Dataset to estimate for.
    complex_dim : int, optional
        Target complex dimension (2 for triangles, 3 for 4-cliques, etc.).
        Default: 2.

    Returns
    -------
    float
        Estimated memory in GB.

    Notes
    -----
    Formula approximation:
    - Triangles (dim=2): O(N × D²) × 12 bytes
    - 4-cliques (dim=3): O(N × D³) × 12 bytes
    Where N = num_nodes or num_graphs, D = avg_degree
    """

    try:
        # Sample first graph to estimate
        sample = dataset[0]

        if hasattr(sample, "edge_index"):
            num_nodes = (
                sample.num_nodes
                if hasattr(sample, "num_nodes")
                else sample.x.shape[0]
            )
            num_edges = sample.edge_index.shape[1]
            avg_degree = num_edges / num_nodes if num_nodes > 0 else 0

            # Estimate structures based on dimension
            # Use max to ensure complexity scaling is always increasing
            num_structures = num_nodes * max(1, (avg_degree**complex_dim))

            # Each structure: ~12 bytes (structure ID + node IDs)
            bytes_per_structure = 12

            # For inductive: multiply by number of graphs
            if not _is_transductive(dataset):
                num_structures *= len(dataset)

            # Convert to GB
            estimated_gb = (num_structures * bytes_per_structure) / 1e9

            return estimated_gb

        return 0.0

    except Exception:
        # If estimation fails, return conservative estimate
        return 1.0


def _should_use_ondisk(
    dataset: torch_geometric.data.Dataset | torch.utils.data.Dataset,
    mode: Literal["auto", "inmemory", "ondisk"],
    available_ram_gb: float | None = None,
    complex_dim: int = 2,
) -> bool:
    """Determine if on-disk preprocessing should be used.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset or torch.utils.data.Dataset
        Dataset to process.
    mode : {"auto", "inmemory", "ondisk"}
        Processing mode.
    available_ram_gb : float, optional
        Available RAM in GB. If None, uses system memory.
        Default: None.
    complex_dim : int, optional
        Target complex dimension for estimation.
        Default: 2.

    Returns
    -------
    bool
        True if on-disk should be used, False for in-memory.
    """
    if mode == "ondisk":
        return True

    if mode == "inmemory":
        return False

    # mode == "auto": decide based on heuristics
    if available_ram_gb is None:
        try:
            import psutil

            available_ram_gb = psutil.virtual_memory().available / 1e9
        except ImportError:
            # Conservative: if can't detect, use on-disk for safety
            available_ram_gb = 4.0

    estimated_memory = _estimate_memory_requirement(dataset, complex_dim)

    # Use on-disk if estimated memory > 70% of available RAM
    threshold = 0.7 * available_ram_gb

    return estimated_memory > threshold


def create_preprocessor(
    dataset: torch_geometric.data.Dataset | torch.utils.data.Dataset,
    data_dir: str | Path,
    transforms_config: DictConfig | None = None,
    mode: Literal["auto", "inmemory", "ondisk"] = "auto",
    available_ram_gb: float | None = None,
    force_reload: bool = False,
    num_workers: int | None = None,
    storage_backend: str = "mmap",
    compression: str | None = "lz4",
    batch_size: int = 32,
    cache_size: int = 100,
    **kwargs,
) -> PreprocessorType:
    """Factory function for creating appropriate preprocessor.

    This function provides a unified interface for both in-memory and on-disk
    preprocessing. It automatically detects whether to use on-disk processing
    based on dataset size and available memory.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset or torch.utils.data.Dataset
        Source dataset to process.
    data_dir : str or Path
        Directory for storing processed data.
    transforms_config : DictConfig, optional
        Configuration for transforms (liftings).
        Default: None.
    mode : {"auto", "inmemory", "ondisk"}, optional
        Processing mode:
        - "auto": Automatically choose based on dataset size and available RAM
        - "inmemory": Force in-memory preprocessing (standard PreProcessor)
        - "ondisk": Force on-disk preprocessing (OnDiskInductivePreprocessor)
        Default: "auto".
    available_ram_gb : float, optional
        Available RAM in GB for auto mode. If None, automatically detected.
        Default: None.
    force_reload : bool, optional
        If True, reprocess all samples even if cache exists (default: False).
        Only used by on-disk preprocessors.
    num_workers : int, optional
        Number of parallel workers for preprocessing (default: None = auto-detect).
        Only used by on-disk preprocessors.
    storage_backend : str, optional
        Storage backend: "mmap" (compressed) or "files" (fast) (default: "mmap").
        Only used by OnDiskInductivePreprocessor.
    compression : str, optional
        Compression algorithm: "lz4", "zstd", or None (default: "lz4").
        Only used by OnDiskInductivePreprocessor with mmap storage.
    batch_size : int, optional
        Batch size for parallel processing (default: 32).
        Only used by on-disk preprocessors.
    cache_size : int, optional
        Number of samples to cache in memory during training (default: 100).
        Only used by OnDiskInductivePreprocessor.
    **kwargs : dict
        Additional arguments passed to the preprocessor.

    Returns
    -------
    PreProcessor or OnDiskInductivePreprocessor
        Appropriate preprocessor instance based on dataset and mode.

    Examples
    --------
    >>> from topobench.data.preprocessor import create_preprocessor
    >>> from omegaconf import OmegaConf
    >>>
    >>> # Automatic mode (recommended)
    >>> preprocessor = create_preprocessor(
    ...     dataset=dataset,
    ...     data_dir="./data",
    ...     transforms_config=config,
    ...     mode="auto"  # Automatically chooses
    ... )
    >>>
    >>> # Force on-disk for large datasets
    >>> preprocessor = create_preprocessor(
    ...     dataset=large_dataset,
    ...     data_dir="./data",
    ...     transforms_config=config,
    ...     mode="ondisk"  # Always use on-disk
    ... )
    >>>
    >>> # Use like any preprocessor
    >>> train, val, test = preprocessor.load_dataset_splits(split_config)

    Notes
    -----
    - For inductive datasets (many graphs), uses OnDiskInductivePreprocessor
    - For transductive datasets (single graph), uses PreProcessor
    - In-memory mode uses standard PreProcessor (current TopoBench default)
    - Auto mode estimates memory requirements and chooses appropriately

    See Also
    --------
    PreProcessor : Standard in-memory preprocessor
    OnDiskInductivePreprocessor : On-disk preprocessor for inductive learning
    """
    from .ondisk_inductive import OnDiskInductivePreprocessor
    from .preprocessor import PreProcessor

    # Determine if on-disk should be used
    use_ondisk = _should_use_ondisk(
        dataset,
        mode,
        available_ram_gb,
        complex_dim=transforms_config.get("complex_dim", 2)
        if transforms_config
        else 2,
    )

    if not use_ondisk:
        # Use standard in-memory PreProcessor
        # Filter out OnDiskInductivePreprocessor-specific kwargs that InMemoryDataset doesn't accept
        inmemory_kwargs = {}
        allowed_inmemory_args = {
            "force_reload",
            "log",
            "transform",
            "pre_transform",
            "pre_filter",
        }

        # Add force_reload if specified (it's allowed by InMemoryDataset)
        if force_reload:
            inmemory_kwargs["force_reload"] = force_reload

        # Add any other allowed kwargs from **kwargs
        inmemory_kwargs.update(
            {
                key: value
                for key, value in kwargs.items()
                if key in allowed_inmemory_args
            }
        )

        return PreProcessor(
            dataset, data_dir, transforms_config, **inmemory_kwargs
        )

    # Use on-disk preprocessing
    is_transductive = _is_transductive(dataset)

    if is_transductive:
        raise NotImplementedError(
            "On-disk preprocessing for transductive learning is not implemented yet."
        )
    # Multiple graphs - use inductive on-disk
    return OnDiskInductivePreprocessor(
        dataset=dataset,
        data_dir=data_dir,
        transforms_config=transforms_config,
        force_reload=force_reload,
        num_workers=num_workers,
        storage_backend=storage_backend,
        compression=compression,
        batch_size=batch_size,
        cache_size=cache_size,
        **kwargs,  # Pass any remaining kwargs
    )
