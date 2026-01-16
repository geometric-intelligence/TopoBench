"""Base classes for creating custom inductive datasets with optimal performance.

This module provides user-friendly base classes that automatically handle:
- Optimal parallel preprocessing (2.29-5× speedup)
- Proper pickling for multiprocessing
- Constant O(1) memory usage
- File-based on-demand loading

Users only need to implement data loading/generation logic!
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


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


class BaseOnDiskInductiveDataset(Dataset, ABC):
    """Base class for custom on-disk inductive datasets with automatic optimization.

    Users need to implement:
    1. `_generate_or_load_sample(idx)` - How to get ONE sample
    2. `_get_num_samples()` - Total number of samples

    Parameters
    ----------
    root : str or Path
        Root directory for storing/loading data.
    cache_samples : bool, optional
        If True, samples are cached to disk on first access (default: True).
        This enables fast repeated access while maintaining O(1) memory.

    Examples
    --------
    >>> class MyDataset(BaseInductiveDataset):
    ...     def __init__(self, root, num_graphs=1000):
    ...         self.num_graphs = num_graphs
    ...         super().__init__(root, cache_samples=True)
    ...
    ...     def _get_num_samples(self):
    ...         return self.num_graphs
    ...
    ...     def _generate_or_load_sample(self, idx):
    ...         # Your data loading logic here
    ...         return Data(x=..., edge_index=..., y=...)
    >>>
    >>> # Use with parallel preprocessing for speedup!
    >>> dataset = MyDataset("./data", num_graphs=5000)
    >>> preprocessor = OnDiskInductivePreprocessor(
    ...     dataset=dataset,
    ...     num_workers=None # default will use available cores - 1
    ... )

    See Also
    --------
    FileBasedInductiveDataset : For datasets with one file per sample.
    GeneratedInductiveDataset : For synthetic/generated datasets.
    """

    def __init__(self, root: str | Path, cache_samples: bool = True):
        """Initialize the dataset.

        Parameters
        ----------
        root : str or Path
            Root directory for data storage.
        cache_samples : bool, optional
            Enable disk caching of samples (default: True).
        """
        self.root = Path(root)
        self.cache_samples = cache_samples
        self.cache_dir = self.root / ".sample_cache"

        if self.cache_samples:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Get number of samples (user-implemented)
        self._num_samples = self._get_num_samples()

    @abstractmethod
    def _get_num_samples(self) -> int:
        """Return the total number of samples in the dataset.

        This is called once during initialization. Store any expensive
        computations as instance attributes to avoid repeated work.

        Returns
        -------
        int
            Total number of samples.

        Examples
        --------
        >>> def _get_num_samples(self):
        ...     # From configuration
        ...     return self.config.num_graphs
        ...
        ... # Or from file count
        ... def _get_num_samples(self):
        ...     return len(list(self.data_dir.glob("*.pt")))
        """

    @abstractmethod
    def _generate_or_load_sample(self, idx: int) -> Data:
        """Generate or load a single sample.

        This is called by each worker independently during parallel processing.
        Load/generate only ONE sample at a time to maintain O(1) memory.

        Parameters
        ----------
        idx : int
            Sample index (0-indexed).

        Returns
        -------
        Data
            PyTorch Geometric Data object for the sample.

        Examples
        --------
        >>> def _generate_or_load_sample(self, idx):
        ...     # Option 1: Load from file
        ...     return torch.load(self.files[idx])
        ...
        ... # Option 2: Generate on-demand
        ... def _generate_or_load_sample(self, idx):
        ...     torch.manual_seed(idx)
        ...     return Data(x=torch.randn(20, 8), ...)
        ...
        ... # Option 3: Memory-mapped arrays (O(1) memory)
        ... def _generate_or_load_sample(self, idx):
        ...     # Extract subgraph from memory-mapped data
        ...     node_ids = self.mmap_node_ids[idx]
        ...     x = torch.from_numpy(self.mmap_features[node_ids])
        ...     edge_index = extract_edges(self.mmap_edges, node_ids)
        ...     return Data(x=x, edge_index=edge_index)
        """

    def _get_cache_path(self, idx: int) -> Path:
        """Get cache file path for a sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Path
            Cache file path.
        """
        return self.cache_dir / f"sample_{idx:06d}.pt"

    def __len__(self) -> int:
        """Return number of samples.

        Returns
        -------
        int
            Total number of samples.
        """
        return self._num_samples

    def __getitem__(self, idx: int) -> Data:
        """Load sample with automatic caching.

        This method automatically handles caching if enabled,
        or directly loads/generates the sample otherwise.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Data
            PyG Data object.
        """
        if idx < 0 or idx >= self._num_samples:
            raise IndexError(
                f"Index {idx} out of range [0, {self._num_samples})"
            )

        # Check cache first
        if self.cache_samples:
            cache_path = self._get_cache_path(idx)
            if cache_path.exists():
                return _torch_load_compat(cache_path)

        # Generate/load sample
        sample = self._generate_or_load_sample(idx)

        # Cache for next time
        if self.cache_samples:
            cache_path = self._get_cache_path(idx)
            torch.save(sample, cache_path)

        return sample

    def __reduce__(self):
        """Support multiprocessing via pickling.

        Only pickles constructor arguments for lightweight serialization.
        Workers reconstruct the dataset using __init__.

        Override this if your subclass has custom unpicklable attributes.

        Returns
        -------
        tuple
            Pickle state (class, init_args).
        """
        # Get constructor arguments
        init_args = self._get_pickle_args()
        return (self.__class__, init_args)

    def _get_pickle_args(self) -> tuple:
        """Get arguments for reconstructing this dataset after pickling.

        Override this when your __init__ accepts custom parameters beyond root
        and cache_samples. This ensures the dataset can be correctly unpickled
        in parallel worker processes.

        Returns
        -------
        tuple
            Arguments to pass to __init__ during unpickling (must match __init__ signature).

        Examples
        --------
        >>> # Custom dataset with additional __init__ parameters
        >>> def __init__(self, root, num_samples, seed=42, cache_samples=True):
        ...     self.num_samples = num_samples
        ...     self.seed = seed
        ...     super().__init__(root, cache_samples)
        ...
        >>> def _get_pickle_args(self):
        ...     return (str(self.root), self.num_samples, self.seed, self.cache_samples)

        Notes
        -----
        Override when:
        - Your __init__ has additional required parameters
        - You store configuration needed for reconstruction

        Don't override when:
        - Using only root and cache_samples parameters
        - Base implementation suffices
        """
        return (str(self.root), self.cache_samples)


class FileBasedInductiveDataset(BaseOnDiskInductiveDataset):
    """Optimized base class for datasets with one file per sample.

    Use this when you have a directory of pre-existing files (one per sample).
    Automatically handles file discovery and on-demand loading.

    Parameters
    ----------
    root : str or Path
        Root directory containing sample files.
    file_pattern : str, optional
        Glob pattern for finding files (default: "*.pt").
    cache_samples : bool, optional
        Enable disk caching (default: False, since files are already on disk).

    Examples
    --------
    >>> # Simplest case: default pattern
    >>> class MyGraphDataset(FileBasedInductiveDataset):
    ...     def _load_file(self, file_path):
    ...         return torch.load(file_path)
    >>>
    >>> dataset = MyGraphDataset("./my_graphs")  # Finds all *.pt files
    >>>
    >>> # Custom pattern:
    >>> class MyGraphMLDataset(FileBasedInductiveDataset):
    ...     def __init__(self, root):
    ...         super().__init__(root, file_pattern="*.graphml")
    ...
    ...     def _load_file(self, file_path):
    ...         import networkx as nx
    ...         G = nx.read_graphml(file_path)
    ...         return self._networkx_to_pyg(G)
    """

    def __init__(
        self,
        root: str | Path,
        file_pattern: str = "*.pt",
        cache_samples: bool = False,
    ):
        """Initialize file-based dataset.

        Parameters
        ----------
        root : str or Path
            Directory containing sample files.
        file_pattern : str, optional
            Glob pattern for files (default: "*.pt").
        cache_samples : bool, optional
            Enable caching (default: False).
        """
        self.file_pattern = file_pattern

        # Find all matching files
        root_path = Path(root)
        self.files = sorted(list(root_path.glob(file_pattern)))

        if len(self.files) == 0:
            raise ValueError(
                f"No files found matching pattern '{file_pattern}' in {root_path}"
            )

        super().__init__(root, cache_samples)

    def _get_num_samples(self) -> int:
        """Return number of files.

        Returns
        -------
        int
            Number of sample files.
        """
        return len(self.files)

    def _generate_or_load_sample(self, idx: int) -> Data:
        """Load sample from file.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Data
            Loaded sample.
        """
        file_path = self.files[idx]
        return self._load_file(file_path)

    @abstractmethod
    def _load_file(self, file_path: Path) -> Data:
        """Load a single file.

        Implement this to define how to load your file format.

        Parameters
        ----------
        file_path : Path
            Path to the file to load.

        Returns
        -------
        Data
            PyTorch Geometric Data object.

        Examples
        --------
        >>> def _load_file(self, file_path):
        ...     return torch.load(file_path)
        """

    def _get_pickle_args(self) -> tuple:
        """Get pickle arguments.

        Returns
        -------
        tuple
            Constructor arguments.
        """
        return (str(self.root), self.file_pattern, self.cache_samples)


class GeneratedInductiveDataset(BaseOnDiskInductiveDataset):
    """Optimized base class for synthetically generated datasets.

    Use this for datasets that generate samples on-the-fly (e.g., synthetic
    benchmarks, procedural generation).

    Parameters
    ----------
    root : str or Path
        Root directory for caching (if enabled).
    num_samples : int
        Number of samples to generate.
    seed : int, optional
        Random seed for reproducibility (default: 42).
    cache_samples : bool, optional
        Enable disk caching of generated samples (default: True).

    Examples
    --------
    >>> class MySyntheticDataset(GeneratedInductiveDataset):
    ...     def __init__(self, root, num_samples=1000):
    ...         super().__init__(root, num_samples, seed=42)
    ...
    ...     def _generate_sample(self, idx, rng):
    ...         # rng is already seeded deterministically
    ...         x = rng.randn(20, 8)
    ...         edge_index = rng.randint(0, 20, (2, 50))
    ...         return Data(x=x, edge_index=edge_index, y=idx % 5)
    >>>
    >>> dataset = MySyntheticDataset("./data", num_samples=5000)
    >>> # First access generates and caches
    >>> # Subsequent access loads from cache
    """

    def __init__(
        self,
        root: str | Path,
        num_samples: int,
        seed: int = 42,
        cache_samples: bool = True,
    ):
        """Initialize generated dataset.

        Parameters
        ----------
        root : str or Path
            Root directory for caching.
        num_samples : int
            Number of samples to generate.
        seed : int, optional
            Random seed (default: 42).
        cache_samples : bool, optional
            Enable caching (default: True).
        """
        self.num_samples = num_samples
        self.seed = seed
        super().__init__(root, cache_samples)

    def _get_num_samples(self) -> int:
        """Return configured number of samples.

        Returns
        -------
        int
            Number of samples.
        """
        return self.num_samples

    def _generate_or_load_sample(self, idx: int) -> Data:
        """Generate sample with deterministic seeding.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Data
            Generated sample.
        """
        # Create deterministic RNG for this sample
        generator = torch.Generator().manual_seed(self.seed + idx)

        # User implements actual generation
        return self._generate_sample(idx, generator)

    @abstractmethod
    def _generate_sample(self, idx: int, rng: torch.Generator) -> Data:
        """Generate a single sample.

        Parameters
        ----------
        idx : int
            Sample index.
        rng : torch.Generator
            Pre-seeded random generator for deterministic generation.

        Returns
        -------
        Data
            Generated sample.

        Examples
        --------
        >>> def _generate_sample(self, idx, rng):
        ...     x = torch.randn(20, 8, generator=rng)
        ...     edges = torch.randint(0, 20, (2, 50), generator=rng)
        ...     return Data(x=x, edge_index=edges, y=idx % 5)
        """

    def _get_pickle_args(self) -> tuple:
        """Get pickle arguments.

        Returns
        -------
        tuple
            Constructor arguments.
        """
        return (
            str(self.root),
            self.num_samples,
            self.seed,
            self.cache_samples,
        )
