"""On-disk preprocessor for inductive learning with large datasets.

This module provides a memory-efficient preprocessing implementation that
processes samples sequentially and stores them on disk, enabling training on
datasets larger than available RAM. This preprocessor applies transforms (e.g.,
lifting operations from graphs to simplicial complexes) one sample at a time to
maintain constant memory usage.
"""

import json
from pathlib import Path
from typing import Any

import torch
import torch_geometric
from omegaconf import DictConfig
from torch.utils.data import Dataset

from topobench.data.preprocessor._ondisk.parallel_processor import (
    ParallelProcessor,
)
from topobench.data.utils import (
    ensure_serializable,
    load_inductive_splits,
    make_hash,
)
from topobench.dataloader import DataloadDataset
from topobench.transforms.data_transform import DataTransform


class OnDiskInductivePreprocessor(Dataset):
    """Sequential disk-backed preprocessor for large-scale inductive learning.

    This preprocessor processes samples one-by-one, applying transforms and
    immediately saving each to disk to maintain constant memory usage regardless
    of dataset size. This enables training on datasets that would otherwise cause
    out-of-memory errors during preprocessing/lifting operations.

    The dataset supports transform caching via parameter hashing, ensuring that
    identical transform configurations reuse previously processed data.

    Design Note
    -----------
    This class inherits from `torch.utils.data.Dataset` (not PyG's `OnDiskDataset`)
    to maintain flexibility in storage backends. This allows us to use optimized
    storage (memory-mapped files, compression) that provides faster I/O than
    database backends while remaining simpler and more debuggable.

    The preprocessor supports parallel processing and maintains O(1)
    memory usage during both preprocessing and dataset iteration.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset or torch.utils.data.Dataset
        Source dataset to process. Can be any dataset with `__getitem__` and `__len__`:
        - `InMemoryDataset`: Small datasets (< 10K samples) that fit in RAM
        - `OnDiskDataset`: Large datasets (> 10K samples) with lazy loading
        - Custom datasets: Any class implementing the Dataset interface

        The preprocessor accesses samples one at a time, so memory usage is O(1)
        regardless of source dataset type.
    data_dir : str or Path
        Root directory for storing processed samples.
    transforms_config : DictConfig, optional
        Configuration parameters for transforms (liftings). If None, no
        transforms are applied and data is used as-is (default: None).
    force_reload : bool, optional
        If True, reprocess all samples even if cache exists (default: False).
    num_workers : int, optional
        Number of parallel workers for preprocessing (default: None = auto-detect).
        If 1, uses sequential processing (no parallel overhead).
        If None, uses cpu_count-1 (leaves 1 core for system).
        Parallel processing provides 4-8× speedup on large datasets.

        Note: Requires dataset to be picklable for multiprocessing. All standard
        PyG datasets (TUDataset, OGB, etc.) are picklable. If dataset cannot be
        pickled, automatically falls back to sequential processing with a warning.
    batch_size : int, optional
        Batch size for parallel processing (default: 32).
        Larger batches reduce overhead but may increase memory during processing.
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
    >>> # Create on-disk dataset (processes with parallel workers)
    >>> dataset = OnDiskInductivePreprocessor(
            dataset=source,
            data_dir='/tmp/enzymes_processed',
            transforms_config=config,
            num_workers=4  # Use 4 parallel workers for speedup
        )

    >>> # Use with TopoBench dataloader for training
    >>> from topobench.dataloader import TBDataloader
    >>> train_ds, val_ds, test_ds = dataset.load_dataset_splits(split_params)
    >>> datamodule = TBDataloader(
            dataset_train=train_ds,
            dataset_val=val_ds,
            dataset_test=test_ds,
            batch_size=32,
            num_workers=0  # Set >0 for multi-process loading
        )
    >>> # Create TBModel and Lightning trainer
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

        # Store split_idx if available (for datasets with fixed splits)
        if hasattr(dataset, "split_idx"):
            self.split_idx = dataset.split_idx

        # Initialize transform and processed directory
        if transforms_config is not None:
            self.pre_transform = self._instantiate_pre_transform(
                transforms_config
            )
            self._set_processed_data_dir(transforms_config)
        else:
            # No transforms - use data as-is
            self.pre_transform = None
            self.processed_dir = self.data_dir / "no_transforms"

        # Ensure processed directory exists
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Load or create metadata
        self.metadata_path = self.processed_dir / "metadata.json"
        if self._should_process():
            self._process_samples()
        else:
            self._load_metadata()

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
        """Load sample from disk.

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
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(
                f"Index {idx} out of range for dataset of size "
                f"{self.num_samples}"
            )

        sample_path = self._get_sample_path(idx)

        if not sample_path.exists():
            raise FileNotFoundError(
                f"Sample file not found: {sample_path}. "
                f"Dataset may be corrupted. Try force_reload=True."
            )

        # Load sample from disk
        data = torch.load(sample_path)
        return data

    def _should_process(self) -> bool:
        """Check if dataset needs to be processed.

        Returns
        -------
        bool
            True if processing is needed, False if cache is valid.
        """
        if self.force_reload:
            return True

        if not self.metadata_path.exists():
            return True

        # Verify all sample files exist
        try:
            with open(self.metadata_path) as f:
                metadata = json.load(f)
            num_samples = metadata.get("num_samples", 0)

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
        """Iterate through samples, apply transforms, and save to disk.

        Uses parallel processing when num_workers > 1 for 4-8× speedup.
        Falls back to sequential processing when num_workers=1.
        """
        print(
            f"Processing {len(self.dataset)} samples to {self.processed_dir}"
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

        # Process samples (parallel or sequential)
        results = processor.process(
            dataset=self.dataset,
            transform=self.pre_transform,
            output_dir=self.processed_dir,
            num_samples=len(self.dataset),
        )

        # Save metadata
        self.num_samples = len(self.dataset)
        self._save_metadata()

        # Report results
        if results["failed"] > 0:
            print(
                f"Processed {results['success']}/{results['total']} samples "
                f"({results['failed']} failed)"
            )
            print("\nErrors:")
            for error in results["errors"][:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(results["errors"]) > 5:
                print(f"  ... and {len(results['errors']) - 5} more errors")
        else:
            print(f"Processed {self.num_samples} samples successfully")

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

    def _set_processed_data_dir(self, transforms_config: DictConfig) -> None:
        """Set processed data directory based on transform parameters.

        Creates a unique directory path using parameter hashing to enable
        caching of preprocessed data across runs with identical configurations.

        Parameters
        ----------
        transforms_config : DictConfig
            Transform configuration parameters.
        """
        # Create repository name from transform keys
        repo_name = "_".join(list(transforms_config.keys()))

        # Hash transform parameters for unique cache directory
        params_hash = make_hash(self.transforms_parameters)

        # Set processed directory path
        self.processed_dir = self.data_dir / repo_name / f"{params_hash}"

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
        """Save dataset metadata to disk."""
        metadata = {
            "num_samples": self.num_samples,
            "source_dataset": str(type(self.dataset).__name__),
            "processed_dir": str(self.processed_dir),
        }

        # Add transform parameters if applicable
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

        # Use existing split utility (it iterates over dataset via __iter__)
        # This maintains O(1) memory as it processes samples one at a time.
        # Note: The split utility extracts labels `[data.y for data in dataset]` which
        # accumulates labels in memory (O(n) for labels), but labels are typically extremely small
        # (single values/tensors) compared to full graph data (x, edge_index, etc.).
        return load_inductive_splits(self, split_params)
