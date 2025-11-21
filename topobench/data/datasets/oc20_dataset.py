"""Dataset class for Open Catalyst 2020 (OC20) family of datasets."""

from __future__ import annotations

import logging
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, Dataset

try:
    import lmdb

    HAS_LMDB = True
except ImportError:
    lmdb = None
    HAS_LMDB = False

logger = logging.getLogger(__name__)


class OC20Dataset(Dataset):
    """Dataset class for Open Catalyst 2020 (OC20) family.

    Supports S2EF (Structure to Energy and Forces) and IS2RE (Initial Structure
    to Relaxed Energy) tasks for catalyst discovery and materials science.

    The OC20 dataset contains DFT calculations for catalyst-adsorbate systems,
    enabling machine learning models to predict energies and forces for
    accelerated materials discovery.

    Parameters
    ----------
    root : str
        Root directory where the dataset is stored.
    name : str
        Name of the dataset.
    parameters : DictConfig
        Configuration parameters for the dataset.

    Attributes
    ----------
    task : str
        Task type: "s2ef", "is2re", or "oc22_is2re".
    train_split : str
        Training split size for S2EF (e.g., "200K", "2M", "20M", "all").
    val_splits : list[str]
        Validation splits to use.
    """

    # S2EF validation splits
    VALID_VAL_SPLITS: ClassVar = [
        "val_id",
        "val_ood_ads",
        "val_ood_cat",
        "val_ood_both",
    ]

    # S2EF train splits
    VALID_TRAIN_SPLITS: ClassVar = ["200K", "2M", "20M", "all"]

    def __init__(
        self,
        root: str,
        name: str,
        parameters: DictConfig,
    ) -> None:
        self.name = name
        self.parameters = parameters

        # Task configuration
        self.task = parameters.get("task", "s2ef").lower()
        self.dtype = self._parse_dtype(parameters.get("dtype", "float32"))
        self.legacy_format = parameters.get("legacy_format", False)
        self.include_test = parameters.get("include_test", True)

        # S2EF-specific configuration
        if self.task == "s2ef":
            self.train_split = parameters.get("train_split", "200K")
            if self.train_split not in self.VALID_TRAIN_SPLITS:
                raise ValueError(
                    f"Invalid S2EF train split: {self.train_split}. "
                    f"Choose from {self.VALID_TRAIN_SPLITS}"
                )

            # Parse validation splits
            val_splits = parameters.get("val_splits", None)
            if val_splits is None:
                self.val_splits = self.VALID_VAL_SPLITS
            elif isinstance(val_splits, str):
                self.val_splits = [val_splits]
            else:
                self.val_splits = list(val_splits)

            # Validate splits
            for vs in self.val_splits:
                if vs not in self.VALID_VAL_SPLITS:
                    raise ValueError(
                        f"Invalid S2EF val split: {vs}. "
                        f"Choose from {self.VALID_VAL_SPLITS}"
                    )

            self.test_split = parameters.get("test_split", "test")

        # Limit for fast experimentation
        self.max_samples = parameters.get("max_samples", None)
        if self.max_samples is not None:
            self.max_samples = int(self.max_samples)
            logger.info(
                f"⚠️  Limiting dataset to {self.max_samples} samples for fast experimentation"
            )

        super().__init__(root)

        # Open LMDB environments
        self._open_lmdbs()

    def __repr__(self) -> str:
        task_info = f"task={self.task}"
        if self.task == "s2ef":
            task_info += f", train={self.train_split}"
        return f"{self.name}(root={self.root}, {task_info}, size={len(self)})"

    @staticmethod
    def _parse_dtype(dtype) -> torch.dtype:
        """Parse dtype parameter to torch.dtype."""
        if isinstance(dtype, str):
            return getattr(torch, dtype)
        return dtype

    def _get_data_paths(self) -> dict[str, list[Path]]:
        """Get paths to LMDB files for each split.

        Returns
        -------
        dict[str, list[Path]]
            Dictionary mapping split names to lists of LMDB file paths.
        """
        root = Path(self.root)
        paths = {"train": [], "val": [], "test": []}

        if self.task == "s2ef":
            # Training data path structure
            train_subdir = f"s2ef_train_{self.train_split}"
            train_dir = (
                root / "s2ef" / self.train_split / train_subdir / train_subdir
            )
            paths["train"] = sorted(train_dir.glob("**/*.lmdb"))

            # Validation data paths
            for val_split in self.val_splits:
                val_subdir = f"s2ef_{val_split}"
                val_dir = root / "s2ef" / "all" / val_subdir / val_subdir
                paths["val"].extend(sorted(val_dir.glob("**/*.lmdb")))

            # Test data path
            if self.include_test:
                test_dir = root / "s2ef" / "all" / "s2ef_test" / "s2ef_test"
                paths["test"] = sorted(test_dir.glob("**/*.lmdb"))

        elif self.task in ["is2re", "oc22_is2re"]:
            # IS2RE datasets have different structure
            base_dir = root / (
                "is2re" if self.task == "is2re" else "oc22_is2re"
            )

            # Try different possible directory structures
            for possible_base in [
                base_dir,
                base_dir / "data" / "is2re",
                root / "data" / "is2re",
            ]:
                if possible_base.exists():
                    paths["train"] = sorted(
                        (possible_base / "train").glob("**/*.lmdb")
                    )
                    paths["val"] = sorted(
                        (possible_base / "val_id").glob("**/*.lmdb")
                    ) or sorted((possible_base / "val").glob("**/*.lmdb"))
                    paths["test"] = sorted(
                        (possible_base / "test_id").glob("**/*.lmdb")
                    ) or sorted((possible_base / "test").glob("**/*.lmdb"))
                    break

        return paths

    def _open_lmdbs(self):
        """Open LMDB files and create split mappings."""
        if not HAS_LMDB:
            raise ImportError(
                "LMDB is required for OC20 dataset. Install with: pip install lmdb"
            )

        paths = self._get_data_paths()

        # Initialize storage
        self.envs = []
        self.cumulative_sizes = [0]
        self.split_idx = {"train": [], "valid": [], "test": []}

        current_idx = 0

        # Open train LMDBs
        for lmdb_path in paths["train"]:
            env, size = self._open_single_lmdb(lmdb_path)
            self.envs.append((lmdb_path, env, size))
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + size)
            self.split_idx["train"].extend(
                range(current_idx, current_idx + size)
            )
            current_idx += size

        # Open validation LMDBs
        for lmdb_path in paths["val"]:
            env, size = self._open_single_lmdb(lmdb_path)
            self.envs.append((lmdb_path, env, size))
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + size)
            self.split_idx["valid"].extend(
                range(current_idx, current_idx + size)
            )
            current_idx += size

        # Open test LMDBs
        for lmdb_path in paths["test"]:
            env, size = self._open_single_lmdb(lmdb_path)
            self.envs.append((lmdb_path, env, size))
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + size)
            self.split_idx["test"].extend(
                range(current_idx, current_idx + size)
            )
            current_idx += size

        # If no test data, reuse validation indices
        if not self.include_test or len(self.split_idx["test"]) == 0:
            self.split_idx["test"] = list(self.split_idx["valid"])

        # Convert to tensors
        self.split_idx = {
            k: torch.tensor(v, dtype=torch.long)
            for k, v in self.split_idx.items()
        }

        logger.info(
            f"Loaded {self.task.upper()} dataset: "
            f"{len(self.split_idx['train'])} train, "
            f"{len(self.split_idx['valid'])} val, "
            f"{len(self.split_idx['test'])} test"
        )

    def _open_single_lmdb(self, lmdb_path: Path) -> tuple:
        """Open a single LMDB file and return (env, size).

        Parameters
        ----------
        lmdb_path : Path
            Path to LMDB file.

        Returns
        -------
        tuple
            (environment, size) tuple.
        """
        env = lmdb.open(
            str(lmdb_path.resolve()),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=1,
        )
        size = env.stat()["entries"]
        return env, size

    def _find_lmdb_and_local_idx(self, idx: int) -> tuple:
        """Find which LMDB contains the given index and the local index within it.

        Parameters
        ----------
        idx : int
            Global dataset index.

        Returns
        -------
        tuple
            (lmdb_index, local_index) tuple.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        # Binary search for the LMDB
        left, right = 0, len(self.envs)
        while left < right - 1:
            mid = (left + right) // 2
            if self.cumulative_sizes[mid] <= idx:
                left = mid
            else:
                right = mid

        lmdb_idx = left
        local_idx = idx - self.cumulative_sizes[lmdb_idx]
        return lmdb_idx, local_idx

    def len(self) -> int:
        """Get dataset length."""
        return self.cumulative_sizes[-1]

    def get(self, idx: int) -> Data:
        """Get data sample at index.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Data
            PyTorch Geometric Data object.
        """
        lmdb_idx, local_idx = self._find_lmdb_and_local_idx(idx)
        lmdb_path, env, _ = self.envs[lmdb_idx]

        with env.begin() as txn:
            cursor = txn.cursor()
            if not cursor.first():
                raise RuntimeError(f"Empty LMDB at {lmdb_path}")

            # Navigate to the target entry
            for _ in range(local_idx):
                if not cursor.next():
                    raise RuntimeError(
                        f"Index {local_idx} out of range in {lmdb_path}"
                    )

            key, value = cursor.item()
            data = pickle.loads(value)

        # Convert to legacy format if needed
        if self.legacy_format and isinstance(data, Data):
            data = Data(
                **{k: v for k, v in data.__dict__.items() if v is not None}
            )

        return data

    def __len__(self) -> int:
        """Get dataset length."""
        return self.len()

    def __getitem__(self, idx: int) -> Data:
        """Get item at index."""
        return self.get(idx)

    def __iter__(self) -> Iterator[Data]:
        """Iterate over dataset."""
        for i in range(len(self)):
            yield self[i]

    def __del__(self):
        """Clean up LMDB environments."""
        if hasattr(self, "envs"):
            for _, env, _ in self.envs:
                env.close()

    @property
    def num_node_features(self) -> int:
        """Number of node features per atom."""
        # Will be determined by the actual data
        return 1  # Atomic numbers

    @property
    def num_classes(self) -> int:
        """Number of classes (regression task)."""
        return 1  # Single regression target (energy)
