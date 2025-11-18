"""Loader for OC20 family datasets (S2EF/IS2RE).

This loader integrates the Open Catalyst 2020 (OC20/OC22) datasets into TopoBench.

Supported tasks:
- S2EF (Structure to Energy and Forces): Predict energy/forces from atomic structure
  - Train splits: 200K, 2M, 20M, all
  - Validation splits: val_id, val_ood_ads, val_ood_cat, val_ood_both (can aggregate)
  - Test split: test (can be optionally skipped with include_test=False)
  - Automatic preprocessing from extxyz/txt to LMDB format
- IS2RE (Initial Structure to Relaxed Energy): Predict relaxed energy from initial structure
  - Pre-split train/val/test datasets

The LMDB backend is integrated directly to avoid external file dependencies.
"""

from __future__ import annotations

import logging
import lzma
import os
import pickle
import shutil
import tarfile
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import lmdb
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

from topobench.data.loaders.base import AbstractLoader
from topobench.data.preprocessor.oc20_s2ef_preprocessor import (
    needs_preprocessing,
    preprocess_s2ef_dataset,
)

logger = logging.getLogger(__name__)

# OC20 dataset split URLs
# S2EF dataset URLs
S2EF_TRAIN_SPLITS = {
    "200K": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_200K.tar",
    "2M": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_2M.tar",
    "20M": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_20M.tar",
    "all": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_all.tar",
}

S2EF_VAL_SPLITS = {
    "val_id": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_id.tar",
    "val_ood_ads": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_ads.tar",
    "val_ood_cat": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_cat.tar",
    "val_ood_both": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_both.tar",
}

S2EF_TEST_SPLIT = "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_test_lmdbs.tar.gz"

# IS2RE dataset URLs (contains train/val/test in one archive)
IS2RE_URL = "https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_val_test_lmdbs.tar.gz"
OC22_IS2RE_URL = "https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/is2res_total_train_val_test_lmdbs.tar.gz"

CACHE_DIR = Path.home() / ".cache" / "oc20"


def _uncompress_xz(file_path: str) -> str:
    if not file_path.endswith(".xz"):
        return file_path

    output_path = file_path.replace(".xz", "")
    try:
        with (
            lzma.open(file_path, "rb") as f_in,
            open(output_path, "wb") as f_out,
        ):
            shutil.copyfileobj(f_in, f_out)
        os.remove(file_path)
        return output_path
    except Exception as e:
        logger.error(f"Error uncompressing {file_path}: {e}")
        return file_path


def _download_and_extract(url: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / os.path.basename(url)

    if not target_file.exists():
        logger.info(f"Downloading {url}...")
        with tqdm(
            unit="B", unit_scale=True, desc=f"Downloading {target_file.name}"
        ) as pbar:

            def report(block_num, block_size, total_size):
                if total_size > 0 and block_num == 0:
                    pbar.total = total_size
                pbar.update(block_size)

            urllib.request.urlretrieve(url, target_file, reporthook=report)

    logger.info(f"Extracting {target_file.name}...")
    if str(target_file).endswith((".tar.gz", ".tgz")):
        with tarfile.open(target_file, "r:gz") as tar:
            tar.extractall(path=target_dir)
    elif str(target_file).endswith(".tar"):
        with tarfile.open(target_file, "r:") as tar:
            tar.extractall(path=target_dir)
    else:
        raise ValueError(f"Unsupported archive format: {target_file}")

    return target_dir


class _OC20LMDBDataset(Dataset):
    """LMDB-based dataset for OC20/OC22.

    Supports:
    - S2EF task with flexible train/val/test split specification
    - IS2RE/OC22_IS2RE tasks with pre-computed train/val/test splits
    """

    def __init__(
        self,
        root: str | Path,
        task: str = "s2ef",
        train_split: str | None = "200K",
        val_splits: list[str] | None = None,
        test_split: str = "test",
        download: bool = True,
        include_test: bool = True,
        dtype: torch.dtype = torch.float32,
        legacy_format: bool = False,
    ):
        """Initialize OC20 LMDB dataset.

        Parameters
        ----------
        root : str | Path
            Root directory for storing datasets.
        task : str
            Task type: "s2ef", "is2re", or "oc22_is2re".
        train_split : Optional[str]
            For S2EF: one of ["200K", "2M", "20M", "all"].
            For IS2RE: ignored (uses precomputed split).
        val_splits : Optional[list[str]]
            For S2EF: list of validation splits to use.
            Can be ["val_id", "val_ood_ads", "val_ood_cat", "val_ood_both"] or subset.
            If None, uses all 4 validation splits.
            For IS2RE: ignored (uses precomputed split).
        test_split : str
            For S2EF: "test" (default).
            For IS2RE: ignored (uses precomputed split).
        download : bool
            Whether to download if not present.
        include_test : bool
            Whether to download/include test split. If False, validation indices are reused for test.
        dtype : torch.dtype
            Data type for tensors.
        legacy_format : bool
            Whether to use legacy PyG Data format.
        """
        super().__init__()
        self.root = Path(root)
        self.task = task.lower()
        self.dtype = dtype
        self.legacy_format = legacy_format
        self.download_flag = download
        self.include_test = include_test

        if self.task == "s2ef":
            if train_split not in S2EF_TRAIN_SPLITS:
                raise ValueError(
                    f"Invalid S2EF train split: {train_split}. "
                    f"Choose from {list(S2EF_TRAIN_SPLITS.keys())}"
                )
            self.train_split = train_split

            # Default: use all validation splits
            if val_splits is None:
                val_splits = list(S2EF_VAL_SPLITS.keys())
            else:
                for vs in val_splits:
                    if vs not in S2EF_VAL_SPLITS:
                        raise ValueError(
                            f"Invalid S2EF val split: {vs}. "
                            f"Choose from {list(S2EF_VAL_SPLITS.keys())}"
                        )
            self.val_splits = val_splits
            self.test_split = test_split

        elif self.task in ["is2re", "oc22_is2re"]:
            # IS2RE datasets have precomputed train/val/test splits
            pass
        else:
            raise ValueError(
                f"Unknown task: {task}. Choose from ['s2ef', 'is2re', 'oc22_is2re']"
            )

        if download:
            self._download_and_prepare()

        self._open_lmdbs()

    def _download_and_prepare(self):
        """Download and prepare the dataset based on task."""
        if self.task == "s2ef":
            self._download_s2ef()
        elif self.task == "is2re":
            self._download_is2re(IS2RE_URL, "is2re")
        elif self.task == "oc22_is2re":
            self._download_is2re(OC22_IS2RE_URL, "oc22_is2re")

    def _download_s2ef(self):
        """Download S2EF train, validation, and test splits."""
        # Download train split
        train_url = S2EF_TRAIN_SPLITS[self.train_split]
        train_dir = self.root / "s2ef" / self.train_split / "train"
        if not train_dir.exists():
            logger.info(f"Downloading S2EF train split: {self.train_split}")
            _download_and_extract(
                train_url, self.root / "s2ef" / self.train_split
            )
            self._decompress_xz_files(self.root / "s2ef" / self.train_split)

        # Download validation splits
        for val_split in self.val_splits:
            val_url = S2EF_VAL_SPLITS[val_split]
            val_dir = self.root / "s2ef" / "all" / val_split
            if not val_dir.exists():
                logger.info(f"Downloading S2EF validation split: {val_split}")
                _download_and_extract(val_url, self.root / "s2ef" / "all")
                self._decompress_xz_files(self.root / "s2ef" / "all")

        # Download test split
        test_dir = self.root / "s2ef" / "all" / "test"
        if self.include_test and not test_dir.exists():
            logger.info("Downloading S2EF test split")
            _download_and_extract(S2EF_TEST_SPLIT, self.root / "s2ef" / "all")
            self._decompress_xz_files(self.root / "s2ef" / "all")
        elif not self.include_test:
            logger.info(
                "Skipping S2EF test split download (include_test=False); will reuse validation as test"
            )

        # Preprocess S2EF data (convert extxyz/txt to LMDB if needed)
        self._preprocess_s2ef()

    def _preprocess_s2ef(self):
        """Preprocess S2EF data from extxyz/txt to LMDB format if needed."""
        # Check if any split needs preprocessing
        train_dir = self.root / "s2ef" / self.train_split / "train"
        needs_any_preprocessing = needs_preprocessing(train_dir, train_dir)

        if not needs_any_preprocessing:
            for val_split in self.val_splits:
                val_dir = self.root / "s2ef" / "all" / val_split
                if needs_preprocessing(val_dir, val_dir):
                    needs_any_preprocessing = True
                    break

        if not needs_any_preprocessing and self.include_test:
            test_dir = self.root / "s2ef" / "all" / "test"
            needs_any_preprocessing = needs_preprocessing(test_dir, test_dir)

        if needs_any_preprocessing:
            logger.info(
                "S2EF data needs preprocessing from extxyz/txt to LMDB format"
            )
            try:
                preprocess_s2ef_dataset(
                    root=self.root,
                    train_split=self.train_split,
                    val_splits=self.val_splits,
                    include_test=self.include_test,
                )
            except ImportError:
                logger.error(
                    "Cannot preprocess S2EF data: fairchem-core or ASE not installed. "
                    "Install with: pip install fairchem-core ase"
                )
                raise
        else:
            logger.info("S2EF data already preprocessed (LMDB files found)")

    def _decompress_xz_files(self, directory: Path):
        """Decompress all .xz files in a directory."""
        xz_files = list(directory.glob("**/*.xz"))
        if xz_files:
            logger.info(
                f"Decompressing {len(xz_files)} .xz files in {directory}..."
            )
            num_workers = max(1, os.cpu_count() - 1)
            # Use threads to avoid pickling/import issues with processes on macOS
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(_uncompress_xz, str(f)) for f in xz_files
                ]
                for future in as_completed(futures):
                    future.result()

    def _download_is2re(self, url: str, name: str):
        """Download IS2RE or OC22 IS2RE dataset."""
        target_dir = self.root / name
        if not target_dir.exists():
            logger.info(f"Downloading {name} dataset")
            _download_and_extract(url, self.root)
            self._decompress_xz_files(self.root)

    def _open_lmdbs(self):
        """Open LMDB files for train/val/test splits."""
        if self.task == "s2ef":
            self._open_s2ef_lmdbs()
        elif self.task in ["is2re", "oc22_is2re"]:
            self._open_is2re_lmdbs()

    def _open_s2ef_lmdbs(self):
        """Open S2EF LMDB files and create split mappings."""
        # Train
        train_dir = self.root / "s2ef" / self.train_split / "train"
        train_lmdbs = self._collect_lmdb_files(train_dir)

        # Validation (can be multiple)
        val_lmdbs = []
        for val_split in self.val_splits:
            val_dir = self.root / "s2ef" / "all" / val_split
            val_lmdbs.extend(self._collect_lmdb_files(val_dir))

        # Test
        test_dir = self.root / "s2ef" / "all" / "test"
        test_lmdbs = (
            self._collect_lmdb_files(test_dir) if self.include_test else []
        )

        # Open all LMDBs and create split index mapping
        self.envs = []
        self.cumulative_sizes = [0]
        self.split_idx = {"train": [], "valid": [], "test": []}

        current_idx = 0

        # Process train LMDBs
        for lmdb_path in train_lmdbs:
            env, size = self._open_single_lmdb(lmdb_path)
            self.envs.append((lmdb_path, env, size))
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + size)
            self.split_idx["train"].extend(
                range(current_idx, current_idx + size)
            )
            current_idx += size

        # Process validation LMDBs
        for lmdb_path in val_lmdbs:
            env, size = self._open_single_lmdb(lmdb_path)
            self.envs.append((lmdb_path, env, size))
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + size)
            self.split_idx["valid"].extend(
                range(current_idx, current_idx + size)
            )
            current_idx += size

        # Process test LMDBs
        for lmdb_path in test_lmdbs:
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
            f"Loaded S2EF dataset: {len(self.split_idx['train'])} train, "
            f"{len(self.split_idx['valid'])} val, {len(self.split_idx['test'])} test"
        )

    def _open_is2re_lmdbs(self):
        """Open IS2RE LMDB files with precomputed splits."""
        # IS2RE datasets have structure: data/is2re/train, data/is2re/val_id, data/is2re/test_id
        # or data/is2re/all/train, etc.
        base_dir = self.root / (
            "is2re" if self.task == "is2re" else "oc22_is2re"
        )

        # Try different possible structures
        possible_structures = [
            base_dir,
            base_dir / "data" / "is2re",
            self.root / "data" / "is2re",
        ]

        found_dir = None
        for poss_dir in possible_structures:
            if poss_dir.exists():
                found_dir = poss_dir
                break

        if found_dir is None:
            raise ValueError(f"Cannot find IS2RE data directory in {base_dir}")

        # Look for train/val/test subdirectories
        train_lmdbs = self._collect_lmdb_files(found_dir / "train")
        val_lmdbs = self._collect_lmdb_files(
            found_dir / "val_id"
        ) or self._collect_lmdb_files(found_dir / "val")
        test_lmdbs = self._collect_lmdb_files(
            found_dir / "test_id"
        ) or self._collect_lmdb_files(found_dir / "test")

        # Open all LMDBs
        self.envs = []
        self.cumulative_sizes = [0]
        self.split_idx = {"train": [], "valid": [], "test": []}

        current_idx = 0

        for lmdb_path in train_lmdbs:
            env, size = self._open_single_lmdb(lmdb_path)
            self.envs.append((lmdb_path, env, size))
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + size)
            self.split_idx["train"].extend(
                range(current_idx, current_idx + size)
            )
            current_idx += size

        for lmdb_path in val_lmdbs:
            env, size = self._open_single_lmdb(lmdb_path)
            self.envs.append((lmdb_path, env, size))
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + size)
            self.split_idx["valid"].extend(
                range(current_idx, current_idx + size)
            )
            current_idx += size

        for lmdb_path in test_lmdbs:
            env, size = self._open_single_lmdb(lmdb_path)
            self.envs.append((lmdb_path, env, size))
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + size)
            self.split_idx["test"].extend(
                range(current_idx, current_idx + size)
            )
            current_idx += size

        # Convert to tensors
        self.split_idx = {
            k: torch.tensor(v, dtype=torch.long)
            for k, v in self.split_idx.items()
        }

        logger.info(
            f"Loaded {self.task.upper()} dataset: {len(self.split_idx['train'])} train, "
            f"{len(self.split_idx['valid'])} val, {len(self.split_idx['test'])} test"
        )

    def _collect_lmdb_files(self, directory: Path) -> list[Path]:
        """Collect all .lmdb files in a directory."""
        if not directory.exists():
            return []
        lmdb_files = sorted(directory.glob("**/*.lmdb"))
        return lmdb_files

    def _open_single_lmdb(self, lmdb_path: Path) -> tuple:
        """Open a single LMDB file and return (env, size)."""
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
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

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
        return self.cumulative_sizes[-1]

    def get(self, idx: int) -> Data:
        lmdb_idx, local_idx = self._find_lmdb_and_local_idx(idx)
        lmdb_path, env, _ = self.envs[lmdb_idx]

        with env.begin() as txn:
            cursor = txn.cursor()
            if not cursor.first():
                raise RuntimeError(f"Empty LMDB at {lmdb_path}")

            for _ in range(local_idx):
                if not cursor.next():
                    raise RuntimeError(
                        f"Index {local_idx} out of range in {lmdb_path}"
                    )

            key, value = cursor.item()
            data = pickle.loads(value)

        if self.legacy_format and isinstance(data, Data):
            data = Data(
                **{k: v for k, v in data.__dict__.items() if v is not None}
            )

        return data

    def __len__(self) -> int:
        return self.len()

    def __getitem__(self, idx: int) -> Data:
        return self.get(idx)

    def __iter__(self) -> Iterator[Data]:
        for i in range(len(self)):
            yield self[i]

    def __del__(self):
        if hasattr(self, "envs"):
            for _, env, _ in self.envs:
                env.close()


class OC20DatasetLoader(AbstractLoader):
    """Load OC20 family datasets.

    This loader supports all OC20/OC22 dataset splits including S2EF and IS2RE tasks.

    Parameters in the Hydra config (dataset.loader.parameters):
    - data_domain: graph
    - data_type: oc20
    - data_name: Logical name for the dataset (e.g., OC20_S2EF_200K)
    - task: "s2ef", "is2re", or "oc22_is2re"

    For S2EF task:
    - train_split: one of ["200K", "2M", "20M", "all"]
    - val_splits: list of validation splits (default: all 4)
      Options: ["val_id", "val_ood_ads", "val_ood_cat", "val_ood_both"]
    - test_split: "test" (default)

    For IS2RE/OC22 tasks:
    - Uses precomputed train/val/test splits from the LMDB archives

    Common parameters:
    - download: whether to download (default: false)
    - legacy_format: whether to use legacy PyG Data format (default: false)
    - dtype: torch dtype (default: "float32")
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        task: str = getattr(self.parameters, "task", "s2ef")
        download: bool = bool(getattr(self.parameters, "download", False))
        legacy_format: bool = bool(
            getattr(self.parameters, "legacy_format", False)
        )
        dtype = getattr(self.parameters, "dtype", "float32")
        dtype_t = (
            getattr(torch, str(dtype)) if isinstance(dtype, str) else dtype
        )

        if task == "s2ef":
            train_split = getattr(self.parameters, "train_split", "200K")
            val_splits_param = getattr(self.parameters, "val_splits", None)

            # Parse val_splits
            if val_splits_param is None:
                val_splits = None  # Use all by default
            elif isinstance(val_splits_param, str):
                # Single validation split as string
                val_splits = [val_splits_param]
            elif isinstance(val_splits_param, (list, tuple)):
                val_splits = list(val_splits_param)
            else:
                val_splits = None

            test_split = getattr(self.parameters, "test_split", "test")
            include_test = bool(getattr(self.parameters, "include_test", True))

            ds = _OC20LMDBDataset(
                root=self.get_data_dir(),
                task="s2ef",
                train_split=train_split,
                val_splits=val_splits,
                test_split=test_split,
                download=download,
                include_test=include_test,
                dtype=dtype_t,
                legacy_format=legacy_format,
            )

        elif task in ["is2re", "oc22_is2re"]:
            ds = _OC20LMDBDataset(
                root=self.get_data_dir(),
                task=task,
                download=download,
                dtype=dtype_t,
                legacy_format=legacy_format,
            )
        else:
            raise ValueError(
                f"Unsupported task '{task}'. Use 's2ef', 'is2re', or 'oc22_is2re'."
            )

        return ds  # type: ignore[return-value]

    def get_data_dir(self) -> Path:
        # Keep default directory convention for TopoBench
        return Path(super().get_data_dir())
