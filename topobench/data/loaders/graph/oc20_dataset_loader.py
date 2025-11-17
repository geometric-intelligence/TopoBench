"""Loader for OC20 family datasets (S2EF/IS2RE).

This loader integrates the Open Catalyst 2020 (OC20/OC22) datasets into TopoBench.
It supports two modes:
- tiny: returns a tiny synthetic PyG dataset for CI/testing (default)
- lmdb: uses the on-disk LMDB datasets from OC20 (optional, requires `lmdb`)

The LMDB backend is integrated directly to avoid external file dependencies.
"""
from __future__ import annotations

import logging
import lzma
import multiprocessing as mp
import os
import pickle
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterator, Optional

from concurrent.futures import ProcessPoolExecutor, as_completed
import lmdb
from tqdm import tqdm

import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, Dataset, InMemoryDataset

from topobench.data.loaders.base import AbstractLoader

logger = logging.getLogger(__name__)

# OC20 dataset split URLs
SPLITS_TO_URL = {
    "s2ef_train_200K": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_200K.tar",
    "s2ef_train_2M": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_2M.tar",
    "s2ef_train_20M": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_20M.tar",
    "s2ef_train_all": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_all.tar",
    "s2ef_val_id": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_id.tar",
    "s2ef_val_ood_ads": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_ads.tar",
    "s2ef_val_ood_cat": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_cat.tar",
    "s2ef_val_ood_both": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_both.tar",
    "s2ef_test": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_test.tar",
    "surfaces": "https://dl.fbaipublicfiles.com/opencatalystproject/data/slab_trajectories.tar",
    "is2re": "https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_val_test_lmdbs.tar.gz",
    "oc22_is2re": "https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/is2res_total_train_val_test_lmdbs.tar.gz",
}

CACHE_DIR = Path.home() / ".cache" / "oc20"


def _uncompress_xz(file_path: str) -> str:
    if not file_path.endswith(".xz"):
        return file_path

    output_path = file_path.replace(".xz", "")
    try:
        with lzma.open(file_path, "rb") as f_in, open(output_path, "wb") as f_out:
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

    def __init__(
        self,
        path: Optional[str | Path] = None,
        split: Optional[str] = "s2ef_train_200K",
        download: bool = True,
        dtype: torch.dtype = torch.float32,
        legacy_format: bool = False,
    ):
        """Initialize OC20 LMDB dataset.

        Parameters
        ----------
        path : Optional[str | Path]
            Path to LMDB directory. If None, uses cache directory.
        split : Optional[str]
            Which OC20 split to load (e.g., "s2ef_train_200K").
        download : bool
            Whether to download if not present.
        dtype : torch.dtype
            Data type for tensors.
        legacy_format : bool
            Whether to use legacy PyG Data format.
        """
        super().__init__()
        self.dtype = dtype
        self.legacy_format = legacy_format

        if path is None:
            if split is None:
                raise ValueError("Must provide either path or split")
            if split not in SPLITS_TO_URL:
                raise ValueError(
                    f"Unknown split: {split}. Available: {list(SPLITS_TO_URL.keys())}"
                )

            url = SPLITS_TO_URL[split]
            dataset_name = os.path.basename(url).split(".")[0]
            path = CACHE_DIR / dataset_name

        self.path = Path(path)

        if download and not self.path.exists():
            if split is None:
                raise ValueError("Cannot download without specifying a split")
            self._download(split)

        if not self.path.exists():
            raise ValueError(f"Dataset not found at {self.path}")

        self._open_lmdbs()

    def _download(self, split: str):
        url = SPLITS_TO_URL[split]
        logger.info(f"Downloading {split} dataset...")
        _download_and_extract(url, self.path)

        xz_files = list(self.path.glob("**/*.xz"))
        if xz_files:
            logger.info(f"Decompressing {len(xz_files)} .xz files...")
            from concurrent.futures import ProcessPoolExecutor, as_completed

            num_workers = max(1, mp.cpu_count() - 1)
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_uncompress_xz, str(f)) for f in xz_files]
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Decompressing"
                ):
                    future.result()


    def _open_lmdbs(self):
        if self.path.is_dir():
            lmdb_paths = sorted(self.path.glob("**/*.lmdb"))
        else:
            lmdb_paths = [self.path]

        if not lmdb_paths:
            raise ValueError(f"No LMDB files found in {self.path}")

        self.envs = []
        self.cumulative_sizes = [0]

        for lmdb_path in lmdb_paths:
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

            self.envs.append((lmdb_path, env, size))
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + size)

        logger.info(
            f"Loaded {len(self.envs)} LMDB files with {len(self)} total entries"
        )

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
                    raise RuntimeError(f"Index {local_idx} out of range in {lmdb_path}")

            key, value = cursor.item()
            data = pickle.loads(value)

        if self.legacy_format and isinstance(data, Data):
            data = Data(**{k: v for k, v in data.__dict__.items() if v is not None})

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


class _TinyOC20Dataset(InMemoryDataset):
    """A tiny synthetic OC20-like dataset for tests and quick runs.

    Each sample is a small "molecule on surface" graph with:
    - x: atom features (random floats)
    - pos: 3D positions
    - z: atomic numbers (ints)
    - y: target energy (scalar regression)
    """

    def __init__(
        self,
        root: str | Path,
        num_samples: int = 64,
        min_nodes: int = 5,
        max_nodes: int = 12,
        num_node_features: int = 6,
        seed: int = 0,
    ) -> None:
        super().__init__(str(root))
        self._num_samples = num_samples
        self._min_nodes = min_nodes
        self._max_nodes = max_nodes
        self._num_node_features = num_node_features
        self._rng = random.Random(seed)
        self._torch_rng = torch.Generator().manual_seed(seed)

        # Generate data list
        data_list: list[Data] = []
        for _ in range(num_samples):
            n = self._rng.randint(self._min_nodes, self._max_nodes)
            pos = torch.randn((n, 3), generator=self._torch_rng)
            x = torch.randn((n, self._num_node_features), generator=self._torch_rng)
            z = torch.randint(low=1, high=86, size=(n,), generator=self._torch_rng)
            # Fully-connected edge index for small graphs
            row = torch.arange(n).repeat_interleave(n)
            col = torch.arange(n).repeat(n)
            edge_index = torch.stack([row, col], dim=0)
            # Scalar target (e.g., energy)
            y = torch.randn(1, generator=self._torch_rng)
            data_list.append(Data(x=x, pos=pos, z=z, edge_index=edge_index, y=y))

        data, slices = self.collate(data_list)
        self.data, self.slices = data, slices

        # Pre-generate split indices for reproducibility (60/20/20)
        idx = list(range(num_samples))
        self._rng.shuffle(idx)
        n_train = int(0.6 * num_samples)
        n_val = int(0.2 * num_samples)
        self.split_idx = {
            "train": torch.tensor(idx[:n_train], dtype=torch.long),
            "valid": torch.tensor(idx[n_train : n_train + n_val], dtype=torch.long),
            "test": torch.tensor(idx[n_train + n_val :], dtype=torch.long),
        }

    @property
    def num_node_features(self) -> int:  # type: ignore[override]
        return self._num_node_features


class OC20DatasetLoader(AbstractLoader):
    """Load OC20 family datasets.

    This loader supports all OC20/OC22 dataset splits including S2EF and IS2RE tasks.

    Parameters in the Hydra config (dataset.loader.parameters):
    - data_domain: graph
    - data_type: oc20
    - data_name: Logical name for the dataset (e.g., OC20_S2EF)
    - mode: "tiny" (default) or "lmdb"
    - split: OC20 split name when mode=="lmdb" (e.g., "s2ef_train_200K", "is2re", etc.)
    - download: whether to download when mode=="lmdb" (default: false)
    - legacy_format: whether to use legacy PyG Data format (default: false)
    - dtype: torch dtype (default: "float32")

    Supported OC20 splits:
    - s2ef_train_200K, s2ef_train_2M, s2ef_train_20M, s2ef_train_all
    - s2ef_val_id, s2ef_val_ood_ads, s2ef_val_ood_cat, s2ef_val_ood_both
    - s2ef_test
    - surfaces
    - is2re
    - oc22_is2re
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        mode: str = getattr(self.parameters, "mode", "tiny")
        
        if mode == "tiny":
            # Fast, dependency-free tiny dataset for CI/tests
            return _TinyOC20Dataset(
                root=self.get_data_dir(),
                num_samples=int(getattr(self.parameters, "num_samples", 64)),
                min_nodes=int(getattr(self.parameters, "min_nodes", 5)),
                max_nodes=int(getattr(self.parameters, "max_nodes", 12)),
                num_node_features=int(
                    getattr(self.parameters, "num_node_features", 6)
                ),
                seed=int(getattr(self.parameters, "seed", 0)),
            )

        if mode == "lmdb":
            split: Optional[str] = getattr(self.parameters, "split", None)
            download: bool = bool(getattr(self.parameters, "download", False))
            legacy_format: bool = bool(
                getattr(self.parameters, "legacy_format", False)
            )
            dtype = getattr(self.parameters, "dtype", "float32")
            dtype_t = getattr(torch, str(dtype)) if isinstance(dtype, str) else dtype

            ds = _OC20LMDBDataset(
                path=None,  # let backend resolve via split/cache
                split=split,
                download=download,
                dtype=dtype_t,
                legacy_format=legacy_format,
            )
            
            # Expose split_idx for TopoBench compatibility
            n = len(ds)
            idx = torch.arange(n)
            n_train = int(0.8 * n)
            n_val = int(0.1 * n)
            ds.split_idx = {  # type: ignore[attr-defined]
                "train": idx[:n_train],
                "valid": idx[n_train : n_train + n_val],
                "test": idx[n_train + n_val :],
            }
            return ds  # type: ignore[return-value]

        raise ValueError(
            f"Unsupported mode '{mode}'. Use 'tiny' (default) or 'lmdb'."
        )

    def get_data_dir(self) -> Path:
        # Keep default directory convention for TopoBench
        return Path(super().get_data_dir())
