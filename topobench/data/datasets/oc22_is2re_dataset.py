"""Dataset class for Open Catalyst 2022 (OC22) IS2RE dataset."""

from __future__ import annotations

import logging
import pickle
from collections.abc import Iterator
from pathlib import Path

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


class OC22IS2REDataset(Dataset):
    """Dataset class for Open Catalyst 2022 (OC22) IS2RE task.

    IS2RE (Initial Structure to Relaxed Energy) is a task for catalyst
    discovery and materials science.

    The OC22 dataset contains DFT calculations for catalyst-adsorbate systems,
    enabling machine learning models to predict energies for accelerated
    materials discovery.

    Parameters
    ----------
    root : str
        Root directory where the dataset is stored.
    name : str
        Name of the dataset.
    parameters : DictConfig
        Configuration parameters for the dataset.
    """

    def __init__(
        self,
        root: str,
        name: str,
        parameters: DictConfig,
    ) -> None:
        self.name = name
        self.parameters = parameters

        # Task configuration
        self.task = "oc22_is2re"
        self.dtype = self._parse_dtype(parameters.get("dtype", "float32"))
        self.legacy_format = parameters.get("legacy_format", False)
        self.include_test = parameters.get("include_test", True)

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
        return f"{self.name}(root={self.root}, task={self.task}, size={len(self)})"

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

        # The downloaded data is extracted to:
        # root/is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/{train,val_id,test_id}/*.lmdb
        base_path = (
            root
            / "is2res_total_train_val_test_lmdbs"
            / "data"
            / "oc22"
            / "is2re-total"
        )

        if base_path.exists():
            # Train data - multiple LMDB files
            train_dir = base_path / "train"
            if train_dir.exists():
                paths["train"] = sorted(train_dir.glob("*.lmdb"))

            # Validation data - using val_id split
            val_dir = base_path / "val_id"
            if val_dir.exists():
                paths["val"] = sorted(val_dir.glob("*.lmdb"))

            # Test data - using test_id split
            test_dir = base_path / "test_id"
            if test_dir.exists():
                paths["test"] = sorted(test_dir.glob("*.lmdb"))

        return paths

    def _open_lmdbs(self):
        """Open LMDB files and create split mappings."""
        if not HAS_LMDB:
            raise ImportError(
                "LMDB is required for OC22 IS2RE dataset. Install with: pip install lmdb"
            )

        paths = self._get_data_paths()

        # Initialize storage
        self.envs = []
        self.cumulative_sizes = [0]
        self.split_idx = {"train": [], "valid": [], "test": []}

        current_idx = 0

        # Open train LMDBs with cumulative max_samples limiting
        train_samples_remaining = (
            self.max_samples if self.max_samples is not None else None
        )
        for lmdb_path in paths["train"]:
            if (
                train_samples_remaining is not None
                and train_samples_remaining <= 0
            ):
                break
            env, size = self._open_single_lmdb(lmdb_path)
            # Apply remaining limit
            if train_samples_remaining is not None:
                size = min(size, train_samples_remaining)
                train_samples_remaining -= size
            self.envs.append((lmdb_path, env, size))
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + size)
            self.split_idx["train"].extend(
                range(current_idx, current_idx + size)
            )
            current_idx += size

        # Open validation LMDBs with cumulative max_samples limiting (10% of max_samples)
        val_samples_remaining = (
            max(1, self.max_samples // 10)
            if self.max_samples is not None
            else None
        )
        for lmdb_path in paths["val"]:
            if (
                val_samples_remaining is not None
                and val_samples_remaining <= 0
            ):
                break
            env, size = self._open_single_lmdb(lmdb_path)
            # Apply remaining limit
            if val_samples_remaining is not None:
                size = min(size, val_samples_remaining)
                val_samples_remaining -= size
            self.envs.append((lmdb_path, env, size))
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + size)
            self.split_idx["valid"].extend(
                range(current_idx, current_idx + size)
            )
            current_idx += size

        # Open test LMDBs with cumulative max_samples limiting (10% of max_samples)
        test_samples_remaining = (
            max(1, self.max_samples // 10)
            if self.max_samples is not None
            else None
        )
        for lmdb_path in paths["test"]:
            if (
                test_samples_remaining is not None
                and test_samples_remaining <= 0
            ):
                break
            env, size = self._open_single_lmdb(lmdb_path)
            # Apply remaining limit
            if test_samples_remaining is not None:
                size = min(size, test_samples_remaining)
                test_samples_remaining -= size
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

        # Convert old PyG Data objects to new format by extracting all attributes
        if isinstance(data, Data):
            try:
                # Check if this is old format data
                # Old PyG format has attributes directly in __dict__ without proper _store
                if "_store" not in data.__dict__ or any(
                    k in data.__dict__ for k in ["x", "edge_index", "pos"]
                ):
                    # Extract all data attributes
                    data_dict = {}
                    # Get all tensor/data attributes from __dict__
                    data_dict = {
                        key: val
                        for key, val in data.__dict__.items()
                        if not key.startswith("_") and val is not None
                    }

                    # Convert y_relaxed to y before creating new Data object
                    if "y_relaxed" in data_dict:
                        data_dict["y"] = torch.tensor(
                            [data_dict["y_relaxed"]]
                        ).float()
                    elif "y" not in data_dict:
                        data_dict["y"] = torch.tensor([float("nan")]).float()

                    # Use atomic numbers as node features (x)
                    if "atomic_numbers" in data_dict:
                        data_dict["x"] = (
                            data_dict["atomic_numbers"].view(-1, 1).float()
                        )
                    elif "x" not in data_dict and "pos" in data_dict:
                        data_dict["x"] = torch.ones(
                            (data_dict["pos"].shape[0], 1)
                        )

                    # Create edge_index from atomic positions using radius graph
                    if "edge_index" not in data_dict and "pos" in data_dict:
                        from torch_geometric.nn import radius_graph

                        data_dict["edge_index"] = radius_graph(
                            data_dict["pos"], r=5.0, max_num_neighbors=50
                        )

                    # Keep only standard PyG attributes
                    standard_attrs = [
                        "x",
                        "edge_index",
                        "edge_attr",
                        "pos",
                        "y",
                        "batch",
                    ]
                    cleaned_dict = {
                        k: v
                        for k, v in data_dict.items()
                        if k in standard_attrs
                    }

                    # Create a completely new Data object with current PyG format
                    data = Data(**cleaned_dict)

            except (AttributeError, KeyError, RuntimeError, TypeError):
                # If extraction fails, pass through
                pass

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
