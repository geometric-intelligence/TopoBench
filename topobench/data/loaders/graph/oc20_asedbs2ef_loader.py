"""Loader for OC20 S2EF dataset using ASE DB backend."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, Dataset

from topobench.data.loaders.base import AbstractLoader
from topobench.data.preprocessor.oc20_s2ef_preprocessor import (
    HAS_ASE,
    AtomsToGraphs,
    needs_preprocessing,
    preprocess_s2ef_split_ase,
)

if HAS_ASE:
    import ase.db

logger = logging.getLogger(__name__)


class OC20ASEDBDataset(Dataset):
    """ASE DB dataset for OC20 S2EF structures.

    Parameters
    ----------
    db_paths : list[str | Path] | None
        Backwards-compatible single list of DBs without explicit splits.
    train_db_paths : list[str | Path] | None
        List of ASE DB file paths for the training split.
    val_db_paths : list[str | Path] | None
        List of ASE DB file paths for the validation split.
    test_db_paths : list[str | Path] | None
        List of ASE DB file paths for the test split.
    max_neigh : int
        Maximum number of neighbors per atom.
    radius : float
        Cutoff radius in Angstroms.
    dtype : torch.dtype
        Torch dtype used for tensors.
    include_energy : bool
        Whether to include energy information.
    include_forces : bool
        Whether to include forces information.
    max_samples : int | None
        Maximum number of samples to load for debugging or fast runs.
    """

    def __init__(
        self,
        db_paths: list[str | Path] | None = None,
        *,
        train_db_paths: list[str | Path] | None = None,
        val_db_paths: list[str | Path] | None = None,
        test_db_paths: list[str | Path] | None = None,
        max_neigh: int = 50,
        radius: float = 6.0,
        dtype: torch.dtype = torch.float32,
        include_energy: bool = True,
        include_forces: bool = True,
        max_samples: int | None = None,
    ):
        """Initialize dataset from ASE DB files.

        See class docstring for parameter descriptions.
        """
        if not HAS_ASE:
            raise ImportError("ASE required for S2EF datasets")

        super().__init__()
        self.dtype = dtype

        # Converter
        self.converter = AtomsToGraphs(
            max_neigh=max_neigh,
            radius=radius,
            r_energy=include_energy,
            r_forces=include_forces,
            r_distances=True,
            r_edges=True,
            r_fixed=True,
        )

        # Normalize input options
        if db_paths is not None and any(
            x is not None
            for x in (train_db_paths, val_db_paths, test_db_paths)
        ):
            raise ValueError(
                "Provide either `db_paths` or the split-specific lists, not both."
            )

        if db_paths is not None:
            train_db_paths = list(db_paths)
            val_db_paths = []
            test_db_paths = []
        else:
            train_db_paths = list(train_db_paths or [])
            val_db_paths = list(val_db_paths or [])
            test_db_paths = list(test_db_paths or [])

        # Track DB files per split
        self._per_split_db_paths: dict[str, list[Path]] = {
            "train": [Path(p) for p in train_db_paths],
            "valid": [Path(p) for p in val_db_paths],
            "test": [Path(p) for p in test_db_paths],
        }
        self.db_paths: list[Path] = (
            self._per_split_db_paths["train"]
            + self._per_split_db_paths["valid"]
            + self._per_split_db_paths["test"]
        )

        # Count total structures and build split indices
        self._num_samples = 0
        self._db_ranges: list[
            tuple[Path, int, int]
        ] = []  # (db_path, start, end)
        self.split_idx: dict[str, list[int]] = {
            "train": [],
            "valid": [],
            "test": [],
        }

        for split_name in ("train", "valid", "test"):
            for db_path in self._per_split_db_paths[split_name]:
                with ase.db.connect(str(db_path)) as db:
                    count = db.count()
                start = self._num_samples
                end = start + count
                self._db_ranges.append((db_path, start, end))
                # Append global indices for this DB to the right split
                self.split_idx[split_name].extend(range(start, end))
                self._num_samples = end

        # Apply max_samples limit if specified (per split, not total)
        if max_samples is not None:
            logger.info(f"Limiting each split to {max_samples} samples")
            # When limiting, we need to:
            # 1. Truncate the split_idx lists
            # 2. Create a mapping from limited indices to new contiguous indices
            # 3. Update _num_samples to reflect the new total
            # This ensures len(dataset) returns the correct value and prevents
            # unnecessary iteration over the full dataset during preprocessing

            # Collect indices from all splits that we want to keep
            all_limited_indices = []
            for split_name in ("train", "valid", "test"):
                if self.split_idx[split_name]:
                    original_len = len(self.split_idx[split_name])
                    new_len = min(max_samples, original_len)
                    self.split_idx[split_name] = self.split_idx[split_name][
                        :new_len
                    ]
                    all_limited_indices.extend(self.split_idx[split_name])
                    logger.info(
                        f"  {split_name}: {original_len} -> {new_len} samples"
                    )

            # Create mapping from old indices to new contiguous indices
            old_to_new_idx = {
                old_idx: new_idx
                for new_idx, old_idx in enumerate(
                    sorted(set(all_limited_indices))
                )
            }

            # Remap split_idx to use new contiguous indices
            for split_name in ("train", "valid", "test"):
                self.split_idx[split_name] = [
                    old_to_new_idx[idx] for idx in self.split_idx[split_name]
                ]

            # Store mapping for __getitem__ to translate back to original indices
            self._index_mapping = sorted(set(all_limited_indices))

            # Update _num_samples to the actual limited size
            self._num_samples = len(self._index_mapping)
            logger.info(
                f"Dataset length limited to {self._num_samples} samples"
            )

        logger.info(
            f"Loaded {len(self.db_paths)} DB files with {self._num_samples} total structures"
        )

    def __len__(self) -> int:
        """Return dataset length.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return self._num_samples

    @property
    def num_node_features(self) -> int:
        """Number of node features per atom.

        Returns
        -------
        int
            Number of node features (atomic numbers by default).
        """
        return 1  # Atomic numbers

    @property
    def data(self):
        """Get combined data view for compatibility with InMemoryDataset API.

        Returns a Data object with x and y attributes representing stacked
        features from a sample of the dataset.

        Returns
        -------
        Data
            A Data object with x and y attributes for API compatibility.
        """
        if not hasattr(self, "_data_cache"):
            # Get a sample to determine feature dimensions
            if len(self) > 0:
                sample = self[0]
                # Create a mock data object with minimal info for compatibility
                self._data_cache = Data(
                    x=sample.x if hasattr(sample, "x") else torch.zeros(1, 1),
                    y=sample.y if hasattr(sample, "y") else torch.zeros(1),
                )
            else:
                self._data_cache = Data(x=torch.zeros(1, 1), y=torch.zeros(1))
        return self._data_cache

    def _get_db_and_idx(self, idx: int) -> tuple[Path, int]:
        """Get DB path and local index for global index.

        Parameters
        ----------
        idx : int
            Global index.

        Returns
        -------
        tuple[Path, int]
            Database path and local index within that database.
        """
        if idx < 0 or idx >= self._num_samples:
            raise IndexError(
                f"Index {idx} out of range [0, {self._num_samples})"
            )
        # Binary search could be used; linear scan is fine for moderate DB counts
        for db_path, start, end in self._db_ranges:
            if start <= idx < end:
                local_idx = (idx - start) + 1  # ASE rows are 1-indexed
                return db_path, local_idx
        raise IndexError(f"Index {idx} not found in DB ranges")

    def __getitem__(self, idx: int) -> Data:
        """Get a single graph by index.

        Parameters
        ----------
        idx : int
            Index of the graph to retrieve.

        Returns
        -------
        Data
            PyTorch Geometric Data object.
        """
        # If we have an index mapping (from max_samples limiting), translate the index
        if hasattr(self, "_index_mapping"):
            if idx < 0 or idx >= len(self._index_mapping):
                raise IndexError(
                    f"Index {idx} out of range [0, {len(self._index_mapping)})"
                )
            idx = self._index_mapping[idx]

        db_path, local_idx = self._get_db_and_idx(idx)

        with ase.db.connect(str(db_path)) as db:
            row = db.get(id=local_idx)
            atoms = row.toatoms()

            # Add metadata
            if hasattr(row, "data") and row.data:
                atoms.info["data"] = row.data

        # Convert to PyG
        data = self.converter.convert(atoms)

        # Cast dtype
        if getattr(data, "pos", None) is not None:
            data.pos = data.pos.to(self.dtype)
        if getattr(data, "edge_attr", None) is not None:
            data.edge_attr = data.edge_attr.to(self.dtype)

        return data


class OC20S2EFDatasetLoader(AbstractLoader):
    """Loader for OC20 S2EF dataset using ASE DB.

    Parameters
    ----------
    parameters : DictConfig
        Configuration dictionary (usually hydra DictConfig) with dataset options.
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load OC20 S2EF dataset from ASE database files.

        Configuration parameters:
        - train_split: "200K", "2M", "20M", or "all"
        - val_splits: list like ["val_id", "val_ood_ads"] or None for all
        - include_test: bool
        - download: bool (download raw data)
        - max_neigh: int (default 50)
        - radius: float (default 6.0 Angstroms)
        - dtype: str (default "float32")

        Returns
        -------
        OC20ASEDBDataset
            Dataset with train/val/test splits.
        """
        if not HAS_ASE:
            raise ImportError("ASE required for OC20 S2EF dataset")

        data_dir = Path(self.get_data_dir())
        train_split = getattr(self.parameters, "train_split", "200K")
        val_splits_param = getattr(self.parameters, "val_splits", None)
        if val_splits_param is None:
            # default: use all 4 validation splits
            val_splits = [
                "val_id",
                "val_ood_ads",
                "val_ood_cat",
                "val_ood_both",
            ]
        elif isinstance(val_splits_param, (list, tuple)):
            val_splits = list(val_splits_param)
        else:
            val_splits = [str(val_splits_param)]

        include_test = bool(getattr(self.parameters, "include_test", False))
        download = bool(getattr(self.parameters, "download", False))
        max_neigh = int(getattr(self.parameters, "max_neigh", 50))
        radius = float(getattr(self.parameters, "radius", 6.0))
        dtype_str = str(getattr(self.parameters, "dtype", "float32"))
        dtype = getattr(torch, dtype_str)

        # Download if needed (raw extxyz/txt files) - not implemented
        if download:
            logger.warning(
                f"S2EF download not implemented. Please download manually to {data_dir}/s2ef/{train_split}/train"
            )

        # Preprocess to ASE DB if needed for train/val/test
        self._ensure_asedb_preprocessed(
            data_dir, train_split, val_splits, include_test, max_neigh, radius
        )

        # Collect DB files
        train_db_files = self._collect_db_files(
            data_dir / "s2ef" / train_split / "train"
        )
        val_db_files: list[Path] = []
        for vs in val_splits:
            val_dir = data_dir / "s2ef" / "all" / vs
            val_db_files.extend(self._collect_db_files(val_dir))
        test_db_files: list[Path] = []
        if include_test:
            test_dir = data_dir / "s2ef" / "all" / "test"
            test_db_files = self._collect_db_files(test_dir)

        if not train_db_files:
            raise RuntimeError(
                f"No ASE DB files found in {data_dir}/s2ef/{train_split}/train. Preprocessing may have failed."
            )

        logger.info(
            f"Loading {len(train_db_files)} train DBs, {len(val_db_files)} val DBs, {len(test_db_files)} test DBs"
        )

        ds = OC20ASEDBDataset(
            train_db_paths=[str(p) for p in train_db_files],
            val_db_paths=[str(p) for p in val_db_files],
            test_db_paths=[str(p) for p in test_db_files],
            max_neigh=max_neigh,
            radius=radius,
            dtype=dtype,
            include_energy=True,
            include_forces=True,
        )

        # Expose split_idx for fixed split handling downstream
        # Already set inside the dataset; nothing else to do here.
        return ds

    def _ensure_asedb_preprocessed(
        self,
        root: Path,
        train_split: str,
        val_splits: list[str],
        include_test: bool,
        max_neigh: int,
        radius: float,
    ) -> None:
        """Ensure ASE DB files are preprocessed for the requested splits.

        Parameters
        ----------
        root : Path
            Root data directory containing the S2EF dataset.
        train_split : str
            Name of the training split (e.g. "200K").
        val_splits : list[str]
            List of validation split names.
        include_test : bool
            Whether to ensure preprocessing for the test split.
        max_neigh : int
            Maximum number of neighbors per atom.
        radius : float
            Cutoff radius for neighbor search in Angstroms.

        Returns
        -------
        None
            Performs preprocessing as a side-effect; no value is returned.
        """
        # Train
        train_dir = root / "s2ef" / train_split / "train"
        if train_dir.exists() and needs_preprocessing(train_dir):
            logger.info(f"Preprocessing {train_dir}")
            preprocess_s2ef_split_ase(
                data_path=train_dir,
                out_path=train_dir,
                num_workers=4,
                max_neigh=max_neigh,
                radius=radius,
            )

        # Validation
        for val_split in val_splits:
            val_dir = root / "s2ef" / "all" / val_split
            if val_dir.exists() and needs_preprocessing(val_dir):
                logger.info(f"Preprocessing {val_dir}")
                preprocess_s2ef_split_ase(
                    data_path=val_dir,
                    out_path=val_dir,
                    num_workers=4,
                    max_neigh=max_neigh,
                    radius=radius,
                )

        # Test
        if include_test:
            test_dir = root / "s2ef" / "all" / "test"
            if test_dir.exists() and needs_preprocessing(test_dir):
                logger.info(f"Preprocessing {test_dir}")
                preprocess_s2ef_split_ase(
                    data_path=test_dir,
                    out_path=test_dir,
                    num_workers=4,
                    max_neigh=max_neigh,
                    radius=radius,
                )

    def _collect_db_files(self, directory: Path) -> list[Path]:
        """Collect all ASE DB files in directory.

        Parameters
        ----------
        directory : Path
            Directory to search.

        Returns
        -------
        list[Path]
            Sorted list of DB file paths.
        """
        if not directory.exists():
            return []
        return sorted(directory.glob("*.db"))

    def get_data_dir(self) -> Path:
        """Get data directory path.

        Returns
        -------
        Path
            Path to data directory.
        """
        return Path(super().get_data_dir())
