"""Loader for OC20 S2EF dataset."""

import logging
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch_geometric.data import Dataset

from topobench.data.datasets.oc20_dataset import OC20Dataset
from topobench.data.loaders.base import AbstractLoader
from topobench.data.utils.oc20_download import (
    download_s2ef_dataset,
)

# Import ASE DB fallback
try:
    from topobench.data.loaders.graph.oc20_asedbs2ef_loader import (
        OC20ASEDBDataset,
    )

    HAS_ASEDB = True
except ImportError:
    HAS_ASEDB = False

logger = logging.getLogger(__name__)


class OC20DatasetLoader(AbstractLoader):
    """Load OC20 S2EF dataset for catalyst discovery and materials science.

    Supports S2EF (Structure to Energy and Forces) to predict energy/forces
    from atomic structure.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - download: Whether to download if not present (default: False)
            - train_split: Training split size ("200K", "2M", "20M", "all")
            - val_splits: List of validation splits or None for all
            - include_test: Whether to download test split (default: True)
            - dtype: Data type for tensors (default: "float32")
            - legacy_format: Use legacy PyG Data format (default: False)
            - max_samples: Limit dataset size for testing (default: None)
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load the OC20 dataset.

        Returns
        -------
        Dataset
            The loaded OC20 dataset with the appropriate configuration.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """
        # Download if requested
        if self.parameters.get("download", False):
            self._download_dataset()

        # Check if we have LMDB files or need ASE DB fallback
        data_root = Path(self.get_data_dir())

        # Try LMDB first
        lmdb_present = any(data_root.glob("**/*.lmdb"))

        if not lmdb_present and HAS_ASEDB:
            # Fallback to ASE DB dataset
            logger.info("No LMDB files found, using ASE DB backend")
            return self._load_asedb_dataset(data_root)

        # Initialize LMDB dataset
        dataset = self._initialize_dataset()
        self.data_dir = self._redefine_data_dir(dataset)
        return dataset

    def _download_dataset(self):
        """Download the S2EF dataset."""
        root = Path(self.get_data_dir())
        train_split = self.parameters.get("train_split", "200K")
        val_splits = self.parameters.get("val_splits", None)
        include_test = self.parameters.get("include_test", True)

        # Parse val_splits
        if val_splits is not None and isinstance(val_splits, str):
            val_splits = [val_splits]

        download_s2ef_dataset(
            root=root,
            train_split=train_split,
            val_splits=val_splits,
            include_test=include_test,
        )

    def _initialize_dataset(self) -> OC20Dataset:
        """Initialize the OC20 dataset.

        Returns
        -------
        OC20Dataset
            The initialized OC20 dataset.

        Raises
        ------
        RuntimeError
            If dataset initialization fails.
        """
        try:
            dataset = OC20Dataset(
                root=str(self.get_data_dir()),
                name=self.parameters.data_name,
                parameters=self.parameters,
            )
            return dataset
        except Exception as e:
            msg = f"Error initializing OC20 dataset: {e}"
            raise RuntimeError(msg) from e

    def _load_asedb_dataset(self, data_root: Path) -> OC20ASEDBDataset:
        """Load dataset using ASE DB backend (fallback when no LMDBs).

        Parameters
        ----------
        data_root : Path
            Root directory for data.

        Returns
        -------
        OC20ASEDBDataset
            Dataset using ASE DB backend.
        """
        train_split = self.parameters.get("train_split", "200K")
        val_splits = self.parameters.get("val_splits", None)
        include_test = self.parameters.get("include_test", True)

        # Parse val_splits
        if val_splits is None:
            val_splits = [
                "val_id",
                "val_ood_ads",
                "val_ood_cat",
                "val_ood_both",
            ]
        elif isinstance(val_splits, str):
            val_splits = [val_splits]

        # Import preprocessing utilities

        # Get preprocessing parameters
        max_neigh = int(self.parameters.get("max_neigh", 50))
        radius = float(self.parameters.get("radius", 6.0))

        # Ensure preprocessing is done for all required splits
        self._ensure_asedb_preprocessed(
            data_root, train_split, val_splits, include_test, max_neigh, radius
        )

        # Collect DB files
        # The data_root might already include the dataset name (e.g., datasets/graph/oc20/OC20_S2EF_200K)
        # or just the base (e.g., datasets/graph/oc20)
        # Try both patterns
        train_subdir_name = f"s2ef_train_{train_split}"

        # Pattern 1: data_root already includes dataset name
        train_dir_pattern1 = (
            data_root
            / "s2ef"
            / train_split
            / train_subdir_name
            / train_subdir_name
        )
        # Pattern 2: data_root is just base, dataset name in separate dir
        if not train_dir_pattern1.exists():
            # Try finding s2ef directory anywhere under data_root
            s2ef_roots = list(data_root.glob("**/s2ef"))
            if s2ef_roots:
                s2ef_root = s2ef_roots[0].parent
                train_dir_pattern1 = (
                    s2ef_root
                    / "s2ef"
                    / train_split
                    / train_subdir_name
                    / train_subdir_name
                )

        train_dbs = (
            sorted(train_dir_pattern1.glob("*.db"))
            if train_dir_pattern1.exists()
            else []
        )

        val_dbs = []
        for vs in val_splits:
            val_subdir_name = f"s2ef_{vs}"
            val_dir = (
                data_root / "s2ef" / "all" / val_subdir_name / val_subdir_name
            )
            if (
                not val_dir.exists()
                and "s2ef_roots" in locals()
                and s2ef_roots
            ):
                val_dir = (
                    s2ef_roots[0].parent
                    / "s2ef"
                    / "all"
                    / val_subdir_name
                    / val_subdir_name
                )
            if val_dir.exists():
                val_dbs.extend(sorted(val_dir.glob("*.db")))

        test_dbs = []
        if include_test:
            test_dir = data_root / "s2ef" / "all" / "s2ef_test" / "s2ef_test"
            if (
                not test_dir.exists()
                and "s2ef_roots" in locals()
                and s2ef_roots
            ):
                test_dir = (
                    s2ef_roots[0].parent
                    / "s2ef"
                    / "all"
                    / "s2ef_test"
                    / "s2ef_test"
                )
            if test_dir.exists():
                test_dbs = sorted(test_dir.glob("*.db"))

        logger.info(
            f"Using ASE DB backend: {len(train_dbs)} train, "
            f"{len(val_dbs)} val, {len(test_dbs)} test DB files"
        )

        # Parse dtype
        dtype = self.parameters.get("dtype", "float32")
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        max_samples = self.parameters.get("max_samples", None)
        if max_samples is not None:
            max_samples = int(max_samples)

        return OC20ASEDBDataset(
            train_db_paths=[str(p) for p in train_dbs],
            val_db_paths=[str(p) for p in val_dbs],
            test_db_paths=[str(p) for p in test_dbs],
            max_neigh=int(self.parameters.get("max_neigh", 50)),
            radius=float(self.parameters.get("radius", 6.0)),
            dtype=dtype,
            include_energy=True,
            include_forces=True,
            max_samples=max_samples,
        )

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
        from topobench.data.preprocessor.oc20_s2ef_preprocessor import (
            needs_preprocessing,
            preprocess_s2ef_split_ase,
        )

        # Find s2ef root directory
        s2ef_roots = list(root.glob("**/s2ef"))
        if not s2ef_roots:
            logger.warning(f"No s2ef directory found under {root}")
            return

        s2ef_root = s2ef_roots[0].parent / "s2ef"

        # Train directory
        train_subdir_name = f"s2ef_train_{train_split}"
        train_dir = (
            s2ef_root / train_split / train_subdir_name / train_subdir_name
        )
        if train_dir.exists() and needs_preprocessing(train_dir):
            logger.info(f"Preprocessing {train_dir}")
            preprocess_s2ef_split_ase(
                data_path=train_dir,
                out_path=train_dir,
                num_workers=4,
                max_neigh=max_neigh,
                radius=radius,
            )

        # Validation directories
        for val_split in val_splits:
            val_subdir_name = f"s2ef_{val_split}"
            val_dir = s2ef_root / "all" / val_subdir_name / val_subdir_name
            if val_dir.exists() and needs_preprocessing(val_dir):
                logger.info(f"Preprocessing {val_dir}")
                preprocess_s2ef_split_ase(
                    data_path=val_dir,
                    out_path=val_dir,
                    num_workers=4,
                    max_neigh=max_neigh,
                    radius=radius,
                )

        # Test directory
        if include_test:
            test_dir = s2ef_root / "all" / "s2ef_test" / "s2ef_test"
            if test_dir.exists() and needs_preprocessing(test_dir):
                logger.info(f"Preprocessing {test_dir}")
                preprocess_s2ef_split_ase(
                    data_path=test_dir,
                    out_path=test_dir,
                    num_workers=4,
                    max_neigh=max_neigh,
                    radius=radius,
                )

    def _redefine_data_dir(self, dataset: Dataset) -> Path:
        """Redefine the data directory based on dataset configuration.

        Parameters
        ----------
        dataset : Dataset
            The OC20 dataset instance.

        Returns
        -------
        Path
            The redefined data directory path.
        """
        return self.get_data_dir()
