"""S2EF preprocessing for OC20/OC22 datasets.

Creates ASE DB files with extracted graph features from provided *.extxyz files
for the S2EF task.

Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import glob
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Try importing ASE
try:
    import ase.db
    import ase.io
    from ase.atoms import Atoms

    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    logger.warning(
        "ASE not installed. S2EF preprocessing will not be available. "
        "Install with: pip install ase"
    )

# Try importing pymatgen for neighbor search
try:
    from pymatgen.io.ase import AseAtomsAdaptor

    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False
    logger.warning(
        "pymatgen not installed. Will use slower ASE neighbor search. "
        "Install with: pip install pymatgen"
    )


class AtomsToGraphs:
    """Convert ASE Atoms objects to PyTorch Geometric Data objects.

    This class handles periodic boundary conditions and creates graph representations
    suitable for machine learning on atomic structures.

    Parameters
    ----------
    max_neigh : int
        Maximum number of neighbors to consider per atom.
    radius : float
        Cutoff radius in Angstroms for neighbor search.
    r_energy : bool
        Whether to include energy in the created Data objects.
    r_forces : bool
        Whether to include forces in the created Data objects.
    r_distances : bool
        Whether to include edge distances as edge attributes.
    r_edges : bool
        Whether to compute edges (can be disabled for debugging).
    r_fixed : bool
        Whether to include fixed atom flags.
    """

    def __init__(
        self,
        max_neigh: int = 50,
        radius: float = 6.0,
        r_energy: bool = True,
        r_forces: bool = True,
        r_distances: bool = True,
        r_edges: bool = True,
        r_fixed: bool = True,
    ):
        self.max_neigh = max_neigh
        self.radius = radius
        self.r_energy = r_energy
        self.r_forces = r_forces
        self.r_distances = r_distances
        self.r_fixed = r_fixed
        self.r_edges = r_edges

    def _get_neighbors_pymatgen(self, atoms: Atoms):
        """Get neighbors using pymatgen (faster for periodic systems).

        Parameters
        ----------
        atoms : ase.atoms.Atoms
            ASE Atoms object for which to compute neighbor lists.

        Returns
        -------
        tuple
            Tuple (c_index, n_index, n_distance, offsets) representing neighbor
            center indices, neighbor indices, distances and periodic offsets.
        """
        if not HAS_PYMATGEN:
            return self._get_neighbors_ase(atoms)

        struct = AseAtomsAdaptor.get_structure(atoms)
        _c_index, _n_index, _offsets, n_distance = struct.get_neighbor_list(
            r=self.radius, numerical_tol=0, exclude_self=True
        )

        # Limit to max_neigh neighbors per atom, sorted by distance
        _nonmax_idx = []
        for i in range(len(atoms)):
            idx_i = (_c_index == i).nonzero()[0]
            idx_sorted = np.argsort(n_distance[idx_i])[: self.max_neigh]
            _nonmax_idx.append(idx_i[idx_sorted])
        _nonmax_idx = np.concatenate(_nonmax_idx)

        _c_index = _c_index[_nonmax_idx]
        _n_index = _n_index[_nonmax_idx]
        n_distance = n_distance[_nonmax_idx]
        _offsets = _offsets[_nonmax_idx]

        return _c_index, _n_index, n_distance, _offsets

    def _get_neighbors_ase(self, atoms: Atoms):
        """Get neighbors using ASE (slower but always available).

        Parameters
        ----------
        atoms : ase.atoms.Atoms
            ASE Atoms object for which to compute neighbor lists.

        Returns
        -------
        tuple
            Tuple (idx_i, idx_j, distances, offsets) representing neighbor
            center indices, neighbor indices, distances and periodic offsets.
        """
        from ase.neighborlist import neighbor_list

        idx_i, idx_j, idx_S, distances = neighbor_list(
            "ijSd", atoms, self.radius, self_interaction=False
        )

        # Limit to max_neigh neighbors per atom
        _nonmax_idx = []
        for i in range(len(atoms)):
            mask = idx_i == i
            dists_i = distances[mask]
            idx_sorted = np.argsort(dists_i)[: self.max_neigh]
            _nonmax_idx.append(np.where(mask)[0][idx_sorted])
        _nonmax_idx = np.concatenate(_nonmax_idx)

        return (
            idx_i[_nonmax_idx],
            idx_j[_nonmax_idx],
            distances[_nonmax_idx],
            idx_S[_nonmax_idx],
        )

    def _reshape_features(self, c_index, n_index, n_distance, offsets):
        """Convert neighbor info to PyTorch tensors.

        Parameters
        ----------
        c_index : array-like
            Center atom indices for edges.
        n_index : array-like
            Neighbor atom indices for edges.
        n_distance : array-like
            Distances between center and neighbor atoms.
        offsets : array-like
            Periodic cell offsets corresponding to edges.

        Returns
        -------
        tuple
            (edge_index, edge_distances, cell_offsets) as PyTorch tensors.
        """
        edge_index = torch.LongTensor(np.vstack((n_index, c_index)))
        edge_distances = torch.FloatTensor(n_distance)
        cell_offsets = torch.LongTensor(offsets)

        # Remove very small distances (self-interactions that slipped through)
        nonzero = torch.where(edge_distances >= 1e-8)[0]
        edge_index = edge_index[:, nonzero]
        edge_distances = edge_distances[nonzero]
        cell_offsets = cell_offsets[nonzero]

        return edge_index, edge_distances, cell_offsets

    def convert(self, atoms: Atoms) -> Data:
        """Convert a single ASE Atoms object to a PyG Data object.

        Parameters
        ----------
        atoms : ase.atoms.Atoms
            ASE Atoms object with positions, atomic numbers, cell, etc.

        Returns
        -------
        torch_geometric.data.Data
            PyG Data object containing node features, positions, edges, and
            optional energy/forces/fixed flags.
        """
        # Basic atomic structure info
        atomic_numbers = torch.LongTensor(atoms.get_atomic_numbers())
        positions = torch.FloatTensor(atoms.get_positions())
        cell = torch.FloatTensor(np.array(atoms.get_cell())).view(1, 3, 3)
        natoms = len(atoms)

        # Create base data object
        # Create node features from atomic numbers (one-hot or simple embedding)
        # For now, use atomic numbers as features (can be enhanced later)
        node_features = atomic_numbers.unsqueeze(1).float()

        data = Data(
            cell=cell,
            pos=positions,
            atomic_numbers=atomic_numbers,
            natoms=natoms,
            z=atomic_numbers,  # Alias for compatibility
            x=node_features,  # Add node features for compatibility with transforms
        )

        # Add edges if requested
        if self.r_edges:
            if HAS_PYMATGEN:
                split_idx_dist = self._get_neighbors_pymatgen(atoms)
            else:
                split_idx_dist = self._get_neighbors_ase(atoms)
            edge_index, edge_distances, cell_offsets = self._reshape_features(
                *split_idx_dist
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets

            if self.r_distances:
                data.distances = edge_distances
                data.edge_attr = edge_distances.view(
                    -1, 1
                )  # For compatibility

        # Add energy if available and requested
        if self.r_energy:
            try:
                energy = atoms.get_potential_energy(apply_constraint=False)
                data.y = torch.FloatTensor([energy])
                data.energy = torch.FloatTensor([energy])
            except Exception:
                # Energy not available (e.g., no calculator)
                pass

        # Add forces if available and requested
        if self.r_forces:
            try:
                forces = atoms.get_forces(apply_constraint=False)
                data.force = torch.FloatTensor(forces)
            except Exception:
                # Forces not available
                pass

        # Add fixed atom flags if requested
        if self.r_fixed:
            fixed_idx = torch.zeros(natoms, dtype=torch.float32)
            if hasattr(atoms, "constraints"):
                from ase.constraints import FixAtoms

                for constraint in atoms.constraints:
                    if isinstance(constraint, FixAtoms):
                        fixed_idx[constraint.index] = 1
            data.fixed = fixed_idx

        # Add metadata from atoms.info if present
        if hasattr(atoms, "info") and atoms.info:
            info_data = atoms.info.get("data", {})
            if "sid" in info_data:
                data.sid = info_data["sid"]
            if "fid" in info_data:
                data.fid = info_data["fid"]
            if "ref_energy" in info_data:
                data.ref_energy = torch.FloatTensor([info_data["ref_energy"]])

        return data


class S2EFPreprocessor:
    """Preprocessor for S2EF data using ASE database format.

    This class handles conversion from extxyz files to ASE database format.
    """

    def write_db(
        self,
        extxyz_paths: list[str | Path | list[str | Path]],
        dbs: dict[str, ase.db.core.Database],
        map_file_to_log: dict[str, str | None] | None = None,
        num_workers: int = 1,
        batch_size: int = 100,
    ):
        """Write ASE DB files from extxyz inputs.

        This stores for each extended XYZ file the Atoms objects and the
        metadata from the corresponding xyz_log file into the ASE Database.

        Parameters
        ----------
        extxyz_paths : list[str | Path | list[str | Path]]
            Paths to the extended XYZ files or lists of paths per DB.
        dbs : dict[str, ase.db.core.Database]
            Mapping from extxyz file keys to ASE Database objects.
        map_file_to_log : dict[str, str | None] | None
            Mapping from extxyz paths to xyz_log paths. If None, no metadata is used.
        num_workers : int, optional
            Number of worker processes to use, by default 1.
        batch_size : int, optional
            Number of structures to write to the ASE DB file at a time, by default 100.

        Returns
        -------
        None
            This function writes DB files as a side effect.
        """
        if not HAS_ASE:
            raise ImportError("ASE is required for S2EF preprocessing")

        if map_file_to_log is None:
            map_file_to_log = {k: None for k in dbs}

        num_workers = min(num_workers, len(extxyz_paths))
        node_counts = []  # Track node counts for each structure

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    self._write_db_worker,
                    file_path,
                    map_file_to_log[db_path],
                    dbs[db_path],
                    batch_size,
                )
                for file_path, db_path in zip(
                    extxyz_paths,
                    map_file_to_log.keys(),
                    strict=True,
                )
            ]
            for future in tqdm(
                as_completed(futures),
                desc="Writing DBs",
                total=len(futures),
                leave=False,
                file=sys.stdout,
                dynamic_ncols=True,
            ):
                db_path, num_atoms, structure_node_counts = future.result()
                node_counts.extend(structure_node_counts)

        # Save node counts
        node_counts_path = (
            Path(list(dbs.keys())[0]).parent.parent / "node_counts.npy"
        )
        np.save(node_counts_path, np.array(node_counts))
        logger.info(
            f"Saved {len(node_counts)} node counts to {node_counts_path}"
        )

    @staticmethod
    def _write_db_worker(
        file_paths: str | Path | list[str | Path],
        xyz_log_path: str | Path | None,
        db: ase.db.core.Database,
        batch_size: int,
    ):
        """Worker function to write atoms to ASE database.

        Parameters
        ----------
        file_paths : str | Path | list[str | Path]
            Path or list of paths to extended XYZ file(s).
        xyz_log_path : str | Path | None
            Path to the log file with metadata or None.
        db : ase.db.core.Database
            ASE database object to write to.
        batch_size : int
            Number of structures to batch together.

        Returns
        -------
        tuple
            (db_path, total_atoms, node_counts).
        """
        if xyz_log_path is not None:
            with open(xyz_log_path) as f:
                xyz_log = f.read().splitlines()
        else:
            xyz_log = None

        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]

        node_counts = []  # Track node counts for each structure
        total_atoms = 0

        for file_path in file_paths:
            atoms_list = ase.io.read(file_path, ":")
            atoms_batch = []
            log_batch = []
            for i, atoms in enumerate(atoms_list):
                if xyz_log is not None and i < len(xyz_log):
                    log_line = xyz_log[i].split(",")
                    log_info = {
                        "sid": int(log_line[0].split("random")[1]),
                        "fid": int(log_line[1].split("frame")[1]),
                        "ref_energy": float(log_line[2]),
                    }
                else:
                    log_info = {}

                atoms_batch.append(atoms)
                log_batch.append(log_info)
                node_counts.append(len(atoms))

                if len(atoms_batch) == batch_size:
                    # Write batch to database
                    for atom, log in zip(atoms_batch, log_batch, strict=True):
                        db.write(atom, data=log)
                    atoms_batch, log_batch = [], []

            total_atoms += len(atoms_list)

        # Write remaining batch
        if atoms_batch:
            for atom, log in zip(atoms_batch, log_batch, strict=True):
                db.write(atom, data=log)

        db_path = db.filename if hasattr(db, "filename") else str(db)
        return db_path, total_atoms, node_counts


def needs_preprocessing(data_path: Path) -> bool:
    """Check if data needs preprocessing (has extxyz but no db files)."""
    has_extxyz = bool(list(data_path.glob("*.extxyz")))
    has_db = bool(list(data_path.glob("*.db")))
    return has_extxyz and not has_db


def preprocess_s2ef_split_ase(
    data_path: Path,
    out_path: Path,
    num_workers: int = 1,
    ref_energy: bool = True,
    test_data: bool = False,
    max_neigh: int = 50,
    radius: float = 6.0,
) -> None:
    """Preprocess S2EF data from extxyz/txt to ASE DB format.

    Parameters
    ----------
    data_path : Path
        Path to directory containing *.extxyz and *.txt files.
    out_path : Path
        Directory to save ASE DB files.
    num_workers : int
        Number of parallel workers for preprocessing.
    ref_energy : bool
        Whether to include reference energies in metadata.
    test_data : bool
        Whether this is test data (no energy/forces in log).
    max_neigh : int
        Maximum number of neighbors per atom.
    radius : float
        Cutoff radius for neighbor search in Angstroms.

    Returns
    -------
    None
        This function writes ASE DB files as a side effect.
    """
    if not HAS_ASE:
        raise ImportError("ASE is required for S2EF preprocessing")

    logger.info(
        f"Preprocessing S2EF data from {data_path} to {out_path} (ASE DB format)"
    )

    # Find all extxyz files
    extxyz_files = sorted(glob.glob(str(data_path / "*.extxyz")))

    if not extxyz_files:
        logger.warning(f"No extxyz files found in {data_path}")
        return

    out_path.mkdir(parents=True, exist_ok=True)

    # Create mapping from extxyz to log files, but only for files that need preprocessing
    map_file_to_log = {}
    dbs = {}
    skipped_count = 0

    for extxyz_file in extxyz_files:
        extxyz_path = Path(extxyz_file)
        base_name = extxyz_path.stem

        # Check if DB file already exists
        db_path = out_path / f"{base_name}.db"
        if db_path.exists():
            skipped_count += 1
            continue  # Skip files that already have DB files

        # Find corresponding txt file
        txt_path = data_path / f"{base_name}.txt"
        if txt_path.exists() and ref_energy and not test_data:
            map_file_to_log[str(extxyz_path)] = str(txt_path)
        else:
            map_file_to_log[str(extxyz_path)] = None

        # Create ASE DB for this file
        dbs[str(extxyz_path)] = ase.db.connect(str(db_path))

    if skipped_count > 0:
        logger.info(
            f"Skipping {skipped_count} files that already have DB files"
        )

    if not map_file_to_log:
        logger.info("All files already preprocessed, skipping...")
        return

    # Write to ASE databases
    preprocessor = S2EFPreprocessor()
    preprocessor.write_db(
        extxyz_paths=list(map_file_to_log.keys()),
        dbs=dbs,
        map_file_to_log=map_file_to_log,
        num_workers=num_workers,
        batch_size=100,
    )

    logger.info(f"Preprocessing complete. DB files saved to {out_path}")


def preprocess_s2ef_dataset_ase(
    data_path: Path,
    num_workers: int = 1,
    splits: list[str] | None = None,
    max_neigh: int = 50,
    radius: float = 6.0,
) -> None:
    """Preprocess entire S2EF dataset with train/val/test splits.

    Parameters
    ----------
    data_path : Path
        Root path containing train/val/test subdirectories.
    num_workers : int
        Number of parallel workers.
    splits : Optional[list[str]]
        List of splits to process. If None, processes all found splits.
    max_neigh : int
        Maximum number of neighbors per atom.
    radius : float
        Cutoff radius for neighbor search in Angstroms.

    Returns
    -------
    None
        This function writes ASE DB files as a side effect.
    """
    if not HAS_ASE:
        raise ImportError("ASE is required for S2EF preprocessing")

    if splits is None:
        # Auto-detect splits
        splits = [d.name for d in data_path.iterdir() if d.is_dir()]

    for split in splits:
        split_path = data_path / split
        if not split_path.exists():
            logger.warning(f"Split {split} not found at {split_path}")
            continue

        logger.info(f"Processing split: {split}")
        is_test = "test" in split.lower()

        preprocess_s2ef_split_ase(
            data_path=split_path,
            out_path=split_path,
            num_workers=num_workers,
            ref_energy=True,
            test_data=is_test,
            max_neigh=max_neigh,
            radius=radius,
        )

    logger.info("Dataset preprocessing complete")
