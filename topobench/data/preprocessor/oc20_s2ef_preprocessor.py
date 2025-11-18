"""S2EF preprocessing, adapted from OC20's preprocess_ef.py.
-----------
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Creates LMDB files with extracted graph features from provided *.extxyz files
for the S2EF task.
"""

from __future__ import annotations

import glob
import logging
import multiprocessing as mp
import os
import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Try importing ASE and fairchem dependencies
try:
    import ase.io
    from fairchem.core.preprocessing import AtomsToGraphs

    HAS_FAIRCHEM = True
except ImportError:
    HAS_FAIRCHEM = False
    logger.warning(
        "fairchem-core or ASE not installed. S2EF preprocessing will not be available. "
        "Install with: pip install fairchem-core ase"
    )


def _write_images_to_lmdb(mp_arg):
    """Write trajectory frames to LMDB (worker function)."""
    if not HAS_FAIRCHEM:
        raise ImportError("fairchem-core is required for S2EF preprocessing")

    (
        a2g,
        db_path,
        samples,
        sampled_ids,
        idx,
        pid,
        data_path,
        ref_energy,
        test_data,
        get_edges,
    ) = mp_arg

    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    pbar = tqdm(
        total=sum(1 for s in samples for line in open(s)),  # noqa: SIM115
        position=pid,
        desc=f"Worker {pid} preprocessing",
        leave=False,
    )

    for sample in samples:
        with open(sample) as fp:
            traj_logs = fp.read().splitlines()

        xyz_idx = os.path.splitext(os.path.basename(sample))[0]
        traj_path = os.path.join(data_path, f"{xyz_idx}.extxyz")

        if not os.path.exists(traj_path):
            logger.warning(f"Missing extxyz file: {traj_path}, skipping")
            continue

        traj_frames = ase.io.read(traj_path, ":")

        for i, frame in enumerate(traj_frames):
            if i >= len(traj_logs):
                logger.warning(
                    f"Log mismatch for {traj_path} frame {i}, skipping"
                )
                continue

            frame_log = traj_logs[i].split(",")
            sid = int(frame_log[0].split("random")[1])
            fid = int(frame_log[1].split("frame")[1])

            data_object = a2g.convert(frame)
            data_object.tags = torch.LongTensor(frame.get_tags())
            data_object.sid = sid
            data_object.fid = fid

            # Subtract off reference energy if needed
            if ref_energy and not test_data and len(frame_log) > 2:
                ref_energy_val = float(frame_log[2])
                data_object.energy -= ref_energy_val

            txn = db.begin(write=True)
            txn.put(
                f"{idx}".encode("ascii"),
                pickle.dumps(data_object, protocol=-1),
            )
            txn.commit()
            idx += 1
            sampled_ids.append(",".join(frame_log[:2]) + "\n")
            pbar.update(1)

    # Save count of objects in lmdb
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    db.sync()
    db.close()
    pbar.close()

    return sampled_ids, idx


def preprocess_s2ef_split(
    data_path: Path,
    out_path: Path,
    num_workers: int = 1,
    ref_energy: bool = True,
    test_data: bool = False,
    get_edges: bool = False,
) -> None:
    """Preprocess S2EF data from extxyz/txt to LMDB format.

    Parameters
    ----------
    data_path : Path
        Path to directory containing *.extxyz and *.txt files.
    out_path : Path
        Directory to save LMDB files.
    num_workers : int
        Number of parallel workers for preprocessing.
    ref_energy : bool
        Whether to subtract reference energies.
    test_data : bool
        Whether this is test data (no energy/forces).
    get_edges : bool
        Whether to precompute and store edge indices (~10x storage).
    """
    if not HAS_FAIRCHEM:
        raise ImportError(
            "fairchem-core and ASE are required for S2EF preprocessing. "
            "Install with: pip install fairchem-core ase"
        )

    logger.info(f"Preprocessing S2EF data from {data_path} to {out_path}")

    # Find all txt files
    xyz_logs = glob.glob(str(data_path / "*.txt"))
    if not xyz_logs:
        raise RuntimeError(f"No *.txt files found in {data_path}")

    num_workers = min(num_workers, len(xyz_logs))

    # Initialize feature extractor
    a2g = AtomsToGraphs(
        max_neigh=50,
        radius=6,
        r_energy=not test_data,
        r_forces=not test_data,
        r_fixed=True,
        r_distances=False,
        r_edges=get_edges,
    )

    # Create output directory
    out_path.mkdir(parents=True, exist_ok=True)

    # Initialize LMDB paths
    db_paths = [
        str(out_path / f"data.{i:04d}.lmdb") for i in range(num_workers)
    ]

    # Chunk trajectories into workers
    chunked_txt_files = np.array_split(xyz_logs, num_workers)

    # Extract features in parallel
    sampled_ids = [[]] * num_workers
    idx = [0] * num_workers

    logger.info(f"Starting preprocessing with {num_workers} workers...")

    with mp.Pool(num_workers) as pool:
        mp_args = [
            (
                a2g,
                db_paths[i],
                chunked_txt_files[i],
                sampled_ids[i],
                idx[i],
                i,
                str(data_path),
                ref_energy,
                test_data,
                get_edges,
            )
            for i in range(num_workers)
        ]
        op = list(
            zip(*pool.imap(_write_images_to_lmdb, mp_args), strict=False)
        )
        sampled_ids, idx = list(op[0]), list(op[1])

    # Write logs
    for j, i in enumerate(range(num_workers)):
        log_path = out_path / f"data_log.{i:04d}.txt"
        with open(log_path, "w") as ids_log:
            ids_log.writelines(sampled_ids[j])

    total_samples = sum(idx)
    logger.info(
        f"Preprocessing complete: {total_samples} samples written to {out_path}"
    )


def needs_preprocessing(raw_dir: Path, processed_dir: Path) -> bool:
    """Check if a split needs preprocessing.

    Parameters
    ----------
    raw_dir : Path
        Directory containing raw extxyz/txt files.
    processed_dir : Path
        Directory where LMDB files should be.

    Returns
    -------
    bool
        True if preprocessing is needed.
    """
    if not raw_dir.exists():
        return False

    # Check if processed directory has LMDB files
    if not processed_dir.exists():
        return True

    lmdb_files = list(processed_dir.glob("*.lmdb"))
    return len(lmdb_files) == 0


def preprocess_s2ef_dataset(
    root: Path,
    train_split: str,
    val_splits: list[str],
    include_test: bool = True,
    num_workers: int | None = None,
) -> None:
    """Preprocess entire S2EF dataset (train/val/test splits).

    Parameters
    ----------
    root : Path
        Root directory containing S2EF data.
    train_split : str
        Train split name (e.g., "200K").
    val_splits : list[str]
        List of validation split names.
    include_test : bool
        Whether to preprocess test split.
    num_workers : Optional[int]
        Number of parallel workers (default: CPU count - 1).
    """
    if not HAS_FAIRCHEM:
        raise ImportError(
            "fairchem-core and ASE are required for S2EF preprocessing. "
            "Install with: pip install fairchem-core ase"
        )

    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    s2ef_root = root / "s2ef"

    # Preprocess train split
    train_raw = s2ef_root / train_split / "train"
    train_processed = train_raw  # Store LMDBs alongside raw data

    if needs_preprocessing(train_raw, train_processed):
        logger.info(f"Preprocessing train split: {train_split}")
        preprocess_s2ef_split(
            train_raw,
            train_processed,
            num_workers=num_workers,
            ref_energy=True,
            test_data=False,
        )
    else:
        logger.info(f"Train split {train_split} already preprocessed")

    # Preprocess validation splits
    for val_split in val_splits:
        val_raw = s2ef_root / "all" / val_split
        val_processed = val_raw

        if needs_preprocessing(val_raw, val_processed):
            logger.info(f"Preprocessing validation split: {val_split}")
            preprocess_s2ef_split(
                val_raw,
                val_processed,
                num_workers=num_workers,
                ref_energy=True,
                test_data=False,
            )
        else:
            logger.info(f"Validation split {val_split} already preprocessed")

    # Preprocess test split
    if include_test:
        test_raw = s2ef_root / "all" / "test"
        test_processed = test_raw

        if needs_preprocessing(test_raw, test_processed):
            logger.info("Preprocessing test split")
            preprocess_s2ef_split(
                test_raw,
                test_processed,
                num_workers=num_workers,
                ref_energy=False,
                test_data=True,
            )
        else:
            logger.info("Test split already preprocessed")
