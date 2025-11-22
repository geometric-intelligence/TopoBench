"""Utilities for downloading and preparing OC20 datasets."""

from __future__ import annotations

import logging
import lzma
import os
import shutil
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)

# OC20 dataset split URLs
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

IS2RE_URL = "https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_val_test_lmdbs.tar.gz"
OC22_IS2RE_URL = "https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/is2res_total_train_val_test_lmdbs.tar.gz"


def uncompress_xz(file_path: str) -> str:
    """Decompress .xz files.

    Parameters
    ----------
    file_path : str
        Path to file to decompress.

    Returns
    -------
    str
        Path to decompressed file.
    """
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


def download_and_extract(
    url: str, target_dir: Path, skip_if_extracted: bool = True
) -> Path:
    """Download and extract a tar archive.

    Parameters
    ----------
    url : str
        URL to download from.
    target_dir : Path
        Directory to extract to.
    skip_if_extracted : bool
        If True, skip extraction if extracted files already exist (default: True).

    Returns
    -------
    Path
        Path to extracted directory.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / os.path.basename(url)

    # Download if needed
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
    else:
        logger.info(f"Archive {target_file.name} already downloaded")

    # Check if extraction is needed
    extraction_marker = target_dir / ".extracted"
    if skip_if_extracted and extraction_marker.exists():
        logger.info(f"Archive {target_file.name} already extracted, skipping")
        return target_dir

    # Extract
    logger.info(f"Extracting {target_file.name}...")
    if str(target_file).endswith((".tar.gz", ".tgz")):
        with tarfile.open(target_file, "r:gz") as tar:
            tar.extractall(path=target_dir)
    elif str(target_file).endswith(".tar"):
        with tarfile.open(target_file, "r:") as tar:
            tar.extractall(path=target_dir)
    else:
        raise ValueError(f"Unsupported archive format: {target_file}")

    # Mark as extracted
    extraction_marker.touch()
    return target_dir


def decompress_xz_files(directory: Path):
    """Decompress all .xz files in a directory.

    Parameters
    ----------
    directory : Path
        Directory to search for .xz files.
    """
    xz_files = list(directory.glob("**/*.xz"))
    if xz_files:
        logger.info(
            f"Decompressing {len(xz_files)} .xz files in {directory}..."
        )
        num_workers = max(1, os.cpu_count() - 1)
        # Use threads to avoid pickling/import issues with processes on macOS
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(uncompress_xz, str(f)) for f in xz_files
            ]
            for future in as_completed(futures):
                future.result()


def download_s2ef_dataset(
    root: Path,
    train_split: str = "200K",
    val_splits: list[str] | None = None,
    include_test: bool = True,
):
    """Download S2EF dataset splits.

    Parameters
    ----------
    root : Path
        Root directory for data storage.
    train_split : str
        Training split size: "200K", "2M", "20M", or "all".
    val_splits : list[str] | None
        List of validation splits to download. If None, downloads all.
    include_test : bool
        Whether to download test split.
    """
    if val_splits is None:
        val_splits = list(S2EF_VAL_SPLITS.keys())

    # Download train split
    train_url = S2EF_TRAIN_SPLITS[train_split]
    train_subdir_name = f"s2ef_train_{train_split}"
    train_dir = (
        root / "s2ef" / train_split / train_subdir_name / train_subdir_name
    )
    if not train_dir.exists():
        logger.info(f"Downloading S2EF train split: {train_split}")
        download_and_extract(train_url, root / "s2ef" / train_split)
        decompress_xz_files(root / "s2ef" / train_split)
    else:
        logger.info(
            f"S2EF train split {train_split} already exists, skipping download"
        )

    # Download validation splits
    for val_split in val_splits:
        val_url = S2EF_VAL_SPLITS[val_split]
        val_subdir_name = f"s2ef_{val_split}"
        val_dir = root / "s2ef" / "all" / val_subdir_name / val_subdir_name
        if not val_dir.exists():
            logger.info(f"Downloading S2EF validation split: {val_split}")
            download_and_extract(val_url, root / "s2ef" / "all")
            decompress_xz_files(root / "s2ef" / "all")
        else:
            logger.info(
                f"S2EF validation split {val_split} already exists, skipping download"
            )

    # Download test split
    if include_test:
        test_subdir_name = "s2ef_test"
        test_dir = root / "s2ef" / "all" / test_subdir_name / test_subdir_name
        if not test_dir.exists():
            logger.info("Downloading S2EF test split")
            download_and_extract(S2EF_TEST_SPLIT, root / "s2ef" / "all")
            decompress_xz_files(root / "s2ef" / "all")
        else:
            logger.info("S2EF test split already exists, skipping download")


def download_is2re_dataset(root: Path, task: str = "is2re"):
    """Download IS2RE or OC22 IS2RE dataset.

    Parameters
    ----------
    root : Path
        Root directory for data storage.
    task : str
        Task name: "is2re" or "oc22_is2re".
    """
    url = IS2RE_URL if task == "is2re" else OC22_IS2RE_URL
    target_dir = root / task

    if not target_dir.exists():
        logger.info(f"Downloading {task.upper()} dataset")
        download_and_extract(url, root)
        decompress_xz_files(root)
    else:
        logger.info(
            f"{task.upper()} dataset already exists, skipping download"
        )
