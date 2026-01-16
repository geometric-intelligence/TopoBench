"""Parallel preprocessing for multi-core speedup.

This module implements parallel sample processing using :class:`ProcessPoolExecutor`
to achieve speedup on multi-core systems by distributing work across worker
processes.
"""

import os
import pickle
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


def _process_batch(
    batch_indices: list[int],
    dataset: Any,
    transform: Any,
    output_dir: Path,
) -> list[tuple[int, bool, str | None]]:
    """Load and process batch of samples inside a worker process.

    Parameters
    ----------
    batch_indices : list[int]
        List of sample indices to process.
    dataset : Any
        Dataset to load samples from (must be pickle-able).
    transform : Any
        Transform to apply.
    output_dir : Path
        Directory to save processed samples.

    Returns
    -------
    list[tuple[int, bool, str | None]]
        List of (index, success, error_message) for each sample.
    """
    results = []

    for idx in batch_indices:
        output_path = output_dir / f"sample_{idx:06d}.pt"

        try:
            data = dataset[idx]

            if transform is not None:
                data = transform(data)

            torch.save(data, output_path)
            del data

            results.append((idx, True, None))
        except Exception as exc:  # noqa: BLE001
            error_msg = f"Sample {idx}: {type(exc).__name__}: {str(exc)}"
            results.append((idx, False, error_msg))

    return results


class ParallelProcessor:
    """Manage and execute parallel sample processing.

    Distributes sample processing across multiple worker processes using
    ProcessPoolExecutor.

    Parameters
    ----------
    num_workers : int | None
        Number of worker processes (default: None = auto-detect).
        If 1, uses sequential processing (no overhead).
        If None, uses cpu_count-1 (leaves 1 core for system).
    batch_size : int
        Samples per worker batch (default: 32).
        Larger = better throughput, more memory per worker.
    show_progress : bool
        Show progress bar (default: True).

    Examples
    --------
    >>> processor = ParallelProcessor(num_workers=4)
    >>> results = processor.process(
    ...     dataset=dataset,
    ...     transform=transform,
    ...     output_dir=Path("./processed"),
    ...     num_samples=10000
    ... )
    >>> print(f"Processed {results['success']} / {results['total']}")
    """

    def __init__(
        self,
        num_workers: int | None = None,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> None:
        """Initialize parallel processor.

        Parameters
        ----------
        num_workers : int | None
            Number of workers (None = auto).
        batch_size : int
            Samples per batch (default: 32).
        show_progress : bool
            Show progress bar (default: True).
        """
        # Auto-detect optimal worker count
        if num_workers is None:
            cpu_count = os.cpu_count() or 1
            # Leave 1 core for system/OS
            num_workers = max(1, cpu_count - 1)

        self.num_workers = max(1, num_workers)
        self.batch_size = batch_size
        self.show_progress = show_progress

    def process(
        self,
        dataset: Any,
        transform: Any,
        output_dir: Path,
        num_samples: int,
    ) -> dict[str, Any]:
        """Execute processing dataset using parallel execution when possible.

        Parameters
        ----------
        dataset : Any
            Source dataset to process.
        transform : Any
            Transform to apply (must be pickle-able).
        output_dir : Path
            Directory to save processed samples.
        num_samples : int
            Total number of samples to process.

        Returns
        -------
        dict[str, Any]
            Processing statistics:
            - total: Total samples.
            - success: Successfully processed.
            - failed: Failed samples.
            - errors: List of error messages.
        """
        if self.num_workers == 1:
            return self._process_sequential(
                dataset, transform, output_dir, num_samples
            )

        return self._process_parallel(
            dataset, transform, output_dir, num_samples
        )

    def _process_sequential(
        self,
        dataset: Any,
        transform: Any,
        output_dir: Path,
        num_samples: int,
    ) -> dict[str, Any]:
        """Execute processing samples sequentially as fallback mode.

        Parameters
        ----------
        dataset : Any
            Source dataset.
        transform : Any
            Transform to apply.
        output_dir : Path
            Directory to save processed samples.
        num_samples : int
            Number of samples.

        Returns
        -------
        dict[str, Any]
            Processing statistics.
        """
        success_count = 0
        errors: list[str] = []

        iterator = range(num_samples)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Processing", unit="sample")

        for idx in iterator:
            output_path = output_dir / f"sample_{idx:06d}.pt"

            try:
                data = dataset[idx]

                if transform is not None:
                    data = transform(data)

                torch.save(data, output_path)
                del data

                success_count += 1
            except Exception as exc:  # noqa: BLE001
                error_msg = f"Sample {idx}: {type(exc).__name__}: {str(exc)}"
                errors.append(error_msg)

        return {
            "total": num_samples,
            "success": success_count,
            "failed": len(errors),
            "errors": errors,
        }

    def _process_parallel(
        self,
        dataset: Any,
        transform: Any,
        output_dir: Path,
        num_samples: int,
    ) -> dict[str, Any]:
        """Execute processing samples in parallel using worker processes.

        Parameters
        ----------
        dataset : Any
            Source dataset (must be pickle-able).
        transform : Any
            Transform to apply.
        output_dir : Path
            Directory to save processed samples.
        num_samples : int
            Number of samples.

        Returns
        -------
        dict[str, Any]
            Processing statistics.
        """
        try:
            # Test if dataset can be pickled (required for multiprocessing)
            pickle.dumps(dataset)
        except (pickle.PicklingError, AttributeError, TypeError) as exc:
            if self.show_progress:
                print(
                    f"\nDataset cannot be pickled ({type(exc).__name__}). "
                    f"Falling back to sequential processing..."
                )
            return self._process_sequential(
                dataset, transform, output_dir, num_samples
            )
        batches = [
            list(range(i, min(i + self.batch_size, num_samples)))
            for i in range(0, num_samples, self.batch_size)
        ]

        success_count = 0
        errors: list[str] = []

        try:
            # Use ProcessPoolExecutor for parallelism
            # Fork is 10-20× faster on Linux; spawn is safer on macOS/Windows
            if sys.platform == "linux":
                mp_context = get_context("fork")
            else:
                mp_context = get_context("spawn")

            with ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=mp_context,
            ) as executor:
                futures = {
                    executor.submit(
                        _process_batch,
                        batch,
                        dataset,
                        transform,
                        output_dir,
                    ): batch
                    for batch in batches
                }

                pbar = None
                if self.show_progress:
                    pbar = tqdm(
                        total=num_samples,
                        desc=f"Processing ({self.num_workers} workers)",
                        unit="sample",
                    )

                for future in as_completed(futures):
                    try:
                        batch_results = future.result()

                        for _idx, success, error in batch_results:
                            if success:
                                success_count += 1
                            else:
                                errors.append(error)

                        if pbar is not None:
                            pbar.update(len(batch_results))
                    except Exception as exc:
                        batch_indices = futures[future]
                        tb_str = "".join(
                            traceback.format_exception(
                                type(exc), exc, exc.__traceback__
                            )[-3:]
                        )

                        error_msg = (
                            f"Batch {batch_indices[0]}-{batch_indices[-1]}: "
                            f"{type(exc).__name__}: {str(exc)}\n{tb_str}"
                        )
                        errors.append(error_msg)

                        if pbar is not None:
                            pbar.update(len(batch_indices))

                if pbar is not None:
                    pbar.close()

        except Exception as exc:
            print(f"\nParallel processing failed: {exc}")
            print("Falling back to sequential processing...")
            return self._process_sequential(
                dataset, transform, output_dir, num_samples
            )

        return {
            "total": num_samples,
            "success": success_count,
            "failed": len(errors),
            "errors": errors,
        }
