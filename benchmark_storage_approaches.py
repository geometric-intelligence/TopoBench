#!/usr/bin/env python3
"""
TopoBench Storage Approach Benchmark.

This script compares the performance of TopoBench's OnDiskInductivePreprocessor
(memory-mapped storage) against PyTorch Geometric's OnDiskDataset (SQL) to demonstrate
superior performance for inductive learning workloads.

Usage:
    python benchmark_storage_approaches.py [--num-samples N] [--size SIZE] [--output FILE]

Arguments:
    --num-samples N: Number of samples to benchmark (default: 100)
    --size SIZE:     Graph size profile: small, medium, large (default: medium)
    --output FILE:   Save results to JSON file (optional)
"""

import argparse
import gc
import json
import pickle
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from torch_geometric.data import Data, InMemoryDataset, OnDiskDataset

from topobench.data.preprocessor.ondisk_inductive import (
    OnDiskInductivePreprocessor,
)


class BenchmarkDataset(InMemoryDataset):
    """Synthetic dataset for benchmarking storage approaches.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    size : str
        Size profile: 'small', 'medium', or 'large'.
    """

    def __init__(self, num_samples: int, size: str = "medium") -> None:
        super().__init__()
        self.num_samples = num_samples

        # Configure graph sizes
        size_configs = {"small": 50, "medium": 200, "large": 1000}
        num_nodes = size_configs[size]

        # Generate synthetic data
        data_list = [
            Data(
                x=torch.randn(num_nodes + i % 10, 16),
                edge_index=torch.randint(0, num_nodes, (2, num_nodes * 3)),
                y=torch.tensor([i % 10]),
            )
            for i in range(num_samples)
        ]
        self.data, self.slices = self.collate(data_list)

    def __len__(self) -> int:
        """Return number of samples.

        Returns
        -------
        int
            Number of samples.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> Data:
        """Return sample at index.

        Parameters
        ----------
        idx : int
            Index of sample.

        Returns
        -------
        Data
            Sample at index.
        """
        return super().get(idx)


def benchmark_topobench_preprocessor(
    dataset, output_dir: Path, num_samples: int
) -> dict[str, Any]:
    """Benchmark TopoBench's OnDiskInductivePreprocessor.

    Parameters
    ----------
    dataset : Dataset
        Source dataset to preprocess.
    output_dir : Path
        Directory for benchmark outputs.
    num_samples : int
        Number of samples in dataset.

    Returns
    -------
    Dict[str, Any]
        Performance metrics.
    """
    results = {"approach": "TopoBench_Preprocessor"}

    # Sequential write performance
    seq_dir = output_dir / "topobench_seq"
    gc.collect()
    start = time.time()

    OnDiskInductivePreprocessor(
        dataset=dataset,
        data_dir=seq_dir,
        transforms_config=None,
        num_workers=1,
    )

    results["write_time_seq"] = time.time() - start

    # Parallel write performance
    par_dir = output_dir / "topobench_par"
    gc.collect()
    start = time.time()

    processor = OnDiskInductivePreprocessor(
        dataset=dataset,
        data_dir=par_dir,
        transforms_config=None,
        num_workers=None,  # Use all available cores
    )

    results["write_time_par"] = time.time() - start

    # Disk space
    disk_usage = sum(
        f.stat().st_size for f in par_dir.rglob("*") if f.is_file()
    )
    results["disk_mb"] = disk_usage / 1024 / 1024

    # Random read performance
    indices = torch.randint(0, len(processor), (100,)).tolist()
    gc.collect()
    start = time.time()

    for idx in indices:
        _ = processor[idx]

    results["random_read_time"] = time.time() - start

    # Sequential read performance
    gc.collect()
    start = time.time()

    for i in range(len(processor)):
        _ = processor[i]
        if i % 100 == 0:
            gc.collect()

    results["seq_read_time"] = time.time() - start

    # Memory usage
    process = psutil.Process()
    gc.collect()
    mem_before = process.memory_info().rss / 1024 / 1024

    for i in range(min(500, len(processor))):
        _ = processor[i]
        if i % 100 == 0:
            gc.collect()

    gc.collect()
    mem_after = process.memory_info().rss / 1024 / 1024
    results["mem_growth_mb"] = mem_after - mem_before

    # Parallel processing capability
    try:
        pickle.dumps(processor)
        results["supports_parallel"] = True
    except Exception:
        results["supports_parallel"] = False

    return results


def benchmark_pyg_ondisk(
    dataset, output_dir: Path, num_samples: int, backend: str = "sqlite"
) -> dict[str, Any]:
    """Benchmark PyTorch Geometric's OnDiskDataset.

    Parameters
    ----------
    dataset : Dataset
        Source dataset to copy.
    output_dir : Path
        Directory for benchmark outputs.
    num_samples : int
        Number of samples in dataset.
    backend : str
        Database backend ('sqlite' or 'rocksdb').

    Returns
    -------
    Dict[str, Any]
        Performance metrics.
    """
    results = {"approach": f"PyG_OnDisk_{backend.upper()}"}

    db_dir = output_dir / f"pyg_{backend}"
    db_dir.mkdir(exist_ok=True)

    try:
        # Check PyTorch Geometric version and OnDiskDataset availability
        import torch_geometric

        pyg_version = torch_geometric.__version__

        # OnDiskDataset was introduced in PyG 2.3.0
        major, minor = map(int, pyg_version.split(".")[:2])
        if major < 2 or (major == 2 and minor < 3):
            results["error"] = (
                f"OnDiskDataset requires PyG >= 2.3.0 (found {pyg_version})"
            )
            return results

        class BenchmarkOnDiskDataset(OnDiskDataset):
            def __init__(self, root, source, backend):
                self.source = source
                super().__init__(root, backend=backend)
                self._process()

            def _process(self) -> None:
                for i in range(len(self.source)):
                    self.append(self.source[i])

        # Write performance
        gc.collect()
        start = time.time()
        pyg_dataset = BenchmarkOnDiskDataset(str(db_dir), dataset, backend)
        results["write_time"] = time.time() - start

        # Disk space
        disk_usage = sum(
            f.stat().st_size for f in db_dir.rglob("*") if f.is_file()
        )
        results["disk_mb"] = disk_usage / 1024 / 1024

        # Random read performance
        indices = torch.randint(0, len(pyg_dataset), (100,)).tolist()
        gc.collect()
        start = time.time()

        for idx in indices:
            _ = pyg_dataset[idx]

        results["random_read_time"] = time.time() - start

        # Sequential read performance
        gc.collect()
        start = time.time()

        for i in range(len(pyg_dataset)):
            _ = pyg_dataset[i]
            if i % 100 == 0:
                gc.collect()

        results["seq_read_time"] = time.time() - start

        # Memory usage
        process = psutil.Process()
        gc.collect()
        mem_before = process.memory_info().rss / 1024 / 1024

        for i in range(min(500, len(pyg_dataset))):
            _ = pyg_dataset[i]
            if i % 100 == 0:
                gc.collect()

        gc.collect()
        mem_after = process.memory_info().rss / 1024 / 1024
        results["mem_growth_mb"] = mem_after - mem_before

        # Parallel processing capability
        try:
            pickle.dumps(pyg_dataset)
            results["supports_parallel"] = True
        except Exception:
            results["supports_parallel"] = False

    except Exception as e:
        results["error"] = str(e)

    return results


def print_results_table(results_topobench: dict, results_pyg: dict) -> None:
    """Print formatted comparison table.

    Parameters
    ----------
    results_topobench : Dict
        TopoBench preprocessor results.
    results_pyg : Dict
        PyG OnDiskDataset results.
    """
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON: TopoBench vs PyTorch Geometric")
    print("=" * 80)

    def format_metric(metric: str) -> None:
        """Format metric comparison.

        Parameters
        ----------
        metric : str
            Metric name.
        """
        tb_val = results_topobench.get(metric)
        pyg_val = results_pyg.get(metric)

        if tb_val is not None and pyg_val is not None:
            if "time" in metric:
                ratio = pyg_val / tb_val
                status = (
                    f"{ratio:.1f}x faster"
                    if ratio > 1
                    else f"{1 / ratio:.1f}x slower"
                )
            elif metric == "disk_mb":
                ratio = pyg_val / tb_val
                status = f"{ratio:.2f}x {'larger' if ratio > 1 else 'smaller'}"
            elif metric == "mem_growth_mb":
                status = f"{pyg_val - tb_val:+.1f} MB {'more' if pyg_val > tb_val else 'less'}"
            else:
                status = ""

            print(f"{metric:<35} {tb_val:<15.2f} {pyg_val:<15.2f} {status}")
        elif tb_val is not None:
            print(f"{metric:<35} {tb_val:<15.2f} {'-':<15} {'-':<15}")
        elif pyg_val is not None:
            print(f"{metric:<35} {'-':<15} {pyg_val:<15.2f} {'-':<15}")

    print(
        f"{'Metric':<35} {'TopoBench':<15} {'PyG OnDisk':<15} {'Comparison'}"
    )
    print("-" * 80)

    # Write time - sequential (fair comparison)
    tb_seq = results_topobench.get("write_time_seq")
    pyg_write = results_pyg.get("write_time")
    if tb_seq is not None and pyg_write is not None:
        ratio = pyg_write / tb_seq
        status = (
            f"{ratio:.1f}x faster" if ratio > 1 else f"{1 / ratio:.1f}x slower"
        )
        print(
            f"{'write_time_seq':<35} {tb_seq:<15.2f} {pyg_write:<15.2f} {status}"
        )
    elif tb_seq is not None:
        print(f"{'write_time_seq':<35} {tb_seq:<15.2f} {'-':<15} {'-':<15}")

    format_metric("write_time_par")
    format_metric("random_read_time")
    format_metric("seq_read_time")
    format_metric("disk_mb")
    format_metric("mem_growth_mb")

    # Parallel capability
    tb_parallel = results_topobench.get("supports_parallel", False)
    pyg_parallel = results_pyg.get("supports_parallel", False)
    tb_status = "✓" if tb_parallel else "✗"
    pyg_status = "✓" if pyg_parallel else "✗"

    print(
        f"{'Parallel Processing':<35} {tb_status:<15} {pyg_status:<15} {'-':<15}"
    )


def main():
    """Run the benchmark suite."""
    parser = argparse.ArgumentParser(
        description="TopoBench Storage Approach Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_storage_approaches.py
  python benchmark_storage_approaches.py --num-samples 500 --size small
  python benchmark_storage_approaches.py --output results.json
        """,
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to benchmark (default: 100)",
    )
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        default="medium",
        help="Graph size profile (default: medium)",
    )
    parser.add_argument("--output", type=str, help="Save results to JSON file")

    args = parser.parse_args()

    print("TopoBench Storage Benchmark")
    print(f"Dataset: {args.num_samples} {args.size}-sized graphs")
    print("=" * 50)

    # Create test dataset
    print("Creating test dataset...", end=" ", flush=True)
    dataset = BenchmarkDataset(num_samples=args.num_samples, size=args.size)
    print(f" {len(dataset)} samples created")

    # Verify picklability for parallel processing
    try:
        pickle.dumps(dataset)
        print(" Dataset is picklable (supports parallel processing)")
    except Exception as e:
        print(f" Dataset not picklable: {e}")
        sys.exit(1)

    results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            tmpdir = Path(tmpdir)

            # Benchmark TopoBench approach
            print("\nBenchmarking TopoBench preprocessor...")
            results["topobench"] = benchmark_topobench_preprocessor(
                dataset, tmpdir, args.num_samples
            )

            # Benchmark PyG approach
            print("\nBenchmarking PyTorch Geometric OnDiskDataset...")
            results["pyg"] = benchmark_pyg_ondisk(
                dataset, tmpdir, args.num_samples, "sqlite"
            )

            # Print comparison
            print_results_table(results["topobench"], results["pyg"])

            # Check if PyG benchmark failed
            if "error" in results["pyg"]:
                print(
                    f"\n⚠️  PyG OnDiskDataset benchmark failed: {results['pyg']['error']}"
                )
                print("   This may be due to PyTorch Geometric API changes.")
                print("   TopoBench benchmark completed successfully.")

            # Save results if requested
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"\n✓ Results saved to {args.output}")

            # Summary (only if PyG worked)
            if "error" not in results["pyg"]:
                read_speedup = (
                    results["pyg"]["random_read_time"]
                    / results["topobench"]["random_read_time"]
                )

                print("\nSummary:")
                print(f"  - {read_speedup:.1f}x faster random reads")
                print(
                    f"  - {'✓' if results['topobench']['supports_parallel'] else '✗'} Parallel processing support"
                )
                print(
                    f"  - {'✓' if results['pyg']['supports_parallel'] else '✗'} PyG parallel processing support"
                )

        finally:
            # Explicit cleanup notification (automatic via TemporaryDirectory context)
            print("\n✓ Temporary files cleaned up")


if __name__ == "__main__":
    main()
