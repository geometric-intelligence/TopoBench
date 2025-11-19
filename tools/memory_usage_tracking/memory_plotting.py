"""Monitor and plot memory usage of test pipelines."""

import argparse
import ast
import csv
import itertools
import os
import subprocess
import sys
import tempfile
import time

import matplotlib.pyplot as plt
import pandas as pd
import psutil

# Constants & paths
SCRIPT_FILE_PATH = "test/pipeline/memory_checks_template.py"
OUTPUT_ROOT = "tools/memory_usage_tracking/outputs"


def dataset_short(name: str) -> str:
    """
    Return the substring after the first slash.

    Parameters
    ----------
    name : str
        Dataset name or path-like string.

    Returns
    -------
    str
        Shortened dataset name.
    """
    parts = name.split("/", 1)
    return parts[1] if len(parts) > 1 else parts[0]


def model_fs(name: str) -> str:
    """
    Make a filesystem-safe name.

    Parameters
    ----------
    name : str
        Original model or dataset name.

    Returns
    -------
    str
        Name with slashes replaced by double underscores.
    """
    return name.replace("/", "__")


def apply_ast_replacements(
    original_content: str,
    dataset_to_inject: str,
    model_to_inject: str,
) -> str:
    """
    Replace module-level DATASET and MODELS assignments.

    Parameters
    ----------
    original_content : str
        Original Python source.
    dataset_to_inject : str
        New value for the DATASET variable.
    model_to_inject : str
        New single model name for the MODELS list.

    Returns
    -------
    str
        Modified Python source code.

    Raises
    ------
    SyntaxError
        If the original source cannot be parsed.
    """
    tree = ast.parse(original_content)

    class ModuleLevelReplacer(ast.NodeTransformer):
        """AST transformer for module-level assignments."""

        def visit_Module(self, node):
            """
            Modify DATASET and MODELS assignments in the module.

            Parameters
            ----------
            node : ast.Module
                Root module node.

            Returns
            -------
            ast.Module
                Modified module node.
            """
            for n in node.body:
                if (
                    isinstance(n, ast.Assign)
                    and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Name)
                ):
                    name = n.targets[0].id
                    if name == "DATASET":
                        n.value = ast.Constant(value=dataset_to_inject)
                    elif name == "MODELS":
                        n.value = ast.List(
                            elts=[ast.Constant(value=model_to_inject)],
                            ctx=ast.Load(),
                        )
            return node

    modified_tree = ModuleLevelReplacer().visit(tree)
    ast.fix_missing_locations(modified_tree)
    return ast.unparse(modified_tree)


def monitor_script(
    script_path: str,
    dataset_to_inject: str,
    model_to_inject: str,
    output_csv: str,
    interval: float = 0.05,
) -> tuple[list[tuple[float, float]], int]:
    """
    Run a pytest script and track its memory usage.

    Parameters
    ----------
    script_path : str
        Path to the template Python test script.
    dataset_to_inject : str
        Dataset string to inject into the template.
    model_to_inject : str
        Model string to inject into the template.
    output_csv : str
        Path where memory usage CSV will be written.
    interval : float, optional
        Sampling interval in seconds, by default 0.05.

    Returns
    -------
    list of tuple of float
        Recorded (time_s, memory_MB) pairs.
    int
        Exit code of the pytest process.
    """
    print(f"Running for model '{model_to_inject}' on dataset '{dataset_to_inject}'")

    project_root = os.getcwd()
    script_abspath = os.path.abspath(script_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

    try:
        with open(script_abspath, encoding="utf-8") as f:
            original_content = f.read()
    except FileNotFoundError:
        print(f"ERROR: Template not found at {script_abspath}", file=sys.stderr)
        return [], -1

    try:
        modified_code = apply_ast_replacements(
            original_content, dataset_to_inject, model_to_inject
        )
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to AST-modify the template: {e}", file=sys.stderr)
        return [], -1

    temp_script_path: str | None = None
    try:
        original_script_dir = os.path.dirname(script_abspath)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            dir=original_script_dir,
            encoding="utf-8",
        ) as temp_f:
            temp_f.write(modified_code)
            temp_script_path = temp_f.name

        process = subprocess.Popen(
            ["pytest", temp_script_path], cwd=project_root, env=env
        )

        ps_proc = psutil.Process(process.pid)
        memory_data: list[tuple[float, float]] = []
        start_time = time.time()

        try:
            while process.poll() is None:
                try:
                    mem_info = ps_proc.memory_info()
                    rss_mb = mem_info.rss / (1024**2)
                    timestamp = time.time() - start_time
                    memory_data.append((timestamp, rss_mb))
                    time.sleep(interval)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
        except KeyboardInterrupt:
            print("Interrupted by user. Terminating child process.")
            process.terminate()

        process.wait()
        return_code = process.returncode
        total_runtime = time.time() - start_time

        print(f"Script finished in {total_runtime:.2f}s with return code {return_code}")

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "memory_MB"])
            writer.writerows(memory_data)
        print(f"Memory usage saved to {output_csv}")

        return memory_data, return_code
    finally:
        if temp_script_path and os.path.exists(temp_script_path):
            try:
                os.remove(temp_script_path)
                print(f"Temporary script {temp_script_path} removed.")
            except OSError as e:
                print(
                    f"Warning: failed to remove temp file {temp_script_path}: {e}",
                    file=sys.stderr,
                )


def plot_normalized_memory(
    model_label: str,
    csv_files: list[str],
    labels: list[str],
    plot_path: str,
    colors: tuple[str, ...] = ("blue", "red", "green"),
) -> None:
    """
    Plot memory usage vs normalized time.

    Parameters
    ----------
    model_label : str
        Label for the model shown in the plot title.
    csv_files : list of str
        Paths to CSV files with time and memory columns.
    labels : list of str
        Legend labels corresponding to each CSV file.
    plot_path : str
        Output path for the saved plot image.
    colors : tuple of str, optional
        Sequence of colors for the lines, by default ("blue", "red", "green").
    """
    plt.figure(figsize=(10, 6))
    color_cycle = itertools.cycle(colors)
    for i, csv_file in enumerate(csv_files):
        label = labels[i] if i < len(labels) else f"run_{i}"
        if not os.path.exists(csv_file):
            print(f"Warning: CSV file not found, skipping plot line: {csv_file}")
            continue
        df = pd.read_csv(csv_file)
        if df.empty or "time_s" not in df.columns or "memory_MB" not in df.columns:
            print(f"Warning: CSV invalid/empty, skipping plot line: {csv_file}")
            continue
        total_duration = df["time_s"].iloc[-1] - df["time_s"].iloc[0]
        if total_duration > 0:
            df["norm_time"] = (df["time_s"] - df["time_s"].iloc[0]) / total_duration
        else:
            n = max(1, len(df) - 1)
            df["norm_time"] = (df.index - df.index[0]) / n
            print(
                f"Warning: zero duration in {csv_file}; using sample index for normalization."
            )
        plt.plot(
            df["norm_time"],
            df["memory_MB"],
            label=label,
            color=next(color_cycle),
        )
    plt.xlabel("Normalized time")
    plt.ylabel("Memory usage [MB]")
    plt.title(f"Memory usage comparison for model: {model_label}")
    plt.xlim(0, 1)
    plt.ylim(0, None)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def plot_raw_time_memory(
    model_label: str,
    csv_files: list[str],
    labels: list[str],
    plot_path: str,
    colors: tuple[str, ...] = ("blue", "red", "green"),
) -> None:
    """
    Plot memory usage vs raw time.

    Parameters
    ----------
    model_label : str
        Label for the model shown in the plot title.
    csv_files : list of str
        Paths to CSV files with time and memory columns.
    labels : list of str
        Legend labels corresponding to each CSV file.
    plot_path : str
        Output path for the saved plot image.
    colors : tuple of str, optional
        Sequence of colors for the lines, by default ("blue", "red", "green").
    """
    plt.figure(figsize=(10, 6))
    color_cycle = itertools.cycle(colors)
    for i, csv_file in enumerate(csv_files):
        label = labels[i] if i < len(labels) else f"run_{i}"
        if not os.path.exists(csv_file):
            print(f"Warning: CSV file not found, skipping plot line: {csv_file}")
            continue
        df = pd.read_csv(csv_file)
        if df.empty or "time_s" not in df.columns or "memory_MB" not in df.columns:
            print(f"Warning: CSV invalid/empty, skipping plot line: {csv_file}")
            continue
        t0 = df["time_s"].iloc[0] if len(df) else 0.0
        df["time_shifted"] = df["time_s"] - t0
        plt.plot(
            df["time_shifted"],
            df["memory_MB"],
            label=label,
            color=next(color_cycle),
        )
    plt.xlabel("Time [s]")
    plt.ylabel("Memory usage [MB]")
    plt.title(f"Memory usage (raw time) for model: {model_label}")
    plt.xlim(0, None)
    plt.ylim(0, None)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def main() -> None:
    """
    Run memory monitoring and plotting from the command line.

    Parses arguments, runs the monitoring pipeline for all requested
    datasets and models and writes CSVs and plots under OUTPUT_ROOT.
    """
    parser = argparse.ArgumentParser(
        description="Monitor and plot memory usage of a test pipeline."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Datasets to compare (e.g., graph/ds1 hypergraph/ds1). "
            "Order is used for plotting and file naming."
        ),
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Models to run (e.g., graph/gcn).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.05,
        help="Sampling interval in seconds.",
    )
    args = parser.parse_args()

    # Display-friendly dataset names and filesystem-safe names
    dataset_disps = [dataset_short(d) for d in args.datasets]
    dataset_fs = [model_fs(d) for d in args.datasets]

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for model in args.models:
        model_label = model
        model_part = model_fs(model_label)

        cfg_folder = "__".join(dataset_fs + [model_part])
        config_dir = os.path.join(OUTPUT_ROOT, cfg_folder)
        os.makedirs(config_dir, exist_ok=True)

        csv_outputs_for_model: list[str] = []
        return_codes: list[int] = []

        for idx, _dataset_disp in enumerate(dataset_disps):
            original_dataset_arg = args.datasets[idx]
            output_csv = os.path.join(
                config_dir, f"{model_part}__{dataset_fs[idx]}.csv"
            )
            _, rc = monitor_script(
                SCRIPT_FILE_PATH,
                original_dataset_arg,
                model_label,
                output_csv,
                interval=args.interval,
            )
            csv_outputs_for_model.append(output_csv)
            return_codes.append(rc)

        for i, rc in enumerate(return_codes):
            if rc != 0:
                print(
                    f"WARNING: Run for model '{model_label}' on dataset "
                    f"'{dataset_disps[i]}' exited with non-zero code {rc}.",
                    file=sys.stderr,
                )

        labels = dataset_disps[:]
        norm_plot_path = os.path.join(config_dir, "memory_plot_normalized.png")
        raw_plot_path = os.path.join(config_dir, "memory_plot_raw.png")

        plot_normalized_memory(
            model_label,
            csv_outputs_for_model,
            labels=labels,
            plot_path=norm_plot_path,
            colors=("blue", "red", "green"),
        )
        plot_raw_time_memory(
            model_label,
            csv_outputs_for_model,
            labels=labels,
            plot_path=raw_plot_path,
            colors=("blue", "red", "green"),
        )


if __name__ == "__main__":
    main()
