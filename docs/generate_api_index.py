"""Generate an index.rst file for the API documentation."""

from pathlib import Path


def organize_by_package_structure(modules):
    """Organize modules by their package structure.

    Parameters
    ----------
    modules : list
        List of module names.

    Returns
    -------
    dict
        Dictionary with package structure and descriptions.
    """
    # Define the main packages with their descriptions
    packages = {
        "topobench.callbacks": {
            "title": "Callbacks",
            "description": "Training callbacks for monitoring and control",
            "modules": [],
        },
        "topobench.data": {
            "title": "Data",
            "description": "Dataset loading, preprocessing, and utilities",
            "modules": [],
        },
        "topobench.dataloader": {
            "title": "Data Loaders",
            "description": "Data loading utilities and batch processing",
            "modules": [],
        },
        "topobench.evaluator": {
            "title": "Evaluators",
            "description": "Model evaluation metrics and tools",
            "modules": [],
        },
        "topobench.loss": {
            "title": "Loss Functions",
            "description": "Loss functions for training topological models",
            "modules": [],
        },
        "topobench.model": {
            "title": "Models",
            "description": "Model definitions and architectures",
            "modules": [],
        },
        "topobench.nn": {
            "title": "Neural Networks",
            "description": "Neural network components: backbones, encoders, readouts, and wrappers",
            "modules": [],
        },
        "topobench.optimizer": {
            "title": "Optimizers",
            "description": "Optimization algorithms and schedulers",
            "modules": [],
        },
        "topobench.transforms": {
            "title": "Transformations",
            "description": "Data transformations and topological lifting operations",
            "modules": [],
        },
        "topobench.utils": {
            "title": "Utilities",
            "description": "Helper functions and utility modules",
            "modules": [],
        },
        "topobench": {
            "title": "Core Package",
            "description": "Main TopoBench package and entry points",
            "modules": [],
        },
    }

    # Organize modules into packages (check most specific first)
    for module in modules:
        # Find which package this module belongs to
        # Sort packages by length (longest first) to match most specific
        sorted_packages = sorted(packages.keys(), key=len, reverse=True)
        for pkg_name in sorted_packages:
            if module == pkg_name or module.startswith(pkg_name + "."):
                packages[pkg_name]["modules"].append(module)
                break

    # Remove empty packages
    return {k: v for k, v in packages.items() if v["modules"]}


def generate_api_index(api_dir, package_name):
    """Generate an index.rst file for the API documentation.

    Parameters
    ----------
    api_dir : str or Path
        Directory containing the API documentation files.
    package_name : str
        Name of the package for which the documentation is generated.
    """
    api_dir = Path(api_dir)

    if not api_dir.exists():
        print(
            f"Warning: API directory {api_dir} does not exist. Skipping index generation."
        )
        return

    modules = []
    for item in api_dir.iterdir():
        if (
            item.suffix == ".rst"
            and item.name != "index.rst"
            and item.name != "modules.rst"
        ):
            module_name = item.stem
            modules.append(module_name)

    if not modules:
        print(f"Warning: No API documentation files found in {api_dir}")
        return

    # Organize modules by package structure
    packages = organize_by_package_structure(modules)

    index_file = api_dir / "index.rst"
    with open(index_file, "w") as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("API Reference\n")
        f.write("=" * 80 + "\n\n")

        f.write(
            "Welcome to the TopoBench API documentation. This section provides detailed\n"
        )
        f.write(
            "documentation for all modules, classes, and functions in the TopoBench package.\n"
        )
        f.write("\n")
        f.write(
            "The documentation is organized following the package structure for easy navigation.\n"
        )
        f.write("\n\n")

        # Overview section with package cards
        f.write("Package Overview\n")
        f.write("-" * 80 + "\n\n")

        # Write a cleaner overview without listing all modules
        for pkg_name, info in sorted(packages.items()):
            # Count submodules (excluding the package itself)
            submodule_count = len(
                [m for m in info["modules"] if m != pkg_name]
            )

            # Create a nice package card
            f.write(f":doc:`{pkg_name}` - **{info['title']}**\n")
            f.write(f"    {info['description']}\n")
            if submodule_count > 0:
                f.write(f"    ({submodule_count} submodules)\n")
            f.write("\n")

        f.write("\n")

        # Detailed sections for each package
        f.write("Detailed Documentation\n")
        f.write("-" * 80 + "\n\n")

        for pkg_name, info in sorted(packages.items()):
            # Main package section
            f.write(f"{info['title']}\n")
            f.write("^" * len(info["title"]) + "\n\n")
            f.write(f"{info['description']}\n\n")

            # Just link to the main package page which will have the details
            f.write(".. toctree::\n")
            f.write("   :maxdepth: 2\n\n")
            f.write(f"   {pkg_name}\n\n")

    print(f"Generated package-structured API index at {index_file}")
    print(f"  Total modules: {len(modules)}")
    print(f"  Main packages: {len(packages)}")
    for pkg_name, info in sorted(packages.items()):
        submodule_count = len([m for m in info["modules"] if m != pkg_name])
        print(f"    - {info['title']}: {submodule_count} submodules")


if __name__ == "__main__":
    # Determine the script location
    script_dir = Path(__file__).parent
    api_dir = script_dir / "api"
    package_name = "topobench"

    print("Generating API index...")
    print(f"  API directory: {api_dir}")
    print(f"  Package name: {package_name}")

    generate_api_index(api_dir, package_name)
    print("Done!")
