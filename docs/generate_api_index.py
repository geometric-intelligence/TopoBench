"""Generate an index.rst file for the API documentation."""

from collections import defaultdict
from pathlib import Path


def categorize_modules(modules):
    """Categorize modules into logical groups.

    Parameters
    ----------
    modules : list
        List of module names.

    Returns
    -------
    dict
        Dictionary mapping category names to lists of modules.
    """
    categories = {
        "Core": {
            "description": "Core TopoBench functionality and main entry points",
            "modules": [],
        },
        "Data Loading & Datasets": {
            "description": "Dataset loaders for different topological domains",
            "modules": [],
        },
        "Neural Network Architectures": {
            "description": "Neural network backbones, encoders, readouts, and wrappers",
            "modules": [],
        },
        "Transformations & Liftings": {
            "description": "Data transformations and topological lifting operations",
            "modules": [],
        },
        "Training & Evaluation": {
            "description": "Loss functions, optimizers, metrics, and evaluation tools",
            "modules": [],
        },
        "Utilities": {
            "description": "Helper functions and utility modules",
            "modules": [],
        },
    }

    for module in modules:
        # Core module
        if module == "topobench":
            categories["Core"]["modules"].append(module)

        # Data-related
        elif any(
            x in module
            for x in [
                "data.loader",
                "data.dataset",
                "dataloader",
                "data.preprocessor",
            ]
        ):
            categories["Data Loading & Datasets"]["modules"].append(module)

        # Neural networks
        elif any(x in module for x in ["nn.", "model."]):
            categories["Neural Network Architectures"]["modules"].append(
                module
            )

        # Transforms and liftings
        elif "transform" in module:
            categories["Transformations & Liftings"]["modules"].append(module)

        # Training/evaluation
        elif any(
            x in module
            for x in ["loss.", "optimizer.", "evaluator.", "callbacks."]
        ):
            categories["Training & Evaluation"]["modules"].append(module)

        # Utilities
        elif "utils" in module or "data.utils" in module:
            categories["Utilities"]["modules"].append(module)

        # Fallback to core
        else:
            categories["Core"]["modules"].append(module)

    # Remove empty categories
    return {k: v for k, v in categories.items() if v["modules"]}


def create_hierarchical_structure(modules):
    """Create a hierarchical structure of modules.

    Parameters
    ----------
    modules : list
        List of module names.

    Returns
    -------
    dict
        Nested dictionary representing the module hierarchy.
    """
    hierarchy = defaultdict(lambda: defaultdict(list))

    for module in modules:
        parts = module.split(".")
        if len(parts) == 1:
            # Top-level module
            hierarchy["_top_level"]["_items"].append(module)
        elif len(parts) == 2:
            # Package level (e.g., topobench.data)
            hierarchy[parts[1]]["_items"].append(module)
        else:
            # Subpackage level (e.g., topobench.data.loaders)
            main_package = parts[1]
            subpackage = ".".join(parts[2:-1]) if len(parts) > 3 else parts[2]
            hierarchy[main_package][subpackage].append(module)

    return hierarchy


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
            module_name = item.stem  # Remove ".rst" extension
            modules.append(module_name)

    if not modules:
        print(f"Warning: No API documentation files found in {api_dir}")
        return

    # Categorize modules
    categories = categorize_modules(modules)

    index_file = api_dir / "index.rst"
    with open(index_file, "w") as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("API Reference\n")
        f.write("=" * 80 + "\n\n")

        f.write(
            "This section contains the complete API documentation for the TopoBench package.\n"
        )
        f.write(
            "The documentation is automatically generated from the source code and organized\n"
        )
        f.write("into logical categories for easy navigation.\n\n")

        # Overview section
        f.write("Overview\n")
        f.write("-" * 80 + "\n\n")
        f.write(
            "TopoBench provides a comprehensive framework for topological deep learning, including:\n\n"
        )
        for category, info in categories.items():
            f.write(f"* **{category}**: {info['description']}\n")
        f.write("\n")

        # Table of contents with categories
        f.write(".. contents:: Quick Navigation\n")
        f.write("   :local:\n")
        f.write("   :depth: 2\n\n")

        # Generate sections for each category
        for category, info in categories.items():
            f.write("\n")
            f.write(category + "\n")
            f.write("-" * len(category) + "\n\n")
            f.write(f"{info['description']}\n\n")

            # Group modules hierarchically within category
            sorted_modules = sorted(info["modules"])

            # Create subsections based on module structure
            subsections = defaultdict(list)
            standalone = []

            for mod in sorted_modules:
                parts = mod.split(".")
                if len(parts) <= 2:
                    standalone.append(mod)
                else:
                    # Group by second-level package
                    key = ".".join(parts[:2])
                    subsections[key].append(mod)

            # Write standalone modules first
            if standalone:
                f.write(".. toctree::\n")
                f.write("   :maxdepth: 1\n\n")
                for mod in standalone:
                    f.write(f"   {mod}\n")
                f.write("\n")

            # Write subsections
            for subsection, mods in sorted(subsections.items()):
                subsection_title = (
                    subsection.replace("topobench.", "")
                    .replace("_", " ")
                    .title()
                )
                f.write(f"{subsection_title}\n")
                f.write("^" * len(subsection_title) + "\n\n")

                f.write(".. toctree::\n")
                f.write("   :maxdepth: 1\n\n")
                for mod in sorted(mods):
                    f.write(f"   {mod}\n")
                f.write("\n")

    print(f"Generated hierarchical API index at {index_file}")
    print(f"  Total modules: {len(modules)}")
    print(f"  Categories: {len(categories)}")
    for category, info in categories.items():
        print(f"    - {category}: {len(info['modules'])} modules")


if __name__ == "__main__":
    # Determine the script location
    script_dir = Path(__file__).parent
    api_dir = script_dir / "api"  # Directory where .rst files are located
    package_name = "topobench"  # Your package name

    print("Generating API index...")
    print(f"  API directory: {api_dir}")
    print(f"  Package name: {package_name}")

    generate_api_index(api_dir, package_name)
    print("Done!")
