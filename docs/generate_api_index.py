"""Generate an index.rst file for the API documentation."""

from pathlib import Path


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

    index_file = api_dir / "index.rst"
    with open(index_file, "w") as f:
        f.write("API Documentation\n")
        f.write("=================\n\n")
        f.write(
            "This section contains the API documentation for the TopoBench package,\n"
        )
        f.write("automatically generated from the source code.\n\n")
        f.write(".. toctree::\n")
        f.write("   :maxdepth: 2\n")
        f.write("   :caption: Modules\n\n")
        for module in sorted(modules):
            f.write(f"   {module}\n")

    print(f"Generated API index at {index_file} with {len(modules)} modules.")


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
