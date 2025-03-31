"""Generate an index.rst file for the API documentation."""

import os


def generate_api_index(api_dir, package_name):
    """Generate an index.rst file for the API documentation.

    Parameters
    ----------
    api_dir : str
        Directory containing the API documentation files.
    package_name : str
        Name of the package for which the documentation is generated.
    """

    modules = []
    for item in os.listdir(api_dir):
        if (
            item.endswith(".rst")
            and item != "index.rst"
            and item != "modules.rst"
        ):
            module_name = item[:-4]  # Remove ".rst" extension
            modules.append(module_name)

    with open(os.path.join(api_dir, "index.rst"), "w") as f:
        f.write("API Documentation\n")
        f.write("=================\n\n")
        f.write(".. toctree::\n")
        f.write("   :maxdepth: 2\n\n")
        for module in sorted(modules):
            f.write(f"   {module}\n")


if __name__ == "__main__":
    api_dir = "api"  # Directory where .rst files are located
    package_name = "topobench"  # Your package name
    generate_api_index(api_dir, package_name)
