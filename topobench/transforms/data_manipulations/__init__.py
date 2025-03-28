"""Data manipulations module with automated exports."""

import inspect
from importlib import util
from pathlib import Path
from typing import Any


class ModuleExportsManager:
    """Manages automatic discovery and registration of data manipulation classes."""

    @staticmethod
    def is_manipulation_class(obj: Any) -> bool:
        """Check if an object is a valid manipulation class.

        Parameters
        ----------
        obj : Any
            The object to check if it's a valid manipulation class.

        Returns
        -------
        bool
            True if the object is a valid manipulation class (non-private class
            defined in __main__), False otherwise.
        """
        return (
            inspect.isclass(obj)
            and obj.__module__ == "__main__"
            and not obj.__name__.startswith("_")
        )

    @classmethod
    def discover_manipulations(cls, package_path: str) -> dict[str, type]:
        """Dynamically discover all manipulation classes in the package.

        Parameters
        ----------
        package_path : str
            Path to the package's __init__.py file.

        Returns
        -------
        dict[str, type]
            Dictionary mapping class names to their corresponding class objects.
        """
        manipulations = {}

        # Get the directory containing the manipulation modules
        package_dir = Path(package_path).parent

        # Iterate through all .py files in the directory
        for file_path in package_dir.glob("*.py"):
            if file_path.stem == "__init__":
                continue

            # Import the module
            module_name = f"{Path(package_path).stem}.{file_path.stem}"
            spec = util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find all manipulation classes in the module
                new_manipulations = {
                    name: obj
                    for name, obj in inspect.getmembers(module)
                    if inspect.isclass(obj)
                    and obj.__module__ == module.__name__
                    and not name.startswith("_")
                }
                manipulations.update(new_manipulations)
        return manipulations


# Create the exports manager
manager = ModuleExportsManager()

# Automatically discover and populate DATA_MANIPULATIONS
DATA_MANIPULATIONS = manager.discover_manipulations(__file__)

# Automatically generate __all__
__all__ = [*DATA_MANIPULATIONS.keys(), "DATA_MANIPULATIONS"]

# For backwards compatibility, also create individual imports
locals().update(DATA_MANIPULATIONS)
