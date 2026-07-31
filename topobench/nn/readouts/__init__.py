"""Readout classes used by the library with canonical automated exports."""

import importlib
import inspect
from pathlib import Path
from typing import Any


class ReadoutExportsManager:
    """Manages automatic discovery and registration of readout classes."""

    @staticmethod
    def is_readout_class(obj: Any) -> bool:
        """Check if an object is a valid readout class.

        Parameters
        ----------
        obj : Any
            The object to check if it's a valid readout class.

        Returns
        -------
        bool
            True if the object is a valid readout class (non-private class defined in __main__), False otherwise.
        """
        return (
            inspect.isclass(obj)
            and obj.__module__ == "__main__"
            and not obj.__name__.startswith("_")
        )

    @classmethod
    def discover_readouts(cls, package_path: str) -> dict[str, type]:
        """Dynamically discover all readout classes in the package.

        Parameters
        ----------
        package_path : str
            Path to the package's __init__.py file.

        Returns
        -------
        dict[str, type]
            Dictionary mapping class names to their corresponding class objects.
        """
        readouts: dict[str, type] = {}
        package_dir = Path(package_path).parent

        for file_path in sorted(package_dir.glob("*.py")):
            if file_path.stem == "__init__":
                continue

            module_name = f"{__package__}.{file_path.stem}"
            module = importlib.import_module(module_name)
            new_readouts = {
                name: obj
                for name, obj in inspect.getmembers(module)
                if inspect.isclass(obj)
                and obj.__module__ == module.__name__
                and not name.startswith("_")
            }
            readouts.update(new_readouts)
        return dict(sorted(readouts.items()))


# Create the exports manager
manager = ReadoutExportsManager()

# Automatically discover and populate READOUT_CLASSES
READOUT_CLASSES = manager.discover_readouts(__file__)

# Automatically generate __all__
__all__ = [*READOUT_CLASSES.keys(), "READOUT_CLASSES"]

# For backwards compatibility, also create individual imports
locals().update(READOUT_CLASSES)
