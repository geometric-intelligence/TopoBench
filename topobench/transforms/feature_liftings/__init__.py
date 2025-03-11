"""Feature lifting transforms with automated exports."""

import inspect
from importlib import util
from pathlib import Path
from typing import Any

from .identity import Identity  # Import Identity for special case


class ModuleExportsManager:
    """Manages automatic discovery and registration of feature lifting classes."""

    @staticmethod
    def is_lifting_class(obj: Any) -> bool:
        """Check if an object is a valid lifting class.

        Parameters
        ----------
        obj : Any
            The object to check if it's a valid lifting class.

        Returns
        -------
        bool
            True if the object is a valid lifting class (non-private class
            defined in __main__), False otherwise.
        """
        return (
            inspect.isclass(obj)
            and obj.__module__ == "__main__"
            and not obj.__name__.startswith("_")
        )

    @classmethod
    def discover_liftings(
        cls, package_path: str, special_cases: dict[Any, type] | None = None
    ) -> dict[str, type]:
        """Dynamically discover all lifting classes in the package.

        Parameters
        ----------
        package_path : str
            Path to the package's __init__.py file.
        special_cases : Optional[dict[Any, type]]
            Dictionary of special case mappings (e.g., {None: Identity}),
            by default None.

        Returns
        -------
        dict[str, type]
            Dictionary mapping class names to their corresponding class objects,
            including any special cases if provided.
        """
        liftings = {}

        # Get the directory containing the lifting modules
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

                # Find all lifting classes in the module
                new_liftings = {
                    name: obj
                    for name, obj in inspect.getmembers(module)
                    if (
                        inspect.isclass(obj)
                        and obj.__module__ == module.__name__
                        and not name.startswith("_")
                    )
                }
                liftings.update(new_liftings)
        # Add special cases if provided
        if special_cases:
            liftings.update(special_cases)

        return liftings


# Create the exports manager
manager = ModuleExportsManager()

# Automatically discover and populate FEATURE_LIFTINGS with special case for None
FEATURE_LIFTINGS = manager.discover_liftings(
    __file__, special_cases={None: Identity}
)

# Automatically generate __all__ (excluding None key)
__all__ = [name for name in FEATURE_LIFTINGS if isinstance(name, str)] + [
    "FEATURE_LIFTINGS"
]

# For backwards compatibility, create individual imports (excluding None key)
locals().update(
    {k: v for k, v in FEATURE_LIFTINGS.items() if isinstance(k, str)}
)
