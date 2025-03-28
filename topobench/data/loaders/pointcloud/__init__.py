"""Init file for PointCloud load module with automated loader discovery."""

import inspect
from importlib import util
from pathlib import Path
from typing import Any, ClassVar


class PointCloudLoaderManager:
    """Manages automatic discovery and registration of PointCloud loader classes."""

    # Base class that all loaders should inherit from (assuming there is one)
    # You may need to adjust this based on your actual base loader class
    BASE_LOADER_CLASS: ClassVar[type] = object

    @staticmethod
    def is_loader_class(obj: Any) -> bool:
        """Check if an object is a valid loader class.

        Parameters
        ----------
        obj : Any
            The object to check if it's a valid loader class.

        Returns
        -------
        bool
            True if the object is a valid loader class (non-private class
            with 'DatasetLoader' in name), False otherwise.
        """
        return (
            inspect.isclass(obj)
            and not obj.__name__.startswith("_")
            and "DatasetLoader" in obj.__name__
        )

    @classmethod
    def discover_loaders(cls, package_path: str) -> dict[str, type[Any]]:
        """Dynamically discover all loader classes in the package.

        Parameters
        ----------
        package_path : str
            Path to the package's __init__.py file.

        Returns
        -------
        Dict[str, Type[Any]]
            Dictionary mapping loader class names to their corresponding class objects.
        """
        loaders = {}

        # Get the directory containing the loader modules
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

                # Find all loader classes in the module
                new_loaders = {
                    name: obj
                    for name, obj in inspect.getmembers(module)
                    if (
                        cls.is_loader_class(obj)
                        and obj.__module__ == module.__name__
                    )
                }
                loaders.update(new_loaders)
        return loaders


# Create the loader manager
manager = PointCloudLoaderManager()

# Automatically discover and populate loaders
POINTCLOUD_LOADERS = manager.discover_loaders(__file__)

POINTCLOUD_LOADERS_list = list(POINTCLOUD_LOADERS.keys())
# Automatically generate __all__
__all__ = [
    # Loader collections
    "POINTCLOUD_LOADERS",
    "POINTCLOUD_LOADERS_list",
    # Individual loader classes
    *POINTCLOUD_LOADERS_list,
]

# For backwards compatibility, create individual imports
locals().update(**POINTCLOUD_LOADERS)
