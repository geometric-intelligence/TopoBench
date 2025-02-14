import inspect
from importlib import util
from pathlib import Path


def discover_objs(package_path, condition=None):
    """Dynamically discover all manipulation classes in the package.

    Parameters
    ----------
    package_path : str
        Path to the package's __init__.py file.
    condition : callable
        `(name, obj) -> bool`

    Returns
    -------
    dict[str, type]
        Dictionary mapping class names to their corresponding class objects.
    """
    if condition is None:

        def condition(name, obj):
            return True

    objs = {}

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
            for name, obj in inspect.getmembers(module):
                if (
                    not inspect.isclass(obj)
                    or name.startswith("_")
                    or obj.__module__ != module.__name__
                ):
                    continue

                if condition(name, obj):
                    objs[name] = obj

    return objs
