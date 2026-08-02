"""Abstract Loader class."""

import hashlib
import json
import random
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from collections.abc import Iterator
from contextlib import contextmanager
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch_geometric
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

CACHE_SOURCE_ATTRIBUTE = "_topobench_processed_cache_source"


def normalize_cache_value(value: Any, *, path: str) -> Any:
    """Return a deterministic, type-preserving JSON-compatible value."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return value
    if isinstance(value, str):
        if "${" in value:
            raise ValueError(f"{path} contains an unresolved interpolation")
        return value
    if isinstance(value, Mapping):
        if (
            all(isinstance(key, str) for key in value)
            and "__cache_type__" not in value
        ):
            return {
                key: normalize_cache_value(
                    value[key],
                    path=f"{path}.{key}",
                )
                for key in sorted(value)
            }
        items = []
        for index, (key, item) in enumerate(value.items()):
            items.append(
                [
                    normalize_cache_value(
                        key,
                        path=f"{path}.items[{index}].key",
                    ),
                    normalize_cache_value(
                        item,
                        path=f"{path}.items[{index}].value",
                    ),
                ]
            )
        items.sort(key=canonical_json_bytes)
        return {"__cache_type__": "mapping", "items": items}
    if isinstance(value, list):
        return [
            normalize_cache_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return {
            "__cache_type__": "tuple",
            "items": [
                normalize_cache_value(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, Sequence):
        raise TypeError(
            f"{path} has unsupported sequence type {type(value).__name__}"
        )
    raise TypeError(
        f"{path} has unsupported cache value type {type(value).__name__}"
    )


def resolve_cache_config(value: Any, *, context: str) -> Any:
    """Resolve OmegaConf containers before canonical cache normalization."""
    if isinstance(value, (DictConfig, ListConfig)):
        try:
            value = OmegaConf.to_container(
                value,
                resolve=True,
                throw_on_missing=True,
            )
        except OmegaConfBaseException as error:
            raise ValueError(
                f"{context} could not be fully resolved: "
                f"{type(error).__name__}"
            ) from error
    return normalize_cache_value(value, path=context.replace(" ", "."))


def _loader_cache_source(
    loader: "AbstractLoader",
    dataset: torch_geometric.data.Dataset | torch.utils.data.Dataset,
) -> dict[str, Any]:
    """Build the loader-owned portion of processed-cache provenance."""
    parameters = loader.effective_cache_parameters(dataset)
    selector = {
        key: parameters.get(key)
        for key in ("data_domain", "data_type", "data_name")
    }
    feature_policy = (
        parameters["feature_policy"]
        if "feature_policy" in parameters
        else normalize_cache_value(
            getattr(dataset, "feature_policy", None),
            path="cache.source.feature_policy",
        )
    )
    representation_version = (
        parameters["representation_version"]
        if "representation_version" in parameters
        else normalize_cache_value(
            getattr(dataset, "representation_version", None),
            path="cache.source.versions.representation",
        )
    )
    parser_version = (
        parameters["parser_version"]
        if "parser_version" in parameters
        else normalize_cache_value(
            getattr(dataset, "parser_version", None),
            path="cache.source.versions.parser",
        )
    )
    return {
        "dataset_selector": selector,
        "loader": {
            "target": f"{type(loader).__module__}.{type(loader).__qualname__}",
            "parameters": parameters,
        },
        "feature_policy": feature_policy,
        "versions": {
            "representation": representation_version,
            "parser": parser_version,
        },
    }


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value deterministically."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using canonical serialization."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@contextmanager
def isolated_rng_state() -> Iterator[None]:
    """Restore caller-global Python, NumPy, and CPU Torch RNG state on exit."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        try:
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)


class AbstractLoader(ABC):
    """Abstract class that provides an interface to load data.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters.
    """

    def __init__(self, parameters: DictConfig):
        self.parameters = parameters
        self.root_data_dir = Path(parameters["data_dir"])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(parameters={self.parameters})"

    def get_data_dir(self) -> str:
        """Return a selector directory confined to the configured root."""
        data_name = self.parameters.data_name
        if not isinstance(data_name, str) or not data_name:
            raise ValueError("data_name must be a non-empty string")
        root = self.root_data_dir.resolve()

        data_dir = (root / data_name).resolve()
        try:
            data_dir.relative_to(root)
        except ValueError as error:
            raise ValueError(
                "data_name must remain within data_dir"
            ) from error
        return str(data_dir)

    def effective_cache_parameters(
        self,
        dataset: torch_geometric.data.Dataset | torch.utils.data.Dataset,
    ) -> dict[str, Any]:
        """Merge configured values over dataset-declared content defaults."""
        parameters = resolve_cache_config(
            self.parameters,
            context="loader parameters",
        )
        defaults = getattr(dataset, "cache_parameters", None)
        if defaults is None:
            return parameters
        normalized_defaults = normalize_cache_value(
            defaults,
            path="loader.parameters.defaults",
        )
        if not isinstance(normalized_defaults, dict):
            raise TypeError("dataset cache_parameters must be a mapping")
        effective_parameters = dict(normalized_defaults)
        effective_parameters.update(parameters)
        return effective_parameters

    @abstractmethod
    def load_dataset(
        self,
    ) -> torch_geometric.data.Dataset | torch.utils.data.Dataset:
        """Load data into a dataset.

        Raises
        ------
        NotImplementedError
            If the method is not implemented.

        Returns
        -------
        Union[torch_geometric.data.Dataset, torch.utils.data.Dataset]
            The loaded dataset, which could be a PyG or PyTorch dataset.
        """
        raise NotImplementedError

    def load(self, **kwargs) -> tuple[torch_geometric.data.Data, str]:
        """Load data.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        tuple[torch_geometric.data.Data, str]
            Tuple containing the loaded data and the data directory.
        """
        dataset = self.load_dataset(**kwargs)
        setattr(
            dataset,
            CACHE_SOURCE_ATTRIBUTE,
            _loader_cache_source(self, dataset),
        )
        data_dir = self.get_data_dir()

        return dataset, data_dir
