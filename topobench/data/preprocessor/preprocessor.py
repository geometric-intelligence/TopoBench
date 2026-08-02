"""Preprocessor for datasets."""

import inspect
import json
import os
import tempfile
import time
from collections.abc import Mapping

import torch
import torch_geometric
from filelock import FileLock
from omegaconf import OmegaConf
from torch_geometric.data import Data, HeteroData
from torch_geometric.io import fs
from tqdm import tqdm

from topobench.data.loaders.base import (
    CACHE_SOURCE_ATTRIBUTE,
    canonical_sha256,
    normalize_cache_value,
)
from topobench.data.utils.split_utils import (
    load_inductive_splits,
    load_transductive_splits,
)
from topobench.transforms.data_transform import DataTransform

SupportedData = Data | HeteroData
PreProcessorInput = (
    torch_geometric.data.Dataset | torch.utils.data.Dataset | SupportedData
)

_PROCESSED_CACHE_SCHEMA = "topobench.processed-cache"
_PROCESSED_CACHE_SCHEMA_VERSION = 2
_CACHE_RECORD_FILE_NAME = "cache_record.json"


def _data_family(
    data: SupportedData,
) -> type[Data] | type[HeteroData]:
    """Return the concrete homogeneous or heterogeneous data family."""
    return HeteroData if isinstance(data, HeteroData) else Data


def _dataset_cache_source(dataset: PreProcessorInput) -> dict:
    """Read loader provenance or build a stable direct-dataset fallback."""
    source = getattr(dataset, CACHE_SOURCE_ATTRIBUTE, None)
    if source is not None:
        return source
    dataset_target = f"{type(dataset).__module__}.{type(dataset).__qualname__}"
    cache_parameters = getattr(dataset, "cache_parameters", {})
    if not isinstance(cache_parameters, Mapping):
        raise TypeError(
            "cache.source.loader.parameters: dataset.cache_parameters "
            "must be a mapping"
        )
    return normalize_cache_value(
        {
            "dataset_selector": {
                "data_domain": None,
                "data_type": "direct",
                "data_name": type(dataset).__name__,
            },
            "loader": {
                "target": None,
                "parameters": cache_parameters,
            },
            "feature_policy": getattr(dataset, "feature_policy", None),
            "versions": {
                "representation": getattr(
                    dataset,
                    "representation_version",
                    None,
                ),
                "parser": getattr(dataset, "parser_version", None),
                "dataset": dataset_target,
            },
        },
        path="cache.source",
    )


def _effective_transform_step(
    name,
    transform_instance: DataTransform,
    supplied_parameters: dict,
    *,
    device: str,
) -> dict:
    """Snapshot effective constructor inputs before transform execution."""
    runtime_transform = getattr(transform_instance, "transform", None)
    if runtime_transform is None:
        target = None
        effective_parameters = dict(supplied_parameters)
    else:
        transform_type = type(runtime_transform)
        target = f"{transform_type.__module__}.{transform_type.__qualname__}"
        effective_parameters = {}
        try:
            signature = inspect.signature(transform_type.__init__)
        except (TypeError, ValueError):
            signature = None
        if signature is not None:
            for parameter in signature.parameters.values():
                if parameter.name == "self" or parameter.kind in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                }:
                    continue
                if parameter.name in supplied_parameters:
                    effective_parameters[parameter.name] = (
                        supplied_parameters[parameter.name]
                    )
                elif parameter.default is not inspect.Parameter.empty:
                    effective_parameters[parameter.name] = parameter.default
        effective_parameters.update(
            {
                key: value
                for key, value in supplied_parameters.items()
                if key not in effective_parameters
            }
        )
        captured_parameters = getattr(
            runtime_transform,
            "parameters",
            None,
        )
        if isinstance(captured_parameters, Mapping):
            for key in tuple(effective_parameters):
                if key in captured_parameters:
                    effective_parameters[key] = captured_parameters[key]
    effective_parameters["preprocessor_device"] = device
    return normalize_cache_value(
        {
            "name": name,
            "target": target,
            "parameters": effective_parameters,
        },
        path=f"transform.{name}",
    )


class PreProcessor(torch_geometric.data.InMemoryDataset):
    """Preprocessor for datasets.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset | torch.utils.data.Dataset | Data | HeteroData
        Dataset of supported PyG objects, or one direct PyG data object. Plain
        Python lists and other arbitrary containers are not supported. Direct
        ``Data`` and ``HeteroData`` inputs use the persisted processing path
        and therefore require ``transforms_config``; the no-transform fast
        path expects a dataset with in-memory ``_data`` and ``slices``.
    data_dir : str
        Path to the directory containing the data.
    transforms_config : DictConfig, optional
        Configuration parameters for the transforms (default: None).
    **kwargs : optional
        Optional additional arguments.
    """

    def __init__(
        self,
        dataset: PreProcessorInput,
        data_dir,
        transforms_config=None,
        **kwargs,
    ):
        self.dataset = dataset
        self.preprocessing_time = 0
        if transforms_config is not None:
            self.transforms_applied = True
            pre_transform = self.instantiate_pre_transform(
                data_dir, transforms_config
            )

            # 1. Ensure the target directory exists so we can place a lock file in it
            os.makedirs(self.processed_data_dir, exist_ok=True)
            lock_path = os.path.join(
                self.processed_data_dir, "preprocessing.lock"
            )

            start_time = time.time()

            with FileLock(lock_path):
                # When Process 1 finishes, Process 2 checks, sees data.pt, and skips.
                super().__init__(
                    self.processed_data_dir, None, pre_transform, **kwargs
                )
                self.save_cache_record()

            end_time = time.time()
            self.preprocessing_time = end_time - start_time

            self.transform = (
                dataset.transform if hasattr(dataset, "transform") else None
            )
            self.load(self.processed_paths[0])
            self.data_list = [data for data in self]
        else:
            self.transforms_applied = False
            super().__init__(data_dir, None, None, **kwargs)
            self.transform = (
                dataset.transform if hasattr(dataset, "transform") else None
            )
            self.data, self.slices = dataset._data, dataset.slices
            self.data_list = [data for data in dataset]

        # Some datasets have fixed splits, and those are stored as split_idx during loading
        # We need to store this information to be able to reproduce the splits afterwards
        if hasattr(dataset, "split_idx"):
            self.split_idx = dataset.split_idx
        if hasattr(dataset, "split_idx_list"):
            self.split_idx_list = dataset.split_idx_list

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory.

        Returns
        -------
        str
            Path to the processed directory.
        """
        return self.root

    @property
    def processed_file_names(self) -> str:
        """Return the name of the processed file.

        Returns
        -------
        str
            Name of the processed file.
        """
        return "data.pt"

    def instantiate_pre_transform(
        self, data_dir, transforms_config
    ) -> torch_geometric.transforms.Compose:
        """Instantiate the pre-transforms.

        Parameters
        ----------
        data_dir : str
            Path to the directory containing the data.
        transforms_config : DictConfig
            Configuration parameters for the transforms.

        Returns
        -------
        torch_geometric.transforms.Compose
            Pre-transform object.
        """
        from torch_geometric.transforms import ToDevice

        if "transform_name" in transforms_config:
            config_items = [
                (transforms_config.transform_name, transforms_config)
            ]
        else:
            config_items = transforms_config.items()

        pre_transforms_list = []
        transform_steps = []

        # Track where the graph currently lives in the pipeline
        current_device = "cpu"

        for name, value in config_items:
            resolved_value = (
                OmegaConf.to_container(
                    value,
                    resolve=True,
                    throw_on_missing=True,
                )
                if OmegaConf.is_config(value)
                else value
            )
            if not isinstance(resolved_value, Mapping):
                raise TypeError(
                    f"transform {name!r} must resolve to a mapping"
                )
            kwargs = dict(resolved_value)

            requested_device = kwargs.pop("preprocessor_device", "cpu")

            target_device = (
                "cuda"
                if requested_device == "cuda" and torch.cuda.is_available()
                else "cpu"
            )

            transform_instance = DataTransform(**kwargs)
            transform_steps.append(
                _effective_transform_step(
                    name,
                    transform_instance,
                    kwargs,
                    device=target_device,
                )
            )

            if target_device != current_device:
                pre_transforms_list.append(ToDevice(target_device))
                current_device = target_device

            pre_transforms_list.append(transform_instance)

        # If the pipeline ends while the graph is still on the GPU,
        # we MUST pull it back to the CPU before PyTorch Geometric saves it to disk.
        if current_device == "cuda":
            pre_transforms_list.append(ToDevice("cpu"))

        pre_transforms = torch_geometric.transforms.Compose(
            pre_transforms_list
        )

        self.set_processed_data_dir(data_dir, transform_steps)
        return pre_transforms

    def set_processed_data_dir(self, data_dir, transform_steps) -> None:
        """Select the processed directory from complete canonical provenance."""
        source = _dataset_cache_source(self.dataset)
        self.cache_record = {
            "schema": _PROCESSED_CACHE_SCHEMA,
            "schema_version": _PROCESSED_CACHE_SCHEMA_VERSION,
            **source,
            "transform": {"steps": transform_steps},
        }
        self.cache_identity = canonical_sha256(self.cache_record)
        self.processed_data_dir = os.path.join(data_dir, self.cache_identity)
        self.cache_record_path = os.path.join(
            self.processed_data_dir,
            _CACHE_RECORD_FILE_NAME,
        )

    def _validate_existing_cache_record(self) -> None:
        """Reject an unreadable or mismatched record at the selected digest."""
        try:
            with open(self.cache_record_path, encoding="utf-8") as file:
                saved_record = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                "cache identity collision: existing cache record is unreadable"
            ) from error
        if saved_record != self.cache_record:
            raise ValueError(
                "cache identity collision: existing record does not match "
                "the canonical cache record"
            )

    def save_cache_record(self) -> None:
        """Atomically persist readable provenance without overwriting a peer."""
        if os.path.exists(self.cache_record_path):
            self._validate_existing_cache_record()
            return

        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{_CACHE_RECORD_FILE_NAME}.",
            dir=self.processed_data_dir,
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(
                    self.cache_record,
                    file,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            try:
                os.link(temporary_path, self.cache_record_path)
            except FileExistsError:
                self._validate_existing_cache_record()
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def process(self) -> None:
        """Process and persist a homogeneous family of supported PyG data.

        Pre-transforms retain PyG's existing semantics: they receive each
        original object and may mutate it in place. This method does not copy
        transform inputs or attempt rollback after a transform failure.
        """
        if isinstance(
            self.dataset,
            (torch_geometric.data.Dataset, torch.utils.data.Dataset),
        ):
            data_list = list(self.dataset)
        elif isinstance(self.dataset, (Data, HeteroData)):
            data_list = [self.dataset]
        else:
            raise TypeError(
                "PreProcessor expects a PyG/PyTorch dataset, Data, or "
                f"HeteroData; received {type(self.dataset).__name__}"
            )

        if not data_list:
            raise ValueError(
                "PreProcessor requires at least one Data or HeteroData item"
            )

        if self.pre_transform is not None:
            print(f"\nApplying transforms to {len(data_list)} graphs...")
        processed: list[SupportedData] = []
        expected_family: type[Data] | type[HeteroData] | None = None
        for item_index, original in enumerate(
            tqdm(
                data_list,
                desc="Processing graphs",
                unit="graph",
            )
        ):
            if not isinstance(original, (Data, HeteroData)):
                raise TypeError(
                    f"Dataset item {item_index} must be Data or HeteroData; "
                    f"received {type(original).__name__}"
                )
            original_family = _data_family(original)
            if expected_family is None:
                expected_family = original_family
            elif original_family is not expected_family:
                raise TypeError(
                    f"Dataset item {item_index} has mixed representation: "
                    f"expected {expected_family.__name__}, "
                    f"received {original_family.__name__}"
                )
            transformed = (
                self.pre_transform(original)
                if self.pre_transform is not None
                else original
            )
            if not isinstance(transformed, (Data, HeteroData)):
                raise TypeError(
                    f"Pre-transform result for dataset item {item_index} "
                    "must be Data or HeteroData; "
                    f"received {type(transformed).__name__}"
                )
            transformed_family = _data_family(transformed)
            if original_family is not transformed_family:
                raise TypeError(
                    "Pre-transform changed representation for dataset item "
                    f"{item_index}: expected {original_family.__name__}, "
                    f"received {transformed_family.__name__}"
                )
            processed.append(transformed)

        self.data_list = processed
        self._data, self.slices = self.collate(processed)
        self._data_list = None  # Reset cache.
        self.save(processed, self.processed_paths[0])

    def load(self, path: str) -> None:
        r"""Load the dataset from the file path `path`.

        Parameters
        ----------
        path : str
            The path to the processed data.
        """
        out = fs.torch_load(path)
        if not isinstance(out, tuple):
            raise TypeError(
                "Processed data artifact must be a tuple; "
                f"received {type(out).__name__}"
            )
        if len(out) not in (2, 3, 4):
            raise ValueError(
                "Processed data artifact must contain 2, 3, or 4 elements; "
                f"received {len(out)}"
            )

        data_cls: type[Data] | type[HeteroData] = Data
        if len(out) == 2:  # Backward compatibility (1).
            data, self.slices = out
        elif len(out) == 3:  # Backward compatibility (2).
            data, self.slices, data_cls = out
        else:  # TU Datasets store additional element (__class__).
            data, self.slices, _, data_cls = out

        if self.slices is not None and not isinstance(self.slices, dict):
            raise TypeError(
                "Processed data slices must be a dictionary or None; "
                f"received {type(self.slices).__name__}"
            )
        if not isinstance(data, (dict, Data, HeteroData)):
            raise TypeError(
                "Processed data payload must be Data, HeteroData, or a "
                f"dictionary; received {type(data).__name__}"
            )
        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
            return
        if not isinstance(data_cls, type) or not issubclass(
            data_cls,
            (Data, HeteroData),
        ):
            received = (
                data_cls.__name__
                if isinstance(data_cls, type)
                else type(data_cls).__name__
            )
            raise TypeError(
                "Processed data class must be Data or HeteroData for a "
                f"dictionary payload; received {received}"
            )
        self.data = data_cls.from_dict(data)

    def load_dataset_splits(
        self, split_params
    ) -> tuple[
        torch.utils.data.Dataset[Data],
        torch.utils.data.Dataset[Data] | None,
        torch.utils.data.Dataset[Data] | None,
    ]:
        """Load the dataset splits.

        Parameters
        ----------
        split_params : dict
            Parameters for loading the dataset splits.

        Returns
        -------
        tuple
            A tuple containing the train, validation, and test datasets.
        """
        if not split_params.get("learning_setting", False):
            raise ValueError("No learning setting specified in split_params")

        if split_params.learning_setting == "inductive":
            return load_inductive_splits(self, split_params)
        elif split_params.learning_setting == "transductive":
            return load_transductive_splits(self, split_params)
        else:
            raise ValueError(
                f"Invalid '{split_params.learning_setting}' learning setting.\
                Please define either 'inductive' or 'transductive'."
            )
