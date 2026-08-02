"""Loaders for TDC (Therapeutics Data Commons) ADME datasets with SMILES to graph conversion."""

import json
import math
import os
import tempfile
from importlib import metadata
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset

try:
    from ogb.utils.mol import smiles2graph
    from tdc.single_pred import ADME

    _ADME_DEPS_AVAILABLE = True
except ImportError:
    _ADME_DEPS_AVAILABLE = False

from topobench.data.features import (
    OGB_ATOM_FEATURE_CARDINALITIES,
    validate_categorical_columns,
)
from topobench.data.loaders.base import (
    AbstractLoader,
    canonical_json_bytes,
    canonical_sha256,
    isolated_rng_state,
)


_PROVENANCE_VERSION = 2
_PHASES = ("train", "valid", "test")

try:
    _PYTDC_VERSION: str | None = metadata.version("PyTDC")
except metadata.PackageNotFoundError:
    _PYTDC_VERSION = None


def _canonical_scalar(value: Any) -> Any:
    """Convert one dataframe scalar to a stable JSON representation."""
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if math.isfinite(number):
            return number
        if math.isnan(number):
            return {"nonfinite_float": "nan"}
        return {
            "nonfinite_float": (
                "positive_infinity" if number > 0 else "negative_infinity"
            )
        }
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return {"isoformat": isoformat()}
    raise TypeError(
        "ADME provenance cannot canonically serialize "
        f"{type(value).__name__}"
    )


def _frame_records(frame: Any) -> list[dict[str, Any]]:
    """Return canonically serializable source rows in dataset order."""
    return [
        {
            str(column): _canonical_scalar(row[column])
            for column in frame.columns
        }
        for _, row in frame.iterrows()
    ]


def _source_data_digest(split: dict[str, Any]) -> str:
    """Hash source rows independently of their assigned split phase."""
    records = [
        record
        for phase in _PHASES
        for record in _frame_records(split[phase])
    ]
    records.sort(key=canonical_json_bytes)
    return canonical_sha256(records)


def _dataset_version(dataset: Any) -> str | None:
    """Return provider dataset-version metadata only when it is concrete."""
    for attribute in ("version", "dataset_version"):
        value = getattr(dataset, attribute, None)
        if isinstance(value, str):
            return value
        if isinstance(value, Integral) and not isinstance(value, bool):
            return str(int(value))
        if isinstance(value, Real) and not isinstance(value, bool):
            return str(float(value))
    return None


def _build_provenance(
    *,
    dataset: Any,
    dataset_name: str,
    split: dict[str, Any],
    method: str,
    seed: int,
) -> dict[str, Any]:
    """Build the complete versioned ADME source and split record."""
    return {
        "provenance_version": _PROVENANCE_VERSION,
        "source": {
            "provider": "PyTDC",
            "provider_version": _PYTDC_VERSION,
            "dataset_name": dataset_name,
            "dataset_version": _dataset_version(dataset),
            "data_digest": _source_data_digest(split),
        },
        "split": {
            "method": method,
            "seed": seed,
            "phase_index_digests": {
                phase: canonical_sha256(
                    [
                        _canonical_scalar(index)
                        for index in split[phase].index.tolist()
                    ]
                )
                for phase in _PHASES
            },
            "phase_data_digests": {
                phase: canonical_sha256(_frame_records(split[phase]))
                for phase in _PHASES
            },
        },
        "representation": {
            "node_feature_encoding": "categorical_one_hot",
            "node_feature_cardinalities": list(
                OGB_ATOM_FEATURE_CARDINALITIES
            ),
            "stored_node_feature_width": len(
                OGB_ATOM_FEATURE_CARDINALITIES
            ),
            "encoded_node_feature_width": sum(
                OGB_ATOM_FEATURE_CARDINALITIES
            ),
        },
    }


def _provenance_matches(
    path: Path,
    expected: dict[str, Any],
) -> bool:
    """Return whether a non-executable JSON record matches completely."""
    try:
        with path.open(encoding="utf-8") as handle:
            cached = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return cached == expected


def _write_provenance(path: Path, record: dict[str, Any]) -> None:
    """Atomically persist canonical JSON with executable bits disabled."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(record))
            handle.write(b"\n")
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


class ADMEDatasetLoader(AbstractLoader):
    """Load TDC ADME datasets with SMILES to graph conversion using OGB featurization.

    This loader:
    1. Loads ADME datasets from TDC (Therapeutics Data Commons)
    2. Converts SMILES strings to PyG graphs using OGB's standard featurization
    3. Uses fixed scaffold splits from TDC
    4. Returns graphs compatible with OGB molecular property prediction

    Node features (nine compact integral OGB categorical columns):
        - Atomic number
        - Chirality
        - Degree
        - Formal charge
        - Number of hydrogens
        - Number of radical electrons
        - Hybridization
        - Is aromatic
        - Is in ring

    Edge features (3-dimensional):
        - Bond type
        - Bond stereochemistry
        - Is conjugated

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the ADME dataset
            - data_type: Type of the dataset (e.g., "ADME")
            - split_method: Explicit PyTDC split method; must be "scaffold"
            - split_seed: Integer seed passed to PyTDC
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)
        split_method = parameters.get("split_method")
        if split_method != "scaffold":
            raise ValueError(
                "ADME split_method must be 'scaffold' so loader behavior "
                "agrees with selector metadata"
            )
        split_seed = parameters.get("split_seed")
        if isinstance(split_seed, bool) or not isinstance(
            split_seed, Integral
        ):
            raise ValueError("ADME split_seed must be an integer")
        self.split_method = split_method
        self.split_seed = int(split_seed)

    def load_dataset(self) -> InMemoryDataset:
        """Load an explicit, reproducible scaffold split without RNG leakage."""
        with isolated_rng_state():
            return self._load_dataset()

    def _load_dataset(self) -> InMemoryDataset:
        """Load, validate, and provenance-check the processed ADME cache."""
        if not _ADME_DEPS_AVAILABLE:
            raise ImportError(
                "ADME datasets require additional dependencies. "
                "Install them with: pip install PyTDC rdkit"
            )

        class _ADMEDataset(InMemoryDataset):
            """Internal cache for already converted ADME graphs."""

            def __init__(self, root, data_name, split_idx, graph_list):
                self.data_name = data_name
                self.split_idx = split_idx
                self._graph_list = graph_list
                super().__init__(root)
                self._data, self.slices = torch.load(
                    self.processed_paths[0],
                    weights_only=False,
                )

            @property
            def processed_file_names(self):
                """Return the representation-specific processed file name."""
                return [f"{self.data_name}_node_categories_v1.pt"]

            def process(self):
                """Collate pre-built graphs and persist the processed cache."""
                self._data, self.slices = self.collate(self._graph_list)
                torch.save((self._data, self.slices), self.processed_paths[0])

            def __repr__(self):
                return f"ADMEDataset({self.data_name}, {len(self)})"

        classification_datasets = {
            "PAMPA_NCATS",
            "HIA_Hou",
            "Pgp_Broccatelli",
            "Bioavailability_Ma",
            "BBB_Martins",
            "CYP1A2_Veith",
            "CYP2C9_Veith",
            "CYP2C19_Veith",
            "CYP2D6_Veith",
            "CYP3A4_Veith",
            "CYP2C9_Substrate_CarbonMangels",
            "CYP2D6_Substrate_CarbonMangels",
            "CYP3A4_Substrate_CarbonMangels",
        }
        regression_datasets = {
            "Caco2_Wang",
            "Lipophilicity_AstraZeneca",
            "Solubility_AqSolDB",
            "HydrationFreeEnergy_FreeSolv",
            "PPBR_AZ",
            "VDss_Lombardo",
            "Half_Life_Obach",
            "Clearance_Hepatocyte_AZ",
            "Clearance_Microsome_AZ",
        }

        dataset_name = str(self.parameters.data_name)
        if dataset_name in classification_datasets:
            is_classification = True
        elif dataset_name in regression_datasets:
            is_classification = False
        else:
            raise ValueError(
                f"Unknown ADME dataset: {dataset_name}. "
                "Please add it to the classification or regression set."
            )

        dataset_root = Path(self.root_data_dir) / dataset_name
        raw_dir = dataset_root / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        source_dataset = ADME(name=dataset_name, path=str(raw_dir))
        split = source_dataset.get_split(
            method=self.split_method,
            seed=self.split_seed,
        )
        train_data = split["train"]
        valid_data = split["valid"]
        test_data = split["test"]
        provenance = _build_provenance(
            dataset=source_dataset,
            dataset_name=dataset_name,
            split=split,
            method=self.split_method,
            seed=self.split_seed,
        )

        processed_path = (
            dataset_root
            / "processed"
            / f"{dataset_name}_node_categories_v1.pt"
        )
        provenance_path = processed_path.with_name(
            f"{processed_path.name}.provenance.json"
        )
        cache_hit = processed_path.is_file() and _provenance_matches(
            provenance_path,
            provenance,
        )
        if not cache_hit:
            processed_path.unlink(missing_ok=True)
            provenance_path.unlink(missing_ok=True)

        graph_list = []
        if not cache_hit:
            for split_data in (train_data, valid_data, test_data):
                for _, row in split_data.iterrows():
                    graph_dict = smiles2graph(row["Drug"])
                    node_features = torch.as_tensor(
                        graph_dict["node_feat"]
                    )
                    validate_categorical_columns(
                        node_features,
                        OGB_ATOM_FEATURE_CARDINALITIES,
                        context=f"{dataset_name} node features",
                    )
                    if is_classification:
                        label_tensor = torch.tensor(
                            int(row["Y"]),
                            dtype=torch.long,
                        )
                    else:
                        label_tensor = torch.tensor(
                            [row["Y"]],
                            dtype=torch.float,
                        )
                    graph_list.append(
                        Data(
                            x=node_features,
                            edge_index=torch.tensor(
                                graph_dict["edge_index"],
                                dtype=torch.long,
                            ),
                            edge_attr=torch.tensor(
                                graph_dict["edge_feat"],
                                dtype=torch.float,
                            ),
                            y=label_tensor,
                            num_nodes=graph_dict["num_nodes"],
                        )
                    )

        split_idx = {
            "train": torch.arange(len(train_data)),
            "valid": torch.arange(
                len(train_data),
                len(train_data) + len(valid_data),
            ),
            "test": torch.arange(
                len(train_data) + len(valid_data),
                len(train_data) + len(valid_data) + len(test_data),
            ),
        }
        dataset = _ADMEDataset(
            root=str(dataset_root),
            data_name=dataset_name,
            split_idx=split_idx,
            graph_list=graph_list,
        )
        if not cache_hit:
            processed_path.chmod(processed_path.stat().st_mode & ~0o111)
            _write_provenance(provenance_path, provenance)

        dataset.split_idx = split_idx
        dataset.split_provenance = provenance
        dataset.split_fingerprint = canonical_sha256(provenance)
        dataset.feature_encoding = "categorical_one_hot"
        dataset.feature_cardinalities = OGB_ATOM_FEATURE_CARDINALITIES
        dataset.provenance_path = str(provenance_path)
        return dataset

    def get_data_dir(self) -> Path:
        """Get the data directory.

        Returns
        -------
        Path
            The path to the dataset directory.
            Format: {root_data_dir}/{dataset_name}/.
            Example: data/graph/ADME/BBB_Martins/.
        """
        return os.path.join(self.root_data_dir, self.parameters.data_name)
