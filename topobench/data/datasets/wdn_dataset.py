"""Dataset class for WDN datasets."""

import json
import os
import os.path as osp
from typing import ClassVar

import pandas as pd
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.io import fs

from topobench.data.utils import download_file_from_link

# Main class for the dataset #


class WDNDataset(InMemoryDataset):
    """Super-class to load datasets from "Large-Scale Multipurpose Benchmark Datasets For Assessing Data-Driven Deep Learning Approaches For Water Distribution Networks" (2023) with some configurables.

    Parameters
    ----------
    root : str
        Root directory where the dataset will be saved.
    parameters : DictConfig
        Configuration parameters for the dataset.

    Attributes
    ----------
    URLS (dict): Name of the specific dataset to be istantiated.
    FILE_FORMAT (dict): File format of the dataset.
    RAW_FILE_NAMES (dict): List of file names of the dataset.
    """

    URL: ClassVar[str] = None
    FILE_FORMAT: ClassVar[str] = "zip"

    def __init__(self, root: str, parameters: DictConfig) -> None:
        self.root = root
        self.parameters = parameters
        super().__init__(root)

        out = fs.torch_load(self.processed_paths[0])
        assert len(out) in (3, 4)

        if len(out) == 3:
            data, self.slices, self.sizes = out
            data_cls = Data
        else:
            data, self.slices, self.sizes, data_cls = out

        if not isinstance(data, dict):
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

        assert isinstance(self._data, Data)

    @property
    def raw_dir(self) -> str:
        """Return the path to the raw directory of the dataset.

        Returns
        -------
        str
            Path to the raw directory.
        """
        return osp.join(self.root, self.parameters.data_name, "raw")

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory of the dataset.

        Returns
        -------
        str
            Path to the processed directory.
        """
        return osp.join(self.root, self.parameters.data_name, "processed")

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw file names for the dataset.

        Returns
        -------
        list[str]
            List of raw file names.
        """
        return [
            "pressure.csv",
            "demand.csv",
            "flowrate.csv",
            "velocity.csv",
            "head.csv",
            "head_loss.csv",
            "friction_factor.csv",
            "attrs.json",
        ]

    @property
    def processed_file_names(self) -> str:
        """Return the processed file name for the dataset.

        Returns
        -------
        str
            Processed file name.
        """
        return "data.pt"

    def download(self) -> None:
        r"""Download the dataset from a URL and saves it to the raw directory.

        Raises:
            FileNotFoundError: If the dataset URL is not found.
        """
        if self.URL is None or self.FILE_FORMAT is None:
            raise FileNotFoundError(
                f"URL or FILE_FORMAT not set for {self.parameters.data_name}"
            )

        download_file_from_link(
            file_link=self.URL,
            path_to_save=self.raw_dir,
            dataset_name=self.parameters.data_name,
            file_format=self.FILE_FORMAT,
        )

        # Extract zip
        path = osp.join(
            self.raw_dir, f"{self.parameters.data_name}.{self.FILE_FORMAT}"
        )
        extract_zip(path, self.raw_dir)

        # Delete zip file
        os.unlink(path)

        # Remove unretained files
        retain_files = getattr(
            self.parameters, "retain_files", self.raw_file_names
        )

        for f in self.raw_file_names:
            if f not in retain_files and osp.exists(osp.join(self.raw_dir, f)):
                os.remove(osp.join(self.raw_dir, f))

    def process(self) -> None:
        r"""Handle the data for the dataset.

        - Builds the graph from metadata
        - Remaps node identifiers to progressive idxs
        - Retrieves the correct temporal dimension
        - Retrieves the regressors and target variables
        - For each scenario, builds:
            - A tensor (num_nodes, num_features, time_stamps)
            for node features;
            - A tensor (num_edges, num_features, time_stamps)
            for edge features;
            - A tensor (*, num_features, times_tamps)
            for target variables accordingly to the target domain.
        - Collated in a PyG Data object each of this graph adding
        an identifier to the related scenario
        - Save processed data.
        """
        attributes_path = osp.join(self.raw_dir, "attrs.json")

        with open(attributes_path) as f:
            attributes_data = json.load(f)

        # --- Build edge_index with edge IDs ---
        adj_list = attributes_data["adj_list"]

        # Extract all unique nodes
        all_nodes = {src for src, _, _ in adj_list} | {
            dst for _, dst, _ in adj_list
        }
        node_id_map = {old: i for i, old in enumerate(sorted(all_nodes))}

        # Remap edges to integers and collect edge IDs
        edge_index_list = []
        edge_ids = []
        for src, dst, eid in adj_list:
            edge_index_list.append((node_id_map[src], node_id_map[dst]))
            edge_ids.append(eid)

        edge_index = (
            torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        )

        # --- Scenarios and time-instants selection ---
        total_scenarios = attributes_data["gen_batch_size"]
        total_duration = attributes_data["duration"]
        num_scenarios = getattr(
            self.parameters, "num_scenarios", total_scenarios
        )
        num_instants = getattr(self.parameters, "num_instants", total_duration)

        # --- Variables to retain ---
        regressors = getattr(self.parameters, "regressors", [])
        targets = getattr(self.parameters, "target", [])

        assert len(targets) == 1, (
            f"Expected exactly one target variable, got {len(targets)}."
        )

        retain_files = list(set(regressors + targets))

        # --- Load all requested CSVs ---
        data_tensors = {}
        csv_columns = {}  # store column names for each CSV
        for file_name in retain_files:
            csv_path = osp.join(self.raw_dir, f"{file_name}.csv")
            if not osp.exists(csv_path):
                continue
            df = pd.read_csv(csv_path, index_col=0)
            csv_columns[file_name] = df.columns.tolist()
            tensor = torch.tensor(df.values, dtype=torch.float32)
            # reshape to (scenarios, duration, features)
            tensor = tensor.reshape(
                total_scenarios, total_duration, df.shape[1]
            )
            # select temporal subset
            tensor = tensor[:num_scenarios, :num_instants, :]
            data_tensors[file_name] = tensor

        # --- Helper function: determine if variable is node-level or edge-level ---
        def is_edge_var(var_name: str) -> bool:
            """Determine whether a variable name corresponds to an edge-level variable.

            Parameters
            ----------
            var_name : str
                The name of the variable to check.

            Returns
            -------
            bool
                ``True`` if the variable is an edge-level variable, ``False`` otherwise.
            """

            return var_name in [
                "flowrate",
                "velocity",
                "head_loss",
                "friction_factor",
            ]

        # --- Reorder node features according to node_id_map ---
        unique_nodes = torch.unique(edge_index)
        node_order = [n.item() for n in unique_nodes]

        graph_samples = []
        for i in range(num_scenarios):
            node_regressors, edge_regressors = [], []
            target_signals = []

            # Node features
            for var_name in regressors + targets:
                if var_name not in data_tensors:
                    continue
                tensor = data_tensors[var_name][
                    i
                ]  # shape [T, num_edges or num_nodes]
                if is_edge_var(var_name):
                    # Reorder columns to match edge_index order via edge_ids
                    tensor = tensor[
                        :,
                        [
                            csv_columns[var_name].index(str(eid))
                            for eid in edge_ids
                        ],
                    ]
                    if var_name in regressors:
                        edge_regressors.append(tensor.unsqueeze(0))
                    else:
                        target_signals.append(tensor.unsqueeze(0))
                else:
                    # Node-level features: reorder according to node_order
                    tensor = tensor[
                        :,
                        [
                            csv_columns[var_name].index(str(n))
                            for n in node_order
                        ],
                    ]
                    if var_name in regressors:
                        node_regressors.append(tensor.unsqueeze(0))
                    else:
                        target_signals.append(tensor.unsqueeze(0))

            # Assemble features
            x = torch.cat(node_regressors, dim=0) if node_regressors else None
            edge_attr = (
                torch.cat(edge_regressors, dim=0) if edge_regressors else None
            )
            y = torch.cat(target_signals, dim=0)

            # Permute to [N, F, T]
            if x is not None and x.dim() == 3:
                x = x.permute(2, 0, 1)
            if edge_attr is not None and edge_attr.dim() == 3:
                edge_attr = edge_attr.permute(2, 0, 1)
            if y is not None and y.dim() == 3:
                y = y.permute(2, 0, 1)

            # Drop last dim if temporal=False
            if not self.parameters.temporal:
                x = x.squeeze(-1) if x is not None else None
                edge_attr = (
                    edge_attr.squeeze(-1) if edge_attr is not None else None
                )
                y = y.squeeze(-1) if y is not None else None

            # Squeeze feature dim for targets (currently only one target allowed)
            y = y.squeeze(1) if y is not None else None

            # Create Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
            )

            data.scenario_id = i
            graph_samples.append(data)

        # --- Collate and save ---
        self.data, self.slices = self.collate(graph_samples)
        self._data_list = None
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )


# Subclasses for each dataset #


class AnytownDataset(WDNDataset):
    """Dataset generated with the Anytown WDN model."""

    URL: ClassVar = "https://zenodo.org/records/11353195/files/simgen_Anytown_20240524_1202_csvdir_20240527_1205.zip?download=1"


class BalermanDataset(WDNDataset):
    """Dataset generated with the Balerma WDN model."""

    URL: ClassVar = "https://zenodo.org/records/11353195/files/simgen_balerman_20240524_1233_csvdir_20240527_1205.zip?download=1"


class CTownDataset(WDNDataset):
    """Dataset generated with the C-Town WDN model."""

    URL: ClassVar = "https://zenodo.org/records/11353195/files/simgen_ctown_20240524_1231_csvdir_20240527_1208.zip?download=1"


class DTownDataset(WDNDataset):
    """Dataset generated with the D-Town WDN model."""

    URL: ClassVar = "https://zenodo.org/records/11353195/files/simgen_d-town_20240525_1755_csvdir_20240527_1210.zip?download=1"


class EXNDataset(WDNDataset):
    """Dataset generated with the EXN WDN model."""

    URL: ClassVar = "https://zenodo.org/records/11353195/files/simgen_EXN_20240525_0928_csvdir_20240527_1237.zip?download=1"


class KY1Dataset(WDNDataset):
    """Dataset generated with the K1 WDN model."""

    URL: ClassVar = "https://zenodo.org/records/11353195/files/simgen_ky1_20240524_1229_csvdir_20240527_1218.zip?download=1"


class KY6Dataset(WDNDataset):
    """Dataset generated with the K6 WDN model."""

    URL: ClassVar = "https://zenodo.org/records/11353195/files/simgen_ky6_20240524_1228_csvdir_20240527_1223.zip?download=1"


class KY8Dataset(WDNDataset):
    """Dataset generated with the K8 WDN model."""

    URL: ClassVar = "https://zenodo.org/records/11353195/files/simgen_ky8_20240524_1228_csvdir_20240527_1225.zip?download=1"


class KY10Dataset(WDNDataset):
    """Dataset generated with the K10 WDN model."""

    URL: ClassVar = "https://zenodo.org/records/11353195/files/simgen_ky10_20240524_1229_csvdir_20240527_1218.zip?download=1"


class LTownDataset(WDNDataset):
    """Dataset generated with the L-Town WDN model."""

    URL: ClassVar = "https://zenodo.org/records/11353195/files/simgen_L-TOWN_Real_20240524_1228_csvdir_20240527_1232.zip?download=1"


class ModenaDataset(WDNDataset):
    """Dataset generated with the Modena WDN model."""

    URL: ClassVar = "https://zenodo.org/records/11353195/files/simgen_moderna_20240524_1230_csvdir_20240527_1212.zip?download=1"
