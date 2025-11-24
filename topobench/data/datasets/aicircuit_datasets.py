"""AICircuit hypergraph dataset definition."""

import os
import shutil
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from topobench.data.utils import download_file_from_link, extract_zip


class AICircuitDataset(InMemoryDataset):
    """Hypergraph dataset for AICircuit analog circuits.

    Parameters
    ----------
    root : str
        Root directory for storing data.
    name : str
        Dataset name.
    parameters : dict
        Loader parameters.
    transform : callable, optional
        Optional transform.
    pre_transform : callable, optional
        Optional pre-transform.
    """

    URLS = {
        "AICircuitAnalog": "1-0-gS9aK2-a-d-Y-g-f-X-e-Y-k-Z-w-Y-k-d-Y-k-d-Y-k-d"
    }
    FILE_FORMAT = {"AICircuitAnalog": "zip"}

    def __init__(
        self, root, name, parameters, transform=None, pre_transform=None
    ):
        """Initialize dataset."""
        self.name = name
        self.parameters = parameters
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False
        )

    @property
    def raw_dir(self):
        """Return raw directory path.

        Returns
        -------
        str
            Raw directory path.
        """
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        """Return processed directory path.

        Returns
        -------
        str
            Processed directory path.
        """
        return os.path.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        """List required raw files.

        Returns
        -------
        list
            Raw file names.
        """
        return ["README.md"]

    @property
    def processed_file_names(self):
        """List processed files.

        Returns
        -------
        list
            Processed file names.
        """
        return ["data.pt"]

    def download(self):
        """Download the dataset from a URL and saves it to the raw directory."""
        # Skip download if raw data already exists
        if os.path.exists(
            os.path.join(self.raw_dir, "Dataset")
        ) and os.path.exists(os.path.join(self.raw_dir, "Simulation")):
            print("Raw data already exists. Skipping download.")
            return

        self.url = "https://github.com/AvestimehrResearchGroup/AICircuit/archive/refs/heads/main.zip"
        self.file_format = "zip"
        download_file_from_link(
            file_link=self.url,
            path_to_save=self.raw_dir,
            dataset_name=self.name,
            file_format=self.file_format,
        )

        folder = self.raw_dir
        filename = f"{self.name}.{self.file_format}"
        path = os.path.join(folder, filename)
        extract_zip(path, folder)
        os.unlink(path)

        # Find the name of the extracted folder
        extracted_folder_name = ""
        for item in os.listdir(self.raw_dir):
            if os.path.isdir(os.path.join(self.raw_dir, item)):
                extracted_folder_name = item
                break

        source_folder = os.path.join(folder, extracted_folder_name)
        for file in os.listdir(source_folder):
            shutil.move(os.path.join(source_folder, file), folder)
        shutil.rmtree(source_folder)

    def _create_component_vocab(self):
        """Build vocabulary of component types.

        Returns
        -------
        dict
            Mapping from component type to ID.
        """
        # Based on scanning the netlists
        vocab = {
            "resistor": 0,
            "capacitor": 1,
            "inductor": 2,
            "nmos": 3,
            "pmos": 4,
            "vsource": 5,
            "isource": 6,
            "balun": 7,
            "vcvs": 8,
            "vccs": 9,
            "cccs": 10,
            "ccvs": 11,
            "diode": 12,
            "bjt": 13,
            "subcircuit": 14,
            "unknown": 15,
        }
        return vocab

    def _get_component_type(self, line, name):
        """Infer component type from netlist line and name.

        Parameters
        ----------
        line : str
            Raw netlist line.
        name : str
            Component name.

        Returns
        -------
        str
            Inferred component type string.
        """
        line = line.lower()
        name = name.lower()

        if "nmos" in line:
            return "nmos"
        if "pmos" in line:
            return "pmos"
        if "resistor" in line:
            return "resistor"
        if "capacitor" in line:
            return "capacitor"
        if "inductor" in line:
            return "inductor"
        if "vsource" in line:
            return "vsource"
        if "isource" in line:
            return "isource"
        if "balun" in line:
            return "balun"

        prefix = name[0]
        if prefix == "r":
            return "resistor"
        if prefix == "c":
            return "capacitor"
        if prefix == "l":
            return "inductor"
        if prefix == "n":
            return "nmos"
        if prefix == "p":
            return "pmos"
        if prefix == "m":
            return "nmos"  # Assume nmos for 'm' if not specified
        if prefix == "v":
            return "vsource"
        if prefix == "i":
            return "isource"
        if prefix == "e":
            return "vcvs"
        if prefix == "g":
            return "vccs"
        if prefix == "f":
            return "cccs"
        if prefix == "h":
            return "ccvs"
        if prefix == "d":
            return "diode"
        if prefix == "q":
            return "bjt"
        if prefix == "x":
            return "subcircuit"

        return "unknown"

    def _get_node_feature(self, node_name: str) -> int:
        """Map a node name to a categorical feature code.

        Parameters
        ----------
        node_name : str
            Name of the node in the netlist.

        Returns
        -------
        int
            Encoded feature ID (0 generic, 1 power, 2 ground, 3 input, 4 output,
            5 bias, 6 gate, 7 drain, 8 source, 9 bulk, 10 clock).
        """
        name = node_name.lower()
        if any(tok in name for tok in ["vdd", "vcc", "power", "pwr"]):
            return 1
        if any(tok in name for tok in ["vss", "gnd", "ground"]):
            return 2
        if "clk" in name:
            return 10
        if "bias" in name or name.startswith("vb"):
            return 5
        if name.startswith(("in", "vin")) or "input" in name:
            return 3
        if name.startswith(("out", "vout")) or "output" in name:
            return 4
        if "gate" in name:
            return 6
        if "drain" in name:
            return 7
        if "source" in name:
            return 8
        if any(tok in name for tok in ["bulk", "body", "substrate"]):
            return 9
        return 0

    def _get_incidence_roles(
        self, component_type: str, nodes: list[str]
    ) -> list[int]:
        """Assign pin-role codes per node for a component.

        Parameters
        ----------
        component_type : str
            Component type identifier.
        nodes : list of str
            Ordered list of node names for this component.

        Returns
        -------
        list of int
            Role codes aligned to nodes (e.g., 1 drain, 2 gate, 3 source).
        """
        roles = []
        comp = component_type.lower()
        if comp in {"nmos", "pmos", "nmos4", "pmos4", "mos", "mos4"}:
            template = [1, 2, 3, 4]  # drain, gate, source, bulk
            for idx, _ in enumerate(nodes):
                roles.append(template[idx] if idx < len(template) else 0)
        elif comp in {"bjt", "npn", "pnp"}:
            template = [11, 12, 13]  # collector, base, emitter
            for idx, _ in enumerate(nodes):
                roles.append(template[idx] if idx < len(template) else 0)
        else:
            # For 2-terminal or other devices, leave as generic
            roles = [0] * len(nodes)
        return roles

    def _parse_params(self, params: list[str]) -> list[float]:
        """Parse parameter tokens into a list of floats.

        Parameters
        ----------
        params : list of str
            Raw parameter tokens from the netlist line.

        Returns
        -------
        list of float
            Numeric parameters (non-numeric tokens skipped).
        """
        values = []
        for tok in params:
            if "=" in tok:
                tok = tok.split("=")[-1]
            tok = tok.strip()
            try:
                values.append(float(tok))
            except ValueError:
                # Non-numeric token (e.g., model name) -> skip
                continue
        return values

    def len(self):
        """Return dataset length.

        Returns
        -------
        int
            Number of graphs.
        """
        if self.slices is None:
            return 1
        return self.slices["x"].size(0) - 1

    def get(self, idx):
        """Get an item by index.

        Parameters
        ----------
        idx : int
            Index of the graph.

        Returns
        -------
        torch_geometric.data.Data
            Retrieved graph object.
        """
        if self.slices is None:
            if idx != 0:
                raise IndexError(
                    "Index out of range for single-graph dataset."
                )
            return self.data
        return super().get(idx)

    def _fix_length(self, tensor, target_len):
        """Pad or truncate last dimension to target_len.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to adjust.
        target_len : int
            Desired length.

        Returns
        -------
        torch.Tensor
            Adjusted tensor.
        """
        cur = tensor.shape[1]
        if cur == target_len:
            return tensor
        if cur > target_len:
            return tensor[:, :target_len]
        pad = torch.zeros(
            (tensor.shape[0], target_len - cur), dtype=tensor.dtype
        )
        return torch.cat([tensor, pad], dim=1)

    def process(self):  # noqa: D401
        """Process raw AICircuit data into hypergraph tensors."""
        Path(self.processed_dir).mkdir(
            parents=True, exist_ok=True
        )  # Ensure processed directory exists
        data_list = []
        component_vocab = self._create_component_vocab()
        parsed_graphs = []

        dataset_root = os.path.join(self.raw_dir, "Dataset")
        netlist_root = os.path.join(self.raw_dir, "Simulation", "Netlists")
        circuit_types = []
        if os.path.exists(dataset_root):
            circuit_types = [
                d
                for d in os.listdir(dataset_root)
                if os.path.isdir(os.path.join(dataset_root, d))
            ]
        if not circuit_types:
            empty = Data(x=torch.empty((0, 0), dtype=torch.float))
            empty_slices = {"x": torch.tensor([0])}
            torch.save((empty, empty_slices), self.processed_paths[0])
            return

        def parse_spice_netlist(netlist_path: str):
            """Parse a SPICE netlist with simple subcircuit expansion.

            Parameters
            ----------
            netlist_path : str
                Path to the netlist file.

            Returns
            -------
            list[dict]
                Parsed component dictionaries.
            """
            components = []
            subckts = {}

            with open(netlist_path) as f:
                lines = f.readlines()

            # Preprocess continuation lines (starting with '+')
            merged = []
            buffer = ""
            for line in lines:
                line = line.strip()
                if not line or line.startswith(("*", "//")):
                    continue
                if line.startswith("+"):
                    buffer += " " + line[1:].strip()
                    continue
                if buffer:
                    merged.append(buffer)
                buffer = line
            if buffer:
                merged.append(buffer)

            idx = 0
            while idx < len(merged):
                line = merged[idx]
                if line.lower().startswith(".subckt"):
                    parts = line.split()
                    name = parts[1]
                    pins = parts[2:]
                    body = []
                    idx += 1
                    while idx < len(merged) and not merged[
                        idx
                    ].lower().startswith(".ends"):
                        body.append(merged[idx])
                        idx += 1
                    subckts[name] = {"pins": pins, "body": body}
                else:
                    components.append(line)
                idx += 1

            def parse_component(line):
                """Parse a single netlist line into a component dictionary.

                Parameters
                ----------
                line : str
                    Raw netlist line.

                Returns
                -------
                dict | None
                    Parsed component or None if skipped.
                """
                tokens = line.replace("(", " ").replace(")", " ").split()
                if not tokens:
                    return None
                name = tokens[0]
                prefix = name[0].lower()
                if prefix == "x":  # subckt instance
                    subckt_name = tokens[-1]
                    nets = tokens[1:-1]
                    return {
                        "type": "subckt",
                        "subckt": subckt_name,
                        "nets": nets,
                        "name": name,
                    }
                elif prefix in {"m"}:  # mos
                    nodes = tokens[1:5]
                    model = tokens[5] if len(tokens) > 5 else "mos"
                    params = tokens[6:] if len(tokens) > 6 else []
                    return {
                        "type": model,
                        "nodes": nodes,
                        "params": params,
                        "name": name,
                    }
                elif prefix in {"q"}:  # bjt
                    nodes = tokens[1:4]
                    model = tokens[4] if len(tokens) > 4 else "bjt"
                    params = tokens[5:] if len(tokens) > 5 else []
                    return {
                        "type": model,
                        "nodes": nodes,
                        "params": params,
                        "name": name,
                    }
                else:
                    # generic two-terminal (R,C,L,V,I, etc.)
                    nodes = tokens[1:3]
                    ctype = tokens[3] if len(tokens) > 3 else tokens[0][0]
                    params = (
                        tokens[4:]
                        if len(tokens) > 4
                        else tokens[3:]
                        if len(tokens) > 3
                        else []
                    )
                    return {
                        "type": ctype,
                        "nodes": nodes,
                        "params": params,
                        "name": name,
                    }

            def expand(instance, prefix=""):
                """Recursively expand components and subcircuits.

                Parameters
                ----------
                instance : dict
                    Component or subcircuit instance.
                prefix : str, optional
                    Name prefix for nested instances.

                Returns
                -------
                list[dict]
                    Expanded component list.
                """
                if instance.get("type") != "subckt":
                    return [instance]
                sub_name = instance["subckt"]
                if sub_name not in subckts:
                    return []  # unknown subckt, skip
                mapping = dict(
                    zip(
                        subckts[sub_name]["pins"],
                        instance["nets"],
                        strict=True,
                    )
                )
                expanded = []
                for line in subckts[sub_name]["body"]:
                    comp = parse_component(line)
                    if comp is None:
                        continue
                    if comp.get("type") == "subckt":
                        # Remap nets and expand deeper
                        remapped = [mapping.get(n, n) for n in comp["nets"]]
                        comp["nets"] = remapped
                        expanded += expand(
                            comp, prefix + instance["name"] + "."
                        )
                    else:
                        comp["nodes"] = [
                            mapping.get(n, n) for n in comp.get("nodes", [])
                        ]
                        comp["name"] = (
                            prefix
                            + instance["name"]
                            + "."
                            + comp.get("name", "")
                        )
                        expanded.append(comp)
                return expanded

            parsed = []
            for line in components:
                comp = parse_component(line)
                if comp is None:
                    continue
                parsed += expand(comp)
            return parsed

        for circuit_type in circuit_types:
            # Read performance data
            csv_path = os.path.join(
                dataset_root, circuit_type, f"{circuit_type}.csv"
            )
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)

            # Read netlist
            netlist_path = os.path.join(netlist_root, circuit_type, "netlist")
            if not os.path.exists(netlist_path):
                continue

            components = parse_spice_netlist(netlist_path)

            node_map = {}
            hyperedge_index = []
            component_features = []
            incidence_roles = []
            hyperedge_params_list = []

            for idx_comp, comp in enumerate(components):
                ctype = comp.get("type", "unknown")
                nodes = comp.get("nodes", [])
                params = comp.get("params", [])
                component_features.append(
                    component_vocab.get(
                        ctype.lower(), component_vocab["unknown"]
                    )
                )
                role_codes = self._get_incidence_roles(ctype, nodes)
                hyperedge_params_list.append(self._parse_params(params))
                for node in nodes:
                    if node not in node_map:
                        node_map[node] = len(node_map)
                    hyperedge_index.append([node_map[node], idx_comp])
                incidence_roles.extend(role_codes[: len(nodes)])

            # Defer tensorization until after global padding is known
            parsed_graphs.append(
                {
                    "component_features": component_features,
                    "hyperedge_index": hyperedge_index,
                    "incidence_roles": incidence_roles,
                    "hyperedge_params_list": hyperedge_params_list,
                    "node_map": node_map,
                    "circuit_type": circuit_type,
                    "graph_attr_raw": df.values[:, :4],
                    "y_raw": df.values[:, 4:],
                }
            )

        # Determine global max param length for padding
        global_max_param_len = 0
        if not parsed_graphs:
            empty = Data(x=torch.empty((0, 0), dtype=torch.float))
            empty_slices = {"x": torch.tensor([0])}
            torch.save((empty, empty_slices), self.processed_paths[0])
            return

        for g in parsed_graphs:
            for p in g["hyperedge_params_list"]:
                if len(p) > global_max_param_len:
                    global_max_param_len = len(p)

        for g in parsed_graphs:
            node_map = g["node_map"]
            component_features = g["component_features"]
            hyperedge_index = g["hyperedge_index"]
            incidence_roles = g["incidence_roles"]
            hyperedge_params_list = g["hyperedge_params_list"]
            circuit_type = g["circuit_type"]

            # node features
            node_features = torch.tensor(
                [
                    self._get_node_feature(name)
                    for name, _ in sorted(
                        node_map.items(), key=lambda kv: kv[1]
                    )
                ],
                dtype=torch.float,
            ).view(-1, 1)

            if not hyperedge_index:
                hyperedge_index = torch.empty((2, 0), dtype=torch.long)
                incidence_roles = torch.empty((0,), dtype=torch.long)
                hyperedge_params = torch.empty(
                    (0, global_max_param_len), dtype=torch.float
                )
            else:
                hyperedge_index = (
                    torch.tensor(hyperedge_index, dtype=torch.long)
                    .t()
                    .contiguous()
                )
                incidence_roles = torch.tensor(
                    incidence_roles, dtype=torch.long
                )
                hyperedge_params = torch.zeros(
                    (len(hyperedge_params_list), global_max_param_len),
                    dtype=torch.float,
                )
                for i, vals in enumerate(hyperedge_params_list):
                    end = min(len(vals), global_max_param_len)
                    if end:
                        hyperedge_params[i, :end] = torch.tensor(
                            vals[:end], dtype=torch.float
                        )

            y = torch.tensor(g["y_raw"], dtype=torch.float)
            graph_attr = torch.tensor(g["graph_attr_raw"], dtype=torch.float)
            graph_attr = self._fix_length(graph_attr, 4)
            y = self._fix_length(y, 3)

            data = Data(
                x=node_features,
                hyperedge_index=hyperedge_index,
                y=y,
                graph_attr=graph_attr,
            )
            data.hyperedge_attr = torch.tensor(
                component_features, dtype=torch.long
            )
            data.incidence_roles = incidence_roles
            data.hyperedge_params = hyperedge_params
            data.name = circuit_type
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
