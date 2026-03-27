"""AnalogGenie hypergraph dataset definition and processing."""

import glob
import os
import shutil
from pathlib import Path

import torch
from torch_geometric.data import Data, InMemoryDataset

from topobench.data.utils import download_file_from_link, extract_zip


class AnalogGenieData(Data):
    """Data object that surfaces missing y as an actual missing attribute."""

    @property
    def y(self):
        """Return target if present, else raise AttributeError.

        Returns
        -------
        torch.Tensor
            Target tensor.

        Raises
        ------
        AttributeError
            If target not set.
        """
        if "y" in self._store:
            return self._store["y"]
        raise AttributeError("'AnalogGenieData' object has no attribute 'y'")

    @y.setter
    def y(self, value):
        """Set target tensor.

        Parameters
        ----------
        value : torch.Tensor
            Target tensor to store.
        """
        self._store["y"] = value


class AnalogGenieDataset(InMemoryDataset):
    """Hypergraph dataset for AnalogGenie circuits.

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
        "AnalogGenie": "1-5-gS9aK2-a-d-Y-g-f-X-e-Y-k-Z-w-Y-k-d-Y-k-d-Y-k-d"
    }
    FILE_FORMAT = {"AnalogGenie": "zip"}

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
        # Check if the 'Dataset' directory (indicating successful download/extraction) already exists
        if os.path.exists(os.path.join(self.raw_dir, "Dataset")):
            print("Raw data already exists. Skipping download.")
            return

        # GitHub repo with .cir files
        self.url = "https://github.com/xz-group/AnalogGenie/archive/refs/heads/main.zip"
        self.file_format = "zip"
        os.makedirs(self.raw_dir, exist_ok=True)
        download_file_from_link(
            file_link=self.url,
            path_to_save=self.raw_dir,
            dataset_name=self.name,  # names the zip AnalogGenie.zip
            file_format=self.file_format,
        )

        folder = self.raw_dir
        filename = f"{self.name}.{self.file_format}"
        path = os.path.join(folder, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Downloaded file not found at {path}")

        extract_zip(path, folder)
        os.unlink(path)  # Remove the zip file after extraction

        # Find the extracted repo folder and move contents up to raw_dir
        extracted_folder_name = ""
        for item in os.listdir(self.raw_dir):
            if os.path.isdir(
                os.path.join(self.raw_dir, item)
            ) and item.lower().startswith("analoggenie"):
                extracted_folder_name = item
                break

        if extracted_folder_name:
            source_folder = os.path.join(folder, extracted_folder_name)
            for file in os.listdir(source_folder):
                shutil.move(os.path.join(source_folder, file), folder)
            shutil.rmtree(source_folder)
        else:
            print(
                "Warning: extracted folder not found; assuming contents already at raw root."
            )

    def _create_component_vocab(self):
        """Build vocabulary of component types.

        Returns
        -------
        dict
            Mapping from component type to ID.
        """
        return {
            "capacitor": 0,
            "nmos4": 1,
            "pmos4": 2,
            "resistor": 3,
            "unknown": 4,
        }

    def _get_incidence_roles(
        self, component_type: str, nodes: list[str]
    ) -> list[int]:
        """Assign pin-role codes per node for a component.

        Parameters
        ----------
        component_type : str
            Component type identifier.
        nodes : list of str
            Node names for this component.

        Returns
        -------
        list of int
            Role codes aligned with nodes.
        """
        roles = []
        comp = component_type.lower()
        if comp in {"nmos4", "pmos4", "nmos", "pmos", "mos", "mos4"}:
            template = [1, 2, 3, 4]  # drain, gate, source, bulk
            for idx, _ in enumerate(nodes):
                roles.append(template[idx] if idx < len(template) else 0)
        elif comp in {"bjt", "npn", "pnp"}:
            template = [11, 12, 13]  # collector, base, emitter
            for idx, _ in enumerate(nodes):
                roles.append(template[idx] if idx < len(template) else 0)
        else:
            roles = [0] * len(nodes)
        return roles

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

    def process(self):  # noqa: D401
        """Process raw AnalogGenie data into hypergraph tensors."""
        Path(self.processed_dir).mkdir(
            parents=True, exist_ok=True
        )  # Ensure processed directory exists
        data_list = []
        component_vocab = self._create_component_vocab()

        # Discover all .cir files in the raw directory
        circuit_files = glob.glob(
            os.path.join(self.raw_dir, "Dataset", "*", "*.cir")
        )
        if not circuit_files:
            empty = AnalogGenieData(x=torch.empty((0, 0), dtype=torch.float))
            empty_slices = {"x": torch.tensor([0])}
            torch.save((empty, empty_slices), self.processed_paths[0])
            return

        def parse_spice_netlist(cir_path):
            """Parse a SPICE netlist with simple subcircuit expansion.

            Parameters
            ----------
            cir_path : str
                Path to the circuit file.

            Returns
            -------
            list[dict]
                Parsed component dictionaries.
            """
            components = []
            subckts = {}
            with open(cir_path) as f:
                lines = f.readlines()

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
                    Parsed component or None.
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
                    # generic two-terminal
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
                    return []
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

        for circuit_file in circuit_files:
            components = parse_spice_netlist(circuit_file)

            node_map = {}
            hyperedge_index = []
            component_features = []
            incidence_roles = []
            hyperedge_params = []

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
                hyperedge_params.append(self._parse_params(params))

                for node in nodes:
                    if node not in node_map:
                        node_map[node] = len(node_map)
                    hyperedge_index.append([node_map[node], idx_comp])
                incidence_roles.extend(role_codes[: len(nodes)])

            # Create node features from node names (power/ground/generic)
            x = torch.tensor(
                [
                    self._get_node_feature(name)
                    for name, _ in sorted(
                        node_map.items(), key=lambda kv: kv[1]
                    )
                ],
                dtype=torch.float,
            ).view(-1, 1)

            if not hyperedge_index:
                # Handle empty graphs
                hyperedge_index = torch.empty((2, 0), dtype=torch.long)
                incidence_roles = torch.empty((0,), dtype=torch.long)
                hyperedge_params_tensor = torch.empty(
                    (0, 0), dtype=torch.float
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
                max_len = max((len(p) for p in hyperedge_params), default=0)
                hyperedge_params_tensor = torch.zeros(
                    (len(hyperedge_params), max_len), dtype=torch.float
                )
                for i, vals in enumerate(hyperedge_params):
                    if not vals:
                        continue
                    end = min(len(vals), max_len)
                    hyperedge_params_tensor[i, :end] = torch.tensor(
                        vals[:end], dtype=torch.float
                    )

            data = AnalogGenieData(x=x, hyperedge_index=hyperedge_index)
            # Add component features as hyperedge attributes
            data.hyperedge_attr = torch.tensor(
                component_features, dtype=torch.long
            )
            data.incidence_roles = incidence_roles
            data.hyperedge_params = hyperedge_params_tensor
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        if not data_list:
            empty = AnalogGenieData(x=torch.empty((0, 0), dtype=torch.float))
            empty_slices = {"x": torch.tensor([0])}
            torch.save((empty, empty_slices), self.processed_paths[0])
            return

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
