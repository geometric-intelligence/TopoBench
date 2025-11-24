"""Dataset class for MIPLIB dataset."""

import gzip
import os
import os.path as osp
import shutil
import zipfile
from typing import ClassVar

from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs

from topobench.data.utils import (
    download_file_from_link,
)


class MIPLIBDataset(InMemoryDataset):
    r"""Dataset class for MIPLIB dataset.

    Parameters
    ----------
    root : str
        Root directory where the dataset will be saved.
    name : str, optional
        Name of the dataset ("benchmark" or "collection") from MIPLIB collection. Default is "benchmark".
    parameters : DictConfig, optional
        Configuration parameters for the dataset.
    force_reload : bool, optional
        If True, deletes the raw directory and forces a fresh download. Default is False.

    Attributes
    ----------
    URLS (dict): Dictionary containing the URLs for downloading the datasets from MIPLIB collection.
    """

    URLS: ClassVar = {
        "benchmark": "https://miplib.zib.de/downloads/benchmark.zip",
        "collection": "https://miplib.zib.de/downloads/collection.zip",
        "solutions": "https://miplib.zib.de/downloads/solutions.zip",
        "miplib2017-v35.solu": "https://miplib.zib.de/downloads/miplib2017-v35.solu",
    }

    def __init__(
        self,
        root: str,
        name: str = "benchmark",
        parameters: DictConfig = None,
        force_reload: bool = False,
    ) -> None:
        # Store original dataset name for URL lookup and directory structure
        dataset_name = name if name in self.URLS else "benchmark"
        self.dataset_name = dataset_name
        self.parameters = parameters

        # Force reload: delete raw directory if it exists
        if force_reload:
            raw_dir = osp.join(root, "miplib", "raw", dataset_name)
            if osp.exists(raw_dir):
                shutil.rmtree(raw_dir)
                print(f"Deleted raw directory: {raw_dir}")

        super().__init__(
            root,
        )

        if osp.exists(self.processed_paths[0]):
            out = fs.torch_load(self.processed_paths[0])
            # assert len(out) == 3 or len(out) == 4
            if len(out) == 3:
                data, self.slices, self.sizes = out
                data_cls = Data
            elif len(out) == 4:
                data, self.slices, self.sizes, data_cls = out

            self.data = data_cls.from_dict(data)

    def __repr__(self) -> str:
        return f"MIPLIBDataset(root={self.root}, name={self.dataset_name}, parameters={self.parameters})"

    @property
    def raw_dir(self) -> str:
        """Return the path to the raw directory of the dataset.

        Returns
        -------
        str
            Path to the raw directory.
        """
        return osp.join(self.root, "miplib", "raw", self.dataset_name)

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory of the dataset.

        Returns
        -------
        str
            Path to the processed directory.
        """
        return osp.join(self.root, "miplib", "processed", self.dataset_name)

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw file names for the dataset.

        Returns
        -------
        list[str]
            List of raw file names.
        """
        # Dynamically discover files to avoid re-downloading
        if osp.exists(self.raw_dir):
            files = [
                osp.relpath(osp.join(root, filename), self.raw_dir)
                for root, _, filenames in os.walk(self.raw_dir)
                for filename in filenames
                if filename.endswith(".mps")
            ]
            return (
                files if files else ["placeholder.mps"]
            )  # Return placeholder if no files found yet
        return ["placeholder.mps"]

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
        self.url = self.URLS[self.dataset_name]

        download_file_from_link(
            file_link=self.url,
            path_to_save=self.raw_dir,
            dataset_name=self.dataset_name,
            file_format="zip",
        )

        # Extract zip file
        folder = self.raw_dir
        filename = f"{self.dataset_name}.zip"
        path = osp.join(folder, filename)

        print("Extracting...")
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(folder)
        print("Extraction complete!")

        # Delete zip file
        os.remove(path)

        # We need to find .gz files and decompress them.
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".gz"):
                    instance_gz = osp.join(root, file)
                    instance_mps = instance_gz.replace(".gz", "")

                    # Decompress
                    with (
                        gzip.open(instance_gz, "rb") as f_in,
                        open(instance_mps, "wb") as f_out,
                    ):
                        shutil.copyfileobj(f_in, f_out)

                    # Remove .gz file to save space
                    os.remove(instance_gz)

    def process(self) -> None:
        r"""Handle the data for the dataset.

        This method loads the MIPLIB data, applies any pre-
        processing transformations if specified, and saves the processed data
        to the appropriate location.
        """
        import torch
        from pyscipopt import Model

        data_list = []

        # Iterate over all files in the raw directory
        # We look for .mps files (which we extracted/renamed from .gz)
        # or .mps.gz if we decided to keep them compressed.
        # The download method extracts zip, then decompresses .gz to .mps (removing .gz extension)

        # Let's find all files in raw_dir
        raw_files = []
        for root, _, files in os.walk(self.raw_dir):
            raw_files.extend(
                osp.join(root, file)
                for file in files
                if file.endswith((".mps", ".mps.gz"))
            )

        if not raw_files:
            print("No .mps or .mps.gz files found in raw directory.")
            return

        for path in raw_files:
            print(f"Processing {path}...")

            # Load model
            model = Model()
            model.readProblem(path)

            # --- Extract Variable (Node) Features ---
            # Features: [Objective Coeff, Lower Bound, Upper Bound, IsContinuous, IsBinary, IsInteger, IsImplInt]
            # Type: 0=Continuous, 1=Binary, 2=Integer, 3=Implicit Integer

            vars = model.getVars()
            num_vars = len(vars)
            var_map = {v.name: i for i, v in enumerate(vars)}

            # 3 numerical features + 4 one-hot encoded type features
            x = torch.zeros((num_vars, 7), dtype=torch.float)

            for i, v in enumerate(vars):
                x[i, 0] = v.getObj()
                x[i, 1] = v.getLbLocal()
                x[i, 2] = v.getUbLocal()

                v_type = v.vtype()
                if v_type == "CONTINUOUS":
                    x[i, 3] = 1.0
                elif v_type == "BINARY":
                    x[i, 4] = 1.0
                elif v_type == "INTEGER":
                    x[i, 5] = 1.0
                elif v_type == "IMPLINT":
                    x[i, 6] = 1.0
                else:
                    pass  # Unknown type, leave all zero

            # --- Extract Constraint (Hyperedge) Features ---
            # Features: [RHS, Sense]
            # Sense: -1=<=, 0==, 1=>= (simplified mapping)

            conss = model.getConss()
            num_conss = len(conss)

            # 1 numerical feature (RHS/Val) + 3 one-hot encoded sense features (<=, =, >=)
            hyperedge_attr = torch.zeros((num_conss, 4), dtype=torch.float)

            # Incidence matrix (sparse)
            # edge_index: [2, num_nonzeros], row 0 = variable index, row 1 = constraint index
            # edge_attr: [num_nonzeros, 1], coefficient

            sources = []  # Variable indices
            targets = []  # Constraint indices
            edge_weights = []  # Coefficients

            for j, c in enumerate(conss):
                # RHS and Sense
                # pyscipopt constraints are often in form lhs <= expr <= rhs
                # We simplify to standard forms if possible or just take rhs/lhs

                # For simplicity, let's look at getRhs() and getLhs()
                # If lhs == rhs, it's equality.
                # If lhs = -infinity, it's <= rhs
                # If rhs = +infinity, it's >= lhs

                rhs = model.getRhs(c)
                lhs = model.getLhs(c)

                # One-hot encode sense: [RHS, <=, =, >=]
                # Sense mapping:
                # <= : index 1
                # =  : index 2
                # >= : index 3

                if lhs == rhs:
                    # Equality
                    hyperedge_attr[j, 0] = rhs
                    hyperedge_attr[j, 2] = 1.0
                elif lhs <= -1e20:  # Effectively -infinity
                    # <= RHS
                    hyperedge_attr[j, 0] = rhs
                    hyperedge_attr[j, 1] = 1.0
                elif rhs >= 1e20:  # Effectively +infinity
                    # >= LHS
                    hyperedge_attr[j, 0] = lhs
                    hyperedge_attr[j, 3] = 1.0
                else:
                    # Range constraint, treat as equality for now or pick one side
                    # Defaulting to equality with RHS for consistency with previous logic
                    hyperedge_attr[j, 0] = rhs
                    hyperedge_attr[j, 2] = 1.0

                # hyperedge_attr[j, 0] is already set above
                # hyperedge_attr[j, 1:] are the one-hot encoding

                # Coefficients
                # getValsLinear returns dictionary {var: coeff}
                coeffs = model.getValsLinear(c)
                for v, coeff in coeffs.items():
                    # v can be a Variable object or a string (name)
                    v_name = v.name if hasattr(v, "name") else v
                    if v_name in var_map:
                        v_idx = var_map[v_name]
                        sources.append(v_idx)
                        targets.append(j)
                        edge_weights.append(coeff)

            edge_index = torch.tensor([sources, targets], dtype=torch.long)
            edge_attr = torch.tensor(edge_weights, dtype=torch.float)

            # Create incidence_hyperedges sparse tensor
            # Shape: (num_nodes, num_hyperedges)
            incidence_hyperedges = torch.sparse_coo_tensor(
                edge_index, edge_attr, size=(num_vars, num_conss)
            )

            # Create Data object
            # We follow TopoBench convention:
            # x: Node features
            # edge_index: Bipartite representation (node_idx, hyperedge_idx)
            # edge_attr: Incidence features (coefficients)
            # incidence_hyperedges: Sparse tensor representation
            # hyperedge_attr: Hyperedge features

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr.unsqueeze(1),
                incidence_hyperedges=incidence_hyperedges,
                hyperedge_attr=hyperedge_attr,
                num_nodes=num_vars,
                num_hyperedges=num_conss,
            )

            data_list.append(data)

        if not data_list:
            print("No valid data objects created.")
            return

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])


if __name__ == "__main__":
    dataset = MIPLIBDataset(
        root="datasets", name="benchmark", force_reload=False
    )
    print(dataset)
