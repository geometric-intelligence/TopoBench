"""Dataset class for MIPLIB dataset."""

import gzip
import os
import os.path as osp
import shutil
import zipfile
from typing import ClassVar

import tqdm
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
        Name of the dataset. Default is "benchmark".
    parameters : DictConfig, optional
        Configuration parameters for the dataset.
    slice : int, optional
        Number of samples to process. Useful for testing to limit dataset size.
    **kwargs : optional
        Additional keyword arguments passed to InMemoryDataset.
        Common options include:
            - force_reload: bool, whether to re-download and re-process the dataset.

    Attributes
    ----------
    URLS (dict): Dictionary containing the URLs for downloading the datasets from MIPLIB collection.
    """

    URLS: ClassVar = {
        "benchmark": "https://miplib.zib.de/downloads/benchmark.zip",
        # "collection": "https://miplib.zib.de/downloads/collection.zip", - too large for now
        "solutions": "https://miplib.zib.de/downloads/solutions.zip",
        "miplib2017-v35.solu": "https://miplib.zib.de/downloads/miplib2017-v35.solu",
    }

    def __init__(
        self,
        root: str,
        name: str = "benchmark",
        parameters: DictConfig = None,
        slice: int | None = None,
        **kwargs,
    ) -> None:
        # Store original dataset name for URL lookup and directory structure
        dataset_name = name if name in self.URLS else "benchmark"
        self.dataset_name = dataset_name
        self.parameters = parameters
        self.slice = slice
        self.task_level = kwargs.get("task_level", "node")
        if parameters is not None and "task_level" in parameters:
            self.task_level = parameters.task_level

        super().__init__(
            root,
            **kwargs,
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
        return f"MIPLIBDataset(root={self.root}, name={self.dataset_name}, parameters={self.parameters}, task_level={self.task_level})"

    @property
    def raw_dir(self) -> str:
        """Return the path to the raw directory of the dataset.

        Returns
        -------
        str
            Path to the raw directory.
        """
        return osp.join(self.root, "raw", self.dataset_name)

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory of the dataset.

        Returns
        -------
        str
            Path to the processed directory.
        """
        return osp.join(self.root, "processed", self.dataset_name)

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

            # Check for solutions marker
            solutions_marker = osp.join(
                self.root,
                "raw",
                "solutions",
                "solutions_downloaded.txt",
            )
            if not osp.exists(solutions_marker):
                # If solutions are missing, return placeholder to trigger download
                return ["placeholder.mps"]

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
        if self.slice is not None:
            return f"data_slice_{self.slice}_{self.task_level}.pt"
        return f"data_{self.task_level}.pt"

    def _extract_zip(
        self,
        zip_path: str,
        extract_to: str,
        suffix: str,
        description: str = "files",
    ) -> None:
        """Extract files from a zip archive, applying slicing if configured.

        Parameters
        ----------
        zip_path : str
            Path to the zip file.
        extract_to : str
            Directory to extract files to.
        suffix : str
            File suffix to filter by (e.g. ".gz").
        description : str, optional
            Description of files for logging, by default "files".
        """
        print(f"Extracting {description}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Get list of all files in zip
            all_files = zip_ref.namelist()
            # Filter for relevant files
            filtered_files = [f for f in all_files if f.endswith(suffix)]
            # Sort to ensure deterministic order
            filtered_files.sort()

            # Apply slice if specified
            if self.slice is not None:
                print(
                    f"Slicing {description}: keeping first {self.slice} files out of {len(filtered_files)}"
                )
                filtered_files = filtered_files[: self.slice]

            # Extract only selected files
            for file in filtered_files:
                zip_ref.extract(file, extract_to)
        print(f"Extraction of {description} complete!")

    def download(self) -> None:
        r"""Download the dataset from a URL and saves it to the raw directory.

        Raises:
            FileNotFoundError: If the dataset URL is not found.
        """
        self.url = self.URLS[self.dataset_name]

        # Check if benchmark files already exist
        mps_files_exist = False
        if osp.exists(self.raw_dir):
            mps_files = [
                f for f in os.listdir(self.raw_dir) if f.endswith(".mps")
            ]
            if mps_files:
                mps_files_exist = True
                print(
                    "Benchmark files already exist. Skipping benchmark download."
                )

        if not mps_files_exist:
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

            self._extract_zip(
                path, folder, ".gz", description="benchmark files"
            )

            # Delete zip file
            os.remove(path)

            # We need to find .gz files and decompress them.
            # Since we only extracted the ones we want, we can just process all .gz files in the folder
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

        # --- Download Solutions ---
        miplib_raw_dir = osp.join(self.root, "raw")
        solutions_dir = osp.join(miplib_raw_dir, "solutions")
        solutions_marker = osp.join(solutions_dir, "solutions_downloaded.txt")

        if osp.exists(solutions_marker):
            print("Solutions already downloaded and processed. Skipping.")
            return

        print("Downloading solutions...")
        download_file_from_link(
            file_link=self.URLS["solutions"],
            path_to_save=miplib_raw_dir,
            dataset_name="solutions",
            file_format="zip",
        )

        sol_zip_path = osp.join(miplib_raw_dir, "solutions.zip")

        if not osp.exists(solutions_dir):
            os.makedirs(solutions_dir)

        self._extract_zip(
            sol_zip_path, solutions_dir, ".sol.gz", description="solutions"
        )

        # Recursively find and extract .sol.gz files
        print("Processing solution files...")
        for root, _, files in os.walk(solutions_dir):
            for file in files:
                if file.endswith(".sol.gz"):
                    gz_path = osp.join(root, file)
                    sol_filename = file.replace(".gz", "")
                    # We want to place the .sol file directly in solutions_dir
                    sol_path = osp.join(solutions_dir, sol_filename)

                    try:
                        with (
                            gzip.open(gz_path, "rb") as f_in,
                            open(sol_path, "wb") as f_out,
                        ):
                            shutil.copyfileobj(f_in, f_out)
                    except Exception as e:
                        print(f"Failed to decompress {gz_path}: {e}")

                    # Remove .gz file
                    os.remove(gz_path)

        print("Solutions extracted and flattened!")

        # Create marker file
        with open(solutions_marker, "w") as f:
            f.write("Solutions downloaded and processed.")
        if osp.exists(sol_zip_path):
            os.remove(sol_zip_path)

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

        # Apply slice if specified (for testing)
        if self.slice is not None:
            raw_files = raw_files[: self.slice]
            print(f"Processing only first {self.slice} files for testing")

        for i, path in enumerate(tqdm.tqdm(raw_files)):
            # Load model
            model = Model()
            model.hideOutput()
            model.readProblem(path)

            instance_name = (
                osp.basename(path).replace(".mps", "").replace(".gz", "")
            )

            solutions_dir = osp.join(self.root, "raw", "solutions")
            sol_path = osp.join(solutions_dir, f"{instance_name}.sol")

            sol = None

            if osp.exists(sol_path):
                try:
                    sol = model.readSolFile(sol_path)
                except Exception as e:
                    print(f"Failed to read solution {sol_path}: {e}")
                    continue  # Skip this MIP if solution reading fails
            else:
                print(f"Warning: Solution file not found for {instance_name}")
                continue  # Skip this MIP if solution doesn't exist

            # --- Extract Variable (Node) Features ---
            # Features: [Objective Coeff, Lower Bound, Upper Bound, IsContinuous, IsBinary, IsInteger, IsImplInt]
            # Type: 0=Continuous, 1=Binary, 2=Integer, 3=Implicit Integer

            vars = model.getVars()
            num_vars = len(vars)
            var_map = {v.name: i for i, v in enumerate(vars)}

            # 3 numerical features + 4 one-hot encoded type features
            x = torch.zeros((num_vars, 7), dtype=torch.float)
            # Labels: optimal value for each variable
            if self.task_level == "node":
                y = torch.zeros((num_vars, 1), dtype=torch.float)

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

                # Set label
                if self.task_level == "node":
                    if sol is not None:
                        y[i, 0] = model.getSolVal(sol, v)
                    else:
                        print(
                            f"Warning: No solution found for {instance_name}, variable {v.name} will have label 0.0"
                        )
                        # Default to 0 if no solution
                        y[i, 0] = 0.0

            if self.task_level == "graph":
                if sol is not None:
                    obj_val = model.getSolObjVal(sol)
                    y = torch.tensor([obj_val], dtype=torch.float)
                else:
                    print(
                        f"Warning: No solution found for {instance_name}, obj_val will be 0.0"
                    )
                    y = torch.tensor([0.0], dtype=torch.float)

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
            data = Data(
                x=x,
                y=y,
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
    """
    Test the MIPLIB dataset.
    `uv run -m topobench.data.datasets.miplib_dataset`
    """
    import rootutils

    root = rootutils.setup_root(
        search_from=".",
        indicator="pyproject.toml",
        pythonpath=True,
        cwd=True,
    )

    dataset = MIPLIBDataset(
        root="datasets", name="benchmark", force_reload=True
    )
    print(dataset)
