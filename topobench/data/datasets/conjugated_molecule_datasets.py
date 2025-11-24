"""Dataset class for conjugated molecular structures."""

import os.path as osp
from collections.abc import Callable

import pandas as pd
import torch
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    extract_zip,
)

from topobench.data.utils import (
    download_file_from_link,
)
from topobench.data.utils.conjugated_utils import (
    get_hypergraph_data_from_smiles,
)


class ConjugatedMoleculeDataset(InMemoryDataset):
    """Dataset class for conjugated molecular structures.

    Parameters
    ----------
    root : str
        Root directory where the dataset will be saved.
    name : str, optional
        Name of the dataset. Default is "conjugated_molecules".
    split : str, optional
        Split of the dataset (e.g., "train", "valid", "test"). Only used for OPV.
    task : str, optional
        Task type. Default is "default". For OPV, can be "polymer" to filter
        molecules with extrapolated properties.
    transform : Callable, optional
        A function/transform that takes in an :obj:`torch_geometric.data.Data` object
        and returns a transformed version. The data object will be transformed before
        every access.
    pre_transform : Callable, optional
        A function/transform that takes in an :obj:`torch_geometric.data.Data` object
        and returns a transformed version. The data object will be transformed before
        being saved to disk.
    pre_filter : Callable, optional
        A function that takes in an :obj:`torch_geometric.data.Data` object and
        returns a boolean value, indicating whether the data object should be
        included in the final dataset.
    **kwargs : optional
        Additional keyword arguments passed to InMemoryDataset.
        Common options include:
            - force_reload: bool, whether to re-download and re-process the dataset.
    """

    URLS = {
        "OCELOTv1": "https://data.materialsdatafacility.org/mdf_open/ocelot_chromophore_v1_v1.1/ocelot_chromophore_v1.csv",
        "OPV": {
            "train": "https://data.nrel.gov/system/files/236/1712697052-smiles_train.csv.gz",
            "valid": "https://data.nrel.gov/system/files/236/1712697052-smiles_valid.csv.gz",
            "test": "https://data.nrel.gov/system/files/236/1712697052-smiles_test.csv.gz",
        },
        "PCQM4MV2": "https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip",
    }

    def __init__(
        self,
        root: str,
        name: str,
        split: str | None = None,
        task: str = "default",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        **kwargs,
    ):
        if name not in self.URLS:
            raise ValueError(f"Unknown dataset name: {name}")
        self.name = name
        self.split = split
        self.task = task
        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            **kwargs,
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    def mean(self, target: int = 0) -> float:
        """Calculate mean of a specific target across the dataset.

        Parameters
        ----------
        target : int
            Index of the target to calculate mean for. Default is 0.

        Returns
        -------
        float
            Mean value of the specified target.
        """
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        if y.dim() == 1:
            return y.mean().item()
        return y[:, target].mean().item()

    def std(self, target: int = 0) -> float:
        """Calculate standard deviation of a specific target across the dataset.

        Parameters
        ----------
        target : int
            Index of the target to calculate std for. Default is 0.

        Returns
        -------
        float
            Standard deviation of the specified target.
        """
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        if y.dim() == 1:
            return y.std().item()
        return y[:, target].std().item()

    @property
    def raw_dir(self) -> str:
        """Return the raw directory.

        Returns
        -------
        str
            Path to the raw directory.
        """
        if self.name == "OPV" and self.split:
            return osp.join(self.root, self.name, "raw", self.split)
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        """Return the processed directory.

        Returns
        -------
        str
            Path to the processed directory.
        """
        if self.name == "OPV" and self.split:
            return osp.join(self.root, self.name, "processed", self.split)
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw file names.

        Returns
        -------
        list[str]
            List of raw file names.
        """
        if self.name == "OCELOTv1":
            return ["ocelot_chromophore_v1.csv"]
        if self.name == "OPV":
            if self.split == "train":
                return ["1712697052-smiles_train.csv.gz"]
            if self.split == "valid":
                return ["1712697052-smiles_valid.csv.gz"]
            if self.split == "test":
                return ["1712697052-smiles_test.csv.gz"]
        if self.name == "PCQM4MV2":
            return ["pcqm4m-v2/raw/data.csv.gz"]  # Extracted path
        return ["merged_data.csv"]

    @property
    def processed_file_names(self) -> str:
        """Return the processed file name.

        Returns
        -------
        str
            Name of the processed file.
        """
        return "data.pt"

    def download(self):
        """Download the dataset."""
        if self.name == "OCELOTv1":
            download_file_from_link(
                self.URLS["OCELOTv1"],
                self.raw_dir,
                dataset_name="ocelot_chromophore_v1",
                file_format="csv",
            )
        elif self.name == "OPV":
            if self.split:
                download_file_from_link(
                    self.URLS["OPV"][self.split],
                    self.raw_dir,
                    dataset_name=f"1712697052-smiles_{self.split}",
                    file_format="csv.gz",
                )
            else:
                for split_name in self.URLS["OPV"]:
                    download_file_from_link(
                        self.URLS["OPV"][split_name],
                        osp.join(self.root, self.name, "raw", split_name),
                        dataset_name=f"1712697052-smiles_{split_name}",
                        file_format="csv.gz",
                    )

        elif self.name == "PCQM4MV2":
            path = osp.join(self.raw_dir, "PCQM4MV2.zip")
            download_file_from_link(
                self.URLS["PCQM4MV2"],
                self.raw_dir,
                dataset_name="PCQM4MV2",
                file_format="zip",
            )
            extract_zip(path, self.raw_dir)
            # The zip extracts to pcqm4m-v2 folder
        else:
            # Placeholder for user provided data
            if not osp.exists(osp.join(self.raw_dir, "merged_data.csv")):
                print(
                    f"Please place 'merged_data.csv' in {self.raw_dir}. "
                    "This file should contain a 'ready_SMILES' column."
                )

    def process(self):
        """Convert data from raw files and save to disk."""

        data_list = []

        if self.name == "OCELOTv1":
            raw_path = osp.join(self.raw_dir, "ocelot_chromophore_v1.csv")
            df = pd.read_csv(raw_path)
            smiles_col = "smiles"
        elif self.name == "OPV":
            # Filename depends on split
            filename = self.raw_file_names[0]
            raw_path = osp.join(self.raw_dir, filename)
            df = pd.read_csv(raw_path)
            smiles_col = "smiles"  # Assuming standard

            # Polymer task: filter molecules with complete extrapolated properties
            if self.task == "polymer":
                df = df.dropna(subset=["gap_extrapolated"])
                print(
                    f"Polymer task: filtered to {len(df)} molecules with gap_extrapolated values"
                )
        elif self.name == "PCQM4MV2":
            # The extracted file is likely in a subdir
            raw_path = osp.join(
                self.raw_dir, "pcqm4m-v2", "raw", "data.csv.gz"
            )
            df = pd.read_csv(raw_path)
            smiles_col = "smiles"
        else:
            raw_path = osp.join(self.raw_dir, "merged_data.csv")
            df = pd.read_csv(raw_path)
            smiles_col = "ready_SMILES"

        if not osp.exists(raw_path):
            raise FileNotFoundError(f"File not found: {raw_path}")

        if smiles_col not in df.columns:
            # Fallback or check for other common names if needed
            if "SMILES" in df.columns:
                smiles_col = "SMILES"
            elif "smile" in df.columns:  # OPV uses singular 'smile'
                smiles_col = "smile"
            elif "ready_SMILES" in df.columns:
                smiles_col = "ready_SMILES"
            else:
                raise ValueError(
                    f"CSV file must contain '{smiles_col}' column. Found: {df.columns}"
                )

        smiles_list = df[smiles_col].tolist()

        for idx, smiles in enumerate(smiles_list):
            try:
                atom_fvs, incidence_list, bond_fvs = (
                    get_hypergraph_data_from_smiles(smiles)
                )
            except (TypeError, ValueError, AttributeError):
                continue

            if not incidence_list:
                continue

            num_nodes = len(atom_fvs)
            # incidence_matrix = create_incidence_matrix(
            #     incidence_list, num_nodes
            # )

            # Convert to tensors
            x = torch.tensor(atom_fvs, dtype=torch.float)

            # Create edge_index from incidence list
            # incidence_list is list of lists of node indices
            sources = []
            targets = []
            for edge_idx, nodes in enumerate(incidence_list):
                for node_idx in nodes:
                    sources.append(node_idx)
                    targets.append(edge_idx)

            edge_index = torch.tensor([sources, targets], dtype=torch.long)

            # Calculate edge order (hyperedge cardinality)
            e_order = torch.tensor(
                [len(nodes) for nodes in incidence_list],
                dtype=torch.long,
            )

            # Hyperedge features (bond features)
            hyperedge_attr = list(bond_fvs)

            hyperedge_attr = torch.tensor(hyperedge_attr, dtype=torch.float)

            # Incidence matrix as sparse tensor
            incidence_hyperedges = torch.sparse_coo_tensor(
                edge_index,
                torch.ones(edge_index.shape[1]),
                size=(num_nodes, len(incidence_list)),
            )

            # Create base data object
            data = Data(
                x=x,
                edge_index=edge_index,
                hyperedge_attr=hyperedge_attr,
                incidence_hyperedges=incidence_hyperedges,
                num_nodes=num_nodes,
                num_hyperedges=len(incidence_list),
                e_order=e_order,  # Edge order tracking
                smi=smiles,  # Store SMILES string
            )

            # Extract target labels based on dataset
            if self.name == "OPV":
                # OPV has 8 regression targets (columns 2-9)
                target = df.iloc[idx, 2:].values.astype(float)
                y = torch.tensor(target, dtype=torch.float).unsqueeze(0)
                data.y = y
            elif self.name == "PCQM4MV2":
                # PCQM4MV2 has single target: homolumogap
                if "homolumogap" in df.columns:
                    y = torch.tensor(
                        [df.loc[idx, "homolumogap"]], dtype=torch.float
                    )
                    data.y = y
            elif self.name == "OCELOTv1":
                # OCELOTv1 - extract targets from columns after identifier and smiles
                # Skip non-numeric columns (identifier, smiles)
                numeric_cols = []
                for col in df.columns:
                    if col not in ["identifier", "smiles", "smile"]:
                        # Try to convert to numeric
                        try:
                            pd.to_numeric(df[col].iloc[0])
                            numeric_cols.append(col)
                        except (ValueError, TypeError):
                            pass

                if len(numeric_cols) > 0:
                    target = df.iloc[idx][numeric_cols].values.astype(float)
                    y = torch.tensor(target, dtype=torch.float).unsqueeze(0)
                    data.y = y

            if self.split:
                data.split = self.split

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


if __name__ == "__main__":
    """
    Run this script to test the ConjugatedMoleculeDataset.
    `uv run python -m topobench.data.datasets.conjugated_molecule_datasets`
    """
    import rootutils

    root = rootutils.setup_root(
        search_from=".",
        indicator="pyproject.toml",
        pythonpath=True,
        cwd=True,
    )

    print("=" * 60)
    print("ConjugatedMoleculeDataset Verification Script")
    print("=" * 60)

    results = []

    print("Testing OCELOTv1 dataset...")
    try:
        dataset = ConjugatedMoleculeDataset(
            root="datasets/test",
            name="OCELOTv1",
        )
        print(
            f"✓ OCELOTv1 dataset loaded successfully with {len(dataset)} samples"
        )
        print(f"  First sample: {dataset[0]}")
    except Exception as e:
        print(f"✗ OCELOTv1 dataset failed: {e}")
        raise

    print("Testing OPV dataset...")
    try:
        dataset = ConjugatedMoleculeDataset(
            root="data/test",
            name="OPV",
            split="train",  # OPV requires a split
        )
        print(f"✓ OPV dataset loaded successfully with {len(dataset)} samples")
        print(f"  First sample: {dataset[0]}")
    except Exception as e:
        print(f"✗ OPV dataset failed: {e}")
        raise

    print("Testing PCQM4MV2 dataset...")
    try:
        dataset = ConjugatedMoleculeDataset(
            root="data/test",
            name="PCQM4MV2",
        )
        print(
            f"✓ PCQM4MV2 dataset loaded successfully with {len(dataset)} samples"
        )
        print(f"  First sample: {dataset[0]}")
    except Exception as e:
        print(f"✗ PCQM4MV2 dataset failed: {e}")
        raise
