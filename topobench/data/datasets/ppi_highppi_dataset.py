"""PPI dataset integrating HIGH-PPI network data with CORUM human protein complexes.

Combines:
- HIGH-PPI SHS27k: PPI network with 7 interaction type features + confidence scores
- CORUM: ~470 experimentally validated human protein complexes as native higher-order structures
- TODO: Add data for node features (embeddings)

Simplicial complex structure:
- 0-cells: 1,553 proteins
- 1-cells: 6,660 PPI edges + CORUM complexes of size 2
- 2+ cells: CORUM complexes of size 3+
"""

import json
import os
import os.path as osp
from typing import ClassVar

import numpy as np
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs

from topobench.data.utils import get_complex_connectivity
from topobench.data.utils.datasets.simplicial.ppi_utils import (
    build_data_features_and_labels,
    build_simplicial_complex_with_features,
    generate_negative_samples,
    load_corum_complexes,
    load_highppi_network,
    load_id_mapping,
)
from topobench.data.utils.io_utils import (
    download_ensembl_biomart_mapping,
    download_file,
    download_folder_from_drive,
)


class PPIHighPPIDataset(InMemoryDataset):
    """HIGH-PPI network integrated with CORUM human protein complexes.

    Combines 6,660 protein-protein interactions from HIGH-PPI SHS27k with ~470
    experimentally validated human protein complexes from CORUM database.

    Parameters
    ----------
    root : str
        Root directory.
    name : str, optional
        Dataset name, default "ppi_highppi".
    parameters : DictConfig, optional
        Config with min_complex_size, max_complex_size, target_ranks, neg_ratio,
        edge_task ("score" or "interaction_type").
    **kwargs : dict
        Additional keyword arguments passed to InMemoryDataset.
    """

    INTERACTION_TYPES: ClassVar[list[str]] = [
        "reaction",
        "binding",
        "ptmod",
        "activation",
        "inhibition",
        "catalysis",
        "expression",
    ]

    # Data source URLs
    HIGHPPI_GDRIVE_FOLDER: ClassVar[str] = (
        "https://drive.google.com/drive/folders/1Yb-fdWJ5vTe0ePAGNfrUluzO9tz1lHIF?usp=sharing"
    )
    CORUM_URL: ClassVar[str] = (
        "https://mips.helmholtz-muenchen.de/fastapi-corum/public/file/download_current_file?file_id=human&file_format=txt"
    )

    # Required raw data filenames
    HIGHPPI_NETWORK_FILE: ClassVar[str] = (
        "protein.actions.SHS27k.STRING.pro2.txt"
    )
    ID_MAPPING_FILE: ClassVar[str] = "ensp_uniprot.txt"
    CORUM_COMPLEXES_FILE: ClassVar[str] = "allComplexes.txt"

    def __init__(
        self,
        root: str,
        name: str = "ppi_highppi",
        parameters: DictConfig = None,
        **kwargs,
    ):
        self.name = name
        self.parameters = parameters or DictConfig({})
        self.min_complex_size = self.parameters.get("min_complex_size", 2)
        self.max_complex_size = self.parameters.get("max_complex_size", 6)
        self.max_rank = self.max_complex_size - 1
        self.neg_ratio = self.parameters.get("neg_ratio", 1.0)
        self.target_ranks = self.parameters.get("target_ranks", [2, 3, 4, 5])
        self.edge_task = self.parameters.get("edge_task", "score")

        self.highppi_edges = []  # List of (p1, p2, interaction_type_vector, confidence_score)
        self.corum_complexes = []  # List of sets of proteins in a complex
        self.all_proteins = set()
        self.ensembl_to_uniprot = {}
        self.uniprot_to_ensembl = {}
        self.official_splits = {}

        super().__init__(root, **kwargs)

        out = fs.torch_load(self.processed_paths[0])
        if len(out) == 3:
            data, self.slices, self.sizes = out
            data_cls = Data
        else:
            data, self.slices, self.sizes, data_cls = out

        if not isinstance(data, dict):
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

        # Ensure data.y is set for single-rank compatibility
        # TODO: Change for B2 submission which will introduce a unified training loop
        if len(self.target_ranks) == 1:
            label_attr = f"cell_labels_{self.target_ranks[0]}"
            if hasattr(self._data, label_attr):
                self._data.y = getattr(self._data, label_attr)

    @property
    def raw_dir(self) -> str:
        """Return the path to the raw directory.

        Returns
        -------
        str
            Path to the raw directory.
        """
        return osp.join(self.root, "raw")

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory.

        Returns
        -------
        str
            Path to the processed directory.
        """
        return osp.join(self.root, "processed")

    @property
    def raw_file_names(self) -> list[str]:
        """Return list of required raw file names.

        Returns
        -------
        List[str]
            Required raw data files.
        """
        return [
            self.HIGHPPI_NETWORK_FILE,
            self.ID_MAPPING_FILE,
            self.CORUM_COMPLEXES_FILE,
        ]

    @property
    def processed_file_names(self) -> list[str]:
        """Return the name of the processed file.

        Filename includes target_ranks to avoid cache conflicts when
        different ranks are requested.

        Returns
        -------
        List[str]
            List containing the name of the processed file.
        """
        # Include target_ranks in filename to prevent cache conflicts
        ranks_str = "_".join(map(str, self.target_ranks))
        return [f"data_ranks_{ranks_str}.pt"]

    def download(self) -> None:
        """Download HIGH-PPI and CORUM data files."""

        # Check if files already exist
        all_exist = all(
            osp.exists(osp.join(self.raw_dir, fname))
            for fname in self.raw_file_names
        )
        if all_exist:
            print("All required files already present")
            return

        print("Downloading HIGH-PPI SHS27k dataset and CORUM complexes...")
        os.makedirs(self.raw_dir, exist_ok=True)

        if not osp.exists(osp.join(self.raw_dir, self.CORUM_COMPLEXES_FILE)):
            print("Downloading CORUM human protein complexes...")
            download_file(
                self.CORUM_URL,
                osp.join(self.raw_dir, self.CORUM_COMPLEXES_FILE),
                verify_ssl=False,
            )
            print("CORUM download complete")

        if not osp.exists(osp.join(self.raw_dir, self.ID_MAPPING_FILE)):
            print("Downloading Ensembl-UniProt ID mapping...")
            download_ensembl_biomart_mapping(
                osp.join(self.raw_dir, self.ID_MAPPING_FILE)
            )
            print("ID mapping download complete")

        if not osp.exists(osp.join(self.raw_dir, self.HIGHPPI_NETWORK_FILE)):
            print("Downloading HIGH-PPI network data from Google Drive...")
            success = download_folder_from_drive(
                self.HIGHPPI_GDRIVE_FOLDER, self.raw_dir, quiet=False
            )
            if not success:
                raise RuntimeError(
                    "Failed to download HIGH-PPI data from Google Drive"
                )
            print("HIGH-PPI download complete")

        # Final verification
        missing_files = [
            fname
            for fname in self.raw_file_names
            if not osp.exists(osp.join(self.raw_dir, fname))
        ]

        if missing_files:
            raise FileNotFoundError(
                f"Failed to download required files: {missing_files}. "
            )

    def process(self):
        """Build simplicial complex: HIGH-PPI edges + CORUM complexes."""
        print("\n" + "=" * 70)
        print(
            "Building PPI simplicial complex from HIGH-PPI and CORUM datasets"
        )
        print("=" * 70)

        # Load Ensembl <-> UniProt ID mapping
        mapping_path = osp.join(self.raw_dir, self.ID_MAPPING_FILE)
        self.ensembl_to_uniprot, self.uniprot_to_ensembl = load_id_mapping(
            mapping_path
        )

        # Load HIGH-PPI network with interaction types and confidence scores
        highppi_path = osp.join(self.raw_dir, self.HIGHPPI_NETWORK_FILE)
        self.highppi_edges, self.all_proteins = load_highppi_network(
            highppi_path, self.INTERACTION_TYPES
        )

        # Load CORUM complexes, filter to SHS27k proteins
        corum_path = osp.join(self.raw_dir, self.CORUM_COMPLEXES_FILE)
        self.corum_complexes = load_corum_complexes(
            corum_path,
            self.all_proteins,
            self.ensembl_to_uniprot,
            self.uniprot_to_ensembl,
            self.min_complex_size,
            self.max_complex_size,
        )

        self._load_splits()

        print("Building simplicial complex...")
        sc, edge_data, cell_data = build_simplicial_complex_with_features(
            self.all_proteins,
            self.highppi_edges,
            self.corum_complexes,
            self.min_complex_size,
            self.max_rank,
        )

        print("Generating negative samples...")
        edge_data, cell_data = generate_negative_samples(
            sc, edge_data, cell_data, self.all_proteins, self.neg_ratio
        )

        print("Extracting features and connectivity...")
        x_dict, labels_dict = build_data_features_and_labels(
            sc,
            edge_data,
            cell_data,
            self.target_ranks,
            self.max_rank,
            edge_task=self.edge_task,
        )

        # Get connectivity
        connectivity = get_complex_connectivity(
            sc, self.max_rank, signed=False
        )

        # Build Data object
        protein_list = sorted(list(sc.nodes))
        protein_to_idx = {p: i for i, p in enumerate(protein_list)}
        n_edges = len(list(sc.skeleton(1)))

        data = Data(
            **x_dict,
            **connectivity,
            **labels_dict,
            num_proteins=len(protein_list),
            num_edges=n_edges,
            num_complexes=len(self.corum_complexes),
            protein_to_idx=protein_to_idx,
        )

        # Add x and y for compatibility with generic tests
        # x_0 uses one-hot encoding, so dimension equals number of proteins
        data.x = x_dict.get(
            "x_0", torch.zeros(0, len(protein_list))
        )  # TODO: This data will not be used for node-level prediction
        if (
            self.target_ranks
            and f"cell_labels_{self.target_ranks[0]}" in labels_dict
        ):
            data.y = labels_dict[f"cell_labels_{self.target_ranks[0]}"]
        else:
            data.y = torch.zeros(len(protein_list), dtype=torch.long)

        # Add official splits if available
        if self.official_splits:
            data.train_mask = torch.tensor(
                self.official_splits.get("train_index", []), dtype=torch.long
            )
            data.val_mask = torch.tensor(
                self.official_splits.get("valid_index", []), dtype=torch.long
            )

        # Save processed data
        print("Saving processed data...")
        self.data, self.slices = self.collate([data])
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )

        print("\n" + "=" * 70)
        print("✅ PROCESSING COMPLETE!")
        print("📊 Dataset statistics:")
        print(f"   - Proteins (0-cells): {len(self.all_proteins)}")
        print(f"   - Labeled edges (1-cells): {len(self.highppi_edges)}")
        print(f"   - CORUM complexes: {len(self.corum_complexes)}")
        print(f"📁 Saved to: {self.processed_paths[0]}")
        print(f"💾 Size: {osp.getsize(self.processed_paths[0]) / 1e6:.1f} MB")
        print("=" * 70 + "\n")

    @property
    def data_list(self):
        """Return list of data objects for TopoBench compatibility.

        Returns
        -------
        list
            List containing single data object (transductive setting).
        """
        return [self._data]

    def get_data_dir(self):
        """Return data directory for split file storage.

        Returns
        -------
        str
            Path to data directory.
        """
        return self.root

    @property
    def split_idx(self):
        """Return train/val/test split indices for split_type='fixed'.

        Used when config has split_type='fixed'. Returns HIGH-PPI's official
        train/val split if it was successfully loaded, otherwise None.

        Returns
        -------
        dict or None
            Dictionary with 'train', 'valid', 'test' keys containing indices,
            or None (triggers random/k-fold splitting based on split_type).
        """
        if hasattr(self, "official_splits") and self.official_splits:
            return {
                "train": np.array(self.official_splits.get("train_index", [])),
                "valid": np.array(self.official_splits.get("val_index", [])),
                "test": np.array(self.official_splits.get("val_index", [])),
            }
        return None

    # TODO: This is not working yet
    def _load_splits(self):
        """Load official train/val split indices from HIGH-PPI.

        Loads splits into self.official_splits which will be used if split_type='fixed'.
        Fails silently if splits are not available (random/k-fold will be used instead).
        """
        split_path = osp.join(self.raw_dir, "train_val_split_1.json")
        if not osp.exists(split_path):
            return

        try:
            with open(split_path) as f:
                content = f.read().strip()
                if len(content) >= 10:  # Basic validation
                    self.official_splits = json.loads(content)
                    print(
                        "Official train/val splits available (use split_type='fixed' to use them)"
                    )
        except (json.JSONDecodeError, Exception):
            pass  # Silently ignore - will use random/k-fold splitting
