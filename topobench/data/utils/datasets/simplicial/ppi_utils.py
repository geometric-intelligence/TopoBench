"""Refactored PPI utilities with cleaner separation of concerns.

Key improvements:
1. Separate topology building from feature/label assignment
2. Consistent data types (no mixed list/float/int in cell_labels)
3. Single-pass iterations (no redundant loops)
4. Clear data flow: topology → features → labels → tensors
"""

import os
import random
from itertools import combinations

import pandas as pd
import torch
from toponetx.classes import SimplicialComplex


def load_id_mapping(
    mapping_path: str,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Load Ensembl ↔ UniProt ID mapping.

    Parameters
    ----------
    mapping_path : str
        Path to ensp_uniprot.txt mapping file.

    Returns
    -------
    ensembl_to_uniprot : dict
        Mapping from Ensembl IDs to UniProt IDs.
    uniprot_to_ensembl : dict
        Reverse mapping (UniProt to list of Ensembl IDs).

    Raises
    ------
    FileNotFoundError
        If mapping file does not exist.
    """
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            f"ID mapping file not found: {mapping_path}. "
            "This file is required to map between Ensembl and UniProt IDs for CORUM complexes."
        )

    ensembl_to_uniprot = {}
    uniprot_to_ensembl = {}

    with open(mapping_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) >= 2:
                ensembl_id = parts[0].strip()
                uniprot_id = parts[1].strip()

                if uniprot_id and uniprot_id not in ("Noneid", "None"):
                    ensembl_to_uniprot[ensembl_id] = uniprot_id

                    if uniprot_id not in uniprot_to_ensembl:
                        uniprot_to_ensembl[uniprot_id] = []
                    uniprot_to_ensembl[uniprot_id].append(ensembl_id)

    return ensembl_to_uniprot, uniprot_to_ensembl


def load_highppi_network(
    file_path: str, interaction_types: list[str]
) -> tuple[list[tuple], set[str]]:
    """Load HIGH-PPI network with interaction types and confidence scores.

    Parameters
    ----------
    file_path : str
        Path to HIGH-PPI SHS27k file.
    interaction_types : list
        List of valid interaction type names.

    Returns
    -------
    highppi_edges : list
        List of (p1, p2, interaction_vector, score) tuples.
    all_proteins : set
        Set of all protein IDs in the network.

    Raises
    ------
    FileNotFoundError
        If HIGH-PPI network file does not exist.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"HIGH-PPI network file not found: {file_path}. "
            "This file contains the protein-protein interaction network."
        )

    df = pd.read_csv(file_path, sep="\t")

    # Rename columns to more intuitive names
    df = df.rename(
        columns={
            "item_id_a": "protein_1",
            "item_id_b": "protein_2",
            "mode": "interaction_type",
            "score": "confidence_score",
        }
    )

    edge_labels = {}
    edge_scores = {}
    all_proteins = set()

    for _, row in df.iterrows():
        p1 = str(row["protein_1"]).strip()
        p2 = str(row["protein_2"]).strip()

        all_proteins.add(p1)
        all_proteins.add(p2)

        edge_key = tuple(sorted([p1, p2]))

        score = float(row["confidence_score"]) / 1000.0
        if edge_key not in edge_scores:
            edge_scores[edge_key] = 0.0
        edge_scores[edge_key] = max(edge_scores[edge_key], score)

        interaction_type = str(row["interaction_type"]).strip()
        if edge_key not in edge_labels:
            edge_labels[edge_key] = [0] * 7
        if interaction_type in interaction_types:
            idx_type = interaction_types.index(interaction_type)
            edge_labels[edge_key][idx_type] = 1

    highppi_edges = [
        (p1, p2, labels, edge_scores[(p1, p2)])
        for (p1, p2), labels in edge_labels.items()
    ]

    return highppi_edges, all_proteins


def load_corum_complexes(
    file_path: str,
    all_proteins: set[str],
    ensembl_to_uniprot: dict[str, str],
    uniprot_to_ensembl: dict[str, list[str]],
    min_size: int,
    max_size: int,
) -> list[set[str]]:
    """Load and filter CORUM protein complexes.

    Parameters
    ----------
    file_path : str
        Path to CORUM allComplexes.txt file.
    all_proteins : set
        Set of proteins in the network (for filtering).
    ensembl_to_uniprot : dict
        Ensembl to UniProt ID mapping.
    uniprot_to_ensembl : dict
        UniProt to Ensembl ID mapping.
    min_size : int
        Minimum complex size.
    max_size : int
        Maximum complex size.

    Returns
    -------
    list[set[str]]
        List of sets, each containing Ensembl protein IDs.

    Raises
    ------
    FileNotFoundError
        If CORUM file does not exist.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"CORUM complexes file not found: {file_path}. "
            "This file is required to load experimentally validated protein complexes."
        )

    df = pd.read_csv(file_path, sep="\t", low_memory=False)

    # CORUM uses 'subunits_uniprot_id' column for UniProt IDs
    if "subunits_uniprot_id" not in df.columns:
        raise ValueError(
            f"Expected column 'subunits_uniprot_id' not found in CORUM file. Available columns: {df.columns.tolist()}"
        )

    # Map proteins to UniProt
    shs27k_uniprot = {
        ensembl_to_uniprot[eid]
        for eid in all_proteins
        if eid in ensembl_to_uniprot
    }

    corum_complexes = []
    for _, row in df.iterrows():
        subunits_str = row["subunits_uniprot_id"]
        if pd.isna(subunits_str):
            continue

        proteins_uniprot = {
            p.strip() for p in subunits_str.split(";") if p.strip()
        }
        proteins_in_network = proteins_uniprot & shs27k_uniprot

        if not (min_size <= len(proteins_in_network) <= max_size):
            continue

        # Convert to Ensembl IDs
        ensembl_complex = set()
        for uniprot_id in proteins_in_network:
            if uniprot_id in uniprot_to_ensembl:
                for ensembl_id in uniprot_to_ensembl[uniprot_id]:
                    if ensembl_id in all_proteins:
                        ensembl_complex.add(ensembl_id)
                        break

        corum_complexes.append(ensembl_complex)

    return corum_complexes


def build_simplicial_complex_with_features(
    all_proteins: set[str],
    highppi_edges: list[tuple],
    corum_complexes: list[set[str]],
    min_complex_size: int,
    max_rank: int,
) -> tuple[SimplicialComplex, dict, dict]:
    """Build simplicial complex with topology and metadata from PPI data.

    Constructs the complex structure and tracks cell data.

    Parameters
    ----------
    all_proteins : set
        Set of all protein IDs.
    highppi_edges : list
        List of (p1, p2, interaction_vector, score) tuples.
    corum_complexes : list
        List of protein complexes as sets.
    min_complex_size : int
        Minimum complex size to include.
    max_rank : int
        Maximum rank to consider.

    Returns
    -------
    sc : SimplicialComplex
        The constructed simplicial complex.
    edge_data : dict
        Edge features {edge_tuple: tensor([7 interaction types, 1 confidence])}.
    cell_data : dict
        Binary labels per rank {rank: {cell_tuple: {-1, 1}}}.
    """
    sc = SimplicialComplex()
    edge_data = {}  # {edge_tuple: tensor([7 types + 1 score])}
    cell_data = {}  # {rank: {cell_tuple: -1 or 1}}

    # Add 0-cells (proteins)
    for protein in sorted(all_proteins):
        sc.add_simplex([protein])

    # Add 1-cells (HIGH-PPI edges) with features
    for p1, p2, interaction_vector, score in highppi_edges:
        edge_tuple = tuple(sorted([p1, p2]))
        sc.add_simplex([p1, p2])

        # Store 8-dim feature vector (7 interaction types + 1 confidence)
        edge_data[edge_tuple] = torch.tensor(
            interaction_vector + [2 * score - 1],
            dtype=torch.float,  # Convert confidence score affinely: [0, 1] -> [-1, 1]
        )

    # Process CORUM complexes top-down (largest first)
    # This ensures lower-rank CORUM complexes can override negative labels
    sorted_complexes = sorted(corum_complexes, key=len, reverse=True)

    for complex_proteins in sorted_complexes:
        # Filter by size and protein membership
        if len(complex_proteins) < min_complex_size:
            continue
        if not complex_proteins.issubset(all_proteins):
            continue

        complex_tuple = tuple(sorted(complex_proteins))
        rank = len(complex_tuple) - 1

        if rank > max_rank:
            continue

        # Add complex to simplicial complex (automatically adds all faces)
        sc.add_simplex(list(complex_tuple))

        if rank == 1:
            edge_data[complex_tuple] = torch.tensor(
                [0, 0, 0, 0, 0, 0, 0, 1.0], dtype=torch.float
            )
            continue

        # Mark this complex as positive
        if rank not in cell_data:
            cell_data[rank] = {}
        cell_data[rank][complex_tuple] = 1

        # Mark all proper sub-faces as negative (not real complexes themselves)
        # Only mark if not already labeled (top-down iteration handles overlaps)
        for sub_rank in range(1, rank):
            if sub_rank not in cell_data:
                cell_data[sub_rank] = {}

            for sub_face in combinations(complex_tuple, sub_rank + 1):
                sub_tuple = tuple(sorted(sub_face))
                if sub_rank == 1 and sub_tuple not in edge_data:
                    edge_data[sub_tuple] = torch.tensor(
                        [0, 0, 0, 0, 0, 0, 0, -1.0], dtype=torch.float
                    )
                elif sub_tuple not in cell_data[sub_rank]:
                    cell_data[sub_rank][sub_tuple] = -1

    return sc, edge_data, cell_data


def generate_negative_samples(
    sc: SimplicialComplex,
    edge_data: dict[tuple, torch.Tensor],
    cell_data: dict[int, dict[tuple, int]],
    all_proteins: set[str],
    neg_ratio: float,
) -> tuple[dict[tuple, torch.Tensor], dict[int, dict[tuple, int]]]:
    """Generate negative samples proportionally across ranks.

    Parameters
    ----------
    sc : SimplicialComplex
        Current simplicial complex.
    edge_data : dict
        Edge features to update with negative edges.
    cell_data : dict
        Existing binary data per rank {rank: {cell_tuple: {-1, 1}}}.
    all_proteins : set
        Set of all protein IDs.
    neg_ratio : float
        Ratio of negative to positive samples.

    Returns
    -------
    edge_data : dict
        Updated with negative edge features.
    cell_data : dict
        Updated data with negative samples added (value=-1).
    """
    random.seed(42)

    all_proteins_list = list(all_proteins)

    # Generate negatives from highest to rank 2 (top-down)
    # Note that edges already have score in [-1, 1] so no need to add more negatives
    for rank in range(sc.dim, 1, -1):
        # Count positive and existing negative samples at this rank
        n_positive = 0
        n_existing_negative = 0
        if rank in cell_data:
            n_positive = sum(
                1 for label in cell_data[rank].values() if label == 1
            )
            n_existing_negative = sum(
                1 for label in cell_data[rank].values() if label == -1
            )
        if n_positive == 0:
            continue

        # Calculate how many more negatives we need (accounting for existing ones)
        n_negative_target = int(n_positive * neg_ratio)
        n_negative_needed = n_negative_target - n_existing_negative
        if n_negative_needed <= 0:
            # Enough negatives
            continue

        # Get existing cells at this rank
        existing_cells = set(cell_data.get(rank, {}).keys())
        existing_cells.update(
            tuple(sorted(cell)) for cell in sc.skeleton(rank)
        )

        # Generate random cells until we have enough negatives
        negatives_added = 0
        max_attempts = n_negative_needed * 100

        for _ in range(max_attempts):
            if negatives_added >= n_negative_needed:
                break

            # Sample random proteins for this rank
            sampled = random.sample(all_proteins_list, rank + 1)
            cell_tuple = tuple(sorted(sampled))

            # Only add if it doesn't exist yet
            if cell_tuple not in existing_cells:
                # Add to complex (automiatically adds faces)
                sc.add_simplex(list(cell_tuple))

                if rank not in cell_data:
                    cell_data[rank] = {}
                # Label as negative
                cell_data[rank][cell_tuple] = -1

                # Mark all proper sub-faces as negative (if they don't exist yet)
                for sub_rank in range(1, rank):
                    if sub_rank not in cell_data:
                        cell_data[sub_rank] = {}

                    for sub_face in combinations(cell_tuple, sub_rank + 1):
                        sub_tuple = tuple(sorted(sub_face))
                        if sub_rank == 1 and sub_tuple not in edge_data:
                            # Edge: create feature vector
                            edge_data[sub_tuple] = torch.tensor(
                                [0, 0, 0, 0, 0, 0, 0, -1.0], dtype=torch.float
                            )
                        elif sub_tuple not in cell_data[sub_rank]:
                            # Higher-order sub-face: mark as negative
                            cell_data[sub_rank][sub_tuple] = -1

                existing_cells.add(cell_tuple)
                negatives_added += 1

        # Calculate total negatives (existing + newly added)
        n_total_negative = n_existing_negative + negatives_added
        print(
            f"  Rank {rank}: {n_positive} positive, {n_total_negative} negative samples"
        )

    return edge_data, cell_data


def build_data_features_and_labels(
    sc: SimplicialComplex,
    edge_data: dict[tuple, torch.Tensor],
    cell_data: dict[int, dict[tuple, int]],
    target_ranks: list[int],
    max_rank: int,
    edge_task: str = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Create feature and label tensors for all ranks.

    Parameters
    ----------
    sc : SimplicialComplex
        The constructed simplicial complex.
    edge_data : dict
        Edge features {edge_tuple: tensor([7 interaction types, 1 confidence])}.
    cell_data : dict
        Binary labels {rank: {cell_tuple: {-1, 1}}}.
    target_ranks : list
        Ranks to predict on.
    max_rank : int
        Maximum rank in configuration.
    edge_task : str, optional
        Edge prediction task: "interaction_type" or "score".
        Only used if rank 1 is in target_ranks.

    Returns
    -------
    x_dict : dict
        Features per rank {f"x_{rank}": tensor}.
    labels_dict : dict
        Labels per target rank {f"cell_labels_{rank}": tensor}.
    """
    actual_max_rank = sc.dim
    x_dict = {}
    labels_dict = {}

    # For each rank: build features and labels
    for rank in range(min(max_rank, actual_max_rank) + 1):
        cells = list(sc.skeleton(rank))
        n_cells = len(cells)

        if n_cells == 0:
            dim = 1 if rank == 0 else (8 if rank == 1 else 1)
            x_dict[f"x_{rank}"] = torch.zeros(0, dim)
            continue

        is_target = rank in target_ranks

        match rank:
            case 0:
                # Nodes: one-hot encoding
                # TODO: Use richer embeddings (ESM, structure, GO annotations)
                x_dict["x_0"] = torch.eye(n_cells)

            case 1:
                # Edges: 8-dim features (7 interaction types + 1 confidence)
                features = []
                labels = [] if is_target else None

                for edge in cells:
                    edge_tuple = tuple(sorted(edge))
                    feat_vec = edge_data[edge_tuple]

                    if is_target:
                        # Split features/labels based on edge_task
                        if edge_task == "interaction_type":
                            labels.append(
                                feat_vec[:7]
                            )  # First 7 dims = labels
                            features.append(
                                feat_vec[7:8]
                            )  # Last dim = feature
                        elif edge_task == "score":
                            # Convert score back from [-1, 1] to [0, 1] for standard regression
                            score_normalized = (feat_vec[7:8] + 1) / 2
                            labels.append(score_normalized)
                            features.append(
                                feat_vec[:7]
                            )  # First 7 dims = features
                    else:
                        # Not a target rank: use all 8 dims as features
                        features.append(feat_vec)

                x_dict["x_1"] = torch.stack(features)

                if is_target:
                    labels_dict["cell_labels_1"] = torch.stack(labels)

            case _:
                # Higher-order cells
                features = []
                labels = [] if is_target else None

                for cell in cells:
                    cell_tuple = tuple(sorted(cell))
                    binary_existence_val = cell_data[rank][cell_tuple]

                    # Features: 0 for target, {-1,+1} for non-target TODO: Bit unsure about this.
                    # Labels (only target): {0, 1} for PyTorch CrossEntropyLoss
                    # TODO: Maybe we should pass some labels as features for true transductivity/semi-supervision?
                    if is_target:
                        # Target rank: features are 0, labels are in {-1, 1}
                        features.append(torch.zeros(1, dtype=torch.float))
                        labels.append(
                            torch.tensor(
                                [binary_existence_val], dtype=torch.float
                            )
                        )
                    else:
                        # Non-target rank: use labels as features {-1, +1}
                        features.append(
                            torch.tensor(
                                [binary_existence_val], dtype=torch.float
                            )
                        )

                x_dict[f"x_{rank}"] = torch.stack(features)

                if is_target:
                    labels_tensor = torch.stack(labels).squeeze()
                    # Convert {-1, +1} → {0, 1} for PyTorch CrossEntropyLoss
                    labels_mapped = ((labels_tensor + 1) / 2).long()
                    labels_dict[f"cell_labels_{rank}"] = labels_mapped
                    n_pos = (labels_mapped == 1).sum().item()
                    n_neg = (labels_mapped == 0).sum().item()
                    print(
                        f"  Rank {rank}: {n_pos} positive, {n_neg} negative labels"
                    )

    # Add empty features for non-existent ranks
    for rank in range(actual_max_rank + 1, max_rank + 1):
        dim = 1 if rank == 0 else (8 if rank == 1 else 1)
        x_dict[f"x_{rank}"] = torch.zeros(0, dim)

    return x_dict, labels_dict
