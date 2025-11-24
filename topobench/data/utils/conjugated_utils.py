"""Utilities for conjugated structures dataset."""

import numpy as np
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem


def contains_conjugated_bond(
    mol: Chem.Mol,
) -> tuple[bool, Chem.ResonanceMolSupplier]:
    """Check if a molecule contains conjugated bonds.

    Parameters
    ----------
    mol : Chem.Mol
        Input molecule.

    Returns
    -------
    tuple[bool, Chem.ResonanceMolSupplier]
        Tuple containing a boolean indicating if the molecule has conjugated bonds
        and the ResonanceMolSupplier object.
    """
    reso = Chem.ResonanceMolSupplier(mol)
    num_he = reso.GetNumConjGrps()
    return num_he > 0, reso


def he_conj(mol: Chem.Mol) -> list[list]:
    """Get incidence list of conjugated structures in a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        Input molecule.

    Returns
    -------
    list[list]
        Incidence list of conjugated structures.
    """
    num_atom = mol.GetNumAtoms()
    reso = Chem.ResonanceMolSupplier(mol)
    num_he = reso.GetNumConjGrps()
    if num_he == 0:
        return []

    incidence_list: list[list] = [[] for _ in range(num_he)]
    for i in range(num_atom):
        _conj = reso.GetAtomConjGrpIdx(i)
        if _conj > -1 and _conj < num_he:
            incidence_list[_conj].append(i)
    return incidence_list


def edge_order(e_idx):
    """Get the order (cardinality) of each edge.

    Parameters
    ----------
    e_idx : list
        List of edge indices.

    Returns
    -------
    list
        List containing the count of each edge index.
    """
    return [e_idx.count(i) for i in range(len(set(e_idx)))]


def get_hypergraph_data_from_smiles(
    smiles_string: str,
) -> tuple[list, list[list], list]:
    """Convert a SMILES string to hypergraph Data object.

    Parameters
    ----------
    smiles_string : str
        SMILES string.

    Returns
    -------
    tuple[list, list[list], list]
        Tuple containing atom feature vectors, incidence list, and bond feature vectors.
    """
    try:
        mol = Chem.MolFromSmiles(smiles_string)
    except TypeError as err:
        print(f"Invalid SMILES: {smiles_string}")
        raise TypeError from err

    # atoms
    atom_fvs = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]

    # bonds
    num_bond_features = 1  # bond type (single, double, triple, conjugated)
    bonds = mol.GetBonds()
    if len(bonds) > 0:  # mol has bonds
        incidence_list: list[list] = [[] for _ in range(len(bonds))]
        bond_fvs: list[list] = [[] for _ in range(len(bonds))]
        for i, bond in enumerate(bonds):
            incidence_list[i] = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
            bond_type = bond_to_feature_vector(bond)[0]
            bond_fvs[i] = [bond_type]

    else:  # mol has no bonds
        print(f"Invalid SMILES: {smiles_string}")
        incidence_list: list[list] = []
        bond_fvs: list[list] = []
        return (atom_fvs, incidence_list, bond_fvs)

    # hyperedges for conjugated bonds
    he_incidence_list = he_conj(mol)  # [[3,4,5], [0,1,2]]
    if len(he_incidence_list) != 0:
        incidence_list.extend(
            he_incidence_list
        )  # [[0,1], [1,2], [3,4], [[3,4,5], [0,1,2]]
        bond_fvs += len(he_incidence_list) * [num_bond_features * [5]]

    return (atom_fvs, incidence_list, bond_fvs)


def create_incidence_matrix(incidence_list, num_nodes):
    """Create an incidence matrix from an incidence list.

    Parameters
    ----------
    incidence_list : list[list]
        List of edges, where each edge is a list of node indices.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    np.ndarray
        Incidence matrix.
    """
    num_edges = len(incidence_list)
    incidence_matrix = np.zeros((num_nodes, num_edges), dtype=int)
    for edge_idx, nodes in enumerate(incidence_list):
        for node in nodes:
            incidence_matrix[node, edge_idx] = 1
    return incidence_matrix
