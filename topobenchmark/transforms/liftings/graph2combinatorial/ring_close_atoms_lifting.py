import networkx as nx
import torch
import torch_geometric
from rdkit import Chem
from rdkit.Chem import Descriptors
from toponetx.classes import CellComplex

from modules.transforms.liftings.graph2cell.base import Graph2CellLifting
from modules.transforms.liftings.graph2combinatorial.base import (
    Graph2CombinatorialLifting,
)


class CombinatorialRingCloseAtomsLifting(Graph2CombinatorialLifting):
    r"""Lifts r-cell features to r+1-cells by molecular rings.
    This rings are obtained by chemical knowledge from rdkit.
    Moreover, the close atoms are defined as the atoms that
    are closer than a threshold distance. This atoms will be
    connected through an hyperedge.


    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.complex_dim = 2

        self.threshold_distance = kwargs.get("threshold_distance", 2)

    #######################################################
    ################### RINGS #############################
    #######################################################

    def get_rings(
        self, data: torch_geometric.data.Data | dict
    ) -> torch.Tensor:
        r"""Returns the ring information for each molecule.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch.Tensor
            The ring information.
        """

        # transform data to molecule and compute rings
        mol = self._generate_mol_from_data(data)
        rings = Chem.GetSymmSSSR(mol)

        return [list(ring) for ring in rings]

    def _generate_mol_from_data(
        self, data: torch_geometric.data.Data | dict
    ) -> Chem.Mol:
        r"""Converts the data to a molecule through the SMILES
            and removes the hydrogens.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        Chem.Mol
            The molecule.
        """
        # mol = Chem.MolFromSmiles(data.smiles)
        # return Chem.rdmolops.RemoveHs(mol)
        return Chem.MolFromSmiles(data.smiles)

    #######################################################
    ################ CLOSE ATOMS ##########################
    #######################################################

    def find_close_atom_groups(
        self, mol: Chem.Mol, data: torch_geometric.data.Data | dict
    ) -> list:
        r"""Finds the groups of atoms that are close to each other within a molecule.

        Parameters
        ----------
        mol : Chem.Mol

        Returns
        -------
        dict of list
            The list of close atom groups.

            Example:
            {
                "C(=O)O": [[0, 1, 2], [2, 3, 9]],
                "C(=O)N": [[3, 4, 5], [5, 6, 7]],
                ...
            }

        """
        # Get the indices of close atoms
        close_atoms = self.find_close_atoms(mol, data)

        # Get the connected components of the graph
        G = nx.Graph()
        G.add_edges_from(close_atoms)
        close_atom_groups = list(nx.connected_components(G))

        # Convert atom indices to SMILES
        substructures = {}
        for group in close_atom_groups:
            group_atoms = list(group)
            smiles = self.get_smiles_for_atoms(mol, group_atoms)
            substructures[smiles] = [list(group_atoms)]

        return substructures

    def find_close_atoms(
        self,
        mol: Chem.Mol,
        data: torch_geometric.data.Data | dict,
        # data: torch_geometric.data.Data | dict
    ) -> list:
        r"""Finds the atoms that are close to each other within a molecule.

        Parameters
        ----------
        data : torch_geometric.data.Data | dict
            The input data containing the positions of the atoms.

        Returns
        -------
        list
            The list of close atoms.
            Example:
            [(0, 1), (1, 2), ...]
        """
        # Get the distances between atoms excluding hydrogen atoms
        # data = self._remove_hydrogens(data)
        distance_matrix = self.get_distance_matrix(mol, data)

        # Get indices of atom pairs that are closer than the threshold
        num_atoms = distance_matrix.size(0)
        return [
            (i, j)
            for i in range(num_atoms)
            for j in range(i + 1, num_atoms)
            if distance_matrix[i, j] < float(self.threshold_distance)
        ]

    def get_distance_matrix(
        self,
        mol: Chem.Mol,
        data: torch_geometric.data.Data | dict,
        # data: torch_geometric.data.Data | dict
    ) -> torch.Tensor:
        r"""Computes the pairwise distances between atoms in a molecule.

        Parameters
        ----------
        data : torch_geometric.data.Data | dict
            The input data containing the positions of the atoms.

        Returns
        -------
        torch.Tensor
            The pairwise distance matrix.
        """
        # Get the positions of the atoms from molecule
        remaining_atom_indices = [atom.GetIdx() for atom in mol.GetAtoms()]
        pos = data.pos[remaining_atom_indices]

        # Compute the pairwise distances between the atoms
        return torch.cdist(pos, pos, p=2)

    def get_smiles_for_atoms(self, mol, atom_indices):
        """Generates SMILES for a group of atoms specified by their indices."""
        atom_map = {i: atom.GetIdx() for i, atom in enumerate(mol.GetAtoms())}
        atoms_to_use = [atom_map[idx] for idx in atom_indices]
        return Chem.MolFragmentToSmiles(mol, atoms_to_use)

    #######################################################
    ################### ATTRIBUTES ########################
    #######################################################

    def get_atom_attributes(self, mol: Chem.Mol) -> dict:
        r"""Returns the atom attributes for each molecule.

        Parameters
        ----------
        mol : Chem.Mol
            The molecule.

        Returns
        -------
        dict
            The atom attributes with the atom id.
            Example:
            {
                0: {"atomic_num": 6, "symbol": "C", "degree": 3, "formal_charge": 0, "hybridization": "SP
                    "is_aromatic": False, "mass": 12.01, "chirality": "CHI_UNSPECIFIED"},

                1: {"atomic_num": 6, "symbol": "C", "degree": 2, "formal_charge": 0, "hybridization": "SP
                    "is_aromatic": False, "mass": 12.01, "chirality": "CHI_UNSPECIFIED"},
                ...
            }
        """
        attr = {}
        for atom in mol.GetAtoms():
            attr[atom.GetIdx()] = {
                "atomic_num": atom.GetAtomicNum(),
                "symbol": atom.GetSymbol(),
                "degree": atom.GetDegree(),
                "formal_charge": atom.GetFormalCharge(),
                "hybridization": str(atom.GetHybridization()),
                "is_aromatic": atom.GetIsAromatic(),
                "mass": atom.GetMass(),
                "chirality": atom.GetChiralTag(),
            }
        return attr

    def get_bond_attributes(self, mol: Chem.Mol) -> torch.Tensor:
        r"""Returns the bond attributes for each molecule.

        Parameters
        ----------
        mol : Chem.Mol
            The molecule.

        Returns
        -------
        dict
            The bond attributes with the bond id.
            Example:
            {
                (0, 1): {"bond_type": "SINGLE", "is_conjugated": False, "is_stereo": False},
                (1, 2): {"bond_type": "DOUBLE", "is_conjugated": False, "is_stereo": False},
                ...
            }
        """
        attr = {}
        for bond in mol.GetBonds():
            attr[(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())] = {
                "bond_type": str(bond.GetBondType()),
                "is_conjugated": bond.GetIsConjugated(),
                "is_stereo": bond.GetStereo(),
            }
        return attr

    def get_ring_attributes(self, mol: Chem.Mol, rings: list) -> dict:
        r"""Returns the ring attributes for each molecule.

        • Ring Size: Number of atoms that form the ring.
        • Aromaticity: A binary feature telling whether the ring is aromatic.
        • Has Heteroatom: A binary feature telling whether the ring contains a heteroatom,
          which is an atom other than carbon, such as nitrogen, oxygen, or sulfur.
        • Saturation: A binary feature telling whether the ring is saturated, meaning all
          the bonds between carbon atoms are single bonds.
        • Hydrophobicity: Tendency of the functional group to repel water.
             Compute based in logp.
        • Electrophilicity: Tendency of the functional group to accept electrons during a chemical reaction.
             Compute based on the number of hydrogen bonds acceptors.
        • Nucleophilicity: Tendency of the functional group to donate electrons during a chemical reaction.
             High nucleophilicity means the functional group readily donates electrons, affecting its reactivity
             with electrophiles. Compute based in number of hydrogen bond donors.
        • Polarity: Distribution of electron density within the functional group. High polarity can significantly
             affect solubility, reactivity, and interactions with other molecules.
             Compute based on the tpsa.
        Parameters
        ----------
        mol : Chem.Mol
            The molecule.

        rings : list
            The rings of the molecule.
            Example: [[0, 1, 2], [1, 2, 3], ...]

        Returns
        -------
        dict
            The ring attributes.
            Example:
            {
                (0, 1, 2): {"ring_size": 3, "aromaticity": False, "has_heteroatom": False,
                            "saturation": True, "hydrophobicity": 0.0, "electrophilicity": 0,
                            "nucleophilicity": 0, "polarity": 0.0},
                (1, 2, 3): {"ring_size": 3, "aromaticity": False, "has_heteroatom": False,
                            "saturation": True, "hydrophobicity": 0.0, "electrophilicity": 0,
                            "nucleophilicity": 0, "polarity": 0.0},
                ...
            }
        """
        attr = {}
        for ring in rings:
            ring_size = len(ring)
            mol_ring = Chem.MolFromSmiles(
                "".join(
                    [mol.GetAtomWithIdx(atom).GetSymbol() for atom in ring]
                )
            )

            try:
                # Attempt to sanitize the molecule
                Chem.SanitizeMol(mol_ring)
            except Exception:
                # Handle specific exception (if needed) or log the error
                # print(f"Failed to sanitize molecule for SMILES '{fg}'")
                continue

            aromaticity = all(
                [atom.GetIsAromatic() for atom in mol_ring.GetAtoms()]
            )
            has_heteroatom = any(
                [atom.GetAtomicNum() != 6 for atom in mol_ring.GetAtoms()]
            )
            saturation = all(
                [
                    bond.GetBondType() == Chem.rdchem.BondType.SINGLE
                    for bond in mol_ring.GetBonds()
                ]
            )
            hydrophobicity = Descriptors.MolLogP(mol_ring)
            electrophilicity = Descriptors.NumHAcceptors(mol_ring)
            nucleophilicity = Descriptors.NumHDonors(mol_ring)
            polarity = Descriptors.TPSA(mol_ring)

            attr[tuple(ring)] = {
                "ring_size": ring_size,
                "aromaticity": aromaticity,
                "has_heteroatom": has_heteroatom,
                "saturation": saturation,
                "hydrophobicity": hydrophobicity,
                "electrophilicity": electrophilicity,
                "nucleophilicity": nucleophilicity,
                "polarity": polarity,
            }

        return attr

    def get_close_atoms_attributes(
        self, mol: Chem.Mol, close_atoms: dict
    ) -> dict:
        r"""Returns the functional groups attributes for each molecule.

        • Conjugation: A binary feature telling whether the functional group
            is part of a conjugated system, which involves alternating double and single bonds.
        • Hydrophobicity: Tendency of the functional group to repel water.
             Compute based in logp.
        • Electrophilicity: Tendency of the functional group to accept electrons during a chemical reaction.
             Compute based on the number of hydrogen bonds acceptors.
        • Nucleophilicity: Tendency of the functional group to donate electrons during a chemical reaction.
             High nucleophilicity means the functional group readily donates electrons, affecting its reactivity
             with electrophiles. Compute based in number of hydrogen bond donors.
        • Polarity: Distribution of electron density within the functional group. High polarity can significantly
             affect solubility, reactivity, and interactions with other molecules.
             Compute based on the tpsa.

        Parameters
        ----------
        mol : Chem.Mol
            The molecule.

        close_atoms : list
            The close atoms groups of the molecule.
            Example:
            {
                "C(=O)O": [[0, 1, 2], [2, 3, 9]],
                "C(=O)N": [[3, 4, 5], [5, 6, 7]],
                ...
            }

        Returns
        -------
        dict
            The functional groups attributes.
            Example:
            {
                (0, 1): {"close_atoms_group": "C(=O)", "num_atoms": 2, "conjugation": False,
                            "hydrophobicity": 0.0, "electrophilicity": 0, "nucleophilicity": 0, "polarity": 0.0},
                (1, 2, 4): {"close_atoms_group": "C(=O)O", "num_atoms": 3, "conjugation": False,
                            "hydrophobicity": 0.0, "electrophilicity": 0, "nucleophilicity": 0, "polarity": 0.0},
                ...
            }
        """
        attr = {}
        for fg, groups in close_atoms.items():
            fg_mol = Chem.MolFromSmarts(fg)
            if fg_mol is None:
                # Skip processing this SMILES string if it cannot be converted
                continue

            try:
                # Attempt to sanitize the molecule
                Chem.SanitizeMol(fg_mol)
            except Exception:
                # Handle specific exception (if needed) or log the error
                # print(f"Failed to sanitize molecule for SMILES '{fg}'")
                continue

            conjugation = all(
                [bond.GetIsConjugated() for bond in fg_mol.GetBonds()]
            )
            hydrophobicity = Descriptors.MolLogP(fg_mol)
            electrophilicity = Descriptors.NumHAcceptors(fg_mol)
            nucleophilicity = Descriptors.NumHDonors(fg_mol)
            polarity = Descriptors.TPSA(fg_mol)
            for group in groups:
                attr[tuple(group)] = {
                    "functional_group": fg_mol,
                    "num_atoms": len(group),
                    "conjugation": conjugation,
                    "hydrophobicity": hydrophobicity,
                    "electrophilicity": electrophilicity,
                    "nucleophilicity": nucleophilicity,
                    "polarity": polarity,
                }

        return attr

    #######################################################
    ################### LIFT ##############################
    #######################################################

    def lift_topology(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        r"""Lifts the topology of a graph to a combinatorial complex.
        Following the steps:
        1. Convert the data to a molecule and remove the hydrogens.
        2. Generate the graph from the molecule. Take into account that this graph
            will not be the same as the starting one, since the hydrogens are removed.
        3. Generate the cell complex from the graph adding the rings as 2-cells.
        4. Group the close atoms and add them as hyperedges.

        Parameters
        ----------
        data : torch_geometric.data.Data | dict
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data | dict
            The lifted data."""

        # Combinaotrial complex is a combination of a cell complex and a hypergraph
        # First create a cell complex and then add hyperedges
        G = self._generate_graph_from_data(data)
        ccc = CellComplex(G)

        # add rings as 2-cells
        rings = self.get_rings(data)
        for ring in rings:
            ccc.add_cell(ring, rank=self.complex_dim)

        mol = self._generate_mol_from_data(data)

        # add atom attributes
        atom_attributes = self.get_atom_attributes(mol)
        ccc.set_cell_attributes(atom_attributes, rank=0)

        # add bond attributes
        bond_attributes = self.get_bond_attributes(mol)
        ccc.set_cell_attributes(bond_attributes, rank=1)

        # add ring attributes
        ring_attributes = self.get_ring_attributes(mol, rings)
        ccc.set_cell_attributes(ring_attributes, rank=2)

        # Create the lifted topology dict for the cell complex
        ccc_lifted_topology = Graph2CellLifting._get_lifted_topology(
            self, ccc, G
        )

        # Hypergraph stuff
        # add close atoms as hyperedges (rank = 1)
        close_atoms = self.find_close_atom_groups(mol, data)
        list_close_atoms = [
            item for sublist in close_atoms.values() for item in sublist
        ]
        edges = [[edge[0], edge[1]] for edge in G.edges]

        # Hyperedges can be edges or list_close_atoms
        hyperedges = edges + list_close_atoms
        # check if there are repeated hyperedges
        # if there are, remove them
        hyperedges = list(set([tuple(sorted(edge)) for edge in hyperedges]))
        hyperedges = sorted(hyperedges)
        # sort the hyperedges
        num_hyperedges = len(hyperedges)

        # create incidence matrix for hyperedges
        incidence_hyperedges = torch.zeros(data.num_nodes, num_hyperedges)
        for i, edge in enumerate(hyperedges):
            for atom in edge:
                incidence_hyperedges[atom, i] = 1

        # add hyperedge attributes
        if close_atoms:
            close_atoms_attributes = self.get_close_atoms_attributes(
                mol, close_atoms
            )
            ccc.set_cell_attributes(
                close_atoms_attributes, rank=2
            )  # if rank = 1, error in the code when close atoms are of more than 2 atoms

        # add hyperedges to the lifted topology
        ccc_lifted_topology["num_hyperedges"] = num_hyperedges
        ccc_lifted_topology["x_0"] = data.x
        ccc_lifted_topology["incidence_hyperedges"] = torch.Tensor(
            incidence_hyperedges
        ).to_sparse_coo()

        return ccc_lifted_topology
