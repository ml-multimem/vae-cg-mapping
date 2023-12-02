import numpy as np
import networkx as nx
import torch
import itertools
from molecular_system.molecular_system import PeriodicMolecularSystem

class Connectivity():
    """Class to handle calculations related to distances and connectivity
    """
    def __init__(self, mol_sys ):
        """
        Initialize the Connectivity class.

        Parameters:
        -----------
        mol_sys: MolecularSystem or PeriodicMolecularSystem
            An instance of the MolecularSystem or PeriodicMolecularSystem class representing the molecular system.

        Attributes:
        ------------
        mol_sys: MolecularSystem or PeriodicMolecularSystem
            The provided molecular system.
        bond_matrix: list
            List to store bond matrix.
        """

        self.mol_sys = mol_sys
        self.bond_matrix = []
    
    def create_bond_matrix(self, bond_list):
        """Create connectivity matrix from list of connected atoms.

        Parameters
        ----------
        i_atoms_list, j_atoms_list: list of int
            Together they epresent bod information: i_atoms_list[i] is bonded to j_atoms_list[i]
        
        Outcome
        -------
        It sets the connectivity_matrix attribute. If particle i is connected to particle j 
        connectivity_matrix[i,j]=1 , if not connected connectivity_matrix[i,j]=0.
        """
        
        #Initialize connectivity matrix 
        bond_matrix=np.zeros([self.mol_sys.n_molecules, self.mol_sys.n_particles_mol, self.mol_sys.n_particles_mol]) 
                                                                                                          
        # Convert to numpy array
        bond_list = np.stack([list(map(int, bond_list[i])) for i in range(len(bond_list))])
        bond_list = np.array(bond_list)
        bond_list = np.reshape(bond_list, (self.mol_sys.n_molecules, -1, 2)) 
        bond_molecule = bond_list[0]
                                                            
        bond_matrix[:,bond_molecule[:,0],bond_molecule[:,1]] = 1
        bond_matrix[:,bond_molecule[:,1],bond_molecule[:,0]] = 1

        return bond_matrix

    def CG_connectivity_matrix(self, moiety_atoms_indices, connectivity_matrix, n_atoms):
        """
        Generate a coarse-grained (CG) connectivity matrix for a subset of atoms within a moiety.

        Parameters:
        -----------
        moiety_atoms_indices: list
            List of indices corresponding to atoms within a moiety.
        connectivity_matrix: np.array
            The connectivity matrix for the entire system.
        n_atoms: int
            Total number of atoms in the system.

        Returns:
        --------
        CG_connectivity_matrix: np.array
            The CG connectivity matrix for the specified moiety atoms i and j. 
            1: i and j are connected - 0: i and j are not connected
        """

        CG_connectivity_matrix=np.zeros((n_atoms,n_atoms))
        for atom_id2 in moiety_atoms_indices:
            for atom_id1 in moiety_atoms_indices:
                if atom_id1!=atom_id2:
                    if connectivity_matrix[atom_id1,atom_id2]==1:
                        CG_connectivity_matrix[atom_id1,atom_id2]=1
                        CG_connectivity_matrix[atom_id2,atom_id1]=1

        return CG_connectivity_matrix

    def hasPathBetweenMoiety(self, CG_connectivity_matrix, node1, node2):
        """
        Check if there is a path between two nodes in a CG connectivity matrix.

        Parameters:
        -----------
        CG_connectivity_matrix: np.array
            The CG connectivity matrix.
        node1: int
            Index of the first node.
        node2: int
            Index of the second node.

        Returns:
        --------
        bool:
            True if there is a path between the two nodes, False otherwise.
        """

        in_moiety_connectivity_graph=nx.Graph(CG_connectivity_matrix)
        return nx.algorithms.shortest_paths.generic.has_path(in_moiety_connectivity_graph, node1, node2)

    def intramolecular_distances(self,mol_sys):
        """
        Calculate intramolecular distances for all pairs of atoms in the molecular system.

        Parameters:
        -----------
        mol_sys: MolecularSystem or PeriodicMolecularSystem
            An instance of the MolecularSystem or PeriodicMolecularSystem class representing the molecular system.

        Returns:
        --------
        dist_matrix: np.array
            A 4D array representing the intramolecular distances between particle pairs.
            Indxing reference: [frame, molecule, particle 1, particle 2]
        """

        coords = torch.tensor(mol_sys.coords)
        n_frames, n_molecules, n_particles_mol, n3 = mol_sys.coords.shape        

        # All pair combinations
        particles_ids = np.arange(0, n_particles_mol, 1, dtype=int)
        distance_pairs = list(itertools.combinations(particles_ids, 2))
        idx_lists = [[feat[i] for feat in distance_pairs]
                    for i in range(2)]

        # List representation of the distances
        dist_vectors = [coords[:, :, idx_lists[i+1], :] 
                    - coords[:, :, idx_lists[i], :]
                    for i in range(2 - 1)]
        dist_vectors = dist_vectors[0]   

        if isinstance(mol_sys, PeriodicMolecularSystem):
            box_lenght = mol_sys.box[:,:,:,1]-mol_sys.box[:,:,:,0]
            box_lenght = np.stack([box_lenght for _ in range(dist_vectors.size()[2])], axis = 2)
            dist_wrapping = np.divide(dist_vectors, box_lenght)
            dist_wrapping = np.multiply(np.fix(dist_wrapping), box_lenght)
            dist_vectors = dist_vectors - dist_wrapping
        dist_list =  torch.norm(dist_vectors, dim=3)  

        # Matrix representation of the distances
        tmp = torch.stack([dist_list[:,:,i] for i in range(dist_list.shape[2])]).float()
        tmp = torch.transpose(tmp,0,1)
        tmp = torch.transpose(tmp,1,2)
        dist_matrix = torch.zeros([n_frames, n_molecules, n_particles_mol, n_particles_mol])
        dist_matrix[:, :, idx_lists[0],idx_lists[1]] = tmp
        dist_matrix[:, :, idx_lists[1],idx_lists[0]] = tmp
        dist_matrix = np.array(dist_matrix)
        
        return dist_matrix



