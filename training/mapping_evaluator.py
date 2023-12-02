import torch.nn as nn
import numpy as np
import networkx as nx

class MappingEvaluator():
    """
    Evaluate mappings generated in terms of acceptability and uniqueness.
    """
    def __init__(self):
        nn.Module.__init__(self)

    def is_acceptable(self, encoder, connectivity):
        """
        Evaluate if the given mapping is acceptable in terms of preserving connectivity.

        Parameters:
        -----------
        encoder: nn.Module
            The encoder used for computing the mapped connectivity.
        connectivity: Connectivity
            The object representing the connectivity information in the atomistic system.

        Returns:
        --------
        acceptable: bool
            True if the mapping is considered acceptable; False otherwise.
        """

        R_effective_CG = encoder.rounded_effective_CG()
        R_effective_CG = R_effective_CG.numpy()
        noise = 0
        signal = 0
        molecule_bond_matrix = connectivity.bond_matrix[0]
        atom_ids = np.nonzero(R_effective_CG)

        # For each moiety, all the assigned atoms are checked pairwise and 
        # if there is not a connectivity path within the moiety, the Noise 
        # value in incremented 8connectivity broken), otherwise the Signal 
        # is incremented (connectivity preserved).

        for moiety in range(len(R_effective_CG)):
            
            idx = atom_ids[1][np.where(atom_ids[0]==moiety)]
            in_moiety_connectivity_matrix = molecule_bond_matrix[np.ix_(idx,idx)]
            moiety_graph = nx.convert_matrix.from_numpy_matrix(in_moiety_connectivity_matrix)
            path = nx.shortest_path(moiety_graph)
            this_signal = sum([len(path[i])-1 for i in range(len(path))])
            signal = signal + this_signal
            max_signal = idx.size*(idx.size-1)
            this_noise = max_signal - this_signal
            noise = noise + this_noise
        
        if noise == 0:
            acceptable = True
        else:
            acceptable = False
        
        return acceptable
    
    def is_unique(self, new_mapping, old_mappings):
        """
        Check if the new mapping is unique compared to a list of old mappings.

        Parameters:
        -----------
        new_mapping: torch.Tensor
            The new mapping to be checked for uniqueness.
        old_mappings: list of torch.Tensor
            List of old mappings to compare against.

        Returns:
        --------
        unique: bool
            True if the new mapping is unique; False otherwise.
        """

        mat_size = sum(new_mapping.size())
        mat1 = np.zeros([mat_size, mat_size])
        ncg1 = new_mapping.size()[0]
        naa1 = new_mapping.size()[1]
        mat1[ncg1:ncg1+ncg1,ncg1:ncg1+naa1] = np.array(new_mapping)
        G1 = nx.from_numpy_array(mat1)

        result = []
        for i in range(len(old_mappings)):
            mat_size = sum(old_mappings[i].size())
            mat2 = np.zeros([mat_size, mat_size])
            ncg2 = old_mappings[i].size()[0]
            naa2 = old_mappings[i].size()[1]
            mat2[ncg2:ncg2+ncg2,ncg2:ncg2+naa2] = np.array(old_mappings[i])
            G2 = nx.from_numpy_array(mat2)

            result.append(not nx.is_isomorphic(G1, G2))
        
        unique = all(i for i in result)

        return unique


