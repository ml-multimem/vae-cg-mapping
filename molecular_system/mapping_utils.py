import torch
import numpy as np
import torch.nn.functional as F
from molecular_system.molecular_system import PeriodicMolecularSystem

class AtomisticToCGMapper():
    """ Class to perform calculations related to mapping from the atomistic to the CG space
    """

    def map_trajectory(self, atom_sys, cg_sys, bead_idx):
        """ Maps an atomistic trajectory to the coarse grained space following a 
        predifined assignment template, using the geometric center or the center 
        of mass to place the coarse grained particles in space. 
        Molecule membership of the CG particles is also calculated, based on the 
        corresponding atomistic property
        
        Parameters
        ----------
        atom_sys: PeriodicMolecularSystem
            It contains the properites of the atomistic trajectory
        cg_sys: PeriodicMolecularSystem
            It is used to store the properites of the atomistic trajectory
            atom_sys mapped to the coarse grained space
        bead_idx: list
            assignment template of n atoms lines. Each line i contains the
            number of the CG bead the atom i belongs to.        
        """

        n_cg_mol = cg_sys.n_particles_mol
        n_aa_mol = cg_sys.n_particles_mol
        n_mol = cg_sys.n_molecules
        
        # Create assignment matrix [n_cg,n_atoms]
        assignment_template=np.zeros([n_cg_mol,len(bead_idx)])
        for i in range(len(bead_idx)):
            # Uncomment the line below to use the geometric center
            #assignment_template[int(bead_idx[i])][i]=1  
            # Uncomment the line below to use the center of mass
            assignment_template[int(bead_idx[i])][i]=atom_sys.masses[int(atom_sys.types[i])] 
        
        # Mapping the masses and the types
        # Assumption: bead type is assigned based on the mass
        masses_cg_molecule = np.round(np.sum(assignment_template,axis=1),decimals=4)
        cg_sys.masses = np.unique(masses_cg_molecule)
        cg_sys.types = np.zeros(len(masses_cg_molecule), dtype=int)
        for i in range(len(cg_sys.masses)):
            idx = np.where(masses_cg_molecule == cg_sys.masses[i])[0]
            cg_sys.types[idx] = i

        # Normalization per number of atoms belonging to a bead
        assignment_template=np.divide(assignment_template.T,assignment_template.sum(1)).T
        # Replicate per number of molecules
        assignment_system = np.stack([assignment_template for _ in range(n_mol)])
        # Replicate per number of frames
        assignment_matrix = np.stack([assignment_system for _ in range(atom_sys.n_frames)])
        # Calculate coordinates of cg bead
        if isinstance(atom_sys, PeriodicMolecularSystem):
            atom_sys.unwrap()
            cg_sys.cords_are_wrapped = False
            cg_sys.box = atom_sys.box

        cg_sys.coords = np.matmul(assignment_matrix,atom_sys.coords)
        cg_sys.forces = np.matmul(assignment_matrix,atom_sys.forces)
        

        # Map masses        

        # Assign molecule membership to the CG particles
        cg_sys.mol_membership=np.zeros([n_cg_mol*n_mol])
        assignment_system = np.reshape(assignment_system,(-1,n_aa_mol))
        for i in range(n_cg_mol*n_mol):
            # Find one atom belonging to the i-th CG particle 
            # and assign the same molecule to the CG particle
            idx_atom = np.argmax(assignment_system[i,:])
            cg_sys.mol_membership[i] = atom_sys.mol_membership[idx_atom]

    def map_forces(self, forces, encoder):
        """Map the effective force acting on each CG particle 
        and take the mean squared value
        
        Parameters:
        forces: torch tensor
            shape [n frames in batch, n atoms, 3] 
            atomistic forces for a batch
        encoder: TODO pass CG directly?
        """
        CG = self.assignment_matrix(encoder)
            # size of CG = [N_cg,n_atoms_per_molecule] 
            # effective_CG represents the probability of each particle (element of n_atoms_per_molecule,) to belong to a specific moiety (element of N_cg)
        f0 = forces.reshape(-1, CG.size()[-1], 3)
        f = torch.matmul(CG, f0)    
            # size of f = [batch,N_cg,3]
            # Calculate the average force components fx, fy, fz acting on the center of the CG particles 
            # (in each element of the batch -> instantaneous)
        mean_force = f.pow(2).sum(2).mean()  # Mean squared instantaneous force
        
        return mean_force

    def assignment_matrix(self, encoder):
        """
        Generate the one-hot assignment matrix for mapping atomistic particles 
        to coarse-grained (CG) beads.

        Parameters:
        -----------
        encoder: nn.Module
            The encoder module containing the temperature scheduler and weight parameters.

        Returns:
        --------
        CG: torch.Tensor
            The assignment matrix for mapping atomistic particles to CG beads, 
            computed using the softmax function on the encoder weight.
        
        """
        temperature = encoder.scheduler.current_temperature()
        CG = F.softmax(encoder.weight.t() / temperature, dim=-1).t()
        return CG
        
    def effective_CG_beads(self, CG):
        """
        Identify the effective CG beads used in the mapping.

        Parameters:
        -----------
        CG: torch.Tensor
            The assignment matrix representing the mapping of atomistic particles to CG beads.

        Returns:
        --------
        list:
            A list of unique CG beads to which atomistic particles are assigned in the mapping.
        """

        assignment_indexes = sorted(np.argmax(CG.detach().cpu().numpy(),axis=0)) 
            #indexes of the GC bead to with each atomistic particle is assigned to .detach()
        used_CG_beads = []
        [used_CG_beads.append(used_CG) for used_CG in assignment_indexes if used_CG not in used_CG_beads]
            #eliminate duplicates from the list of indexes
        return used_CG_beads