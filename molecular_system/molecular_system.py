import numpy as np

class MolecularSystem():
    """A class to contain all the properties of a molecular system.
    """
    def __init__(self, n_particles_mol, n_molecules, n_frames = 1):
        self.n_particles_mol = n_particles_mol # N particles per molecule
        self.n_molecules = n_molecules
        self.n_frames = n_frames
        self.n_particles_tot = self.n_particles_mol*self.n_molecules
        self.masses = []  # N masses per particle type
       
        self.coords = [] # Shape [n_frames, n_molecules, n_particles_per_mol ,3]
        self.forces = [] # Shape [n_frames, n_molecules, n_particles_per_mol ,3]
        self.types = [] # Shape [n_particles_tot]
        self.elements_list = [] # Shape [n_particles_mol]
        self.mol_membership = [] # Shape [n_particles_tot]
        self.neighbors = [] #list of lists [n_frames][neighbors of this particle]
        self.bonds_list = [] # Shape [n_bonds,2]
        self.angles_list = [] # Shape [n_angles,3]
        self.dihedrals_list = [] # Shape [n_dihedrals,4]
        self.impropers_list = [] # Shape [n_impropers,4]
    
class PeriodicMolecularSystem(MolecularSystem):
    """A class to contain all the properties of a molecular system 
    under periodic boundary conditions. Inherits all attributs of 
    MolecularSystem and adds:
    - the box attribute to store the position of the boundary during the simulation
    - the images attribute to represent wrapped coordinates
    - a flag to signal if the self.coords attribute contains wrapped or unwrapped coordinates
    - wrap and unwrap methods to switch between wrapped and unwrapped coordinates 
    """
    def __init__(self, n_particles_mol, n_molecules, n_frames):
        super().__init__(n_particles_mol, n_molecules, n_frames)
        self.box = [] # Shape [n_frames, n_molecules, 3,2] (3-> x,y,z) (2-> min, max)
        self.images = [] # Shape [n_frames, n_molecules, n_particles_mol,3]
        self.cords_are_wrapped = True
    
    def unwrap(self):
        """Change coordinates format from wrapped to upwrapped"""

        if self.cords_are_wrapped == False:
            raise Exception("Coordinates are already unwrapped") 
        else:
            self.cords_are_wrapped = False
            box_lenght = self.box[:,:,:,1]-self.box[:,:,:,0]
            box_lenght = np.stack([box_lenght for _ in range(self.n_particles_mol)], axis = 2)
            self.coords = self.coords + np.multiply(self.images,box_lenght)

    
    def wrap(self):
        """Change coordinates format from upwrapped to wrapped"""
        if self.images == []:
            self.calculate_images()

        if self.cords_are_wrapped == True:
            raise Exception("Coordinates are already unwrapped") 
        else:
            self.cords_are_wrapped = True
            box_lenght = self.box[:,:,:,1]-self.box[:,:,:,0]
            box_lenght = np.stack([box_lenght for _ in range(self.n_particles_mol)], axis = 2)
            self.coords = self.coords - np.multiply(self.images,box_lenght)

    def calculate_images(self):
        """Calculate periodic image index"""
        box_lenght = self.box[:,:,:,1]-self.box[:,:,:,0]
        box_lenght = np.stack([box_lenght for _ in range(self.n_particles_mol)], axis = 2)
        self.images = np.fix(np.divide(self.coords,box_lenght))
