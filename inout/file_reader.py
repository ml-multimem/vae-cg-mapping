import numpy as np
from parameters import Parameters
from molecular_system.molecular_system import PeriodicMolecularSystem

class MolecularFilesReader():
    """Class to read from files containing molecular trajectories 
       or structure in various formats. Supported:
       - read_dump: LAMMPS dump file in the format id mol type q x y z ix iy iz fx fy fz
       - read_data: LAMMPS data file in the "Full" format
       - read_pdb: read the coordinates from ATOM records 
    """

    def read_file(self, filename):
        """Return lines read from filename"""
        with open(filename,"r") as f:
            lines = f.readlines()
        f.close()
        return lines
    
    def read_dump(self, dumpfile_path, mol_sys):
        """ Reads a trajectory from a LAMMPS dump file.
        
        Parameters
        ----------
        dumpfile_path: str
            path to the LAMMPS dump file, inclding file name 
            Expected format : id mol type q x y z ix iy iz fx fy fz
            Wrapped coordinates
            If the trajectory contains more frames than the n_frames specified 
            in the input, the first n_frames will be read
        mol_sys: PeriodicMolecularSystem
            It will contain the properites of the trajectory read
        
        Outcome
        -------
        Sets the attributes that change per particle and per frame (mol_sys.coords, 
        mol_sys.forces, mol_sys.images) and the periodic box, mol_sys.box.
        Notes:
        - Forces are in kcal/mol
        - Coordinates are wrapped and in Angstroem
        """

        # Number of snapshot to read from the trajectory and total number of particles
        n_frames = mol_sys.n_frames
        n_particles_mol = mol_sys.n_particles_mol
        n_molecules = mol_sys.n_molecules
        n_particles_tot = mol_sys.n_particles_tot

        header=9    # Lenght of the header in the LAMMPS dump file
        current_frame=0

        # Initialize property matrices
        mol_sys.forces=np.zeros([n_frames, n_molecules, n_particles_mol, 3])
        mol_sys.coords=np.zeros([n_frames, n_molecules, n_particles_mol, 3])
        mol_sys.images=np.zeros([n_frames, n_molecules, n_particles_mol, 3])
        #if isinstance(mol_sys, PeriodicMolecularSystem):
        temp_box = np.zeros([3,2])
        mol_sys.box = np.zeros([n_frames, n_molecules, 3,2])

        dump_lines = self.read_file(dumpfile_path)    # Import the whole dump file   
                                                # TODO: split the import in case of very big dump files
        lines_per_frame = n_particles_tot+header    # Total lines for each snapshot
        lines_to_read = lines_per_frame*n_frames    # Total lines to consider 
            
        for line in range(0, lines_to_read, n_particles_tot+header):
            # Read the 3 box dimensions in all frames and separately, to 
            # support NPT trajectories and non-cubic cells
            if isinstance(mol_sys, PeriodicMolecularSystem):
                for i in range(3):
                    temp_box[i][:] = np.array(dump_lines[line+5+i].split()[:]).astype(float)
                
                for imol in range(n_molecules):    
                    mol_sys.box[current_frame, imol, : , :] = temp_box
            
            # Read per-atom and per-frame properties: id mol type q x y z ix iy iz fx fy fz
            # Same format assumed both for periodic and non perioci systems
            j = 0
            for imol in range(n_molecules):  #Assumes that dump is sorted
                for ipart in range (n_particles_mol):
                    mol_sys.coords[current_frame][imol][ipart][:]=\
                        np.array(dump_lines[line+header+j].split()[4:7]).astype(float)
                    mol_sys.images[current_frame][imol][ipart][:]=\
                        np.array(dump_lines[line+header+j].split()[7:10]).astype(int)
                    mol_sys.forces[current_frame][imol][ipart][:]=\
                        np.array(dump_lines[line+header+j].split()[10:13]).astype(float)
                    j=j+1
                    
            current_frame = current_frame +1

    def read_data(self, datafile_path, mol_sys):
        """ Reads molecular properties and connectivity from a LAMMPS data file.
        
        Parameters
        ----------
        datafile_path: str
            path to the LAMMPS dump file, inclding file name 
            Expected format : id mol type q x y z ix iy iz fx fy fz
            Wrapped coordinates
            If the trajectory contains more frames than the n_frames specified 
            in the input, the first n_frames will be read
        mol_sys: MolecularSystem or subclasses
            It will contain the properites read
        
        Outcome
        -------
        Sets the attributes that change per particle or particle type, but not 
        per frame mol_sys.masses, mol_sys.mol_membership, mol_sys.types.
        Reads in the connectivity information as list of pairs, triplets, quadruplets
        of particles participating in a connection (Bond, Angles, Dihedrals..)
        Notes:
        - Molecule membership and types are numbered starting from 0
        - Atom ids for the connectivity are numbered starting from 0
        """

        data_lines = self.read_file(datafile_path)

        # find the values for the counters 
        for i in range(len(data_lines)):
            if len(data_lines[i].split()) > 1:
                if data_lines[i].split()[1] == 'atoms':
                    n_atoms = int(data_lines[i].split()[0])
                if data_lines[i].split()[1] == 'bonds':
                    n_bonds = int(data_lines[i].split()[0])
                if data_lines[i].split()[1] == 'angles':
                    n_angles = int(data_lines[i].split()[0])
                if data_lines[i].split()[1] == 'dihedrals':
                    n_dihedrals = int(data_lines[i].split()[0])   
                if data_lines[i].split()[1] == 'impropers':
                    n_impropers = int(data_lines[i].split()[0])   

                if data_lines[i].split()[1] == 'atom':  #atom types
                    n_atom_types = int(data_lines[i].split()[0])
                if data_lines[i].split()[1] == 'bond':  #bond types
                    n_bond_types = int(data_lines[i].split()[0])
                if data_lines[i].split()[1] == 'angle':  #angle types
                    n_angle_types = int(data_lines[i].split()[0])
                if data_lines[i].split()[1] == 'dihedral':  #dihedral types
                    n_dihedral_types = int(data_lines[i].split()[0])
                if data_lines[i].split()[1] == 'impropers':  #impropers types
                    n_impropers_types = int(data_lines[i].split()[0])
                

        mol_sys.masses = np.zeros([n_atom_types])
        for i in range(len(data_lines)):
            if len(data_lines[i].split()) > 0:
                if data_lines[i].split()[0] == 'Masses':
                    for j in range(i+2,i+2+n_atom_types):
                        mol_sys.masses[int(data_lines[j].split()[0])-1]=float(data_lines[j].split()[1])
                    break

        mol_sys.types = np.zeros([n_atoms])
        mol_sys.mol_membership=np.zeros([n_atoms])
        mol_sys.elements_list={}
        for i in range(len(data_lines)):
            if len(data_lines[i].split()) > 0:
                if data_lines[i].split()[0] == 'Atoms':
                    for j in range(i+2,i+2+n_atoms):
                        current_type = int(data_lines[j].split()[2])-1
                        mol_sys.types[int(data_lines[j].split()[0])-1]=current_type
                        mol_sys.mol_membership[int(data_lines[j].split()[0])-1]=int(data_lines[j].split()[1])-1
                        mol_sys.elements_list[str(int(data_lines[j].split()[0])-1)] = \
                                self.get_element(mol_sys.masses[current_type])
                    break    

        mol_sys.bonds_list = np.zeros([n_bonds,2])
        for i in range(len(data_lines)):
            if len(data_lines[i].split()) > 0:
                if data_lines[i].split()[0] == 'Bonds':
                    z=0
                    for j in range(i+2,i+2+n_bonds):
                        mol_sys.bonds_list[z][0]=int(data_lines[j].split()[2])-1
                        mol_sys.bonds_list[z][1]=int(data_lines[j].split()[3])-1
                        z+=1
                    break

        mol_sys.angles_list = np.zeros([n_angles,3])
        for i in range(len(data_lines)):
            if len(data_lines[i].split()) > 0:
                if data_lines[i].split()[0] == 'Angles':
                    z=0
                    for j in range(i+2,i+2+n_angles):
                        mol_sys.angles_list[z][0]=int(data_lines[j].split()[2])-1
                        mol_sys.angles_list[z][1]=int(data_lines[j].split()[3])-1
                        mol_sys.angles_list[z][2]=int(data_lines[j].split()[4])-1
                        z+=1
                    break

        mol_sys.dihedrals_list = np.zeros([n_dihedrals,4])
        for i in range(len(data_lines)):
            if len(data_lines[i].split()) > 0:
                if data_lines[i].split()[0] == 'Dihedrals':
                    z=0
                    for j in range(i+2,i+2+n_dihedrals):
                        mol_sys.dihedrals_list[z][0]=int(data_lines[j].split()[2])-1
                        mol_sys.dihedrals_list[z][1]=int(data_lines[j].split()[3])-1
                        mol_sys.dihedrals_list[z][2]=int(data_lines[j].split()[4])-1
                        mol_sys.dihedrals_list[z][3]=int(data_lines[j].split()[5])-1
                        z+=1
                    break

        mol_sys.impropers_list = np.zeros([n_impropers,4])
        for i in range(len(data_lines)):
            if len(data_lines[i].split()) > 0:
                if data_lines[i].split()[0] == 'Impropers':
                    z=0
                    for j in range(i+2,i+2+n_impropers):
                        mol_sys.impropers_list[z][0]=int(data_lines[j].split()[2])-1
                        mol_sys.impropers_list[z][1]=int(data_lines[j].split()[3])-1
                        mol_sys.impropers_list[z][2]=int(data_lines[j].split()[4])-1
                        mol_sys.impropers_list[z][3]=int(data_lines[j].split()[5])-1
                        z+=1
                    break

    def read_pdb(self, pdbfile_path, mol_sys):
        """Read atomic coordinates from a PDB file and update a MolecularSystem or PeriodicMolecularSystem object.

        Parameters:
        -----------
        pdbfile_path: str
            Path to the PDB file containing atomic coordinates.
        mol_sys: MolecularSystem or PeriodicMolecularSystem
            The MolecularSystem object to be updated with the read atomic coordinates.

        Returns:
        --------
        mol_sys: MolecularSystem or PeriodicMolecularSystem
            Updated MolecularSystem or PeriodicMolecularSystem object with atomic coordinates read from the PDB file.
        """

        # Read the lines from the PDB file
        pdb_lines = self.read_file(pdbfile_path)

        # Initialize the total number of particles
        mol_sys.n_particles_tot = 0

        # Iterate through the lines in the PDB file
        for i in range(len(pdb_lines)):
            
            # Check if the line corresponds to an ATOM record
            if pdb_lines[i].split()[0] == 'ATOM':
                # Increment the total number of particles
                mol_sys.n_particles_tot = mol_sys.n_particles_tot +1
                # Update the atomic coordinates in the MolecularSystem or PeriodicMolecularSystem
                mol_sys.coords[0][i][1]=float(pdb_lines[i].split()[6])
                mol_sys.coords[0][i][1]=float(pdb_lines[i].split()[7])
                mol_sys.coords[0][i][1]=float(pdb_lines[i].split()[8])
                
        return mol_sys
        
    def get_element(self, mass):
        """Get the element symbol based on the given atomic mass.

        Parameters:
        -----------
        mass: float
            Atomic mass of the element.

        Returns:
        --------
        element_name: str
            Symbol of the element corresponding to the provided atomic mass.
            If the provided mass does not match any known element, returns "Unknown".
        """
        element = {
            "1": "H",
            "2": "He",
            "3": "Li",
            "4": "Be",
            "5": "B",
            "6": "C",
            "7": "N",
            "8": "O",
            "9": "F",
            "10": "Ne",
            "11": "Na",
            "12": "Mg",
            "13": "Al",
            "14": "Si",
            "15": "P",
            "16": "S",
            "17": "Cl",
            "18": "Ar",
            "19": "K",
            "20": "Ca",
            # Add more elements as needed
            }
        
        # Round the provided mass to the nearest integer and convert it to a string
        # then access an element using its mass
        return element[str(int(round(mass)))]

class InputParser():
    def __init__(self):
        pass
    
    def parse(self,filename):
        """Read and parse an input file to extract parameters.

        Parameters:
        -----------
        filename: str
            The name of the input file to be parsed.

        Returns:
        --------
        parameters: Parameters
            An instance of the Parameters class containing the parsed parameters.
        """
        
        with open(filename,"r") as f:
            #Return lines read from filename
            input_lines = f.readlines()
        f.close()
        parameters = Parameters()

        for i in range(len(input_lines)):
            if len(input_lines[i].split()) > 0: # Skip empty lines
                if not input_lines[i].split()[0] == '####': # Skip header lines marked with ####
                    parameters[input_lines[i].split()[0]] = input_lines[i].split()[1]
                  
        return parameters
