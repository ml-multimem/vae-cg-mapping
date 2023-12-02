import os, shutil
from datetime import datetime
from molecular_system.molecular_system import MolecularSystem, PeriodicMolecularSystem
from molecular_system.mapping_utils import *
from inout.file_reader import InputParser, MolecularFilesReader
 
class Initializer():
    """Class to read the input file and initialize the variables 
    necessary for a specific task from specific
    trajectory formats. Currently only LAMMPS format is supported:
    - initialize_from_lammps
    """

    def __init__(self, filename):
        """Call InputParser to read the input file and store 
        parameter names and values in self.parameters.        
        """
        self.parameters = InputParser().parse(filename)
        self.reader = MolecularFilesReader()

    def initialize_for_mapping(self, cmdl_args):
        """ Reads the atomistic trajectory and properties from a 
        LAMMPS dump and data files and initializes a corresponding 
        PeriodicMolecularSystem object with the read values:
         - coordinates, images, box
         - forces
         - number of atoms, molecules, frames
         - masses, atom types, molecule membership
         - bonds list
                    
        Returns
        -------
        atom_sys: PeriodicMolecularSystem of MolecularSystem
            Initialized PeriodicMolecularSystem or MolecularSystem corresponding to the atomistic trajectory
        feature: 
            Selected feature for the mapping model: cooridinates or full distances matrix
        parameters: Parameters
            A dictionary-like object of model parameters and settings read from the input file
        """
        parameters = self.overwrite_from_command_line(self.parameters, cmdl_args)
        if not hasattr(parameters, 'overwrite'):
            parameters["overwrite"] = "no"

        atom_sys = self.initialize_from_lammps(parameters)
               
        parameters["shuffle"] = True # Shuffle the instances during training
        parameters["Tstart"] = 4 # Starting temperature in Gumbel-Softmax annealing
        parameters["molname"] = parameters["trajectory"].partition(".")[0] # Used for output file names
        parameters["task"] = "mapping"

        parameters = self.create_output_folder(parameters) 

        return atom_sys, parameters

    def initialize_from_lammps(self, parameters):
        """Read from LAMMPS data and dump files and store the 
        properties in a PeriodicMolecularSystem or MolecularSystem object.
        NOTE: Only systems with one molecule type supported.

        Arguments:
        ---------
        parameters: dict
            Dictionary with filenames and import parameters read 
            from the input file. It should containy:
            - n_particles_mol: number of particles of one molecule
            - n_molecules: number of molecules in the system
            - n_frames: number of frames to read. They will be taken 
                        from the beginning of the file
            - directory: relative path where the lammps files are located
            - trajectory: name of the lammps dump file
            - data_file: name of the lammps data file

        Returns:
        -------
        mol_sys: PeriodicMolecularSystem or MolecularSystem
            Object initialized with all properties read from the lammps files
        
        """
        # Instanciate the atomistic trajectory and fill it with data from the LAMMPS dump file
        n_particles_mol = int(parameters["n_atoms_mol"])
        n_molecules = int(parameters["n_molecules"])
        n_frames = int(parameters["n_frames"])

        if eval(parameters["periodic"]) == True:
            mol_sys = PeriodicMolecularSystem(n_particles_mol, n_molecules, n_frames)
        else:
            mol_sys = MolecularSystem(n_particles_mol, n_molecules, n_frames)
        self.reader.read_dump(parameters["directory"]+parameters["trajectory"],mol_sys)  
        self.reader.read_data(parameters["directory"]+parameters["data_file"],mol_sys) 
        return mol_sys 
   
    def overwrite_from_command_line(self, parameters, cmdl_args):
        """Overwrite parameters from the command line arguments.

        Arguments:
        ---------
        parameters: dict
            Original dictionary of parameters read from the input file
        cmdl_args: argparse.Namespace
            Command line arguments parsed using argparse

        Returns:
        -------
        parameters: dict
            Updated dictionary with values overwritten from command line arguments
        """
        i = 0
        for attribute in dir(cmdl_args):
            if not attribute.startswith('_'):  # exclude attributes that do not come from the input file
                cmd_value = getattr(cmdl_args, dir(cmdl_args)[i])
                if cmd_value is not None:
                    parameters[attribute] = str(cmd_value) 
            i=i+1                   
        return parameters

    def create_output_folder(self, parameters):
        """Create the output folder for the current run.

        Arguments:
        ---------
        parameters: dict
            Dictionary of parameters including 'directory', 'task', and 'overwrite'

        Returns:
        -------
        parameters: dict
            Updated dictionary with the 'res_folder' key containing the path to the output folder
        """
        # Overwrite the content in the default folder
        if eval(parameters["overwrite"]) == True:
            res_folder = parameters["directory"] + "default_output_" + parameters["task"] + "/"
            parameters["res_folder"] = res_folder
            if not os.path.exists(res_folder): #create the directory if it does not exists
                os.mkdir(res_folder)    
            elif os.listdir(res_folder):            # make a temporary backup if the directory 
                bak_folder = res_folder+"bak/"      # exists and it is not empty
                if os.path.exists(bak_folder): # remove the bak directory if it exists
                    shutil.rmtree(bak_folder)  
                shutil.copytree(res_folder, bak_folder)            

        # Create a new folder for the run
        elif eval(parameters["overwrite"]) == False:
            timestamp = str(datetime.utcnow().strftime('%Y_%m_%d_%H%M%S'))
            res_folder = parameters["directory"] + parameters["task"] + "_" + timestamp + "/"
            parameters["res_folder"] = res_folder
            if not os.path.exists(res_folder):
                os.mkdir(res_folder)   
            
        return parameters