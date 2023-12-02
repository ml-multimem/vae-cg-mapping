import torch
import logging

class MolecularFilesWriter():
    """Class to write files containig molecular trajectories 
       or structure in various formats. Supported:
       - write_xyz: trajectory file in the xyz format
       - write_pdb: pdb file with residue information
    """
    def __init__(self, logger_name = None):
        self.logger_name = logger_name

    def write_xyz(self, file_name_path, coords):
        """Write a xyz trajectory file based on the values of the self.coords attribute.

        Parameters:
        ---------_
        file_name_path: str
            relative path where to save the xyz file, including filename and extension
        coords: numpy array or torch tensor
            shape [number of frames, number of particles, 3]

        Reference for the format: https://www.ks.uiuc.edu/Research/vmd/plugins/molfile/xyzplugin.html
        """
        if self.logger_name is not None:
            logger = logging.getLogger(self.logger_name)
            logger.info("Writing xyz file to " + file_name_path)

        file = open(file_name_path,'w')
        particles_tot = coords.shape[1]

        for i, frame in enumerate(coords): 
            file.write( str(particles_tot) + '\n')
            file.write('Atoms. Timestep: '+ str(i)+'\n')

            for particle in frame:
                if particle.shape[0] == 4:
                    try:
                        file.write(str(int(particle[0])) + " " + str(particle[1]) + " " + str(particle[2]) + " " + str(particle[3]) + "\n")
                    except:
                        file.write(str(particle[0]) + " " + str(particle[1]) + " " + str(particle[2]) + " " + str(particle[3]) + "\n")
                elif particle.shape[0] == 3:
                    file.write("1" + " " + str(particle[0]) + " " + str(particle[1]) + " " + str(particle[2]) + "\n")
                else:
                    if self.logger_name is not None:
                        logger.exception("Could not write .xyz file. Wrong format of the coordinates")
                    else:
                        raise Exception("Could not write .xyz file. Wrong format of the coordinates")
                    
        file.close()
        
    def write_pdb(self, molname, pathtosave, n_particles_mol, n_mol, xyz, res_id = [], elements = None, box = [100, 100, 100]):
        """Write a pdb file with residue information.

        Parameters:
        ---------_
        molname: str
            name for the output file, with extension
        pathtosave: str
            relative path to save the file
        n_particles: numpy array or torch tensor
            number of particles in the system 
        n_mol: numpy array or torch tensor
            number of molecules in the system 
        xyz: numpy array or torch tensor
            coordinates with shape [n_particles * n_mol, 3]
        res_id: list
            residue list with shape [n_particles * n_mol]

        Reference: https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
        https://zhanglab.ccmb.med.umich.edu/COFACTOR/pdb_atom_format.html#CRYST1

        TODO implement atom name writing and element symbol
        """

        if self.logger_name is not None:
            logger = logging.getLogger(self.logger_name)
            logger.info("Writing pdb file to " + pathtosave + molname)

        if res_id == []:
            res_id = ["RES"] * n_mol * n_particles_mol

        #xyz [Ncg,3]
        pdbfile = open(pathtosave + molname,'w')  
        pdbfile.write("COMPND   MOLECULE: "+ molname + "\n")
        
        cryst1line = "CRYST1"
        boxsidex = str(round(box[0],3)).rjust(9, ' ')
        boxsidey = str(round(box[1],3)).rjust(9, ' ')
        boxsidez = str(round(box[2],3)).rjust(9, ' ')
        boxangle = str(round(90,2)).rjust(7, ' ')
        cryst1line = cryst1line + boxsidex + boxsidey + boxsidez + boxangle + boxangle + boxangle + " "
                    # space at the end is col 55
        spacegroup = "P 1"
        spacegroup = spacegroup.rjust(10, ' ') 
        cryst1line = cryst1line + spacegroup
        zvalue = "1"
        zvalue = zvalue.rjust(4, ' ') 
        cryst1line = cryst1line + zvalue + "\n"
        pdbfile.write(cryst1line)   
        
        i = 0
        for imol in range (n_mol):
            for ipart in range (n_particles_mol):

                pdbline = "ATOM" + "  " #the spaces are col 5,6
                    # ATOM record            cols 1-4        no justification        character

                pdbline = pdbline + str(i+1).rjust(5, ' ') + ' '  #the space is col 12
                    # Atom serial number     cols 7-11       right justification     integer

                atom_name = "X"
                pdbline = pdbline + atom_name.ljust(4,' ')
                    # Atom name              cols 13-16      left justification      character

                alt_loc_ind =" "
                pdbline = pdbline + alt_loc_ind
                    # Alternate locator      cols 17         no justification        character
                    # indicator

                residue_name = ' '.join(map(str,res_id[ipart])).replace(" ", "")
                pdbline = pdbline + residue_name.rjust(3,' ') + ' '  #the space is col 21
                    # Residue name           cols 18-20      right justification     character

                chain_id = str(0)  #NOTE: it is only one col and it is a character
                pdbline = pdbline + chain_id
                    # Chain identifier       cols 22         right justification     character

                residue_sequence_number = str(imol) # used as the molecule id
                pdbline = pdbline + residue_sequence_number.rjust(4, ' ')
                    # Residue sequence num   cols 23-26      right justification     integer

                code_for_insertions_of_residues = ' '
                pdbline = pdbline + code_for_insertions_of_residues + '   ' #the spaces are col 28-30
                    # Code for insertions    cols 27         no justification        character  
                    # of residues

                x_coord = str(round(xyz[i,0].item(),3)).rjust(8, ' ')
                y_coord = str(round(xyz[i,1].item(),3)).rjust(8, ' ')
                z_coord = str(round(xyz[i,2].item(),3)).rjust(8, ' ')

                pdbline = pdbline + x_coord + y_coord + z_coord
                    # x coordinate           cols 31-38      right justification     real (8.3)
                    # y coordinate           cols 39-46      right justification     real (8.3)
                    # z coordinate           cols 47-54      right justification     real (8.3)

                occupancy = str(round(1.00,2)).rjust(6, ' ')   # fixed value at 1
                pdbline = pdbline + occupancy
                    # Occupancy             cols 55-60      right justification     real (6.2)

                temperature_factor = str(round(0.00,2)).rjust(6, ' ') # fixed value at 0
                pdbline = pdbline + temperature_factor + '       '  #the spaces are col 67-73
                    # Temperature factor    cols 61-66      right justification     real (6.2)

                segment_id = "" 
                pdbline = pdbline + segment_id.ljust(4,' ')
                    # Segment ID            cols 73-76      left justification     character

                if elements is None:
                    element_symbol = "Q" 
                else:
                    element_symbol = elements[str(i)]
                pdbline = pdbline + element_symbol.rjust(2,' ') 
                    # Element symbol        cols 77-78      right justification     character

                charge = "  \n" 
                pdbline = pdbline + charge
                    # Charge                cols 79-80      right justification     character

                pdbfile.write(pdbline)
                i = i+1

        finalline = "MASTER        0    0    0    0    0    0    0    0    0    0" 
        finalline = finalline + str(n_particles_mol*n_mol).rjust(5, ' ') + "    0\n"
        pdbfile.write(finalline)

        pdbfile.write("END") 
        pdbfile.close()   

    def write_assignments(self, molname, pathtosave, R_effective_CG):
        """Write the mapping to CG beads to a file, to be used
        as input in the force matchin task. Format: one header 
        line followed by N atoms lines containing the number of 
        the CG particle they are assigned to. Numbering starts from zero.        

        Parameters:
        ---------_
        molname: str
            name for the output file, with extension
        pathtosave: str
            relative path to save the file
        R_effective_CG: torch tensor
            shape = [number of effective CG, number of atoms]
            It should contain only zeros and ones. There is only one 
            1 in each column j, corresponding to the CG bead the atom j is 
            assigned to.

        """
        if self.logger_name is not None:
            logger = logging.getLogger(self.logger_name)
            logger.info("Writing CG assignment file for " + molname) 

        n_aa = R_effective_CG.size()[1]
        
        file = open(pathtosave + molname+".map", "w")
        file.write("#Assigned to bead\n")
        for j in range(n_aa):
            idx = torch.argmax(R_effective_CG[:,j])
            file.write(str(idx.item())+"\n")
            
        file.close
