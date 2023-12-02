import numpy as np
import torch
import os
import pathlib
from molecular_system.molecular_system import PeriodicMolecularSystem
from inout.file_writer import MolecularFilesWriter
import math
import logging

class Renderer():
    """Class to create VMD renderings"""
    def __init__(self, molname,
                 render_path,
                 logger_name, 
                 device):
        """ Initialize the Renderer.

        Parameters:
        -----------
        molname: str
            Name of the molecular system.
        render_path: str
            Path to save rendered files.
        logger_name: str
            Name of the logger for logging messages.
        device: torch.device
            The device to use for rendering.

        Attributes:
        -----------
        molname: str
            Name of the molecular system.
        render_path: str
            Path to save rendered files.
        writer: MolecularFilesWriter
            Instance of MolecularFilesWriter for writing PDB files.
        logger_name: str
            Name of the logger for logging messages.
        device: torch.device
            The device to use for rendering.
        
        """
        self.molname = molname
        self.render_path = render_path
        if not os.path.exists(render_path):
            os.mkdir(render_path) 
        self.writer = MolecularFilesWriter(logger_name)
        self.logger_name = logger_name
        self.device = device

    def mapping_backmapping(self, model, mol_sys):
        """Render atomistic system + CG mapping and backmapping.

        Parameters:
        -----------
        model: nn.Module
            Instance of the model containing encoder and decoder.
        mol_sys: PeriodicMolecularSystem
            The atomistic molecular system to render.

        Raises:
        -------
        Exception:
            If VMD rendering fails, an exception is logged, and the program continues.
        """

        logger = logging.getLogger(self.logger_name)

        R_effective_CG = model.encoder.rounded_effective_CG()

        if isinstance(mol_sys, PeriodicMolecularSystem):
            box = mol_sys.box[0,0,:,1] - mol_sys.box[0,0,:,0]
            mol_sys.unwrap()
        else:
            box = [100, 100, 100]

        n_cg = R_effective_CG.size()[0]
        xyz = torch.tensor(mol_sys.coords[0]).float().to(self.device)
        encoded = model.encoder.true_forward(xyz)  
        encoded_effective = model.encoder.effective_mapping(xyz) 
        encoded_effective = torch.reshape(encoded_effective, (-1,3))      
        backmapped_xyz = model.decoder(encoded)
        backmapped_xyz = torch.transpose(backmapped_xyz,1,2).squeeze()
        backmapped_xyz = torch.reshape(backmapped_xyz, (-1,3))


        # Mapping
        self.create_vmd_commands_CG(self.molname, self.render_path)
        res_id_cg, res_id_aa = self.res_id_list(R_effective_CG)
        
        elements = None
        self.writer.write_pdb(self.molname + "_CG.pdb", 
                              self.render_path, 
                              n_cg, 
                              mol_sys.n_molecules, 
                              encoded_effective, 
                              res_id_cg, 
                              elements, 
                              box)

        self.writer.write_pdb(self.molname + "_AA.pdb", 
                              self.render_path, 
                              mol_sys.n_particles_mol, 
                              mol_sys.n_molecules, 
                              np.reshape(mol_sys.coords[0], (-1,3)), 
                              res_id_aa,
                              mol_sys.elements_list, 
                              box)
        
        try:
            logger.info("Rendering mapping picture for %s", self.molname)
            cmdstring = "vmd -pdb \"" + self.render_path + self.molname + "_CG.pdb" + "\" -e " + self.render_path + "vmd_commands_CG"
            os.system(cmdstring)

            #fix path in the vmd scene file
            vmdinfile = self.render_path + self.molname + "_scene.tmp"
            vmdoutfile = self.render_path + self.molname + "_scene.vmd"
            with open(vmdinfile) as fin, open(vmdoutfile, "w+") as fout:
                for line in fin:
                    absolute_path = str(pathlib.Path().absolute())
                    absolute_path = absolute_path.replace("\\","/")
                    line = line.replace(self.render_path, absolute_path+"/"+ self.render_path)
                    fout.write(line)
            os.remove(vmdinfile)
        except Exception as exceptionWhichOccurred:
            logger.exception("ERROR: Could not call VMD successfully for the following reason:\n%s\nIntermediate files saved anyway. Continuting normally..."%(str(exceptionWhichOccurred)))
        
        # Backmapping        
        self.writer.write_pdb(self.molname + "_BM.pdb", 
                              self.render_path, 
                              mol_sys.n_particles_mol, 
                              mol_sys.n_molecules, 
                              backmapped_xyz, 
                              res_id_aa,
                              mol_sys.elements_list, 
                              box)
        self.create_vmd_commands_AA(self.molname, self.render_path)  
        
        try:
            logger.info("Rendering backmapping picture for %s",self.molname)
            cmdstring = "vmd -pdb \"" +  self.render_path + self.molname + "_BM.pdb" + "\" -e " + self.render_path + "vmd_commands_AA"
            os.system(cmdstring)    #
        except Exception as exceptionWhichOccurred:
            logger.exception("ERROR: Could not call VMD successfully for the following reason:\n%s\nIntermediate files saved anyway. Continuting normally..."%(str(exceptionWhichOccurred)))

    def create_vmd_commands_CG(self,molname,pathtodata):
        """Write options for VMD rendering of atomistic PDB + coarse
        grained PDB, colored by residue name to highlight the CG
        moiety membership. 
        Options written to a file called "vmd_commands_CG"
        The options specify the format of the output rendered file
        depending on available renderers on different operating systems.
        
        Arguments:
        ---------
        molname: str
            used to create file names
        pathtodata: str
            relative path to save "vmd_commands_CG"
        """

        vmdfile = open(pathtodata+"vmd_commands_CG",'w')
        vmdfile.write("mol modcolor 0 0 ResName" + "\n")
        vmdfile.write("mol modstyle 0 0 VDW 1.000000 12.000000" + "\n")
        vmdfile.write("mol modmaterial 0 0 Transparent" + "\n")
        vmdfile.write("mol new " + pathtodata + molname + "_AA.pdb" + "\n")
        vmdfile.write("mol addrep 1" + "\n")
        vmdfile.write("mol modcolor 0 1 ResName" + "\n")
        vmdfile.write("mol modstyle 0 1 CPK 1.000000 0.300000 12.000000 12.000000" + "\n")
        vmdfile.write("display resetview" + "\n")
        
        if os.name == 'nt':
            vmdfile.write("render snapshot " + pathtodata + molname + "_mapping.bmp" + "\n")
        elif os.name == 'posix':
            vmdfile.write("render TachyonInternal " + pathtodata + molname + "_mapping.tga" + "\n")
        
        vmdfile.write("save_state "+ pathtodata + molname + "_scene.tmp" + "\n")
        vmdfile.write("exit")
        vmdfile.close()   

    def create_vmd_commands_AA(self,molname,pathtodata):
        """Write options for VMD rendering of backmapped atomistic 
        PDB + coarse colored by residue name to highlight the CG
        moiety membership. 
        Options written to a file called "vmd_commands_AA"
        The options specify the format of the output rendered file
        depending on available renderers on different operating systems.
        
        Arguments:
        ---------
        molname: str
            used to create file names
        pathtodata: str
            relative path to save "vmd_commands_AA"
        """
        vmdfile = open(pathtodata+"vmd_commands_AA",'w')
        vmdfile.write("mol modcolor 0 0 ResName" + "\n")
        vmdfile.write("mol modstyle 0 0 CPK 1.000000 0.000000 12.000000 12.000000" + "\n")
        vmdfile.write("display resetview" + "\n")
        
        if os.name == 'nt':
            vmdfile.write("render snapshot " + pathtodata + molname +"_backmapping.bmp" + "\n")
        elif os.name == 'posix':
            vmdfile.write("render TachyonInternal " + pathtodata + molname +"_backmapping.tga" + "\n")
        
        #vmdfile.write("save_state "+ pathtodata + molname + "_" + progressive_id + "_scene_AA_backmapped.tmp" + "\n")
        vmdfile.write("exit")
        vmdfile.close()  

    def res_id_list(self, R_effective_CG):
        """Create the mapping list to color the rendering based on
        moiety membership.

        Arguments: 
        ---------
        R_effective_CG: torch tensor
            shape = [number of effective CG, number of atoms]
            It should contain only zeros and ones. There is only one 
            1 in each column j, corresponding to the CG bead the atom j is 
            assigned to.
        n_molecules: int
            number of molecules in the system
        
        Returns:
        -------
        res_id_cg: list of str
            list containing the names given to each CG moiety for 1 molecule 
            Naming convention A00, A01, A02.... until A99
            then B00, B01, B02....
        res_id_aa: list of str
            list containing the name of the CG moiety each atom 
            is assigned to for 1 molecule
        """
        n_cg = R_effective_CG.size()[0]
        n_aa = R_effective_CG.size()[1]
        res_id_cg = [] 
        res_id_aa = [] 
        for i in range(n_cg):
            letter = math.trunc(i / 100) + 1
            rem = i % 100
            if rem < 10:
                res_id_cg.append([chr(ord('@')+letter)+"0"+str(rem)])
            else:
                res_id_cg.append([chr(ord('@')+letter)+str(rem)])            

        for j in range(n_aa):
            idx = np.argmax(R_effective_CG[:,j])
            res_id_aa.append(res_id_cg[idx])
        
        return res_id_cg, res_id_aa
