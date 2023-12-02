#################################################
"""
Comments 

"""

import os
import torch
from torch.utils.data import DataLoader

import numpy as np
import logging

from inout.logger import Logger
from inout.initializer import Initializer
from inout.renderer import Renderer
from inout.argparser import init_parser
from inout.renderer import Renderer
from inout.plotter import Plotter
from inout.file_writer import MolecularFilesWriter

from training.scheduler import TScheduler
from training.trainer import ModelTrainer 
from training.dataset import *
from training.mapping_evaluator import MappingEvaluator

from models.mapping_model import *
from models.layers import *

from molecular_system.connectivity import Connectivity


def main():
    #################################################
    # Global settings

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    SEED = 2345
    if SEED is not None:
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(SEED)

    #################################################
    # Initialize command line arguments parser based on the "argparse" module
    # and parse command line arguments
    parser = init_parser()
    cmdl_args = parser.parse_args()

    if cmdl_args.input_file is None:
        input_file = 'vae_cg_mapping.in'
    else:
        input_file = cmdl_args.input_file

    #################################################
    # Input & Preprocessing

    atom_sys, parameters = Initializer(input_file).initialize_for_mapping(cmdl_args)
    connect = Connectivity(atom_sys)
    connect.bond_matrix = connect.create_bond_matrix(atom_sys.bonds_list)  
    # Select the input feature for the mapping model 
    if parameters["feature"]=="distances":
        feature = connect.intramolecular_distances(atom_sys)
    elif parameters["feature"]=="coordinates":
        feature = atom_sys.coords

    # Configure logger
    log_level = logging.INFO

    #Log molecular system information
    logger = Logger(name = "maplog", level = log_level, filename = parameters["res_folder"] + 'mapping.log')
    logger.info("MAPPING TEST")
    logger.info("System: " + parameters["molname"])
    logger.info("N frames: " + str(atom_sys.n_frames) + " - N molecules: " + str(atom_sys.n_molecules))
    logger.info("RUN PARAMETERS")
    logger.write_parameters(parameters)

    # Examine CUDA support
    if parameters["device"] == 'cpu':
        device = "cpu"
        logger.info("Using CPU backend.")
    elif parameters["device"] == 'cuda':
        if torch.cuda.is_available():
        # and use first device if available
            device = "cuda:0"
            logger.info("Using CUDA backend.")
        else:
            device = "cpu"
            logger.info("CUDA not available, using CPU backend instead.")

    device = torch.device(device)

    ###############################################
    # Split data in train and test sets

    train_amount = float(parameters["train_amount"])

    if eval(parameters["periodic"]):
        train_dataset, test_dataset = periodic_dataset_mapping(atom_sys, feature, train_amount, device)
    else:
        train_dataset, test_dataset = not_periodic_dataset_mapping(atom_sys, feature, train_amount, device)

    train_loader = DataLoader(train_dataset, batch_size=int(parameters["batch_size"]), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=int(parameters["batch_size"]), shuffle=True)

    num_batches = round(train_dataset.lenght / int(parameters["batch_size"]) + 0.5) + \
                round(test_dataset.lenght / int(parameters["batch_size"]) + 0.5)

    #################################################
    # Produce the required number of acceptable mappings
    acceptable_results = 0
    mapping_attempt = 0
    molname = parameters["molname"]
    acceptable_mappings = []

    while (acceptable_results < int(parameters["number_of_outputs"])) \
         & (mapping_attempt < int(parameters["max_attempts"])):

        logger.info("Mapping attempt: " + str(mapping_attempt+1) + "/" + parameters["max_attempts"])
        parameters["molname"] = molname + "_" + str(acceptable_results+1)

        #################################################
        # Build model architecture and trainer

        scheduler = TScheduler(Tstart = float(parameters["Tstart"]),
                            tmin = float(parameters["tmin"]),
                            n_epochs = int(parameters["n_epochs"]),
                            decay_ratio = float(parameters["decay_ratio"]))

        encoder = GumbelVarEncoder(input_dim = atom_sys.n_particles_mol, 
                                out_dim = int(parameters["n_CG_mol"]), 
                                scheduler = scheduler, 
                                noise_scaling = float(parameters["noise_scaling"]), 
                                device = device)
        decoder = GumbelDecoder(int(parameters["n_CG_mol"]), atom_sys.n_particles_mol, int(parameters["n_layers"]))
        loss_function = LossSelector(parameters).select_function(encoder, connect)
        model = AutoEncoder(encoder, decoder, loss_function)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=float(parameters["learning_rate"]))

        trainer = ModelTrainer(model, optimizer, scheduler, parameters, parameters["task"], logger.name)

        mapping_evaluator = MappingEvaluator()

        #################################################
        # Train the model
        trainer.train(int(parameters["n_epochs"]), train_loader, test_loader)

        #################################################
        # Evaluate the output
        acceptable = mapping_evaluator.is_acceptable(encoder, connect)

        if acceptable:

            R_effective_CG = encoder.rounded_effective_CG()
            
            if len(acceptable_mappings)>0:
                unique = mapping_evaluator.is_unique(R_effective_CG, acceptable_mappings)
            else:
                unique = True

            if unique:
                acceptable_mappings.append(R_effective_CG)
                acceptable_results = acceptable_results + 1  
                logger.info("Mapping was succesful, result " + str(acceptable_results)+"/"+parameters["number_of_outputs"]+" found")  
                R_effective_CG = encoder.rounded_effective_CG()

                #################################################
                # Graphical output

                if eval(parameters["VMD_flag"]) == True:
                    renderer = Renderer(parameters["molname"],
                                        parameters["res_folder"], 
                                        logger_name = logger.name, 
                                        device = device)
                    renderer.mapping_backmapping(model, atom_sys)

                if eval(parameters["plot_flag"]) == True:
                    plotter = Plotter(parameters["molname"],
                                    parameters["res_folder"])
                    logger.info("Plotting assignment matrix")
                    plotter.plot_assignment(encoder, atom_sys.elements_list)

                    loss_function.reduce_loss(num_batches)
                    losses, legend = plotter.get_losses(trainer, loss_function)
                    plotter.plot_loss(losses, legend)

                    plotter.scatter_plot(np.arange(0,len(trainer.n_cg_training),1),trainer.n_cg_training,
                                        "Epochs","N CG","_effective_cg")

                writer = MolecularFilesWriter(logger.name)
                writer.write_assignments(parameters["molname"], parameters["res_folder"], R_effective_CG)
                
            else:
                logger.info("Mapping was not unique")  
        else:
            logger.info("Mapping was not acceptable")

        mapping_attempt = mapping_attempt +1

    if mapping_attempt >= int(parameters["max_attempts"]):
        logger.info("Maximum number of mapping attempts reached")

    print("Done")

# Run main function, if appropriate
if __name__ == "__main__":
    main()
