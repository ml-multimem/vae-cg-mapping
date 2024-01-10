# vae-cg-mapping

Automatic Coarse Grained mapping of organic molecules using Autoencorder Neural Networks.


<img src="https://github.com/ml-multimem/vae-cg-mapping/blob/main/schema.png" width=50% height=50%>


**Reference paper**:
- D. Nasikas, E. Ricci, G. Giannakopoulos, V. Karkaletsis, D.N. Theodorou, N. Vergadou. "Investigation of Machine Learning-based Coarse-Grained Mapping Schemes for Organic Molecules". In: *Proceedings of the 12th Hellenic Conference on Artificial Intelligence (SETN2022)*, ACM, New York, NY, USA, 2022: pp. 1–8. https://doi.org/10.1145/3549737.3549792.

## Overview

Train an Autoencorder Neural Network to produce one or multiple Coarse Grained mapping(s) for an atomistic system.
The model is trained on an atomistic trajectory, and the produced mappings are evaluated for uniqueness and connectivity preservation.

## Table of Contents
- [Features](#features)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Input Parameters](#input-parameters)

## Features

- Implementation of multicomponent loss functions to account for reconstruction error, mean force normalization, and connectivity preservation
- Produce multiple unique mappings at a given resolution level (number of CG beads)
- Save CG mapping as moiety assignment list, CG molecule .pdb file, and VMD render 

## Usage

1. Fill in the required entries in the input script "vae_cg_mapping.in" and supply the input data in the required format (more on supported format(s) below).
2. Run the script: `python main_vae_cg_mapping.py`
3. Find the results into a subfolder of your input data folder, named "mapping_<current_date_and_time>"

## Dependencies
Tested with the following package versions:
- python = 3.7.6
- torch = 1.7.0
- numpy = 1.21.6
- matplotlib = 3.5.1
- logging = 0.5.1.2
- networkx = 2.2
- argparse = 1.1

## Input Parameters

- `directory`: Relative path to the simulation data.
- `trajectory`: LAMMPS .dump file with the atomistic trajectory. Expected format : id mol type q x y z ix iy iz fx fy fz
- `data_file`: LAMMPS .data file name. Expected format: "full"
- `n_frames`: Number of frames to load from the trajectory.
- `n_atoms_mol`: Number of atoms per molecule.
- `n_molecules`: Number of molecules.
- `periodic`: Boolean indicating whether the system has periodic boundaries.


- `learning_rate`: Learning rate for the model.
- `decay_ratio`: Decay ratio for the temperature parameter in Gumbel-Softmax.
- `batch_size`: Batch size for training.
- `patience`: Training patience (consecutive epochs with loss change below min_change).
- `min_change`: Minimum change in loss to continue training.
- `n_epochs`: Number of training epochs.
- `device`: Specify the backend for training (cpu or cuda).
  

- `n_CG_mol`: Number of CG moieties per molecule (latent dimension of teh autoencoder).
- `n_layers`: Number of layers in the decoder.
- `feature`: Feature used as input for the model (accepted values: coordinates, distances).
- `forces_weight`: Weighting factor for the force term in the loss function.
- `loss_selector`: Loss function to use (accepted values: only_rec, only_forces, rec_and_forces, normal_rec_and_forces, normal_rec_forces_connect).
- `tmin`: Lower limit for the temperature parameter in Gumbel-Softmax.
- `noise_scaling`: Scaling factor for the Gumbel noise during training.
- `train_amount`: Fraction of data used for training.


- `number_of_outputs`: Number of candidate mappings to generate.
- `max_attempts`: Maximum number of mapping trials.
- `overwrite`: Overwrite previous output (yes) or create a new folder for the current run (no).
- `save_every`: Frequency of saving the model (0 means only final output).
- `VMD_flag`: Output VMD renderings (True or False).
- `plot_flag`: Output loss plots and assignment (True or False).
- `save_also`: Other property to export during training (e.g., n_CG).

Inspired by [Wang, Gómez-Bombarelli, npj Comput Mater 5, 125 (2019)](https://github.com/learningmatter-mit/Coarse-Graining-Auto-encoders)
