import argparse

def init_parser():
    
    """Initialize a command line arguments parser using the "argparse" module.

    The following arguments are included:

    | Shortcut | Full name           | Destination      | Type   | Help                                                                                                                           |
    |----------|---------------------|------------------|--------|--------------------------------------------------------------------------------------------------------------------------------|
    | -in      | --input_file        | input_file       | str    | Input file name                                                                                                                |
    | -ow      | --overwrite         | overwrite        | str    | Overwrite previous output (True) or create a new folder for the current run (False)                                            |
    | -dir     | --directory         | directory        | str    | Relative path to the simulation data                                                                                           |
    | -tr      | --trajectory        | trajectory       | str    | LAMMPS .dump file with the atomistic trajectory. Expected format: id mol type q x y z ix iy iz fx fy fz                        |
    | -df      | --data_file         | data_file        | str    | LAMMPS .data file name                                                                                                         |
    | -nf      | --n_frames          | n_frames         | int    | Number of frames to load from the trajectory, starting from the beginning. Type = int                                          |
    | -nam     | --n_atoms_mol       | n_atoms_mol      | int    | Number of atoms per molecule. Type = int. Currently only systems with all equal molecules are supported                        |
    | -nm      | --n_molecules       | n_molecules      | int    | Number of molecules. Type = int                                                                                                |
    | -pb      | --periodic          | periodic         | str    | Does the systems have periodic boundaries? True or False                                                                       |
    | -lr      | --learning_rate     | learning_rate    | float  | Learning rate. Type = float                                                                                                    |
    | -dr      | --decay_ratio       | decay_ratio      | float  | Decay ratio. Type = float                                                                                                      |
    | -bs      | --batch_size        | batch_size       | int    | Batch size. Type = int                                                                                                         |
    | -pt      | --patience          | patience         | int    | Training patience = consecutive epochs with loss change below min_change will cause training to stop. Type = int               |
    | -mc      | --min_change        | min_change       | float  | After a number of epochs (defined by "patience") with the loss change below min_change, the training will stop. Type = float   |
    | -ne      | --n_epochs          | n_epochs         | int    | Number of training epochs. Type = int                                                                                          |
    | -d       | --device            | device           | str    | Specify the backend for training: cpu or cuda. If cuda is unavailable, it defaults to cpu.                                     |
    | -ncg     | --n_CG_mol          | n_CG_mol         | int    | Number of CG moieties per molecule (size of the latent dimension). Type = int                                                  |
    | -nl      | --n_layers          | n_layers         | int    | Number of layers in the decoder. Type = int                                                                                    |
    | -f       | --feature           | feature          | str    | Feature used as input for the model. Accepted values: coordinates, distances.                                                  |
    | -rho     | --forces_weight     | forces_weight    | float  | Weighting factor for the force term in the loss function.                                                                      |
    | -ls      | --loss_selector     | loss_selector    | str    | Loss function to use. Accepted values: only_rec, only_forces, rec_and_forces, normal_rec_and_forces, normal_rec_forces_connect.|
    | -tm      | --tmin              | tmin             | float  | Lower limit for the temperature parameter in gumbel softmax (Initial value 4). Type = float                                    |
    | -ns      | --noise_scaling     | noise_scaling    | float  | Scaling factor for the gumbel-noise used during training (default 0.1). Type = float                                           |
    | -se      | --save_every        | save_every       | int    | Frequency of saving the model: 0 means only final output. Type = int                                                           |
    | -rd      | --run_description   | run_description  | str    | Description of the run scope                                                                                                   |
    | -out     | --number_of_outputs | number_of_outputs| int    | Number of candidate mappings to generate. Type = int                                                                           |
    | -max     | --max_attempts      | max_attempts     | int    | Maximum number of mapping trials. Type = int                                                                                   |
    | -vmd     | --VMD_flag          | VMD_flag         | str    | Output VMD renderings. True or False                                                                                           |
    | -sv      | --save_also         | save_also        | int    | Other property to export during training. Accepted values: n_CG (number of used GC beads - type = int)                         |                    
    | -plt     | --plot_flag         | plot_flag        | str    | Output loss plots and assignment. True or False                                                                                |
    
    """

    parser = argparse.ArgumentParser(add_help=False)

    #### File names ####

    parser.add_argument('-in', '--input_file',
                        dest='input_file',
                        action = 'store',
                        help='Input file name',
                        type=str
                        )
    parser.add_argument('-ow', '--overwrite',
                        dest='overwrite',
                        action = 'store',
                        help='Overwrite previous output (True) or create a new folder for the current run (False)',
                        type=str
                        )
    parser.add_argument('-dir', '--directory',
                        dest='directory',
                        action = 'store',
                        help='Relative path to the simulation data',
                        type=str
                        )
    parser.add_argument('-tr', '--trajectory',
                        dest='trajectory',
                        action = 'store',
                        help='LAMMPS .dump file with the atomistic trajectory. Expected format : id mol type q x y z ix iy iz fx fy fz',
                        type=str
                        )
    parser.add_argument('-df', '--data_file',
                        dest='data_file',
                        action = 'store',
                        help='LAMMPS .data file name',
                        type=str
                        )

    #### Simulation data ####

    parser.add_argument('-nf', '--n_frames',
                    dest='n_frames',
                    action = 'store',
                    help='Number of frames to load from the trajectory, starting from the beginnning. Type = int',
                    type=int
                    )
    parser.add_argument('-nam', '--n_atoms_mol',
                dest='n_atoms_mol',
                action = 'store',
                help='Number of atoms per molecule. Type = int. Currently only systems with all equal molecules are supported',
                type=int
                )                
    parser.add_argument('-nm', '--n_molecules',
                    dest='n_molecules',
                    action = 'store',
                    help='Number of molecules. Type = int',
                    type=int
                    )
    parser.add_argument('-pb', '--periodic',
                    dest='periodic',
                    action = 'store',
                    help='Does the systems have periodic boundaries? True or False',
                    type = str,
                    )

    #### Training Parameters ####

    parser.add_argument('-lr', '--learning_rate',
                    dest='learning_rate',
                    action = 'store',
                    help='Learning rate. Type = float',
                    type=float
                    )
    parser.add_argument('-dr', '--decay_ratio',
                    dest='decay_ratio',
                    action = 'store',
                    help='Decay ratio. Type = float',
                    type=float
                    )
    parser.add_argument('-bs', '--batch_size',
                    dest='batch_size',
                    action = 'store',
                    help='Batch size. Type = int',
                    type=int
                    )
    parser.add_argument('-pt', '--patience',
                    dest='patience',
                    action = 'store',
                    help='Training patience = consecutive epochs with loss change below min_change will cause training to stop. Type = int',
                    type=int
                    )
    parser.add_argument('-mc', '--min_change',
                    dest='min_change',
                    action = 'store',
                    help='After a number of epochs (defined by "patience") with the loss change below min_change the training will stop. Type = float',
                    type=float
                    )
    parser.add_argument('-ne', '--n_epochs',
                    dest='n_epochs',
                    action = 'store',
                    help='Number of training epochs. Type = int',
                    type=int
                    )

    parser.add_argument('-d', '--device',
                    dest='device',
                    action = 'store',
                    help='Specify the backend for training: cpu or cuda. If cuda is unavailable, it defaults to cpu.',
                    type=str
                    )

    parser.add_argument('-ncg', '--n_CG_mol',
                    dest='n_CG_mol',
                    action = 'store',
                    help='Number of CG moieties per molecule (size of the latent dimension). Type = int',
                    type=int
                    )
    
    parser.add_argument('-nl', '--n_layers',
                    dest='n_layers',
                    action = 'store',
                    help='Number of layers in the decoder. Type = int',
                    type=int
                    )
    
    parser.add_argument('-f', '--feature',
                    dest='feature',
                    action = 'store',
                    help='Feature used as input for the model. Accepted values: coordinates, distances.',
                    type=str
                    )
    
    parser.add_argument('-rho', '--forces_weight',
                    dest='forces_weight',
                    action = 'store',
                    help='Weighting factor for the force term in the loss function.',
                    type=float
                    )
    
    parser.add_argument('-ls', '--loss_selector',
                    dest='loss_selector',
                    action = 'store',
                    help='Loss function to use. Accepted values: only_rec, only_forces, rec_and_forces, normal_rec_and_forces, normal_rec_forces_connect.',
                    type=str
                    )
    
    parser.add_argument('-tm', '--tmin',
                    dest='tmin',
                    action = 'store',
                    help='Lower limit for the temperature parameter in gumbel softmax (Initial value 4). Type = float',
                    type=float
                    )   
                     
    parser.add_argument('-ns', '--noise_scaling',
                    dest='noise_scaling',
                    action = 'store',
                    help='Scaling factor for the gumbel-noise used during training (default 0.1). Type = float',
                    type=float
                    )   

    #### Output Parameters ####           

    parser.add_argument('-se', '--save_every',
                    dest='save_every',
                    action = 'store',
                    help='Frequency of saving the model: 0 means only final output. Type = int',
                    type=int
                    ) 
    parser.add_argument('-rd', '--run_description',
                    dest='run_description',
                    action = 'store',
                    help='Description of the run scope',
                    type=str
                    )
   
    parser.add_argument('-out', '--number_of_outputs',
                    dest='number_of_outputs',
                    action = 'store',
                    help='Number of candidate mappings to generate. Type = int',
                    type = int,
                    ) 
    parser.add_argument('-max', '--max_attempts',
                    dest='max_attempts',
                    action = 'store',
                    help='Maximum number of mapping trials. Type = int',
                    type = int,
                    ) 
    parser.add_argument('-vmd', '--VMD_flag',
                    dest='VMD_flag',
                    action = 'store',
                    help='Output VMD renderings. True or False',
                    type = str,
                    )       
    parser.add_argument('-sv', '--save_also',
                    dest='save_also',
                    action = 'store',
                    help='Other property to export during training. Accepted values: n_CG (number of used GC beads - type = int)',
                    type=int
                    ) 
    parser.add_argument('-plt',
                    dest='plot_flag',
                    action = 'store',
                    help='Output loss plots and assignment. True or False',
                    type = str,
                    ) 
    
    return parser