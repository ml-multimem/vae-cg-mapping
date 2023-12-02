from torch.nn.modules.loss import _Loss
import torch.nn as nn
import numpy as np
import math
import torch
import networkx as nx

class LossSelector():
    def __init__(self, parameters):
        """Extract parameters to pass to the loss function for initialization
        """
        self.loss_selector = parameters["loss_selector"]
        self.forces_weight = float(parameters["forces_weight"])
    
    def select_function(self, encoder, connectivity):
        """Choose and initialize the loss function for the mapping task.
        Possible choices:
        - only_rec: reconstruction loss calculated using the Mean Squared Error    
                    loss = loss_rec
        - only_forces: mean squared force in the CG space loss
                    loss = loss_mf
        - rec_and_forces: reconstruction + mean squared force in the CG space loss
                    weighted according to the forces_weight parameter
                    loss = loss_rec + forces_weight * loss_mf
        - normal_rec_and_forces: as above, but normalized using the largest 
                    loss value seen until each epoch. 
                    forces_weight should be between 0 and 1
                    loss = loss_rec_norm * (1- forces_weight) + 
                                 forces_weight * loss_mf_norm
        - normal_rec_forces_connect: as above, but with added penalty if 
                    unconnected atoms are assigned to the same moiety.
                    loss_connect is computed using a Noise to Signal definition
                    loss = oss_rec_norm * (1- forces_weight) + 
                                 forces_weight * loss_mf_norm + loss_connect
        
        Parameters:
        -----------                        
        encoder: nn.Module
            The encoder used for mapping atomistic forces to coarse-grained forces.
        connectivity: Connectivity object
            Object containing the information about the atomistic connectivity.

        Returns:
        -------
        loss_function: the initialized loss function
        
        """
        if self.loss_selector == "only_rec":
            loss_function = ReconstructionLoss()
        elif self.loss_selector == "only_forces":
            loss_function = AverageForceLoss(encoder)
        elif self.loss_selector == "rec_and_forces":
            loss_function = RecAndForceLoss(encoder, self.forces_weight)
        elif self.loss_selector == "normal_rec_and_forces":
            loss_function = RecAndForceNormalizedLoss(encoder, self.forces_weight)
        elif self.loss_selector == "normal_rec_forces_connect":
            loss_function = RecForceConnectLoss(encoder, self.forces_weight, connectivity)
        else:    
            raise Exception("Loss function %s undefined", self.loss_selector) 
        return loss_function

class ReconstructionLoss(_Loss):
    """ Custom reconstruction loss for autoencoders.
        This class extends the PyTorch `_Loss` class and is designed for
        evaluating the reconstruction loss during training of autoencoders."""
    
    def __init__(self):
        
        """
        Attributes:
        -----------
        loss_reconstruction: list
            A list to store the trend of reconstruction losses during training.
        criterion: nn.MSELoss
            Mean Squared Error (MSE) loss function for evaluating the reconstruction loss.

        """
        nn.Module.__init__(self)

        #Store the trend during training
        self.loss_reconstruction = []
        self.criterion = nn.MSELoss()

    def eval_loss(self, *args):
        """
        Evaluate the reconstruction loss.

        Parameters:
        -----------
        *args: tuple
            Variable-length argument list, where args[0] contains predictions and args[1] contains the batch.

        Returns:
        --------
        loss_rec: torch.Tensor
            Reconstruction loss.
        """

        args = args[0]
        predictions = args[0]
        batch = args[1]
        labels = batch["input"]
        loss_rec = self.criterion(predictions, labels)
        self.loss_reconstruction.append(loss_rec.detach())

        return loss_rec
    
    def reduce_loss(self,batch_size):
        """
        Store the average epoch reconstruction loss.

        Parameters:
        -----------
        batch_size: int
            Size of the batch used during training.
        """

        old_loss = self.loss_reconstruction
        self.loss_reconstruction = []
        for i in range(0,len(old_loss)-batch_size,batch_size):
            self.loss_reconstruction.append(np.mean(np.float_(old_loss[i:i+batch_size])))
            
class AverageForceLoss(_Loss):
    """
    Custom loss for training autoencoders with force information.
    This class extends the PyTorch `_Loss` class.
    """
    def __init__(self, encoder):
        """
        Attributes:
        -----------
        encoder: nn.Module
            The encoder used for mapping atomistic forces to coarse-grained forces.
        loss_mean_forces: list
            A list to store the trend of average force losses during training.

        """
        nn.Module.__init__(self)
        #self.mapper = AtomisticToCGMapper()
        self.encoder = encoder
        #Store the trend during training
        self.loss_mean_forces = []

    def eval_loss(self, *args):
        """
        Evaluate the average force loss.

        Parameters:
        -----------
        *args: tuple
            Variable-length argument list, where args[0] contains predictions and args[1] contains the batch.

        Returns:
        --------
        loss_mf: torch.Tensor
            Average mean force loss.
        """

        args = args[0]
        batch = args[1]
        forces = batch["forces"].detach()
            # It is necessary to detach this from the graph otherwise 
            # the "retain_graph = True" error is triggered
        CG = self.encoder.CG()
        f0 = forces.reshape(-1, CG.size()[-1], 3)
        f = torch.matmul(CG, f0)   
        loss_mf = f.pow(2).sum(2).mean() #mean square scalar force
        self.loss_mean_forces.append(loss_mf.detach())

        return loss_mf

    def get_encoder(self):
        """
        Get the encoder used in the loss.

        Returns:
        --------
        encoder: nn.Module
            The encoder used for mapping atomistic forces to coarse-grained forces.
        """
        return self.encoder
    
    def reduce_loss(self, batch_size):
        """
        Store the average epoch mean force loss.

        Parameters:
        -----------
        batch_size: int
            Size of the batch used during training.
        """

        old_loss = self.loss_mean_forces
        self.loss_mean_forces = []
        for i in range(0,len(old_loss)-batch_size,batch_size):
            self.loss_mean_forces.append(np.mean(np.float_(old_loss[i:i+batch_size])))

class RecAndForceLoss(_Loss):
    """
    Combined loss function for training autoencoders with both reconstruction and force information.
    This class extends the PyTorch `_Loss` class.
    """
    def __init__(self, encoder, forces_weight):
        """
        
        Attributes:
        -----------
        encoder: nn.Module
            The encoder used for mapping atomistic forces to coarse-grained forces.
        forces_weight: float
            Weighting factor for the average force loss in the combined loss.
        rec_loss: ReconstructionLoss
            Instance of the ReconstructionLoss class for handling reconstruction loss.
        avg_force_loss: AverageForceLoss
            Instance of the AverageForceLoss class for handling average force loss.
        loss_reconstruction: list
            A list to store the trend of reconstruction losses during training.
        loss_mean_forces: list
            A list to store the trend of average force losses during training.
        """

        nn.Module.__init__(self)

        self.rec_loss = ReconstructionLoss()
        self.avg_force_loss = AverageForceLoss(encoder)
        self.forces_weight = forces_weight
        self.loss_reconstruction = []
        self.loss_mean_forces = []

    def eval_loss(self, *args):
        """
        Evaluate the combined reconstruction and average force errors.

        Parameters:
        -----------
        *args: tuple
            Variable-length argument list containing the input data.

        Returns:
        --------
        loss: torch.Tensor
            Combined loss of reconstruction and average force.
        """
        loss_rec = self.rec_loss.eval_loss(args[0])
        loss_mf = self.avg_force_loss.eval_loss(args[0])
        loss = loss_rec + loss_mf * self.forces_weight
        return loss
    
    def reduce_loss(self,batch_size):
        """
        Store the average epoch mean force and reconstruction loss components.

        Parameters:
        -----------
        batch_size: int
            Size of the batch used during training.
        """
        self.rec_loss.reduce_loss(batch_size)
        self.loss_reconstruction = self.rec_loss.loss_reconstruction
        self.avg_force_loss.reduce_loss(batch_size)
        self.loss_mean_forces = self.avg_force_loss.loss_mean_forces

class RecAndForceNormalizedLoss(_Loss):
    """
    Combined loss function for training autoencoders with normalized reconstruction and force information.
    This class extends the PyTorch `_Loss` class.
    """
    def __init__(self, encoder, forces_weight):
        """
        Attributes:
        -----------
        encoder: nn.Module
            The encoder used for mapping atomistic forces to coarse-grained forces.
        forces_weight: float
            Weighting factor for the average force loss in the combined loss.
        rec_loss: ReconstructionLoss
            Instance of the ReconstructionLoss class for handling reconstruction loss.
        avg_force_loss: AverageForceLoss
            Instance of the AverageForceLoss class for handling average force loss.
        unit_of_reconstruction_loss: float
            Unit used for normalization of the reconstruction loss.
        unit_of_force_loss: float
            Unit used for normalization of the average force loss.
        loss_reconstruction: list
            A list to store the trend of normalized reconstruction losses during training.
        loss_mean_forces: list
            A list to store the trend of normalized average force losses during training.
        """

        nn.Module.__init__(self)

        self.rec_loss = ReconstructionLoss()
        self.avg_force_loss = AverageForceLoss(encoder)
        self.forces_weight = forces_weight
        self.unit_of_reconstruction_loss = 0
        self.unit_of_force_loss = 0
        self.loss_reconstruction = []
        self.loss_mean_forces = []

    def eval_loss(self, *args):
        """
        Evaluate the combined normalized reconstruction and average force errors.

        Parameters:
        -----------
        *args: tuple
            Variable-length argument list containing the input data.

        Returns:
        --------
        loss: torch.Tensor
            Combined loss of normalized reconstruction and average force.
        """

        loss_rec = self.rec_loss.eval_loss(args[0])
        loss_mf = self.avg_force_loss.eval_loss(args[0])
        
        # Check and update the maximum value encountered so far
        max_loss_rec=float(max(self.unit_of_reconstruction_loss,loss_rec))
        max_loss_mf=float(max(self.unit_of_force_loss,loss_mf))
        self.unit_of_reconstruction_loss = max_loss_rec
        self.unit_of_force_loss = max_loss_mf
        
        # Normalize and store
        loss_rec = loss_rec / max_loss_rec
        self.loss_reconstruction.append(loss_rec.detach())
        loss_mf = loss_mf / max_loss_mf
        self.loss_mean_forces.append(loss_mf.detach())
        loss = (1-self.forces_weight)*loss_rec + self.forces_weight * loss_mf 
        
        return loss

    def reduce_loss(self,batch_size):
        """
        Store the average epoch mean normalized force and reconstruction loss components.

        Parameters:
        -----------
        batch_size: int
            Size of the batch used during training.
        """

        old_loss = self.loss_reconstruction
        self.loss_reconstruction = []
        for i in range(0,len(old_loss)-batch_size,batch_size):
            self.loss_reconstruction.append(np.mean(np.float_(old_loss[i:i+batch_size])))

        old_loss = self.loss_mean_forces
        self.loss_mean_forces = []
        for i in range(0,len(old_loss)-batch_size,batch_size):
            self.loss_mean_forces.append(np.mean(np.float_(old_loss[i:i+batch_size])))

class RecForceConnectLoss(_Loss):
    """
    Combined loss function for training autoencoders with normalized reconstruction, average force,
    and connectivity preservation.
    This class extends the PyTorch `_Loss` class.
    """
    def __init__(self, encoder, forces_weight, connectivity):
        """
        Attributes:
        -----------
        encoder: nn.Module
            The encoder used for mapping atomistic forces to coarse-grained forces.
        forces_weight: float
            Weighting factor for the average force loss in the combined loss.
        connectivity: Connectivity object
            Object containing the atomistic connectivity information.
        rec_and_force_norm_loss: RecAndForceNormalizedLoss
            Instance of the RecAndForceNormalizedLoss class for handling normalized reconstruction and force losses.
        unit_of_reconstruction_loss: float
            Unit used for normalization of the reconstruction loss.
        unit_of_force_loss: float
            Unit used for normalization of the average force loss.
        loss_connectivity: list
            A list to store the trend of connectivity preservation losses during training.
        loss_reconstruction: list
            A list to store the trend of normalized reconstruction losses during training.
        loss_mean_forces: list
            A list to store the trend of normalized average force losses during training.
        """

        nn.Module.__init__(self)

        self.rec_and_force_norm_loss = RecAndForceNormalizedLoss(encoder, forces_weight)
        self.forces_weight = forces_weight
        self.unit_of_reconstruction_loss = 0
        self.unit_of_force_loss = 0
        self.connectivity = connectivity
        self.encoder = encoder

        #Store the trend during training   
        self.loss_connectivity = []   
        self.loss_reconstruction = []
        self.loss_mean_forces = []   
    
    def eval_loss(self, *args):
        """
        Evaluate the combined error of normalized reconstruction, average force, and connectivity preservation.

        Parameters:
        -----------
        *args: tuple
            Variable-length argument list containing the input data.

        Returns:
        --------
        loss: torch.Tensor
            Combined loss of normalized reconstruction, average force, and connectivity preservation.
        """

        loss_rec_forces = self.rec_and_force_norm_loss.eval_loss(args[0])
        loss_NtoS = self.compute_loss_NtoS(self.encoder)
        self.loss_connectivity.append(loss_NtoS)
        loss = loss_rec_forces + loss_NtoS
        
        return loss

    def compute_loss_NtoS(self, encoder):
        """
        Compute the loss for preserving connectivity (Noise to Signal approach) using the provided encoder.

        Parameters:
        -----------
        encoder: nn.Module
            The encoder used for computing the mapped connectivity.

        Returns:
        --------
        loss_NtoS: float
            The computed Noise to Signal ratio for correct connectivity preservation.
        """

        R_effective_CG = encoder.rounded_effective_CG()
        R_effective_CG = R_effective_CG.numpy()
        noise = 0
        signal = 0
        molecule_bond_matrix = self.connectivity.bond_matrix[0]
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
        
        loss_NtoS = self.NtoS(noise,signal)
        
        return loss_NtoS

    def NtoS(self, noise, signal):
        """
        Compute the normalized-to-signal (N-to-S) loss using noise and signal components.

        Parameters:
        -----------
        noise: int
            The noise component (broken connectivity).
        signal: int
            The signal component (preserved connectivity).

        Returns:
        --------
        loss_NtoS: float
            The computed N-to-S loss.
        """

        MIN_PENALTY = 3 
        return max(0.0, MIN_PENALTY + math.log10((10e-8 + noise)/(10e-8 + signal)))

    def reduce_loss(self, batch_size):
        """
        Store the average epoch mean normalized force, reconstruction, and connectivity loss components.

        Parameters:
        -----------
        batch_size: int
            Size of the batch used during training.
        """

        self.rec_and_force_norm_loss.reduce_loss(batch_size)
        self.loss_reconstruction = self.rec_and_force_norm_loss.loss_reconstruction
        self.loss_mean_forces = self.rec_and_force_norm_loss.loss_mean_forces

        old_loss = self.loss_connectivity
        self.loss_connectivity = []
        for i in range(0,len(old_loss)-batch_size,batch_size):
            self.loss_connectivity.append(np.mean(np.float_(old_loss[i:i+batch_size])))

