"""nn.Modules to assemble the architectures"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np

class GumbelDecoder(nn.Module):
    def __init__(self, input_dim, out_dim, n_layers):
        """Gumbel Decoder module for neural network architectures with ReLU activation.

        Parameters:
        -----------
        input_dim: int
            Dimension of the input layer.
        out_dim: int
            Dimension of the output layer.
        n_layers: int
            Number of layers in the decoder. 1 and 2 supported
        TODO: add option to pass a different activation.

        Attributes:
        -----------
        decoder: nn.Module
            Neural network module representing the decoder architecture.
        """
        nn.Module.__init__(self)

        if n_layers == 1:
            self.decoder = nn.Linear(input_dim, out_dim) 
        elif n_layers == 2:
            self.decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim), 
            nn.ReLU(),
            nn.Linear(input_dim, out_dim),
            nn.ReLU())

    def forward(self, encoded):
        """Forward pass of the GumbelDecoder.

        Parameters:
        -----------
        encoded: torch.Tensor
            Input tensor to be decoded.

        Returns:
        --------
        decoded: torch.Tensor
            Output tensor after decoding.
        """
        decoded = self.decoder(encoded) 
        decoded = torch.transpose(decoded,1,2)       
        return decoded

class GumbelVarEncoder(nn.Module):
    """Gumbel Variational Encoder module for neural network architectures."""

    def __init__(self, input_dim, out_dim, scheduler, noise_scaling = 0.1, device = "cpu"):
        """ Parameters:
            -----------
            input_dim: int
                Dimension of the input.
            out_dim: int
                Dimension of the output.
            scheduler: Scheduler
                Learning rate scheduler for temperature in Gumbel-Softmax.
            noise_scaling: float, optional
                Scaling factor for the noise in Gumbel-Softmax, default is 0.1.
            device: str, optional
                Device to run the module on, default is "cpu".

            Attributes:
            -----------
            input_dim: int
                Dimension of the input.
            out_dim: int
                Dimension of the output.
            scheduler: Scheduler
                Learning rate scheduler for temperature in Gumbel-Softmax.
            weight: nn.Parameter
                Weight parameter for Gumbel-Softmax.
            noise_scaling: float
                Scaling factor for the noise in Gumbel-Softmax.
            device: str
                Device to run the module on.
            """
        nn.Module.__init__(self)

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.scheduler = scheduler
        self.weight = Parameter(torch.rand(self.out_dim, self.input_dim).to(device))
        # Since it does not contain any nn.Layer module, it does not have 
        # the default weight attribute  
        self.noise_scaling = noise_scaling
        self.device = device
        
    def gumbel_softmax(self, logits):
        """Apply Gumbel-Softmax sampling to the logits.

        Parameters:
        -----------
        logits: torch.Tensor
            Logit values for Gumbel-Softmax sampling.

        Returns:
        --------
        y: torch.Tensor
            Gumbel-Softmax sampled values.
        """
        temperature = self.scheduler.current_temperature()
        y = self.gumbel_softmax_sample(logits, temperature)
        return y

    def gumbel_softmax_sample(self, logits, temperature):
        """Sample Gumbel-Softmax distribution with given logits and temperature.

        Parameters:
        -----------
        logits: torch.Tensor
            Logit values for Gumbel-Softmax sampling.
        temperature: float
            Temperature parameter for Gumbel-Softmax.

        Returns:
        --------
        y: torch.Tensor
            Sampled values from Gumbel-Softmax distribution.
        """
        noise = self.sample_gumbel(logits.size())
        y = logits + noise * self.noise_scaling
        # below: softmax normalizing over columns (dim=-1), which are the CG moieties
        return F.softmax(y / temperature, dim=-1)

    def sample_gumbel(self, shape, eps=1e-20):
        """Sample values from Gumbel distribution.

        Parameters:
        -----------
        shape: tuple
            Shape of the output tensor.
        eps: float, optional
            Small constant for numerical stability, default is 1e-20.

        Returns:
        --------
        noise: torch.Tensor
            Sampled values from Gumbel distribution.
        """
        U = torch.rand(shape).to(self.device)
        # above: uij - below: gij
        return -torch.log(-torch.log(U + eps) + eps)
    
    def forward(self, xyz): 
        """Forward pass of the GumbelVarEncoder.

        Parameters:
        -----------
        xyz: torch.Tensor
            Input tensor to be encoded.

        Returns:
        --------
        encoded: torch.Tensor
            Encoded tensor using Gumbel-Softmax.
        """
        probabilities = torch.maximum(self.weight.t(),torch.zeros(self.weight.t().size()).to(self.device)+1e-6)
        logits = torch.log(probabilities)
        CG = self.gumbel_softmax(logits).t()
        # Cij
        # size of CG = [N_cg,n_atoms_per_molecule] 
        Eij = CG/CG.sum(1).unsqueeze(1)
        # Eij
        encoded = torch.matmul(Eij.expand(xyz.shape[0], self.out_dim, self.input_dim), xyz)
        return torch.transpose(encoded,1,2)

    def true_forward(self, xyz):
        """Forward pass without Gumbel-Softmax sampling.

        Parameters:
        -----------
        xyz: torch.Tensor
            Input tensor to be encoded.

        Returns:
        --------
        encoded: torch.Tensor
            Encoded tensor without Gumbel-Softmax.
        """
        # xyz need to be on the right device, cpu or cuda
        CG = self.CG()
        Eij = CG/CG.sum(1).unsqueeze(1)
        n_cg = Eij.size()[0]
        n_aa = Eij.size()[1]
        encoded = torch.matmul(Eij.expand(xyz.shape[0], n_cg, n_aa), xyz)
        return torch.transpose(encoded,1,2)

    def effective_mapping(self, xyz):
        """Compute the effective mapping.

        Parameters:
        -----------
        xyz: torch.Tensor
            Input tensor to be encoded.

        Returns:
        --------
        effective_encoded: torch.Tensor
            Effective encoded tensor.
        """
        Eff_R_Eij = self.Eff_R_Eij().to(self.device)
        n_cg = Eff_R_Eij.size()[0]
        n_aa = Eff_R_Eij.size()[1]
        effective_encoded = torch.matmul(Eff_R_Eij.expand(xyz.shape[0], n_cg, n_aa), xyz)
        return effective_encoded

    def CG(self):
        temperature = self.scheduler.current_temperature()
        CG = F.softmax(self.weight.t() / temperature, dim=-1).t()
        return CG

    def effective_CG(self):
        """Compute the effective CG mapping.

        Returns:
        --------
        effective_CG: torch.Tensor

        """
        CG = self.CG()
        assignment_indexes = sorted(np.argmax(CG.detach().cpu().numpy(),axis=0)) 
            #indexes of the GC bead to with each atomistic particle is assigned to .detach()
        used_CG_beads = []
        [used_CG_beads.append(used_CG) for used_CG in assignment_indexes if used_CG not in used_CG_beads]
            #eliminate duplicates from the list of indexes
        effective_CG = CG[used_CG_beads,:] 
        return effective_CG
        
    def rounded_effective_CG(self):
        """Compute the rounded effective (one hot) assignments.

        Returns:
        --------
        R_effective_CG: torch.Tensor

        """
        effective_CG = self.effective_CG()
        n_cg = effective_CG.size()[0]
        n_aa = effective_CG.size()[1]
        R_effective_CG = torch.zeros([n_cg,n_aa])
        idx = np.argmax(effective_CG.detach().cpu().numpy(),axis=0)
        for i_aa in range(n_aa):
            R_effective_CG[idx[i_aa],i_aa] = 1

        return R_effective_CG
    
    def Eff_R_Eij(self):
        """Compute the effective rounded Eij matrix.

        Returns:
        --------
        Eff_Eij: torch.Tensor
            Effective rounded Eij matrix.
            
        """
        R_effective_CG = self.rounded_effective_CG()
        Eff_Eij = R_effective_CG/R_effective_CG.sum(1).unsqueeze(1)
        return Eff_Eij

# Activations
class ShiftedSoftplus(nn.Module):
    """Shifted Softplus activation module for neural network architectures."""

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, input_tensor):
        return nn.functional.softplus(input_tensor) - np.log(2.0)
