import torch.nn as nn
from training.loss import *

class AutoEncoder(nn.Module):
    """AutoEncoder class representing an autoencoder neural network."""
    def __init__(self, encoder, decoder, loss_function):
        """
        Parameters:
        -----------
        encoder: nn.Module
            The encoder module responsible for encoding input data.
        decoder: nn.Module
            The decoder module responsible for decoding encoded data.
        loss_function: LossFunction
            The loss function used for computing the reconstruction loss.

        """
        nn.Module.__init__(self)
        
        self.encoder = encoder
        self.decoder = decoder
        self.loss_function = loss_function
    
    def forward(self, *args):
        """
        Forward pass through the autoencoder.

        Parameters:
        -----------
        *args: Variable-length argument list
            Input arguments. Expected to contain a dictionary 'batch' with a key 'input' representing the input data.

        Returns:
        --------
        decoded: torch.Tensor
            Decoded output from the autoencoder.

        """

        batch = args[0]
        data = batch["input"]
        encoded = self.encoder(data)
        decoded = self.decoder(encoded)

        return decoded

    def loss(self, *args):
        """
        Compute the reconstruction loss using the provided loss function.

        Parameters:
        -----------
        *args: Variable-length argument list
            Input arguments. Expected to contain a dictionary 'batch' with a key 'input' representing the input data.

        Returns:
        --------
        loss: torch.Tensor
            Loss computed using the provided loss function.
        """

        loss = self.loss_function.eval_loss(args[0])

        return loss
