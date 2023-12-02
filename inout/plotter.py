import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import inspect

class Plotter():
    """Class to plot training output"""
    
    def __init__(self, molname,
                 plot_directory):
        """
        Initialize the Plotter object.

        Parameters:
        -----------
        molname: str
            Name of the molecule for which the plots will be generated.
        plot_directory: str
            Directory to save the plots. If the directory does not exist, it will be created.
        """
        self.molname = molname
        self.plot_directory = plot_directory
        if not os.path.exists(plot_directory):
            os.mkdir(plot_directory) 

    def scatter_plot(self, x, y, xlabel = None, ylabel= None, plotname = "plot"):
        """Save a scatter plot of series x and y

        Arguments:
        ---------
        x, y: 1-dimensional numpy array or torch tensor
        xlabel, ylabel, plotname: str
            used to set the axes labels and the name of the plot
        """

        plt.figure()
        plt.rcParams['font.size'] = 18
        plt.plot(x,y)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        plt.savefig(self.plot_directory + self.molname + plotname+".png",
                    bbox_inches="tight")
        plt.close('all')
        plt.clf()
        plt.cla()

    def plot_assignment(self, encoder, elements_list):
        """
        Save the assignment matrix with a colormap to indicate the confidence of the assignment.

        Parameters:
        -----------
        encoder: torch module
            The encoder module.
        elements_list: dict
            Dictionary mapping element indices to their symbols.
        """

        CG = encoder.CG()
        n_cg = CG.size()[0]
        n_aa = CG.size()[1]
        if n_cg == 1:   
            CG = torch.ones(n_cg,n_aa)

        plt.figure(figsize=(25, 10))
        plt.rcParams['font.size'] = 18
        plt.imshow(CG.detach().cpu().numpy() , aspect='auto', vmin=0, vmax=1)
        plt.colorbar()
        #plt.xticks(np.arange(n_aa), ["A" + str(i+1) for i in range(n_aa)])
        plt.xticks(np.arange(n_aa), [elements_list[str(i)] for i in range(n_aa)])
        plt.yticks(np.arange(n_cg), ["CG" + str(i+1) for i in range(n_cg)])
        plt.savefig(self.plot_directory+self.molname+"_assignments.png")
        plt.close('all')

        effective_CG = encoder.rounded_effective_CG()
        n_cg = effective_CG.size()[0]
        n_aa = effective_CG.size()[1]
        if n_cg == 1:   
            effective_CG = torch.ones(n_cg,n_aa)

        plt.figure(figsize=(25, 10))
        plt.rcParams['font.size'] = 18
        plt.imshow(effective_CG.detach().cpu().numpy() , aspect='auto', vmin=0, vmax=1)
        plt.colorbar()
        #plt.xticks(np.arange(n_aa), ["A" + str(i+1) for i in range(n_aa)])
        plt.xticks(np.arange(n_aa), [elements_list[str(i)] for i in range(n_aa)])
        plt.yticks(np.arange(n_cg), ["CG" + str(i+1) for i in range(n_cg)])
        plt.savefig(self.plot_directory+self.molname+"_effective_assignments.png")
        plt.close('all')

    def plot_loss(self, losses, legend):
        """
        Plot the loss over the training set and the test set during training, as well as the various loss components.

        Parameters:
        -----------
        losses: list of numpy array or torch tensor
            Shape [number of losses][number of epochs].
            The first one is the loss over the training set, the second one is the loss over the test set, then there are the loss components in alphabetical order.
        legend: list of str
            Names for the series in the plot.
        """

        if len(losses)>2:

            fig = plt.figure(figsize=(25, 10))
            plt.rcParams['font.size'] = 18

            with plt.style.context('bmh'):
                ax1 = fig.add_subplot(121)
                for i in range(2):
                    ax1.plot(np.arange(0,len(losses[i]),1), losses[i], label=legend[i])
            ax1.legend(loc='best')        
            ax1.set(xlabel='Epochs', ylabel='Loss')        
    
            with plt.style.context('bmh'):
                ax2 = fig.add_subplot(122)
                for i in range(2,len(losses)):
                    ax2.plot(np.arange(0,len(losses[i]),1), losses[i], label=legend[i])
            ax2.legend(loc='best')
            ax2.set(xlabel='Epochs', ylabel='Loss')

            fig.suptitle('Losses during training')
        
        else:
            fig = plt.figure(figsize=(15, 10))
            plt.rcParams['font.size'] = 18

            with plt.style.context('bmh'):
                for i in range(2):
                    plt.plot(np.arange(0,len(losses[i]),1), losses[i], label=legend[i])
            plt.legend(loc='best')  
            plt.xlabel('Epochs')
            plt.ylabel('Loss')  

            fig.suptitle('Losses during training')

        
        plt.savefig(self.plot_directory + self.molname + "_loss.png")
    
    def get_losses(self, trainer, loss_function):
        """Extract the loss attributes and the corresponding names.

        Returns:
        -------
        losses: list of numpy array or torch tensor
            shape [number of losses][number of epochs]
            the first one is the loss over the trainig set, the second 
            one is the loss over the test set, then there are the loss 
            components in alphabetical order
        legend: list of str
            names for the series in the plot
        """

        losses = [trainer.epoch_train_loss, trainer.epoch_test_loss]
        legend = ["Training", "Test"]
        
        for attribute in inspect.getmembers(loss_function):
            # to remove private and protected functions
            if attribute[0].startswith('loss'):
                #get the values of all the attributes of the object
                losses.append(getattr(loss_function,attribute[0]))
                legend.append(attribute[0])
                
        return losses, legend



       

        
