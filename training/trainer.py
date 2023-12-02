from cmath import inf
import torch
import numpy as np
from training.model_evaluator import ModelEvaluator
from molecular_system.mapping_utils import AtomisticToCGMapper
import os
import logging

class ModelTrainer():
    """
    Class for training a PyTorch model.
    """
    
    def __init__(self, model, 
                       optimizer, 
                       scheduler,
                       parameters,
                       task, 
                       logger_name):

        """
        Initialize the ModelTrainer.

        Parameters:
        -----------
        model: nn.Module
            The PyTorch model to be trained.
        optimizer: torch.optim.Optimizer
            The optimizer used for updating the model's weights.
        scheduler: TScheduler
            The temperature scheduler used for Gumbel distribution sampling during training.
        parameters: dict
            A dictionary containing training parameters.
        task: str
            The specific task for which the model is trained.
        logger_name: str
            The name of the logger for logging training information.
        """

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = int(parameters["save_every"])
        self.save_model = True
        self.directory = parameters["res_folder"] + "model/"
        self.patience = int(parameters["patience"])
        self.min_change = float(parameters["min_change"])
        self.epoch_train_loss = []
        self.epoch_test_loss = []
        self.loss_prv_epoch = self.min_change + 1.0  # Just to make sure it is above the min change
        self.task = task
        self.logger_name = logger_name
        self.parameters = parameters
        if "save_also" in dir(parameters):
            self.save_properties = parameters["save_also"]
            if self.save_properties == "n_CG":
                self.mapper = AtomisticToCGMapper()  
                self.n_cg_training = []   
        self.evaluator = ModelEvaluator(model)
        self.best_loss = inf
    
    def train(self, *args):
        """
        Train the model for a specified number of epochs using the provided data loaders.

        Parameters:
        -----------
        args: variable arguments
            Variable arguments containing at least:
            - the number of epochs [0] 
            - the training data loader [1]
            - the test data loader [2].
        """

        n_epochs = args[0]
        train_loader = args[1]
        test_loader = args[2]
        logger = logging.getLogger(self.logger_name)
        logger.info("Training the model")

        # Initialize patience
        patience_so_far = 0
         
        for epoch in range(n_epochs): #TODO count epochs from 1

            train_loss = self.train_epoch(train_loader)
            if hasattr(self, 'save_properties'):
                if self.save_properties == "n_CG":
                    n_CG_effective = len(self.mapper.effective_CG_beads(self.model.encoder.CG().detach()))
                    self.n_cg_training.append(n_CG_effective)
                    #train_loss = train_loss/n_CG_effective

            self.epoch_train_loss.append(train_loss)

            test_loss = self.evaluator.evaluate(test_loader)
            #test_loss = 0
            self.epoch_test_loss.append(test_loss)

            logger.info("Epoch: {: <5} Training loss: {: <20} Test loss: {: <20}".format(epoch, train_loss, test_loss))

            #self.scheduler.step()
            if (epoch+1) % 100 == 0:
                print("Epoch ", epoch+1)

            if self.save_every>0 and (epoch+1) % self.save_every == 0:
                if train_loss < self.best_loss:
                    self.save_checkpoint(self.model, self.optimizer, epoch, train_loss)

            patience_so_far, exit_flag = self.check_patience(train_loss, patience_so_far)
            if exit_flag==True:
                break
                 
        self.save_checkpoint(self.model, self.optimizer, epoch, train_loss)
        
        logger.info("Training completed")

        return

    def train_epoch(self, train_loader):
        """
        Train the model for one epoch using the provided training data loader.

        Parameters:
        -----------
        train_loader: torch.utils.data.DataLoader
            The training data loader.

        Returns:
        --------
        train_loss: float
            The average training loss for the epoch.
        """

        train_loss = 0.00
                        
        for num, batch in enumerate(train_loader):
            predictions = self.model(batch)
            batch_loss = self.model.loss((predictions, batch))

            # Backpropagation and weight update
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            
            train_loss = train_loss + batch_loss.detach().item()

        train_loss = train_loss / (num+1)

        return train_loss

    def check_patience(self, loss_epoch, patience_so_far):
        """
        Check whether the training should be stopped based on the patience criterion.

        Parameters:
        -----------
        loss_epoch: float
            The training loss for the current epoch.
        patience_so_far: int
            The number of consecutive epochs with minimal improvement in the loss.

        Returns:
        --------
        patience_so_far: int
            Updated number of consecutive epochs with minimal improvement in the loss. 
        exit_flag: Bool
            Flag indicating whether to exit training.
        """

        # Make sure we examine the change in the loss
        loss_diff = abs(loss_epoch - self.loss_prv_epoch) # Using abs to take into account also increases
        self.loss_prv_epoch = loss_diff # Update the current loss
        logger = logging.getLogger(self.logger_name)
        if  loss_diff < self.min_change:
            patience_so_far += 1 # We start/continue to wait
            logger.info("INFO: Waiting the loss removal difference to be better (%d out of %d patience)... Now it is %s while the threshold is %s."%(patience_so_far, self.patience, str(loss_diff), str(self.min_change)))
        else:
            # Update the user if close to losing patience
            if patience_so_far > 0:
                logger.info("INFO: Nice! Got an improvement. Patience restored.")
            patience_so_far = 0 # Reset it
        exit_flag = False
        # If we have reached our patience limit
        if patience_so_far > self.patience:
            # Shout forget it and stop the loop
            logger.info("WARNING: Waited for %d epochs and the change was below %s... Stopping.\n"%(self.patience, str(self.min_change)))
            exit_flag = True

        return patience_so_far, exit_flag     

    def save_checkpoint(self, model, optimizer, epoch, loss):

        state = { 'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'parameters': self.parameters
                }
        if not os.path.exists(self.directory):
            os.mkdir(self.directory) 
        if epoch < int(self.parameters["n_epochs"])-1: #TODO count epochs from 1
            model_path = self.directory + self.parameters["molname"] + "_best_model_" + self.task + ".pt"  
        else:
            model_path = self.directory + self.parameters["molname"] + "_final_model_" + self.task + ".pt"  
            
        torch.save(state, model_path)  # save checkpoint


    
            
