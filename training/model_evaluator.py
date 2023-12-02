class ModelEvaluator():
    """
    Evaluate a PyTorch model on a given data loader.
    It computes the average loss over all batches in the data loader.
    """
    def __init__(self, model):
        """
        Parameters:
        -----------
        model: nn.Module
            The PyTorch model to be evaluated.

        """
        self.model = model

    def evaluate(self, data_loader):
        """
        Evaluate the model on the provided data loader and return the average loss.

        Parameters:
        -----------
        data_loader: torch.utils.data.DataLoader
            The PyTorch DataLoader containing the test dataset.

        Returns:
        --------
        test_loss: float
            The average loss computed over all batches in the data loader.
        """

        test_loss = 0.00
                            
        for num, batch in enumerate(data_loader):
                            
            predictions = self.model.forward(batch)
            batch_loss = self.model.loss((predictions, batch))
            
            test_loss = test_loss + batch_loss.detach()

        test_loss = test_loss / (num+1)

        return test_loss
