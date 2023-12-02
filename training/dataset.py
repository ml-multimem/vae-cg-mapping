from torch.utils.data import Dataset
import inspect
import numpy as np
import torch

class MolecularDataset(Dataset):
    """
    Custom PyTorch Dataset for handling molecular data with flexible attributes.
    Create a dataset with flexible number of objects
    One of the objects must be called "input" and it 
    will be used in the loss calculation during the during 
    """

    def __init__(self, **kwargs):
        """
        Parameters:
        -----------
        **kwargs: keyword arguments
            Keyword arguments representing different attributes of the dataset.
            At least one attribute must be named "input" and will be used for loss calculation during training.

        Attributes:
        -----------
        length: int
            Number of samples in the dataset.
        
        """

        for key, value in kwargs.items():
            setattr(self, key, value)
        # TODO: check if "input" was defined
        # The [0] dimension of all arguments is the batch size
        self.lenght = value.shape[0]
        
    def __len__(self):
        """
        Return the length of the dataset.

        Returns:
        --------
        length: int
            Number of samples in the dataset.
        """
        return self.lenght

    def __getitem__(self,idx):
        """
        Retrieve the item at the specified index.

        Parameters:
        -----------
        idx: int
            Index of the item to retrieve.

        Returns:
        --------
        item_values: dict
            Dictionary containing attribute values for the specified index.

        """
        item_values = {}
        # getmembers() returns all the members of an object 
        for attribute in inspect.getmembers(self):
            # to remove private and protected functions
            if not attribute[0].startswith('_'):
                #to remove other methods and properties
                if isinstance(attribute[1],np.ndarray):
                    #get the values of all the attributes of the object
                    item_values[attribute[0]] = getattr(self,attribute[0])[idx]
                elif torch.is_tensor(attribute[1]):
                    #get the values of all the attributes of the object
                    item_values[attribute[0]] = getattr(self,attribute[0])[idx]
                   
        return item_values

def training_indices(N_samples, train_amount):
    """
    Generate random training and testing indices.

    Parameters:
    -----------
    N_samples: int
        Total number of samples.
    train_amount: float in ]0,1[
        Percentage of samples to be used for training.

    Returns:
    --------
    train_idx: numpy array
        Randomly selected training indices.
    test_idx: numpy array
        Remaining indices for testing.
    """

    indices = np.random.permutation(N_samples) 
    train_idx = int(train_amount * N_samples) 

    return indices[0:train_idx], indices[train_idx:N_samples]

def format_for_dataset(input_data, shape, idx, device):
    """
    Format data for the MolecularDataset.

    Parameters:
    -----------
    input_data: numpy array or torch.Tensor
        Input data to be formatted.
    shape: tuple
        Desired shape of the data.
    idx: numpy array
        Indices to select from the input data.
    device: str
        Device to move the formatted data to.

    Returns:
    --------
    formatted_data: torch.Tensor
        Formatted data.
    """

    formatted_data = torch.reshape(torch.tensor(input_data), shape)
    formatted_data = formatted_data.float().to(device)

    return formatted_data[idx]

def periodic_dataset_mapping(mol_sys, feature, train_amount, device):
    """
    Create a dataset for mapping a periodic system, including box information.

    Parameters:
    -----------
    mol_sys: PeriodicMolecularSystem
        PeriodicMolecularSystem object storing the trajctory information.
    feature: numpy array
        Feature data for mapping.
    train_amount: float
        Percentage of samples to be used for training.
    device: str
        Device to move the data to.

    Returns:
    --------
    train_dataset: MolecularDataset
        Training dataset.
    test_dataset: MolecularDataset
        Testing dataset.
    """
    N_samples = mol_sys.n_frames * mol_sys.n_molecules
    N_particles = mol_sys.n_particles_mol
    train_idx, test_idx = training_indices(N_samples, train_amount)

    train_dataset = MolecularDataset(input = format_for_dataset(feature, (N_samples, N_particles, -1), train_idx, device),
                                     forces = format_for_dataset(mol_sys.forces, (N_samples, N_particles, 3), train_idx, device), 
                                     box = format_for_dataset(mol_sys.box, (N_samples, 3, 2), train_idx, device))
                                     # Setting the chosen feature as the labels for training
    test_dataset = MolecularDataset(input = format_for_dataset(feature, (N_samples, N_particles, -1), test_idx, device),
                                     forces = format_for_dataset(mol_sys.forces, (N_samples, N_particles, 3), test_idx, device), 
                                     box = format_for_dataset(mol_sys.box, (N_samples, 3, 2), test_idx, device))

    return train_dataset, test_dataset

def not_periodic_dataset_mapping(mol_sys, feature, train_amount, device):
    """
    Create a dataset for mapping a non-periodic system, without box information.

    Parameters:
    -----------
    mol_sys: MolecularSystem
        MolecularSystem object storing the trajctory information.
    feature: numpy array
        Feature data for mapping.
    train_amount: float
        Percentage of samples to be used for training.
    device: str
        Device to move the data to.

    Returns:
    --------
    train_dataset: MolecularDataset
        Training dataset.
    test_dataset: MolecularDataset
        Testing dataset.
    """
    N_samples = mol_sys.n_frames * mol_sys.n_molecules
    N_particles = mol_sys.n_particles_mol
    train_idx, test_idx = training_indices(N_samples, train_amount)

    train_dataset = MolecularDataset(input = format_for_dataset(feature, (N_samples, N_particles, -1), train_idx, device),
                                     forces = format_for_dataset(mol_sys.forces, (N_samples, N_particles, 3), train_idx, device))
                                     # Setting the chosen feature as the labels for training
    test_dataset = MolecularDataset(input = format_for_dataset(feature, (N_samples, N_particles, -1), test_idx, device),
                                    forces = format_for_dataset(mol_sys.forces, (N_samples, N_particles, 3), test_idx, device))
                                  
    return train_dataset, test_dataset
