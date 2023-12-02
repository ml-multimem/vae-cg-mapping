import logging
import torch
import inspect

# Logging settings
class Logger(logging.getLoggerClass()):
    """Custom logger for setting up logging and writing parameters to a log file."""
    
    """Set up logging"""
    def __init__(self, name = __name__, 
                 level = logging.DEBUG, 
                 filename = "standard.log"):
        """
        Initialize the Logger object.

        Parameters:
        -----------
        name: str, optional
            The logger name. Defaults to the name of the calling module.
        level: int, optional
            The logging level. Defaults to logging.DEBUG.
        filename: str, optional
            The name of the log file. Defaults to "standard.log".

        Note:
        ------
        If the logging level is set to logging.DEBUG, additional settings are applied:
        - Enabling backward compatibility warnings from PyTorch.
        - Enabling anomaly detection in PyTorch autograd.
        """
        super().__init__(name)
        
        if level == logging.DEBUG:
            torch.utils.backcompat.broadcast_warning.enabled=True
            torch.autograd.set_detect_anomaly(True)

        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(filename),
                logging.StreamHandler()
            ]
        )

    def info(self, msg):
        """
        Log an informational message.

        Parameters:
        -----------
        msg: str
            The message to be logged.
        """
        logger = logging.getLogger(self.name)
        logger.info(msg)

    def write_parameters(self, parameters):
        """
        Write the attributes contained in the parameters dictionary to the log file.

        Parameters:
        -----------
        parameters: dict
            A dictionary-like object containing parameters to be logged.
        """
        logger = logging.getLogger(self.name)
        for attribute in inspect.getmembers(parameters):
            if not attribute[0].startswith('_'):  # exclude attributes that do not come from the input file
                if not attribute[0].startswith('len'): # exclude the len attribute, if present
                    logger.info( attribute[0] + "  -  " + str(attribute[1]) )

