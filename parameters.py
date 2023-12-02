class Parameters():
    """Class to store the parameters read from the input file"""

    def __getitem__(self, key):
        """Get the value of a parameter using square bracket notation.

        Parameters:
        -----------
        key: str
            The name of the parameter.

        Returns:
        --------
        value:
            The value of the specified parameter.
        """
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Set the value of a parameter using square bracket notation.

        Parameters:
        -----------
        key: str
            The name of the parameter.
        value:
            The value to be assigned to the parameter.
        """
        setattr(self, key, value)
    
    def len(self):
        """Get the number of parameters stored in the object.

        Returns:
        --------
        length: int
            The number of parameters stored in the object.
        """
        return len(self.__dict__)

    def merge(self, p1, p2):
        """Merge two sets of parameters into a single set.

        Parameters:
        -----------
        p1: Parameters
            The first set of parameters to be merged.
        p2: Parameters
            The second set of parameters to be merged.

        Note:
        -----
        This method combines parameters from both sets, prioritizing parameters from the first set (`p1`) 
        in case of conflicts.

        """

        for attribute in dir(p1):
            if not attribute.startswith('_'):  # exclude attributes that do not come from the input file
                if not attribute.startswith('len'): # exclude the len attribute, if present
                    if not attribute.startswith('merge'):
                        setattr(self, attribute, p1[attribute])
        
        for attribute in dir(p2):
            if not attribute.startswith('_'):  # exclude attributes that do not come from the input file
                if not attribute.startswith('len'): # exclude the len attribute, if present
                    if not attribute.startswith('merge'):
                        setattr(self, attribute, p2[attribute])
                    
