from abc import ABC, abstractmethod


class Model(ABC):
    """ Abstract base class for simulation models

        Model classes implement a discrete state space model
        The state of the model is maintained outside the model object
    """

    @abstractmethod
    def __init__(self):
        """ Model initialization
        """
        self.parameters = {}
        self.output = {}

    @abstractmethod
    def update_parameters(self, parameters):
        """ To update model parameters

        Parameters
        ----------
        parameters : dict
            parameters dict as defined by the model
        """
        pass

    @abstractmethod
    def initialize_state(self, x):
        """ generate an initial state based on user parameters """
        pass

    @abstractmethod
    def update_state(self, u, x):
        """update the state based on input u and state x"""
        pass

    @abstractmethod
    def calculate_output(self, u, x):
        """calculate output based on input u and state x"""
        pass

    @abstractmethod
    def get_output(self):
        """get output of the model"""
        return self.output
