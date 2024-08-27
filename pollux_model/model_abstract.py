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
        self.state = {}
        self.output = {}

    def update_parameters(self, parameters):
        """ To update model parameters

        Parameters
        ----------
        parameters : dict
            parameters dict as defined by the model
        """
        for key, value in parameters.items():
            self.parameters[key] = value

    @abstractmethod
    def initialize_state(self, x):
        """ generate an initial state based on user parameters """
        pass

    @abstractmethod
    def calculate_output(self, u):
        """calculate output based on input u"""
        pass

    def get_output(self):
        """get output of the model"""
        return self.output
