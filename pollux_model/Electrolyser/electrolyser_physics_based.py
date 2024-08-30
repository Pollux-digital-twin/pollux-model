from pollux_model.model_abstract import Model
from abc import ABC, abstractmethod

from pollux_model.Electrolyser.correlations.electrolyser_pydolphin import Electrolyser_pydolphin_DeGroot

class Electrolyser(Model):
    """ Abstract base class for simulation models

        Model classes implement a discrete state space model
        The state of the model is maintained outside the model object
    """


    def __init__(self):
        """ Model initialization
        """
        self.parameters = {}
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


    def initialize_state(self, x):
        """ generate an initial state based on user parameters """
        pass


    def update_state(self, u, x):
        """update the state based on input u and state x"""
        pass


    def calculate_output(self, u=0, x=0):
        """calculate output based on input u and state x"""
        self.output = self.calculate_production_rate(u,x)


    def get_output(self):
        """get output of the model"""
        return self.output

    def calculate_production_rate(self, u=0, x=0):
        """
          Calculate the corrosion rate.
        """

        model = self.parameters['electrolyser_model']

        if model == 'ML_forrest':
            pass
        elif model == "Physics_based":
            electrolyser_model = Electrolyser_pydolphin_DeGroot()



        electrolyser_model.update_parameters(self.parameters)
        electrolyser_model.calculate_output(u,x)
        return electrolyser_model.get_output()


if __name__ == '__main__':

    model = Electrolyser()

