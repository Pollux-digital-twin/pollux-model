from pollux_model.model_abstract import Model
from pollux_model.model_example.correlation.correlation_example import CorrelationExample


class Model1(Model):
    def __init__(self):
        self.parameters = {}
        self.output = {}
        self.parameter_1 = 'parameter_1_value' # Add any extra parameters needed for the current model

    def update_parameters(self, parameters):
        """ To update model parameters

        Parameters
        ----------
        parameters: dict
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

    def calculate_output(self, u, x):
        """calculate output based on input u and state x"""

        # OUTPUT calculations
        # ...

        # Assign output to self
        self.output['output_value_key_1'] = 'output_value_1'
        self.output['output_value_key_2'] = 'output_value_2'