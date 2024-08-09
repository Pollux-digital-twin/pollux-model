from pollux_model.model_abstract import Model


class Model1(Model):
    def __init__(self):
        self.parameters = {}
        self.output = {}

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

        input_model = u['input_key_1']

        # OUTPUT calculations
        output_model = self._internal_function(input_model)

        # Assign output to self
        self.output['output_key_1'] = output_model

    def _internal_function(self, input_function):
        """model function defined internally"""
        output_function = input_function
        return output_function

    def get_output(self):
        """get output of the model"""
        return self.output
