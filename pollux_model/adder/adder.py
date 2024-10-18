from pollux_model.model_abstract import Model
import numpy as np

class Adder(Model):
    def __init__(self):
        super().__init__()

    def initialize_state(self, x):
        """ generate an initial state based on user parameters """
        pass
    
    def calculate_output(self):
        # self.output['output'] = self.input['input_0'] + self.input['input_1']
        if len(self.input) == 2:
            self.output['output'] = self.input['input_0'] + self.input['input_1']
        else:
            raise ValueError("adder requires exactly 2 inputs.")