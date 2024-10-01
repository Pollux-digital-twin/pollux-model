from pollux_model.model_abstract import Model
import numpy as np

class PowerDemand(Model):
    def __init__(self, time_function):
        super().__init__()
        self.time_function = time_function
        self.current_time = 0

    def initialize_state(self, x):
        """ generate an initial state based on user parameters
            """
        pass
    
    def calculate_output(self):
        self.output['power_demand'] = self.time_function(self.current_time)
        self.output['power_difference'] = self.output['power_demand'] - self.input['power_input']
        
    def update_time(self, time_step):
        self.current_time += time_step