from pollux_model.model_abstract import Model
import numpy as np

# class PowerSupply(Model):
#     def __init__(self):
#         super().__init__()
   
#     # def __init__(self, value):
#     #     super().__init__()
#     #     self.output = value  # TODO: This is for testing, it should be timeseries
        
#     def initialize_state(self, x):
#         """ generate an initial state based on user parameters
#             """
#         pass
    
#     def calculate_output(self):
#         self.output['power_output'] = self.input['power_supply']

class PowerSupply(Model):
    def __init__(self, time_function):
        super().__init__()
        self.time_function = time_function
        self.current_time = 0
        
    def initialize_state(self, x):
        """ generate an initial state based on user parameters
            """
        pass
    
    def calculate_output(self):
        self.output['power_supply'] = self.time_function(self.current_time)
        
    def update_time(self, time_step):
        self.current_time += time_step