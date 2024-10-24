import numpy as np

# def step_function(t, step_size, constants):
#     """
#     A stepwise constant function that outputs predefined constants for each step.
    
#     Parameters:
#     t : float or int
#         The input value for which the function is evaluated.
#     step_size : float
#         The size of the step intervals.
#     constants : list
#         A list of constants.
    
#     Returns:
#     float 
#         The constant value corresponding to the input step.
#     """
#     t = np.array(t)
    
#     # Determine which step interval x falls into
#     # step_index = int(x // step_size)
#     step_indices = np.floor(t / step_size).astype(int)
    
#      # Clip the step indices to the range of the constants list
#     step_indices = np.clip(step_indices, 0, len(constants) - 1)
    
#     return np.array(constants)[step_indices]

class StepFunction:
    def __init__(self, values, step_size):
        self.values = values  # Step values (list or array)
        self.step_size = step_size  # Fixed step size
    
    def get_step_size(self):
        # Return the constant step size
        return self.step_size
    
    def evaluate(self, t):
        t = np.array(t)
        # Evaluate the step function at a point x based on the step size
        step_indices = np.floor(t / self.step_size).astype(int)
        
        # Clip the step indices to the range of the constants list
        step_indices = np.clip(step_indices, 0, len(self.values) - 1)
    
        return np.array(self.values)[step_indices]
