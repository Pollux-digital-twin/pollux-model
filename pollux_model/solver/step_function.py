import numpy as np
def step_function(x, step_size, constants):
    """
    A stepwise constant function that outputs predefined constants for each step.
    
    Parameters:
    x : float or int
        The input value for which the function is evaluated.
    step_size : float
        The size of the step intervals.
    constants : list
        A list of constants.
    
    Returns:
    float 
        The constant value corresponding to the input step.
    """
    x = np.array(x)
    
    # Determine which step interval x falls into
    # step_index = int(x // step_size)
    step_indices = np.floor(x / step_size).astype(int)
    
     # Clip the step indices to the range of the constants list
    step_indices = np.clip(step_indices, 0, len(constants) - 1)
    
    # Return the corresponding constant for each index
    #return np.array([constants[i] for i in step_indices])
    return np.array(constants)[step_indices]

    # # If the index exceeds the number of constants, return the last constant
    # if step_index >= len(constants):
    #     return constants[-1]
    
    # return constants[step_index]