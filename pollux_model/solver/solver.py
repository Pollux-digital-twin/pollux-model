from pollux_model.power_supply_demand.power_supply import PowerSupply
from pollux_model.power_supply_demand.power_demand import PowerDemand
from pollux_model.hydrogen_demand.hydrogen_demand import HydrogenDemand
from pollux_model.splitter.splitter import Splitter
from pollux_model.gas_storage.hydrogen_tank_model import HydrogenTankModel

import numpy as np

class Solver:
    def __init__(self, time_vector):
        self.connections = []
        self.time_vector = time_vector
        self.time_step = np.diff(time_vector)[0]  # Assuming constant time step
        self.inputs = {}  # Dictionary to store inputs of each component over time
        self.outputs = {}  # Dictionary to store outputs of each component over time

    def connect(self, predecessor,  successor, predecessor_output, successor_input):
        """Connect the output of the predecessor component to one of the successor's input ports."""
        self.connections.append((predecessor,  successor, predecessor_output, successor_input))

    def run(self):
        for t in self.time_vector:
            processed_components = set()
            """Process each connection in the system."""
            for predecessor, successor, predecessor_output, successor_input in self.connections:

                predecessor.calculate_output()  # First, calculate the predecessor to get its output
                successor.input[successor_input] = predecessor.output[predecessor_output] # Pass the output to the successor's input
                successor.calculate_output()  # Calculate the successor component

                # Store outputs for each component at each time step
                for component in [predecessor, successor]:
                    if component not in processed_components:
                        if (isinstance(predecessor, (PowerSupply, PowerDemand, HydrogenDemand, Splitter, HydrogenTankModel)) and predecessor.current_time < t):
                            predecessor.update_time(self.time_step)
                
                        # A component can be occuring multiple times in a (predecessor, successor) pair but should be adressed only once 
                        processed_components.add(component)
                        if component not in self.outputs:
                            self.outputs[component] = []
                        self.outputs[component].append(list(component.output.values())) #converted to list, appending dict fails
                        
                        if component not in self.inputs:
                            self.inputs[component] = []
                        self.inputs[component].append(list(component.input.values()))
                        
                        # print(component)
                        # print(component.output.values)
