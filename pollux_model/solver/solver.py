from pollux_model.power_supply_demand.power_supply import PowerSupply
from pollux_model.power_supply_demand.power_demand import PowerDemand
from pollux_model.hydrogen_demand.hydrogen_demand import HydrogenDemand
from pollux_model.splitter.splitter import Splitter
from pollux_model.adder.adder import Adder
from pollux_model.gas_storage.hydrogen_tank_model import HydrogenTankModel

import numpy as np

class Solver:
    def __init__(self, time_vector, components, components_with_control):
        self.connections = []
        self.time_vector = time_vector
        self.components = components # Dictionary with key component name and value component object
        self.components_with_control = components_with_control # list of the components with control
        self.time_step = np.diff(time_vector)[0]  # Assuming constant time step
        self.inputs = {}  # Dictionary to store inputs of each component over time
        self.outputs = {}  # Dictionary to store outputs of each component over time

    def connect(self, predecessor,  successor, predecessor_output, successor_input):
        #Connect the output of the predecessor component to one of the successor's input.
        self.connections.append((predecessor,  successor, predecessor_output, successor_input))

    def run(self, control):
        # control is unscaled
        # clean outputs/inputs which is needed when run is called multiple times
        control=np.array(control)
        for component_name in self.components:
            component = self.components[component_name]
            # self.outputs[self.components[component_name]] = []
            # self.outputs[component] = np.zeros(len(self.time_vector), len(component.output.values()))
            # self.inputs[self.components[component_name]] = []
            # self.outputs[component] = np.zeros(len(self.time_vector), len(component.input.values()))
            # components with input profiles or control profiles TODO make more generic
            if (isinstance(component, (PowerSupply, PowerDemand, HydrogenDemand, Splitter, HydrogenTankModel, Adder))):
                component.set_time(0) #reset time
            if (isinstance(component, (HydrogenTankModel))):
                component.reset_current_mass() #reset initial storage H2 mass
               
                
        number_of_components_with_control = len(self.components_with_control)
        control_reshaped = control.reshape(number_of_components_with_control, -1)
        for ii in range(number_of_components_with_control):
            self.components[self.components_with_control[ii]].update_time_function(control_reshaped[ii])
        
        time_index = -1
        for t in self.time_vector:
            time_index = time_index + 1
            processed_components = set()
            #Process each connection in the system.
            for predecessor, successor, predecessor_output, successor_input in self.connections:
                for component in [predecessor, successor]:
                    # components with input profiles or control profiles TODO make more generic
                    # if (isinstance(component, (PowerSupply, PowerDemand, HydrogenDemand, Splitter, HydrogenTankModel)) and component.current_time < t):
                    if (isinstance(component, (PowerSupply, PowerDemand, HydrogenDemand, Splitter, HydrogenTankModel, Adder))):
                        # print(f"component: {component}")
                        # predecessor.set_time(t)
                        # component.update_time(self.time_step)
                        component.set_time(t)

                predecessor.calculate_output()  # First, calculate the predecessor to get its output
                successor.input[successor_input] = predecessor.output[predecessor_output] # Pass the output to the successor's input
                successor.calculate_output()  # Calculate the successor component

                # Store outputs for each component at each time step
                for component in [predecessor, successor]:
                    # if component not in processed_components:
                    #     # components with input profiles or control profiles TODO make more generic
                    #     # if (isinstance(predecessor, (PowerSupply, PowerDemand, HydrogenDemand, Splitter, HydrogenTankModel)) and predecessor.current_time < t):
                    #     # if (isinstance(component, (HydrogenDemand))):
                    #     # #     print(f"component: {component}")
                    #     # #     # predecessor.update_time(self.time_step)
                    #     #     component.update_time(self.time_step)
                    #     #     component.set_time(t + self.time_step)
                
                    #     # A component can occur multiple times in a (predecessor, successor) pair but should be adressed only once 
                    #     processed_components.add(component)
                    #     if component not in self.outputs:
                    #         self.outputs[component] = []
                    #     self.outputs[component].append(list(component.output.values())) #converted to list, appending dict fails
                        
                    #     if component not in self.inputs:
                    #         self.inputs[component] = []
                    #     self.inputs[component].append(list(component.input.values()))
                    
                    if component not in self.outputs:
                        self.outputs[component] = np.zeros((len(self.time_vector), len(component.output.values())))
                    self.outputs[component][time_index] = list(component.output.values()) #converted to list, appending dict fails
                    
                    if component not in self.inputs:
                        self.inputs[component] = np.zeros((len(self.time_vector), len(component.input.values())))
                    self.inputs[component][time_index] = list(component.input.values())
                        
                        # print(component)
                        # print(component.output.values)
