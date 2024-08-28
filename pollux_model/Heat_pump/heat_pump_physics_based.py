# Imports

from NREL_components.heat_pump_model import *
#from NREL_components.utilities.unit_defs import ureg, Q_
from pollux_model.Heat_pump.NREL_components.utilities.unit_defs import ureg, Q_

from pollux_model.model_abstract import Model
import numpy as np

class Heat_pump_nrel(Model):

    def __init__(self,yaml_file_path):
        self.parameters = {}
        self.load_yaml(yaml_file_path) # load yaml content into a dict
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

    def initialize_state(self, x=0):
        """ generate an initial state based on user parameters """
        self.heat_pump_component = heat_pump()
        self.heat_pump_component.construct_yaml_input_quantities('NREL_components\heat_pump_model_inputs.yml')
        pass

    def update_state(self, u, x):
        """update the state based on input u and state x"""
        pass

    def calculate_output(self, u=0, x=0):
        #heat_pump_component = heat_pump()
        n_hrs = 1
        #heat_pump_component.construct_yaml_input_quantities('NREL_components\heat_pump_model_inputs.yml')
        self.heat_pump_component.hot_temperature_desired = Q_(np.array([150] * n_hrs), 'degC')
        self.heat_pump_component.hot_temperature_minimum = Q_(np.array([100] * n_hrs), 'degC')
        self.heat_pump_component.cold_temperature_available = Q_(np.array([80] * n_hrs), 'degC')
        self.heat_pump_component.carnot_efficiency_factor = Q_('0.55')
        self.heat_pump_component.carnot_efficiency_factor_flag = False
        self.heat_pump_component.refrigerant = 'n-Pentane'

        #heat_pump_component.run_all("hp_test")
        # self.output['hydrogen_production'] = result[0][1]
        # self.output['oxygen_production'] = result[0][0]

    def run_simulations(self,heat_pump_mod):
        heat_pump_mod.run_all('hp_test')


    def get_output(self):
        """get output of the model"""
        return self.output

    def load_yaml(self,yaml_file_path):
        """"
        Load a yaml file into a self.parameters dictionary

        """
        with open(yaml_file_path,'r') as file:
            self.parameters = yaml.safe_load(file)

    def read_config(self,file_path):
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    def update_config(self,config):
        for key in config.keys():
            new_value = input(f"Enter new value for {key} (leave blank to keep current value '{config[key]}'): ")
            if new_value:
                config[key] = new_value









if __name__ =='__main__':


    correlation_result = Heat_pump_nrel(r'C:\Users\ntagkrasd\Pollux\pollux-model\pollux_model\Heat_pump\NREL_components\heat_pump_model_inputs.yml')
    # input_data = correlation_result.collect_user_inputs()
    correlation_result.initialize_state()
    print(correlation_result.parameters)
    correlation_result.calculate_output()
    correlation_result.run_simulations(correlation_result.heat_pump_component)
   # correlation_result.run_simulations(correlation_result.heat_pump_component)
    # result = correlation_result.prediction(input_data)