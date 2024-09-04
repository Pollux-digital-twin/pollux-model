# Imports
import os
import yaml
from pollux_model.Heat_pump.NREL_components.heat_pump_model import heat_pump
# from NREL_components.utilities.unit_defs import ureg, Q_
from pollux_model.Heat_pump.NREL_components.utilities.unit_defs import Q_

from pollux_model.model_abstract import Model
import numpy as np


class Heat_pump_nrel(Model):
    """"
    This is the class with the specific format that we want and is a replicate of the
    NREL Heat pump model
    """

    def __init__(self):
        self.parameters = {}
        self.script_dir = os.path.dirname(__file__)
        self.yaml_file_path = os.path.join(self.script_dir,
                                           'NREL_components', 'heat_pump_model_inputs.yml')
        self.load_yaml(self.yaml_file_path)  # load yaml content into a dict

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
        """ generate an initial state based on user parameters
         # ALWAYS the heat pump should be instantiated bya yaml file with
          this name(heat_pump_model_inputs.yml)

         """
        self.heat_pump_component = heat_pump()  # instantiate the heat pump from NREL
        # self.input_file_path = os.path.join(self.script_dir,'NREL_components',
        # 'heat_pump_model_inputs.yml')
        self.heat_pump_component.construct_yaml_input_quantities(self.yaml_file_path)

    def update_state(self, u=0, x=0):
        """update the state based on input u and state x"""
        self.heat_pump_component.hot_temperature_desired = Q_(
            np.array([u['desired outlet hot temperature']] * self.parameters['n_hrs']), 'degC')
        self.heat_pump_component.hot_temperature_minimum = Q_(
            np.array([u['minimum outlet hot temperature']] * self.parameters['n_hrs']), 'degC')
        self.heat_pump_component.cold_temperature_available = Q_(
            np.array([u['available outlet cold temperature']] * self.parameters['n_hrs']), 'degC')
        self.heat_pump_component.carnot_efficiency_factor = Q_('0.55')
        self.heat_pump_component.carnot_efficiency_factor_flag = False
        self.heat_pump_component.refrigerant = u['refrigerant']
        pass

    def calculate_output(self, u=0, x=0):
        self.run_simulations(self.heat_pump_component)
        self.output['Power Demand(kW)'] = self.heat_pump_component.average_power_in.m
        self.output['Hot mass flow rate(kg/s)'] = (
            self.heat_pump_component.hot_mass_flowrate_average.m)

        # heat_pump_component.run_all("hp_test")
        # self.output['hydrogen_production'] = result[0][1]
        # self.output['oxygen_production'] = result[0][0]

    def get_output(self):
        """get output of the model"""
        return self.output

    def load_yaml(self, yaml_file_path):
        """"
        Load a yaml file into a self.parameters dictionary

        """
        with open(yaml_file_path, 'r') as file:
            self.parameters = yaml.safe_load(file)

    def read_config(self, file_path):
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def update_config(self, config):
        for key in config.keys():
            new_value = input(f"Enter new value for {key} "
                              f"(leave blank to keep current value '{config[key]}'): ")
            if new_value:
                config[key] = new_value

    @staticmethod
    def run_simulations(heat_pump_mod):
        heat_pump_mod.run_all('hp_test')
