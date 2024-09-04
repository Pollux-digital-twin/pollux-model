import os
import yaml
from pollux_model.heat_pump.NREL_components.heat_pump_model import HeatPumpModel
from pollux_model.heat_pump.NREL_components.utilities.unit_defs import Q_

from pollux_model.model_abstract import Model
import numpy as np


class HeatpumpNREL(Model):
    """"
    This is the class with the specific format that we want and is a replicate of the
    NREL Heat pump model
    """

    def __init__(self):
        super().__init__()

        self.parameters['script_dir'] = os.path.dirname(__file__)
        self.parameters['yaml_file_path'] = os.path.join(self.parameters['script_dir'],
                                                         'NREL_components',
                                                         'heat_pump_model_inputs.yml')
        self._load_yaml_parameters(
            self.parameters['yaml_file_path'])  # load yaml content into a dict

        self.model = HeatPumpModel()  # instantiate the heat pump from NREL
        self.model.construct_yaml_input_quantities(self.parameters['yaml_file_path'])

    def update_parameters(self, parameters):
        """ To update model parameters

        Parameters
        ----------
        parameters: dict
            parameters dict as defined by the model
        """
        for key, value in parameters.items():
            self.parameters[key] = value

        self.model.refrigerant = self.parameters['refrigerant']
        self.model.carnot_efficiency_factor = Q_('0.55')
        self.model.carnot_efficiency_factor_flag = False

    def initialize_state(self, x):
        """ generate an initial state based on user parameters
         """
        pass

    def calculate_output(self, u):
        self.model.hot_temperature_desired = Q_(
            np.array([u['hot_temperature_desired']] * self.parameters['n_hrs']), 'degC')
        self.model.hot_temperature_minimum = Q_(
            np.array([u['hot_temperature_minimum']] * self.parameters['n_hrs']), 'degC')
        self.model.cold_temperature_available = Q_(
            np.array([u['cold_temperature_available']] * self.parameters['n_hrs']), 'degC')

        self.model.run_simulation()

        self.output['power_demand'] = self.model.average_power_in.m * 1000
        self.output['hot_mass_flow_rate'] = self.model.hot_mass_flowrate_average.m

    def _load_yaml_parameters(self, yaml_file_path):
        """"
        Load a yaml file into a self.parameters dictionary

        """
        with open(yaml_file_path, 'r') as file:
            yaml_parameters = yaml.safe_load(file)

        for key in yaml_parameters.keys():
            self.parameters[key] = yaml_parameters[key]

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
