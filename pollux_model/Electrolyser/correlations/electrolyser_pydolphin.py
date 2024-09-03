# Imports
from pollux_model.model_abstract import Model
import numpy as np

from thermo.chemical import Chemical
from scipy.optimize import root_scalar


# electrolyser model
class Electrolyser_pydolphin_DeGroot(Model):

    def __init__(self, ):
        self.parameters = {}
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
        if self.parameters['cell_type'] == 'alk_ref_cell':
            self.parameters['power_single_cell'] = 6222
        elif self.parameters['cell_type'] == 'low_power_cell':
            self.parameters['power_single_cell'] = 4000
        elif self.parameters['cell_type'] == 'medium_power_cell':
            self.parameters['power_single_cell'] = 10000
        elif self.parameters['cell_type'] == 'high_power_cell':
            self.parameters['power_single_cell'] = 16000

    def initialize_state(self, x):
        """ generate an initial state based on user parameters """
        pass

    def update_state(self, u, x):
        """update the state based on input u and state x"""
        pass

    def calculate_output(self, u=0, x=0):
        self.calc_prod_rates(u, x)
        self.output['hydrogen_production'] = self.parameters['mass_H2']
        self.output['oxygen_production'] = self.parameters['mass_O2']

    def get_output(self):
        """get output of the model"""
        return self.output

    def calc_prod_rates(self, u, x):

        T_cell = u['T_cell']
        p_cathode = u['p_cathode']
        p_anode = u['p_anode']
        p_0_H2O = u['p_0_H2O']
        capacity = u['capacity']
        power_input = u['power_input']
        # PVT properties of H2, O2 and water at current pressure and temperature.
        PVT_H2 = Chemical('hydrogen')
        PVT_O2 = Chemical('oxygen')
        PVT_H2O = Chemical('water')

        PVT_H2.calculate(T=T_cell, P=p_cathode)
        PVT_O2.calculate(T=T_cell, P=p_anode)
        PVT_H2O.calculate(T=T_cell, P=p_0_H2O)

        self.parameters['N_cells'] = np.ceil(capacity / self.parameters['power_single_cell'])
        self.parameters['power_cell_real'] = power_input / self.parameters[
            'N_cells']  # * self.power_multiplier  # todo: the power multiplier
        # can be extended to include active and non active stacks,
        # for now just give the independent stacks
        self.calc_i_cell()
        # wteta faraday assume to be constant
        # Production rates [mol/s]
        #
        self.parameters['prod_rate_H2'] = (self.parameters['N_cells']) * self.I_cell_array / (
                2 * self.parameters['Faraday_const']) * self.parameters['eta_Faraday_array']
        self.parameters['prod_rate_O2'] = (self.parameters['N_cells']) * self.I_cell_array / (
                4 * self.parameters['Faraday_const']) * self.parameters['eta_Faraday_array']
        self.parameters['prod_rate_H2O'] = (self.parameters['N_cells']) * self.I_cell_array / (
                2 * self.parameters['Faraday_const'])

        # Massflows [kg/s].
        self.parameters['massflow_H2'] = self.parameters['prod_rate_H2'] * PVT_H2.MW * 1e-3
        self.parameters['massflow_O2'] = self.parameters['prod_rate_O2'] * PVT_O2.MW * 1e-3
        self.parameters['massflow_H2O'] = self.parameters['prod_rate_H2O'] * PVT_H2O.MW * 1e-3

        # Densities [kg/m^3].
        self.parameters['rho_H2'] = PVT_H2.rho
        self.parameters['rho_O2'] = PVT_O2.rho
        self.parameters['rho_H2O'] = PVT_H2O.rho

        # Flowrates [m^3/s].
        self.parameters['flowrate_H2'] = self.parameters['massflow_H2'] / self.parameters['rho_H2']
        self.parameters['flowrate_O2'] = self.parameters['massflow_O2'] / self.parameters['rho_O2']
        self.parameters['flowrate_H2O'] = (self.parameters['massflow_H2O'] /
                                           self.parameters['rho_H2O'])

        # Integrate massflows to obtain masses of H2, O2 and H20 in this period [kg].
        # Note: it assumes constant operating conditions in the time-step
        self.parameters['mass_H2'] = self.parameters['massflow_H2'] * self.parameters['delta_t']
        self.parameters['mass_O2'] = self.parameters['massflow_O2'] * self.parameters['delta_t']
        self.parameters['mass_H2O'] = self.parameters['massflow_H2O'] * self.parameters['delta_t']

    def calc_i_cell(self):

        I_current_sol = root_scalar(
            self.root_I_cell, bracket=[1.0, 30000],
            method='brentq',
            args=(
                self.parameters['power_cell_real'],
            )
        )
        self.I_cell_array = I_current_sol.root

    def root_I_cell(self, I_cell, power_cell):

        self.parameters['E_total_cell'] = \
            self.compute_potentials(
                I_cell, self.parameters['A_cell'])

        root_expr = power_cell / (self.parameters['E_total_cell']) - I_cell

        return root_expr

    @staticmethod
    def compute_potentials(I_cell, A_cell):

        # A_cell = 0.436
        I_cell_in = I_cell / 1e4 / A_cell
        # Voltage efficiency WITH COEFFICIENTS
        E_total_cel = (-0.160249069 * I_cell_in ** 4 + 0.734073995 * I_cell_in ** 3 -
                       1.168543948 * I_cell_in ** 2 + 1.048496283 * I_cell_in + 1.46667069)

        return E_total_cel
