# Imports
from pollux_model.model_abstract import Model
import numpy as np

from thermo.chemical import Chemical
from scipy.optimize import root_scalar


# electrolyser model
class Electrolyser_pydolphin_DeGroot(Model):

    def __init__(self, power_input):
        self.parameters = {}
        self.output = {}
        self.capacity = 100 * 1e6
        self.cell_type = 'low_power_cell'
        if self.cell_type == 'alk_ref_cell':
            self.power_single_cell = 6222
        elif self.cell_type == 'low_power_cell':
            self.power_single_cell = 4000
        elif self.cell_type == 'medium_power_cell':
            self.power_single_cell = 10000
        elif self.cell_type == 'high_power_cell':
            self.power_single_cell = 16000

        self.eta_Faraday_array = 1  # just a constant, in reality is a variable
        self.Faraday_const = 96485.3329  # Faraday constant [(s A)/mol]
        self.delta_t = 3600
        self.T_cell = 273.15 + 40
        self.p_cathode = 10e5
        self.p_anode = 10e5
        self.p_0_H2O = 10e5

        self.power_input = power_input
        self.power_multiplier = 5

        self.parameters['electrolyser_model'] = "Physics_based"


    def update_parameters(self, parameters):
        """ To update model parameters

        Parameters
        ----------
        parameters: dict
            parameters dict as defined by the model
        """
        for key, value in parameters.items():
            self.parameters[key] = value

    def initialize_state(self, x):
        """ generate an initial state based on user parameters """
        pass

    def update_state(self, u, x):
        """update the state based on input u and state x"""
        pass

    def calculate_output(self, u=0, x=0):
        result = self.calc_prod_rates()
        self.output['hydrogen_production'] = self.mass_H2
        self.output['oxygen_production'] = self.mass_O2


    def get_output(self):
        """get output of the model"""
        return self.output

    def calc_prod_rates(self):

        # PVT properties of H2, O2 and water at current pressure and temperature.
        PVT_H2 = Chemical('hydrogen')
        PVT_O2 = Chemical('oxygen')
        PVT_H2O = Chemical('water')

        PVT_H2.calculate(T=self.T_cell, P=self.p_cathode)
        PVT_O2.calculate(T=self.T_cell, P=self.p_anode)
        PVT_H2O.calculate(T=self.T_cell, P=self.p_0_H2O)

        self.N_cells = np.ceil(self.capacity / self.power_single_cell)
        self.power_cell_real = self.power_input / self.N_cells * self.power_multiplier
        self.calc_i_cell()
        ### wteta faraday assume to be constant
        # Production rates [mol/s]
        #
        self.prod_rate_H2 = (self.N_cells / self.power_multiplier) * self.I_cell_array / (2 * self.Faraday_const) \
                            * self.eta_Faraday_array
        self.prod_rate_O2 = (self.N_cells / self.power_multiplier) * self.I_cell_array / (4 * self.Faraday_const) \
                            * self.eta_Faraday_array
        self.prod_rate_H2O = (self.N_cells / self.power_multiplier) * self.I_cell_array / (2 * self.Faraday_const)

        # Massflows [kg/s].
        self.massflow_H2 = self.prod_rate_H2 * PVT_H2.MW * 1e-3
        self.massflow_O2 = self.prod_rate_O2 * PVT_O2.MW * 1e-3
        self.massflow_H2O = self.prod_rate_H2O * PVT_H2O.MW * 1e-3

        # Densities [kg/m^3].
        self.rho_H2 = PVT_H2.rho
        self.rho_O2 = PVT_O2.rho
        self.rho_H2O = PVT_H2O.rho

        # Flowrates [m^3/s].
        self.flowrate_H2 = self.massflow_H2 / self.rho_H2
        self.flowrate_O2 = self.massflow_O2 / self.rho_O2
        self.flowrate_H2O = self.massflow_H2O / self.rho_H2O

        # Integrate massflows to obtain masses of H2, O2 and H20 in this period [kg].
        # Note: it assumes constant operating conditions in the time-step
        self.mass_H2 = self.massflow_H2 * self.delta_t
        self.mass_O2 = self.massflow_O2 * self.delta_t
        self.mass_H2O = self.massflow_H2O * self.delta_t


    def calc_i_cell(self):

        I_current_sol = root_scalar(
            self.root_I_cell, bracket=[1.0, 30000],
            method='brentq',
            args=(
                self.power_cell_real,
            )
        )
        self.I_cell_array = I_current_sol.root

    def root_I_cell(self, I_cell, power_cell):

        self.E_total_cell = \
            self.compute_potentials(
                I_cell)

        root_expr = power_cell / (self.E_total_cell) - I_cell

        return root_expr

    @staticmethod
    def compute_potentials(I_cell):

        A_cell = 0.436
        I_cell_in = I_cell / 1e4 / A_cell
        # Voltage efficiency WITH COEFFICIENTS
        E_total_cel = -0.160249069 * I_cell_in ** 4 + 0.734073995 * I_cell_in ** 3 - 1.168543948 * I_cell_in ** 2 + 1.048496283 * I_cell_in + 1.46667069

        return E_total_cel

if __name__ =='__main__':


    correlation_result = Electrolyser_pydolphin_DeGroot(2118181.8181)
    res = correlation_result.calculate_output()
    print(correlation_result.output)
