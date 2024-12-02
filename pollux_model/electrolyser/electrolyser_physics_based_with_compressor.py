from scipy.optimize import root_scalar
from pollux_model.model_abstract import Model
from pollux_model.electrolyser.electrolyser_physics_based import ElectrolyserDeGroot
from pollux_model.compressor.compressor import Compressor
import math


class ElectrolyserWithCompressor(ElectrolyserDeGroot):
    """ Abstract base class for simulation models

        Model classes implement a discrete state space model
        The state of the model is maintained outside the model object
    """

    def __init__(self):
        """ Model initialization
        """
        # super().__init__()
        ElectrolyserDeGroot.__init__(self)

    # def update_parameters(self, parameters):
    #     """ To update model parameters

    #     Parameters
    #     ----------
    #     parameters : dict
    #         parameters dict as defined by the model
    #     """
    #     for key, value in parameters.items():
    #         self.parameters[key] = value

    def initialize_state(self, x):
        """ generate an initial state based on user parameters """
        pass

    def calculate_output(self):
        """calculate output based on input u"""
        u = self.input
        solution = root_scalar(self._objective_function, bracket=[0.01 * u['power_input'], 0.9 * u['power_input']],
                               method='brentq')

        v = dict(u)  # make a copy
        v['power_input'] = u['power_input'] - solution.root
        ElectrolyserDeGroot._calc_prod_rates(self, v)

        #  checking
        mass_flow = self.output['massflow_H2']  # kg/s
        compressor_power = Compressor._power_calculation(self, mass_flow)
        rel_tolerance = 0.001
        if not math.isclose(compressor_power, solution.root, rel_tol=rel_tolerance):
            raise ValueError(f"{compressor_power} and {solution.root} are not equal within a tolerance of {rel_tolerance}.")

        self.output['power_electrolyser'] = v['power_input']
        self.output['power_compressor'] = solution.root

    def _objective_function(self, x):
        u = self.input
        v = dict(u)  # make a copy
        v['power_input'] = u['power_input'] - x
        ElectrolyserDeGroot._calc_prod_rates(self, v)
        mass_flow = self.output['massflow_H2']  # kg/s
        compressor_power = Compressor._power_calculation(self, mass_flow)
        return compressor_power - x
