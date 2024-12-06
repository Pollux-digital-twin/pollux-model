from scipy.optimize import root_scalar
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
        super().__init__()
        self.compressor = Compressor()

    def initialize_state(self, x):
        """ generate an initial state based on user parameters """
        pass

    def calculate_output(self):
        """calculate output based on input"""
        solution = root_scalar(self._objective_function,
                               bracket=[0.01 * self.input['power_input'],
                                        0.9 * self.input['power_input']], method='brentq')

        self.input['power_input'] = self.input['power_input'] - solution.root
        ElectrolyserDeGroot._calc_prod_rates(self)

        #  checking
        mass_flow = self.output['massflow_H2']  # kg/s
        self.compressor.input['mass_flow'] = mass_flow
        self.compressor.calculate_output()
        output_compressor = self.compressor.get_output()
        rel_tolerance = 0.001
        if not math.isclose(output_compressor['compressor_power'], solution.root,
                            rel_tol=rel_tolerance):
            raise ValueError(f"{output_compressor['compressor_power']} and {solution.root} "
                             f"are not equal within a tolerance of {rel_tolerance}.")

        self.output['power_electrolyser'] = self.input['power_input']
        self.output['power_compressor'] = solution.root

    def _objective_function(self, x):
        u = self.input
        v = dict(u)  # make a copy
        self.input['power_input'] = u['power_input'] - x
        ElectrolyserDeGroot._calc_prod_rates(self)
        self.input = v
        mass_flow = self.output['massflow_H2']  # kg/s
        self.compressor.input['mass_flow'] = mass_flow
        self.compressor.calculate_output()
        output_compressor = self.compressor.get_output()

        return output_compressor['compressor_power'] - x
