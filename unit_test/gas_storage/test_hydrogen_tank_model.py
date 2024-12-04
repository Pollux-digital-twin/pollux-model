import unittest
from pollux_model.gas_storage.hydrogen_tank_model import HydrogenTankModel
from pollux_model.solver.step_function import StepFunction
import numpy as np


class TestHydrogenStorage(unittest.TestCase):

    def test_calculate_fill_level(self):
        # ARRANGE
        zeros_array = np.zeros(1)
        step_function = StepFunction(zeros_array, 1)
        # The HydrogenTankModel has a control vector as input argument
        # A dummy control is used for this test
        hydrogentank = HydrogenTankModel()
        hydrogentank.set_time_function(step_function)

        param = dict()
        param['timestep'] = 3600  # 1 hour in seconds
        hydrogentank.update_parameters(param)

        # ACT
        u = dict()
        u['mass_flow_in'] = 1 / 3600  # 1 kg/hr to kg/s
        hydrogentank.input = u
        hydrogentank.calculate_output()

        # ASSERT
        y = hydrogentank.get_output()

        self.assertAlmostEqual(y['fill_level'], 0.166, delta=0.01)
