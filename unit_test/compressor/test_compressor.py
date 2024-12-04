import unittest
from pollux_model.compressor.compressor import Compressor


class TestCompressor(unittest.TestCase):

    def test_calculate_power(self):

        param = dict()
        param['specific_heat_ratio'] = 1.41  # = 1.41 for hydrogen
        param['inlet_temperature'] = 298  # K
        param['inlet_pressure'] = 1E5  # Pa
        param['outlet_pressure'] = 20E6  # Pa
        param['R'] = 4124  # J/(kg K), gas constant = 4124 for hydrogen
        compressor = Compressor()
        compressor.update_parameters(param)

        u = dict()
        u['mass_flow'] = 0.01  # kg/s
        compressor.input = u

        compressor.calculate_output()

        y = compressor.get_output()
        self.assertAlmostEqual(y['compressor_power'], 81500.09267, delta=0.01)
