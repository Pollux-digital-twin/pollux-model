import unittest
from pollux_model.power_input.power_curve import DataProcessor


class TestDataProcessor(unittest.TestCase):

    def test_calculate_model(self):
        # ARRANGE
        param = dict()

        param['file_path'] = None  # just a constant, in reality is a variable

        model = DataProcessor()
        model.update_parameters(param)

        # define the input values
        u = dict()
        u['T_cell'] = 273.15 + 40  # cell temperature in K
        u['p_cathode'] = 10e5  # cathode pressure in Pa
        u['p_anode'] = 10e5  # anode pressure in Pa
        u['p_0_H2O'] = 10e5  # Pa
        u['power_input'] = 2118181.8181  # input power in Watt

        # ACT
        model.calculate_output(u)

        # ASSERT
        y = model.get_output()

        expected_data_type = 'custom_solar'
        expected_power = 100

        self.assertAlmostEqual(y['data_type'], expected_data_type, )
        self.assertAlmostEqual(y['power'], expected_power, delta=0.1)
