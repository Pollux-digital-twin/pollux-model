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

        # ACT
        model.calculate_output(u)

        # ASSERT
        y = model.get_output()

        expected_data_type = 'custom_solar'
        expected_power = 1

        self.assertAlmostEqual(y['data_type'], expected_data_type, )
        self.assertAlmostEqual(y['power'], expected_power, delta=0.1)
