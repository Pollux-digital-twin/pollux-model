import unittest
from pollux_model.model_example.model_example import Model1


class TestModel(unittest.TestCase):

    def test_calculate_model(self):
        # ARRANGE
        param = dict()
        param['parameter_key_1'] = 'parameter_value_1'

        model_instance = Model1()
        model_instance.update_parameters(param)

        # define the input values
        u = dict()
        u['input_key_1'] = 0.999

        # calculate  output
        model_instance.calculate_output(u)

        # get output
        y = model_instance.get_output()

        self.assertAlmostEqual(y['output_key_1'], 1, delta=0.1)
