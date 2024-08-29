import unittest
from pollux_model.Electrolyser.electrolyser_physics_based import Electrolyser


class TestElectrolyser_pydolphin_DeGroot(unittest.TestCase):

    def test_get_inputs(self):
        # ARRANGE
        #todo: difference between param and u?
        param = dict()
        # param['temperature(C)'] = 25
        # param['pressure(bar)'] = 1
        # param['power(W)'] = 9.995438e+07
        param['electrolyser_model'] = 'Physics_based'


        model = Electrolyser()
        model.update_parameters(param)

        # define the state IF NEEDED
        x = []

        # define the input values
        u = dict()
        u['temperature(C)'] = 25 # unit:C
        u['pressure(bar)'] = 1 #unit:bar
        u['power(W)'] = 9.995438e+07 #unit:Watt

        # calculate  output
        model.calculate_output(u, x)
#
        #ACT
        # get output
        y = model.get_output()
        expected_hydrogen_rate = 16640
        expected_oxygen_rate = 2096
        self.assertAlmostEqual(y['hydrogen_production'], expected_hydrogen_rate, delta=1)
        self.assertAlmostEqual(y['oxygen_production'], expected_oxygen_rate, delta=1)

        print(y)


# Run the unittest from this file
if __name__ == '__main__':
    unittest.main()
