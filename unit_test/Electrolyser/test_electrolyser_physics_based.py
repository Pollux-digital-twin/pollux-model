import unittest
from pollux_model.Electrolyser.electrolyser_physics_based import Electrolyser


class TestElectrolyser_pydolphin_DeGroot(unittest.TestCase):

    def test_get_inputs(self):
        # ARRANGE
        #todo: difference between param and u?
        param = dict()
        # param['capacity'] = 100 * 1e6

        param['eta_Faraday_array'] = 1  # just a constant, in reality is a variable
        param['Faraday_const'] = 96485.3329  # Faraday constant [(s A)/mol]
        param['delta_t'] = 3600
        param['A_cell'] = 0.436
        param['cell_type'] = 'low_power_cell'
        param['electrolyser_model'] = "Physics_based"



        model = Electrolyser()
        model.update_parameters(param)

        # define the state IF NEEDED
        x = []

        # define the input values
        u = dict()
        u['capacity'] = 100 * 1e6
        u['T_cell'] = 273.15 + 40
        u['p_cathode'] = 10e5
        u['p_anode'] = 10e5
        u['p_0_H2O'] = 10e5
        u['power_input'] = 2118181.8181 #unit:Watt
        #u['power_multiplier'] = 5 # independent stacks
        # calculate  output
        model.calculate_output(u, x)
#
        #ACT
        # get output
        y = model.get_output()
        expected_hydrogen_rate = 53.81
        expected_oxygen_rate = 427.11
        self.assertAlmostEqual(y['hydrogen_production'], expected_hydrogen_rate, delta=1)
        self.assertAlmostEqual(y['oxygen_production'], expected_oxygen_rate, delta=3)

        print(y)


# Run the unittest from this file
if __name__ == '__main__':
    unittest.main()
