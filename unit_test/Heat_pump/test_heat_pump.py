import unittest
from pollux_model.Heat_pump.heat_pump_physics_based import Heat_pump_nrel


class TestHeat_pump_nrel(unittest.TestCase):

    def test_get_inputs(self):
        # ARRANGE
        param = dict()
        # param['temperature(C)'] = 25
        # param['pressure(bar)'] = 1
        # param['power(W)'] = 9.995438e+07
        param['heat_pump_model'] = 'physics'
        model = Heat_pump_nrel()
        model.update_parameters(param)

        # define the state IF NEEDED
        x = []

        # define the input values
        u = dict()

        u['desired outlet hot temperature'] = 150
        u['minimum outlet hot temperature'] = 100
        u['available outlet cold temperature'] = 80
        u['refrigerant'] = 'n-Pentane'

        # initialise the model
        model.initialize_state()
        # update
        model.update_state(u)

        # calculate  output
        model.calculate_output(u, x)
        #
        # ACT
        # get output
        y = model.get_output()
        expected_hot_mass_flow_rate = 991
        expected_power_demand = 41953.7
        self.assertAlmostEqual(y['Hot mass flow rate(kg/s)'], expected_hot_mass_flow_rate, delta=1)
        self.assertAlmostEqual(y['Power Demand(kW)'], expected_power_demand, delta=1)

        print(y)
