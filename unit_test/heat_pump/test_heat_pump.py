import unittest
from pollux_model.heat_pump.heat_pump_physics_based import HeatpumpNREL


class TestHeatpumpNREL(unittest.TestCase):

    def test_get_inputs(self):
        # ARRANGE
        param = dict()
        param['refrigerant'] = 'n-Pentane'
        model = HeatpumpNREL()
        model.update_parameters(param)

        # define the input values
        u = dict()

        u['hot_temperature_desired'] = 150
        u['hot_temperature_minimum'] = 100
        u['cold_temperature_available'] = 80

        # ACT
        model.calculate_output(u)

        # ASSERT
        y = model.get_output()

        expected_hot_mass_flow_rate = 991
        expected_power_demand = 41953765

        self.assertAlmostEqual(y['hot_mass_flow_rate'], expected_hot_mass_flow_rate, delta=1)
        self.assertAlmostEqual(y['power_demand'], expected_power_demand, delta=1)
