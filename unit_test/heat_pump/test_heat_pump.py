import unittest
from pollux_model.heat_pump.heat_pump_physics_based import HeatpumpNREL


class TestHeatpumpNREL(unittest.TestCase):

    def test_heat_pump(self):
        # test 1:input:power electricity ,output:expected heat requirement
        # ARRANGE
        param = dict()
        param['refrigerant'] = 'n-Pentane'
        model = HeatpumpNREL()
        model.update_parameters(param)

        # define the input values
        u = dict()

        u['hot_temperature_desired'] = 150
        u['hot_temperature_return'] = 100
        u['cold_temperature_available'] = 80
        u['cold_deltaT'] = 40

        u['process_heat_requirement'] = 'NaN'
        u['hot_mass_flowrate'] = 'NaN'
        u['electricity_power_in'] = 419537.6

        model.input = u
        # ACT
        model.calculate_output()

        # ASSERT
        y = model.get_output()

        expected_hot_mass_flow_rate = 4.69
        expected_cold_mass_flow_rate = 5.55
        expected_heat_requirement = 999999.87
        expected_actual_COP = 2.38
        expected_cold_temperature_return = 37
        #
        self.assertAlmostEqual(y['hot_mass_flow_rate'], expected_hot_mass_flow_rate, delta=0.1)
        self.assertAlmostEqual(y['cold_mass_flow_rate'], expected_cold_mass_flow_rate, delta=0.1)
        self.assertAlmostEqual(y['process_heat_requirement'],
                               expected_heat_requirement, delta=1)
        self.assertAlmostEqual(y['actual_COP'], expected_actual_COP, delta=0.1)
        self.assertAlmostEqual(y['cold_temperature_return'], expected_cold_temperature_return,
                               delta=0.1)

        # test 2:input:expected heat requirement ,output:power electricity
        # ARRANGE
        param = dict()
        param['refrigerant'] = 'n-Pentane'
        model = HeatpumpNREL()
        model.update_parameters(param)

        # define the input values
        u = dict()

        u['hot_temperature_desired'] = 150
        u['hot_temperature_return'] = 100
        u['cold_temperature_available'] = 80
        u['cold_deltaT'] = 40

        u['process_heat_requirement'] = 999999.87
        u['hot_mass_flowrate'] = 'NaN'
        u['electricity_power_in'] = 'NaN'

        model.input = u
        # ACT
        model.calculate_output()

        # ASSERT
        y = model.get_output()

        expected_hot_mass_flow_rate = 4.69
        expected_cold_mass_flow_rate = 5.55
        expected_electricity_power_in = 419537.6
        expected_actual_COP = 2.38
        expected_cold_temperature_return = 37
        #
        self.assertAlmostEqual(y['hot_mass_flow_rate'], expected_hot_mass_flow_rate, delta=0.1)
        self.assertAlmostEqual(y['cold_mass_flow_rate'], expected_cold_mass_flow_rate, delta=0.1)
        self.assertAlmostEqual(y['electricity_power_in'],
                               expected_electricity_power_in, delta=1)
        self.assertAlmostEqual(y['actual_COP'], expected_actual_COP, delta=0.1)
        self.assertAlmostEqual(y['cold_temperature_return'], expected_cold_temperature_return,
                               delta=0.1)

        # test 3:input:expected hot mass flow rate ,output:power electricity,heat requirement
        # ARRANGE
        param = dict()
        param['refrigerant'] = 'n-Pentane'
        model = HeatpumpNREL()
        model.update_parameters(param)

        # define the input values
        u = dict()

        u['hot_temperature_desired'] = 150
        u['hot_temperature_return'] = 100
        u['cold_temperature_available'] = 80
        u['cold_deltaT'] = 40

        u['process_heat_requirement'] = 'NaN'
        u['hot_mass_flowrate'] = 4.69
        u['electricity_power_in'] = 'NaN'

        model.input = u
        # ACT
        model.calculate_output()

        # ASSERT
        y = model.get_output()

        expected_heat_requirement = 999032.277983237
        expected_cold_mass_flow_rate = 5.55
        expected_electricity_power_in = 419131.6
        expected_actual_COP = 2.38
        expected_cold_temperature_return = 37
        #
        self.assertAlmostEqual(y['process_heat_requirement'],
                               expected_heat_requirement, delta=1)
        self.assertAlmostEqual(y['cold_mass_flow_rate'], expected_cold_mass_flow_rate, delta=0.1)
        self.assertAlmostEqual(y['electricity_power_in'],
                               expected_electricity_power_in, delta=1)
        self.assertAlmostEqual(y['actual_COP'], expected_actual_COP, delta=0.1)
        self.assertAlmostEqual(y['cold_temperature_return'], expected_cold_temperature_return,
                               delta=0.1)
