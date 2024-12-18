import unittest
from pollux_model.electrolyser.electrolyser_physics_based import ElectrolyserDeGroot


class TestElectrolyserPhysicsBased(unittest.TestCase):

    def test_calculate_model(self):
        # ARRANGE
        param = dict()

        param['eta_Faraday_array'] = 1  # just a constant, in reality is a variable
        param['Faraday_const'] = 96485.3329  # Faraday constant [(s A)/mol]
        param['delta_t'] = 3600  # timestep in seconds
        param['A_cell'] = 0.436  # area in m2
        param['cell_type'] = 'low_power_cell'
        param['capacity'] = 100 * 1e6  # capacity in Watt

        model = ElectrolyserDeGroot()
        model.update_parameters(param)

        # define the input values
        u = dict()
        u['T_cell'] = 273.15 + 40  # cell temperature in K
        u['p_cathode'] = 10e5  # cathode pressure in Pa
        u['p_anode'] = 10e5  # anode pressure in Pa
        u['p_0_H2O'] = 10e5  # Pa
        u['power_input'] = 2118181.8181  # input power in Watt

        model.input = u
        # ACT
        model.calculate_output()

        # ASSERT
        y = model.get_output()

        expected_hydrogen_rate = 50.2  # 53.81
        expected_oxygen_rate = 398.8  # 427.11

        self.assertAlmostEqual(y['mass_H2'], expected_hydrogen_rate, delta=0.1)
        self.assertAlmostEqual(y['mass_O2'], expected_oxygen_rate, delta=0.1)
