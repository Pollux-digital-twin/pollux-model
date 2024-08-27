import unittest
from pollux_model.gas_storage.hydrogen_tank_model import HydrogenTankModel


class TestHydrogenStorage(unittest.TestCase):

    def test_calculate_fill_level(self):
        # ARRANGE
        hydrogentank = HydrogenTankModel()

        param = dict()
        param['timestep'] = 3600  # 1 hour in seconds
        hydrogentank.update_parameters(param)

        # ACT
        u = dict()
        u['mass_flow'] = 1 / 3600  # 1 kg/hr to kg/s
        hydrogentank.calculate_output(u)

        # ASSERT
        y = hydrogentank.get_output()

        self.assertAlmostEqual(y['fill_level'], 0.166, delta=0.01)
