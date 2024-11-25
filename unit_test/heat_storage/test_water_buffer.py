import unittest
from pollux_model.heat_storage.water_buffer import WaterBufferTankModel


class TestHeatStorage(unittest.TestCase):

    def test_calculate_fill_level_charge(self):
        # ARRANGE
        buffertank = WaterBufferTankModel()

        param = dict()
        param['timestep'] = 3600  # 1 hour in seconds
        buffertank.update_parameters(param)

        x = dict()
        x['current_volume'] = 0.5
        buffertank.initialize_state(x)

        # ACT
        u = dict()
        u['volume_flow'] = 0.02 / 3600  # 1 m3/hr to m3/s
        buffertank.calculate_output(u)

        # ASSERT
        y = buffertank.get_output()

        self.assertAlmostEqual(y['fill_level'], 0.52, delta=0.01)

    def test_calculate_fill_level_discharge(self):
        # ARRANGE
        buffertank = WaterBufferTankModel()

        param = dict()
        param['timestep'] = 3600  # 1 hour in seconds
        buffertank.update_parameters(param)

        x = dict()
        x['current_volume'] = 0.5
        buffertank.initialize_state(x)

        # ACT
        u = dict()
        u['volume_flow'] = -0.02 / 3600  # 1 m3/hr to m3/s
        buffertank.calculate_output(u)

        # ASSERT
        y = buffertank.get_output()

        self.assertAlmostEqual(y['fill_level'], 0.48, delta=0.01)
