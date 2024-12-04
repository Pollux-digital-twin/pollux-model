import unittest
from pollux_model.adder.adder import Adder


class TestAdder(unittest.TestCase):

    def test_add(self):
        # ARRANGE
        adder = Adder()

        u = dict()
        u['input_0'] = 0.1
        u['input_1'] = 0.9

        adder.input = u

        # ACT
        adder.calculate_output()

        # ASSERT
        y = adder.get_output()
        self.assertEqual(y['output'], 1)
