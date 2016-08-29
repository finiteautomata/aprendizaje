#! coding: utf-8
import unittest
import numpy as np
from tree_aa import entropy, information_gain

class EntropyTest(unittest.TestCase):
    def test_for_zero_entropy(self):
        self.assertAlmostEqual(entropy([0, 0, 0]), 0.0)

    def test_for_max_entropy(self):
        # We need two bits to represent this
        self.assertAlmostEqual(entropy([0, 1, 2, 3]), 2.0)

    def test_for_just_one_element(self):
        self.assertAlmostEqual(entropy([1]), 0.0)

class InformationGainTest(unittest.TestCase):
	def test_for_maximum_gain(self):
		"""
		En este caso, los valores de la columna se corresponden exactamente con sí o por no.

		Espero que sea exactamente igual a la entropía (1.0)
		"""
		column = np.array(['Rain', 'Rain', 'Sunny', 'Sunny'])
		y = np.array([False, False, True, True])

		self.assertAlmostEqual(information_gain(column, y), 1.0)

if __name__ == '__main__':
    unittest.main()
