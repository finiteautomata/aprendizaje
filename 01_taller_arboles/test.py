import unittest
from tree_aa import entropy

class EntropyTest(unittest.TestCase):
    def test_for_zero_entropy(self):
        self.assertAlmostEqual(entropy([0, 0, 0]), 0.0)

    def test_for_max_entropy(self):
        # We need two bits to represent this
        self.assertAlmostEqual(entropy([0, 1, 2, 3]), 2.0)

    def test_for_just_one_element(self):
        self.assertAlmostEqual(entropy([1]), 0.0)

if __name__ == '__main__':
    unittest.main()
