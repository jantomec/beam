import unittest
import numpy as np
import mathematics as math


class TestMathematics(unittest.TestCase):

    def test_normalized(self):
        correct = np.array([1.0, 0.0, 0.0])
        u = math.normalized(np.array([2.0, 0, 0]))
        self.assertTrue(np.allclose(u, correct, rtol=1e-10))

if __name__ == '__main__':
    unittest.main()