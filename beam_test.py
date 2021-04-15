import unittest
import numpy as np
import frame
import cantilever


class TestSimoElement(unittest.TestCase):

    def test_cantilever(self):
        correct = np.zeros(3)
        u = cantilever.cantilever(printing=False)[-1,:,-1]
        self.assertTrue(np.allclose(u, correct, rtol=1e-10))

    def test_frame(self):
        correct = np.array([[ 0.,-0.77500735,-1.28238162,-1.36128958,-1.57274618],
                            [ 0., 0.30816282, 0.20936808, 0.08018192, 0.68076979],
                            [ 0.,-0.07214447,-0.10100747,-0.93952252,-2.22247546]])
        u = frame.frame(t_end=30.0, printing=False)[-1]
        self.assertTrue(np.allclose(u, correct, rtol=1e-10))

if __name__ == '__main__':
    unittest.main()
