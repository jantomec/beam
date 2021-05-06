import sys
sys.path.insert(0, 'C:/Users/jan.tomec/Documents/THREAD/beam')
import numpy as np
import unittest
from beam_tests import test_static_cantilever
from beam_tests import test_static_cantilever_displacement_control
from beam_tests import test_dynamic_frame

class TestSimoElement(unittest.TestCase):

    def test_static_cantilever(self):
        correct = np.zeros(3)
        cantilever = test_static_cantilever.case()
        cantilever.printing = False
        cantilever.solve()
        u = cantilever.coordinates[:,-1] + cantilever.displacement[-1][:,-1]
        self.assertTrue(np.allclose(u, correct, rtol=1e-10))

    def test_static_cantilever_displacement_control(self):
        correct = np.array([0.00106344, 0.0, 0.05166995])
        cantilever = test_static_cantilever_displacement_control.case()
        cantilever.printing = False
        cantilever.solve()
        u = cantilever.displacement[-1][:,1]
        self.assertTrue(np.allclose(u, correct, rtol=1e-8))

    def test_dynamic_frame(self):
        correct = np.array([[ 0.        ,  0.08215249,  0.29149011, -0.07170301, -0.7119306 ],
                            [ 0.        ,  1.57180122,  4.18189905,  2.4362738 ,  0.05252605],
                            [ 0.        , -0.26587026, -1.01934113, -0.37816349,  0.1400477 ]])
        frame = test_dynamic_frame.case()
        frame.printing = False
        frame.solve()
        u = frame.displacement[-1]
        self.assertTrue(np.allclose(u, correct, rtol=1e-8))

if __name__ == '__main__':
    unittest.main()
