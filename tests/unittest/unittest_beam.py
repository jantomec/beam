import numpy as np
import unittest
from tests import cantilever, frame, patch

class TestSimoElement(unittest.TestCase):

    def test_static_cantilever(self):
        correct = np.zeros(3)
        case = cantilever.case1()
        case.printing = False
        case.solve()
        u = case.coordinates[:,-1] + case.displacement[-1][:,-1]
        self.assertTrue(np.allclose(u, correct, rtol=1e-10))

    def test_static_cantilever_displacement_control(self):
        correct = np.array([0.00106344, 0.0, 0.05166995])
        case = cantilever.case2()
        case.printing = False
        case.solve()
        u = case.displacement[-1][:,1]
        self.assertTrue(np.allclose(u, correct, rtol=1e-8))

    def test_dynamic_frame(self):
        correct = np.array([[ 0.        ,  0.08215249,  0.29149011, -0.07170301, -0.7119306 ],
                            [ 0.        ,  1.57180122,  4.18189905,  2.4362738 ,  0.05252605],
                            [ 0.        , -0.26587026, -1.01934113, -0.37816349,  0.1400477 ]])
        case = frame.case()
        case.printing = False
        case.solve()
        u = case.displacement[-1]
        self.assertTrue(np.allclose(u, correct, rtol=1e-8))
    
    def test_contact_patch(self):
        correct = 0.0
        case = patch.case()
        case.printing = False
        case.solve()

        gN = np.array(case.gap_function)
        u = np.linalg.norm(gN[1:,:,1])
        self.assertTrue(np.allclose(u, correct, rtol=1e-8))
        
if __name__ == '__main__':
    unittest.main()
