import os
import sys

cwd = os.getcwd()
folder = os.path.basename(cwd)
while folder != "beam":
    cwd = os.path.dirname(cwd)
    folder = os.path.basename(cwd)
    if len(cwd) == 0:
        print("Root directory was not found. Try inserting the path manually with 'sys.path.insert(0, absolute_path_to_root)'")
        sys.exit()
print("Root directory:", cwd)
sys.path.insert(0, cwd)


import numpy as np
import unittest
from beam_tests import test_cantilever, test_frame, test_patch

class TestSimoElement(unittest.TestCase):

    def test_static_cantilever(self):
        correct = np.zeros(3)
        cantilever = test_cantilever.case1()
        cantilever.printing = False
        cantilever.solve()
        u = cantilever.coordinates[:,-1] + cantilever.displacement[-1][:,-1]
        self.assertTrue(np.allclose(u, correct, rtol=1e-10))

    def test_static_cantilever_displacement_control(self):
        correct = np.array([0.00106344, 0.0, 0.05166995])
        cantilever = test_cantilever.case2()
        cantilever.printing = False
        cantilever.solve()
        u = cantilever.displacement[-1][:,1]
        self.assertTrue(np.allclose(u, correct, rtol=1e-8))

    def test_dynamic_frame(self):
        correct = np.array([[ 0.        ,  0.08215249,  0.29149011, -0.07170301, -0.7119306 ],
                            [ 0.        ,  1.57180122,  4.18189905,  2.4362738 ,  0.05252605],
                            [ 0.        , -0.26587026, -1.01934113, -0.37816349,  0.1400477 ]])
        frame = test_frame.case()
        frame.printing = False
        frame.solve()
        u = frame.displacement[-1]
        self.assertTrue(np.allclose(u, correct, rtol=1e-8))
    
    def test_contact_patch(self):
        correct = 0.0
        system = test_patch.case()
        system.printing = False
        system.solve()

        gN = np.array(system.gap_function)
        u = np.linalg.norm(gN[:,:,1])
        self.assertTrue(np.allclose(u, correct, rtol=1e-8))
        
if __name__ == '__main__':
    unittest.main()
