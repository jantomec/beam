import unittest
import numpy as np
import interpolation as intp


class TestInterpolation(unittest.TestCase):

    def test_lagrange_poly(self):
        correct = np.array([[ 0.195, -0.105],
                            [ 0.91 ,  0.91 ],
                            [-0.105,  0.195]])
        u = intp.lagrange_poly(degree=2, eval_pts=[-0.3,0.3])
        self.assertTrue(np.allclose(u, correct, rtol=1e-10))

    def test_lagrange_poly_d(self):
        correct = np.array([[-0.8, -0.2],
                            [ 0.6, -0.6 ],
                            [ 0.2,  0.8]])
        u = intp.lagrange_poly_d(degree=2, eval_pts=[-0.3,0.3])
        self.assertTrue(np.allclose(u, correct, rtol=1e-10))
    
    def test_lagrange_poly_dd(self):
        correct = np.array([[ 1.0,  1.0],
                            [-2.0, -2.0],
                            [ 1.0,  1.0]])
        u = intp.lagrange_poly_dd(degree=2, eval_pts=[-0.3,0.3])
        self.assertTrue(np.allclose(u, correct, rtol=1e-10))
    
    def test_lagrange_poly_d3(self):
        correct = np.zeros((3,2))
        u = intp.lagrange_poly_d3(degree=2, eval_pts=[-0.3,0.3])
        self.assertTrue(np.allclose(u, correct, rtol=1e-10))

if __name__ == '__main__':
    unittest.main()