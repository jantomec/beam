import unittest
import numpy as np
import interpolation as intp
import projection as proj


class TestProjection(unittest.TestCase):

    def test_nearest_point_projection(self):
        correct = np.array([-0.23045933, 0.38268789, -0.0406458, -0.4820473])
        X = np.array([[0,   0,   0  ],
                      [1,   0,   0.4],
                      [1.5, 0.4, 1.0],
                      [1.2, 1.0, 1.7]]).T
        P = np.array([1.5, 0.0, 0.0])
        interpolation_d0 = lambda n, x: intp.lagrange_poly(degree=n, eval_pts=x)
        interpolation_d1 = lambda n, x: intp.lagrange_poly_d(degree=n, eval_pts=x)
        interpolation_d2 = lambda n, x: intp.lagrange_poly_dd(degree=n, eval_pts=x)
        u = proj.nearest_point_projection(
            interpolation_d0, interpolation_d1, interpolation_d2,
            X, P,
            s0=0, TOLER=1e-8, MAXITER=10
        )
        self.assertTrue(np.allclose(u, correct, rtol=1e-10))

if __name__ == '__main__':
    unittest.main()