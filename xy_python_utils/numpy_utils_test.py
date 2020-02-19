#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Mar 13, 2014.

import unittest

from numpy_utils import *
from unittest_utils import *

class NumpyUtilsTest(unittest.TestCase):
    """Unit test for NumpyUtils."""
    def test_null(self):
        np.random.seed(0)
        tol = 1e-12
        A = np.random.randn(3)
        nA = null(A)
        r = np.dot(A, nA)
        check_near(r, np.zeros(2), tol)
        A = np.random.randn(2,5)
        nA = null(A)
        r = np.dot(A, nA)
        check_near(r, np.zeros((2,3)), tol)

    def test_meshgrid_nd(self):
        np.random.seed(0)
        tol = 1e-12
        x = np.random.randn(5)
        y = np.random.randn(3)
        X, Y = np.meshgrid(x, y)
        X2, Y2 = meshgrid_nd(x, y)
        check_near(X-X2.T, np.zeros(X.shape), tol)
        check_near(Y-Y2.T, np.zeros(Y.shape), tol)
        z = np.random.randn(8)
        X, Y, Z = meshgrid_nd(x, y, z)
        self.assertEqual(X.shape, (5,3,8))
        self.assertEqual(Y.shape, (5,3,8))
        self.assertEqual(Z.shape, (5,3,8))

if __name__ == "__main__":
    unittest.main()
