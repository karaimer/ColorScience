#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Feb 15, 2014.

import numpy as np
import unittest

from unittest_utils import *

class UnitTestUtilsTest(unittest.TestCase):
    """Unit test for unittest_utils."""
    def test_check_near(self):
        np.random.seed(0)
        v1 = [np.array([1.0]),
              np.random.randn(4),
              np.random.randn(5,7)]
        v2 = [v1[0]+np.random.randn(1)*0.5,
              v1[1]+np.random.randn(4)*0.5,
              v1[2]+np.random.randn(5,7)*0.5]
        for i in range(len(v1)):
            # Check for success case.
            self.assertTrue(check_near(v1[i], v2[i], 1.0))
            self.assertTrue(check_near_abs(v1[i], v2[i], np.sqrt(v1[i].size)))
            self.assertTrue(check_near_rel(v1[i], v2[i], 1.0))
            # Check for fail case.
            self.assertRaises(Exception, check_near, v1[i], v2[i], 0.1)
            self.assertRaises(Exception, check_near_abs, v1[i], v2[i], 0.1)
            self.assertRaises(Exception, check_near_rel, v1[i], v2[i], 0.1)
            # Check for fail case but no exception raised.
            self.assertFalse(
                check_near(v1[i], v2[i], 0.1, raise_exception=False))
            self.assertFalse(
                check_near_abs(v1[i], v2[i], 0.1, raise_exception=False))
            self.assertFalse(
                check_near_rel(v1[i], v2[i], 0.1, raise_exception=False))

    def test_check_gradient(self):
        np.random.seed(0)
        fcn = lambda x: \
              np.sin(x[0]) - np.cos(x[1]) + x[0]**2*np.exp(-5*x[1])
        dfcn = lambda x: \
               np.array([np.cos(x[0]) + 2*x[0]*np.exp(-5*x[1]),
                         np.sin(x[1]) - 5*x[0]**2*np.exp(-5*x[1])])
        self.assertTrue(check_gradient(fcn, dfcn, 2))
        self.assertRaises(Exception, check_gradient, fcn, dfcn, 2, m=0, M=0)
        self.assertFalse(
            check_gradient(fcn, dfcn, 2, m=0, M=0, raise_exception=False))
        fdf = lambda x: (fcn(x), dfcn(x))
        self.assertTrue(check_gradient(fdf, None, 2))
        self.assertRaises(Exception, check_gradient, fdf, [], 2, m=0, M=0)
        self.assertFalse(
            check_gradient(fdf, True, 2, m=0, M=0, raise_exception=False))

    def test_check_jacobian(self):
        np.random.seed(0)
        fcn = lambda x: np.array(
            [np.sin(x[0]), -np.cos(x[1]), x[0]**2 * np.exp(-5*x[1])])
        dfcn = lambda x: np.array(
            [[          np.cos(x[0]),                          0],
             [                     0,               np.sin(x[1])],
             [2*x[0]*np.exp(-5*x[1]), -5*x[0]**2*np.exp(-5*x[1])]])
        self.assertTrue(check_jacobian(fcn, dfcn, 2))
        self.assertRaises(Exception, check_jacobian, fcn, dfcn, 2, m=0, M=0)
        self.assertFalse(
            check_jacobian(fcn, dfcn, 2, m=0, M=0, raise_exception=False))
        fdf = lambda x: (fcn(x), dfcn(x))
        self.assertTrue(check_jacobian(fdf, None, 2))
        self.assertRaises(Exception, check_jacobian, fdf, [], 2, m=0, M=0)
        self.assertFalse(
            check_jacobian(fdf, True, 2, m=0, M=0, raise_exception=False))

if __name__ == "__main__":
    unittest.main()
