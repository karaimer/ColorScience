#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Mar 19, 2014.

import numpy as np
import unittest

from numerical_differentiation import *
from unittest_utils import *

class NumericalDifferentiationTest(unittest.TestCase):
    """Unit test for NumericalDifferentiation."""
    def _testNumJacCost(self, x, jac):
        y = sum(x**2)
        f = np.array([np.sin(y), np.cos(y), np.exp(-y**2), np.log(y)])
        if not jac:   return f
        Jy = np.array([np.cos(y), -np.sin(y), -2*y*np.exp(-y**2), 1/y])
        J = 2*np.outer(Jy, x)
        return (f, J)

    def testNumericalJacobian(self):
        np.random.seed(0)
        x0 = np.random.randn(5)
        # Forward difference.
        fcn = lambda x: self._testNumJacCost(x, False)
        nJ, nf0 = numerical_jacobian(fcn, x0, dx=1e-4, return_f0=True)
        af0, aJ = self._testNumJacCost(x0, True)
        check_near(af0, nf0, 1e-12)
        check_near(nJ, aJ, 1e-3)
        # Central difference (higher accuracy).
        fcn = lambda x: self._testNumJacCost(x, False)
        nJ = numerical_jacobian(fcn, x0, dx=1e-4, method="central")
        check_near(nJ, aJ, 1e-6)
