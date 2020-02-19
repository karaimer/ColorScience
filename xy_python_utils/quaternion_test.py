#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Mar 18, 2014.

import numpy as np
import unittest

from quaternion import *
from unittest_utils import *

class QuaternionTest(unittest.TestCase):
    """Unit test for Quaternion."""
    def testQuadHProd(self):
        o = np.array([1, 0, 0, 0])
        i = np.array([0, 1, 0, 0])
        j = np.array([0, 0, 1, 0])
        k = np.array([0, 0, 0, 1])

        tol = 1e-12

        # ii = jj = kk = ijk = -1.
        check_near(quatHProd(i,i), -o, tol)
        check_near(quatHProd(j,j), -o, tol)
        check_near(quatHProd(k,k), -o, tol)
        check_near(quatHProd(quatHProd(i,j),k), -o, tol)

        # ij = k, ji = -k.
        check_near(quatHProd(i,j),  k, tol)
        check_near(quatHProd(j,i), -k, tol)

        # jk = i, kj = -i.
        check_near(quatHProd(j,k),  i, tol)
        check_near(quatHProd(k,j), -i, tol)

        # ki = j, ik = -j.
        check_near(quatHProd(k,i),  j, tol)
        check_near(quatHProd(i,k), -j, tol)

        # ||q|| = sqrt(q q*)
        np.random.seed(0)
        tol = 1e-12
        nTest = 100
        for iTest in xrange(nTest):
            q = np.random.randn(4)
            p = quatConj(q)
            n = np.array([np.dot(q,q), 0, 0, 0])
            check_near(quatHProd(q,p), n, tol)
            check_near(quatHProd(p,q), n, tol)

    def testQuatRecip(self):
        np.random.seed(0)
        tol = 1e-12
        o = np.array([1,0,0,0])
        nTest = 100
        for iTest in xrange(nTest):
            q = np.random.randn(4)
            p = quatRecip(q)

            check_near(quatHProd(p,q), o, tol)
            check_near(quatHProd(q,p), o, tol)

    def testQuatToRotMatx(self):
        np.random.seed(0)
        tol = 1e-12
        nTest = 100
        for iTest in xrange(nTest):
            # Rotate the vector by two different methods and check whether the
            # results are the same.
            v = np.random.randn(3)
            q = np.random.randn(4)
            q /= np.linalg.norm(q)
            R = quatToRotMatx(q)

            check_near(rotVecByQuat(v, q), np.dot(R, v), tol)

    def testQuatFromRotMatx(self):
        np.random.seed(0)
        tol = 1e-12
        nTest = 100
        for iTest in xrange(nTest):
            # Convert back and forth between quaternion and rotation matrix.
            q = np.random.randn(4)
            q /= np.linalg.norm(q)
            R = quatToRotMatx(q)
            q2 = quatFromRotMatx(R)
            assert check_near(q, q2, tol, raise_exception=False) or \
                check_near(q, -q2, tol, raise_exception=False)
    def testRotVecByAxisAng(self):
        tol = 1e-12
        x = np.array([1,0,0])
        z = np.array([0,0,1])
        theta = np.pi / 3
        r = np.array([np.cos(theta), np.sin(theta), 0])
        check_near(rotVecByAxisAng(x, z, theta), r, tol)
