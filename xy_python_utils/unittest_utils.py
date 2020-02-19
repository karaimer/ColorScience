#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Feb 15, 2014.

"""Utility functions for unit test."""

import numpy as np

def check_near(v1, v2, tol, raise_exception=True):
    """Check whether scalar/vector/matrix 'v1' and 'v2' are close to each other
    under tolerance tol, in the sense that::


        (absolute)   ||v1 - v2|| <= tol,   **or**
        (relative)   ||v1 - v2|| / max(||v1||, ||v2||, eps) <= tol,

    where ||.|| is the Frobenius norm."""
    diff = v1 - v2
    errAbs = np.linalg.norm(diff)
    errRel = errAbs / max(np.linalg.norm(v1), np.linalg.norm(v2), np.spacing(1))
    if (errAbs <= tol) or (errRel <= tol):
        return True
    else:
        errMsg = "".join([
            "||v1-v2|| = %f\n" % errAbs,
            "||v1-v2|| / max(||v1||, ||v2||, eps) = %f\n" % errRel,
            "Tolerance = %f\n" % tol])
        if (raise_exception):
            raise Exception(errMsg)
        return False

def check_near_abs(v1, v2, tol, raise_exception=True):
    """Same as 'check_near' but only check in the "absolute" sense."""
    diff = v1 - v2
    errAbs = np.linalg.norm(diff)
    if (errAbs <= tol):
        return True
    else:
        errMsg = "".join([
            "||v1-v2|| = %f\n" % errAbs,
            "Tolerance = %f\n" % tol])
        if (raise_exception):
            raise Exception(join(errMsg))
        return False

def check_near_rel(v1, v2, tol, raise_exception=True):
    """Same as 'check_near' but only check in the "relative" sense."""
    diff = v1 - v2
    errRel = np.linalg.norm(diff) / \
             max(np.linalg.norm(v1), np.linalg.norm(v2), np.spacing(1))
    if (errRel <= tol):
        return True
    else:
        errMsg = "".join([
            "||v1-v2|| / max(||v1||, ||v2||, eps) = %f\n" % errRel,
            "Tolerance = %f\n" % tol])
        if (raise_exception):
            raise Exception(errMsg)
        return False

def check_gradient(fcn, dfcn, N, x0=None, dx=None, delta=1e-4, m=0.01, M=10,
                   raise_exception=True):
    """Numerically check whether `dfcn` calculates the gradient of `fcn`.

    More specifically, this function checks whether the following quantities are
    close to each other

      * `f(x) - f(x0)`
      * `(x-x0) \cdot f'(x0)`

    We consider them to be close enough if **either one** of the following is
    true

      1. the absolute difference is smaller than `(m * ||x-x0||)`;
      2. the relative difference is smaller than `(M * ||x-x0||)`.

    Parameters
    ----------
    fcn: function handler
        Takes a single (vector or scalar) as input and outputs a scalar.
    dfcn: function handler
        Takes a single (vector or scalar) as input and outputs a vector output
        for gradient of 'fcn'. NOTE: Another option is to let `dfcn=None` (or
        something else that is not callable, e.g. []), and fcn return a 2-tuple
        for both fucntion value and its gradient.
    N: int
        The dimensionality of input to the fucntion, which is a Nx1 vector.
    x0:
        The initial input point evaluated by the function, with default
        {randn(N)}.
    dx, delta:
        The direction of evaluation point moves, such that::

            x = x0 + delta*dx

        with 'dx' a unit Nx1 vector and 'delta' a scalar.
    m, M: float, optional
        The thresholds described above.

    """
    # Set default.
    if (not x0):
        x0 = np.random.randn(N)
    if (not dx):
        dx = np.random.randn(N)
        dx /= np.linalg.norm(dx)
    # Evaluate the functions.
    x = x0 + delta*dx
    if hasattr(dfcn, "__call__"):
        y0 = fcn(x0)
        dy0 = dfcn(x0)
        y = fcn(x)
    else:
        y0, dy0 = fcn(x0)
        y, _ = fcn(x)
    # Do the check.
    v1 = y - y0
    v2 = np.dot(x-x0, dy0)
    if (check_near_abs(v1, v2, m*delta, raise_exception=False) or
        check_near_rel(v1, v2, M*delta, raise_exception=False)):
        return True
    else:
        errMsg = "".join([
            "f(x) - f(x0) = %e\n" % v1,
            "(x-x0) * f'(x0) = %e\n" % v2,
            "Absolute difference = %e" % abs(v1-v2),
            " > m * ||x-x0||",
            " = %s * %s = %s\n" % (str(m), str(delta), str(m*delta)),
            "Relative difference = %e > " % (abs(v1-v2)/max(abs(v1),abs(v2))),
            " > M*||x-x0|| ",
            " = %s * %s = %s\n" % (str(M), str(delta), str(M*delta))])
        if (raise_exception):
            raise Exception(errMsg)
        return False

def check_jacobian(fcn, dfcn, N, x0=None, dx=None, delta=1e-4, m=0.01, M=10,
                   raise_exception=True):
    """Numerically check whether `dfcn` calculates the Jacobian of `fcn`.

    More specifically, whether the following vectors are close to each other

      * `f(x) - f(x0)`
      * `J(x0) \cdot (x-x0)`

    We consider them to be close enough if **either one** of the following is
    true

      1. "absolutely" close with tolerance `m*||x-x0||` (see `check_near_abs`);
      2. "relatively" close with tolerance `M*||x-x0||` (see `check_near_rel`).

    Parameters
    -----------
    fcn: function handler
        Takes a single (vector or scalar) as input and outputs a vector.
    dfcn: function handler
        Takes a single (vector or scalar) as input and outputs a matrix for
        Jacobian of `fcn`. NOTE: Another option is to let dfcn=None (or
        something else that is not callable, e.g. `[]`), and `fcn` return a
        2-tuple for both fucntion value and its Jacobian.
    The rest is the same as `check_gradient`.

    """
    # Set default.
    if (not x0):
        x0 = np.random.randn(N)
    if (not dx):
        dx = np.random.randn(N)
        dx /= np.linalg.norm(dx)
    # Evaluate the functions.
    x = x0 + delta*dx
    if hasattr(dfcn, "__call__"):
        y0 = fcn(x0)
        J0 = dfcn(x0)
        y = fcn(x)
    else:
        y0, J0 = fcn(x0)
        y, _ = fcn(x)
    # Do the check.
    v1 = y - y0
    v2 = np.dot(J0, x-x0)
    if (check_near_abs(v1, v2, m*delta, raise_exception=False) or
        check_near_rel(v1, v2, M*delta, raise_exception=False)):
        return True
    else:
        absErr = np.linalg.norm(v1-v2)
        relErr = absErr / max(np.linalg.norm(v1),np.linalg.norm(v2))
        errMsg = "".join([
            "Absolute difference = %e" % absErr,
            " > m * ||x-x0||",
            " = %s * %s = %s\n" % (str(m), str(delta), str(m*delta)),
            "Relative difference = %e > " % relErr,
            " > M*||x-x0|| ",
            " = %s * %s = %s\n" % (str(M), str(delta), str(M*delta))])
        if (raise_exception):
            raise Exception(errMsg)
        return False
