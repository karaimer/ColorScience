#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Mar 13, 2014.

"""Some extended utility functions for 'numpy' module."""

import numpy as np

def null(A, tol = 1e-12):
    """Return the null space of matrix or vector `A`, such that::

        dot(A, null(A)) == eps(M, N)

    Each column `r` of `null(A)` is a unit vector, and `||dot(A, r)|| < tol`.
    """
    A = np.atleast_2d(A)
    _, s, vt = np.linalg.svd(A)
    nnz = (s >= tol).sum()
    return vt[nnz:].T

def meshgrid_nd(*arrs):
    """Multi-dimensional meshgrid.

    Parameters
    ----------
    x, y, z, ...: ndarray
        Multiple 1-D arrays representing the coordinates of the grid.

    Returns
    -------
    X, Y, Z, ... : ndarray
        Multi-dimensional arrays of shape (len(x), len(y), len(z), ...). Note
        that there is a discrepancy to the original 2D meshgrid, where the
        output array shape is swapped, i.e. (len(y), len(x)). Specifically, if::

            X, Y = meshgrid(x, y)
            X2, Y2 = meshgrid_nd(x, y)

        then we have `X == X2.T` and `Y == Y2.T`.

    Examples
    --------
    >>> X, Y, Z = np.meshgrid([1,2,3], [10,20], [-2,-3,-4,-5])
    >>> X
    array([[[1, 1, 1, 1],
            [1, 1, 1, 1]],
           [[2, 2, 2, 2],
            [2, 2, 2, 2]],
           [[3, 3, 3, 3],
            [3, 3, 3, 3]]])
    >>> Y
    array([[[10, 10, 10, 10],
            [20, 20, 20, 20]],
           [[10, 10, 10, 10],
            [20, 20, 20, 20]],
           [[10, 10, 10, 10],
            [20, 20, 20, 20]]])
    >>> Z
    array([[[-2, -3, -4, -5],
            [-2, -3, -4, -5]],
           [[-2, -3, -4, -5],
            [-2, -3, -4, -5]],
           [[-2, -3, -4, -5],
            [-2, -3, -4, -5]]])

    """
    shape = map(len, arrs)
    ndim = len(arrs)

    ans = []
    for i, arr in enumerate(arrs):
        slc = [1] * ndim
        slc[i] = shape[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(shape):
            if j != i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)

    return tuple(ans)
