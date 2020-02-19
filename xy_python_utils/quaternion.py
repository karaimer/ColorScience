#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Mar 18, 2014.

"""Utility functions for quaternion and spatial rotation.

A quaternion is represented by a 4-vector `q` as::

  q = q[0] + q[1]*i + q[2]*j + q[3]*k.

The validity of input to the utility functions are not explicitly checked for
efficiency reasons.

========  ================================================================
Abbr.      Meaning
========  ================================================================
quat      Quaternion, 4-vector.
vec       Vector, 3-vector.
ax, axis  Axis, 3- unit vector.
ang       Angle, in unit of radian.
rot       Rotation.
rotMatx   Rotation matrix, 3x3 orthogonal matrix.
HProd     Hamilton product.
conj      Conjugate.
recip     Reciprocal.
========  ================================================================
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def quatConj(q):
    """Return the conjugate of quaternion `q`."""
    return np.append(q[0], -q[1:])

def quatHProd(p, q):
    """Compute the Hamilton product of quaternions `p` and `q`."""
    r = np.array([p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3],
                  p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2],
                  p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1],
                  p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0]])
    return r

def quatRecip(q):
    """Compute the reciprocal of quaternion `q`."""
    return quatConj(q) / np.dot(q,q)

def quatFromAxisAng(ax, theta):
    """Get a quaternion that performs the rotation around axis `ax` for angle
    `theta`, given as::

        q = (r, v) = (cos(theta/2), sin(theta/2)*ax).

    Note that the input `ax` needs to be a 3x1 unit vector."""
    return np.append(np.cos(theta/2), np.sin(theta/2)*ax)

def quatFromRotMatx(R):
    """Get a quaternion from a given rotation matrix `R`."""
    q = np.zeros(4)

    q[0] = ( R[0,0] + R[1,1] + R[2,2] + 1) / 4.0
    q[1] = ( R[0,0] - R[1,1] - R[2,2] + 1) / 4.0
    q[2] = (-R[0,0] + R[1,1] - R[2,2] + 1) / 4.0
    q[3] = (-R[0,0] - R[1,1] + R[2,2] + 1) / 4.0

    q[q<0] = 0   # Avoid complex number by numerical error.
    q = np.sqrt(q)

    q[1] *= np.sign(R[2,1] - R[1,2])
    q[2] *= np.sign(R[0,2] - R[2,0])
    q[3] *= np.sign(R[1,0] - R[0,1])

    return q

def quatToRotMatx(q):
    """Get a rotation matrix from the given unit quaternion `q`."""
    R = np.zeros((3,3))

    R[0,0] = 1 - 2*(q[2]**2 + q[3]**2)
    R[1,1] = 1 - 2*(q[1]**2 + q[3]**2)
    R[2,2] = 1 - 2*(q[1]**2 + q[2]**2)

    R[0,1] = 2 * (q[1]*q[2] - q[0]*q[3])
    R[1,0] = 2 * (q[1]*q[2] + q[0]*q[3])

    R[0,2] = 2 * (q[1]*q[3] + q[0]*q[2])
    R[2,0] = 2 * (q[1]*q[3] - q[0]*q[2])

    R[1,2] = 2 * (q[2]*q[3] - q[0]*q[1])
    R[2,1] = 2 * (q[2]*q[3] + q[0]*q[1])

    return R

def rotVecByQuat(u, q):
    """Rotate a 3-vector `u` according to the quaternion `q`. The output `v` is
    also a 3-vector such that::

        [0; v] = q * [0; u] * q^{-1}

    with Hamilton product."""
    v = quatHProd(quatHProd(q, np.append(0, u)), quatRecip(q))
    return v[1:]

def rotVecByAxisAng(u, ax, theta):
    """Rotate the 3-vector `u` around axis `ax` for angle `theta` (radians),
    counter-clockwisely when looking at inverse axis direction. Note that the
    input `ax` needs to be a 3x1 unit vector."""
    q = quatFromAxisAng(ax, theta)
    return rotVecByQuat(u, q)

def quatDemo():
    # Rotation axis.
    ax = np.array([1.0, 1.0, 1.0])
    ax = ax / np.linalg.norm(ax)

    # Rotation angle.
    theta = -5*np.pi/6

    # Original vector.
    u = [0.5, 0.6, np.sqrt(3)/2];
    u /= np.linalg.norm(u)

    # Draw the circle frame.
    nSamples = 1000
    t = np.linspace(-np.pi, np.pi, nSamples)
    z = np.zeros(t.shape)
    fig = plt.figure()
    fig_ax = fig.add_subplot(111, projection="3d", aspect="equal")
    fig_ax.plot(np.cos(t), np.sin(t), z, 'b')
    fig_ax.plot(z, np.cos(t), np.sin(t), 'b')
    fig_ax.plot(np.cos(t), z, np.sin(t), 'b')

    # Draw rotation axis.
    fig_ax.plot([0, ax[0]*2], [0, ax[1]*2], [0, ax[2]*2], 'r')

    # Rotate the `u` vector and draw results.
    fig_ax.plot([0, u[0]], [0, u[1]], [0, u[2]], 'm')
    v = rotVecByAxisAng(u, ax, theta)
    fig_ax.plot([0, v[0]], [0, v[1]], [0, v[2]], 'm')

    # Draw the circle that is all rotations of `u` across `ax` with different
    # angles.
    v = np.zeros((3, len(t)))
    for i,theta in enumerate(t):
        v[:,i] = rotVecByAxisAng(u, ax, theta)
    fig_ax.plot(v[0,:], v[1,:], v[2,:], 'm')

    fig_ax.view_init(elev=8, azim=80)
    plt.show()

if __name__ == "__main__":
    quatDemo()
