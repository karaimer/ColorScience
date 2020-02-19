#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Apr 24, 2015.

import numpy as np
import sys

from data import *

def color_space_transform(src_data, src_space, dst_space):
    """Transform an image from a one color space to another color space.

    Parameters
    ----------
    src_data: ndarray
        The input data to be transformed, of either one of following form:
          1. a `3xN` matrix, with each column representing a color vector.
          2. an `MxNx3` or `MxNx4` image. The alpha channel will be preserved if
             present.

    src_space, dst_space: string
        Color spaces to be transformed from and to. Current supported color
        spaces are: `"CIE-XYZ"`, `"CIE-xyY"`, `"sRGB-linear"`, `"sRGB"` and
        `"CIE-L*a*b*"`.

    Returns
    -------
    dst_data : ndarray
        The output data after transformation. It will be an `ndarray` of
        the same size as `src_data`.

    """
    if len(src_data.shape) == 3:
        # The 'src_data' is an image, convert it into 3xN color matrix.
        (M,N,C) = src_data.shape
        assert (C == 3) or (C == 4)
        src_data2 = src_data[:,:,:3].reshape(M*N, 3).T

        # Recursively call this function on the 3xN matrix.
        dst_data2 = color_space_transform(src_data2, src_space, dst_space)

        # Put the transformed data back into image form.
        dst_data = np.zeros(src_data.shape)
        dst_data[:,:,:3] = dst_data2.T.reshape(M,N,3)
        if (C == 4):
            dst_data[:,:,3] = src_data[:,:,3]
        return dst_data

    try:
        # Find a transform function from `src_space` to `dst_space` and run it.
        _src_space = _color_space_name[src_space]
        _dst_space = _color_space_name[dst_space]
        _transform_fcn = getattr(
            sys.modules[__name__],
            "_transform_%s_to_%s" % (_src_space, _dst_space))
        return _transform_fcn(src_data)
    except AttributeError:
        # There is no such transform function defined. In this case we will use
        # an intermediate space `itm_space`, try convert the data from
        # `src_space` to `itm_space` and then from `itm_space` to `dst_space`.
        if src_space == "sRGB" or dst_space == "sRGB":
            itm_space = "sRGB-linear"
        else:
            itm_space = "CIE-XYZ"
        if src_space == itm_space or dst_space == itm_space:
            raise Exception("Unknown transform from '%s' to '%s'." %
                            (src_space, dst_space))
        itm_data = color_space_transform(src_data, src_space, itm_space)
        dst_data = color_space_transform(itm_data, itm_space, dst_space)
        return dst_data

_color_space_name = {
    "CIE-XYZ": "xyz",
    "CIE-xyY": "xyy",
    "sRGB": "srgb",
    "sRGB-linear": "srgblin",
    "CIE-L*a*b*": "lab",
}

def _transform_xyy_to_xyz(src_data):
    """Convert data from CIE-xyY color space to CIE-XYZ color space."""
    assert src_data.shape[0] == 3, "Input data must be 3xN matrix."
    dst_data = np.zeros(src_data.shape)
    # Y = Y.
    dst_data[1,:] = src_data[2,:]
    # X = Y / y * x.
    validTag = src_data[1,:] > 0
    dst_data[0,validTag] = src_data[2,validTag] / src_data[1,validTag] * \
                           src_data[0,validTag]
    # Z = Y / y * (1 - x - y).
    dst_data[2,validTag] = src_data[2,validTag] / src_data[1,validTag] * \
                           (1 - src_data[0,validTag] - src_data[1,validTag])
    return dst_data

def _transform_xyz_to_xyy(src_data):
    """Convert data from CIE-XYZ color space to CIE-xyY color space."""
    assert src_data.shape[0] == 3, "Input data must be 3xN matrix."
    dst_data = np.zeros(src_data.shape)
    # Y = Y.
    dst_data[2] = src_data[1,:]
    # x = X / (X + Y + Z).
    # y = Y / (X + Y + Z).
    s = np.sum(src_data, axis = 0)
    validTag = s > 0
    dst_data[:2, validTag] = src_data[:2,validTag] / s[validTag]
    return dst_data

def _transform_xyz_to_srgblin(src_data):
    return np.dot(xyz_to_srgb_matrix, src_data)

def _transform_srgblin_to_xyz(src_data):
    return np.dot(srgb_to_xyz_matrix, src_data)

def _transform_srgblin_to_srgb(src_data):
    return srgb_gamma(src_data)

def _transform_srgb_to_srgblin(src_data):
    return srgb_inverse_gamma(src_data)

def _transform_xyz_to_lab(src_data):
    """Convert data from CIE-XYZ color space to CIE-L*a*b* color space.

    See: https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
    Accessed on: Apr 24, 2015.
    """
    assert src_data.shape[0] == 3, "Input data must be 3xN matrix."
    Xn = d65_xyz[0] / d65_xyz[1]
    Yn = 1.
    Zn = d65_xyz[2] / d65_xyz[1]
    f = _lab_f

    dst_data = np.empty(src_data.shape)
    # L* = 116 f(Y / Yn) - 16.
    dst_data[0:] = 116 * f(src_data[1,:] / Yn) - 16
    # a* = 500 [f(X / Xn) - f(Y / Yn)].
    dst_data[1:] = 500 * (f(src_data[0,:] / Xn) - f(src_data[1,:] / Yn))
    # b* = 200 [f(Y / Yn) - f(Z / Zn)].
    dst_data[2:] = 200 * (f(src_data[1,:] / Yn) - f(src_data[2,:] / Zn))

    return dst_data

def _lab_f(t):
    f = np.empty(t.shape)
    part1 = (t > ((6. / 29.) ** 3))
    f[part1] = t[part1] ** (1. / 3.)
    part2 = ~part1
    f[part2] = 1. / 3. * (29. / 6.) ** 2 * t[part2] + 4. / 29.
    return f

def _transform_lab_to_xyz(src_data):
    """Convert data from CIE-L*a*b* color space to CIE-XYZ color space.

    See: https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
    Accessed on: Apr 24, 2015.
    """
    assert src_data.shape[0] == 3, "Input data must be 3xN matrix."
    Xn = d65_xyz[0] / d65_xyz[1]
    Yn = 1.
    Zn = d65_xyz[2] / d65_xyz[1]
    f_inv = _lab_f_inv

    dst_data = np.empty(src_data.shape)
    # X = Xn * f_inv((L* + 16) / 116 + a* / 500).
    dst_data[0:] = Xn * f_inv((src_data[0,:]+16)/116 + src_data[1,:]/500)
    # Y = Yn * f_inv((L* + 16) / 116)
    dst_data[1:] = Yn * f_inv((src_data[0,:]+16)/116)
    # Z = Zn * f_inv((L* + 16) / 116 - b* / 200)
    dst_data[2:] = Zn * f_inv((src_data[0,:]+16)/116 - src_data[2,:]/200)

    return dst_data

def _lab_f_inv(t):
    f_inv = np.empty(t.shape)
    part1 = (t > (6. / 29.))
    f_inv[part1] = t[part1] ** 3.
    part2 = ~part1
    f_inv[part2] = 3. * ((6. / 29.) ** 2) * (t[part2] - 4. / 29.)
    return f_inv
