#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Oct 28, 2014.

import os
import numpy as np

_this_file_path = os.path.dirname(__file__)
_data_path = _this_file_path + "/data"

# Chromaticity coordinates (normalized xyz) for primaries of sRGB color space.
# Accessed from: http://www.color.org/chardata/rgb/srgb.pdf.
# Accessed on: Nov 30, 2014.
srgb_red_xyz = np.array([0.64, 0.33, 0.03])
srgb_green_xyz = np.array([0.30, 0.60, 0.10])
srgb_blue_xyz = np.array([0.15, 0.06, 0.79])
srgb_white_xyz = np.array([0.3127, 0.3290, 0.3583])

# The color transform matrix from CIE XYZ space to linear sRGB space.
# Accessed from: http://www.color.org/chardata/rgb/srgb.pdf.
# Accessed on: Oct 28, 2014.
xyz_to_srgb_matrix = np.array([
    [ 3.2406, -1.5372, -0.4986],
    [-0.9689,  1.8758,  0.0415],
    [ 0.0557, -0.2040,  1.0570]
])
srgb_to_xyz_matrix = np.linalg.inv(xyz_to_srgb_matrix)

def srgb_gamma(linear_data):
    """The per-channel, nonlinear transfer function used in sRGB.

    The conversion formula is::

      C = 12.92 * C_linear,                     if C_linear <= 0.0031308
          1.055 * C_linear ** (1/2.4) - 0.055,  otherwise

    | Accessed from: http://www.color.org/chardata/rgb/srgb.pdf.
    | Accessed on: Oct 28, 2014."""
    nonlinear_data = np.empty(linear_data.shape)
    part1 = (linear_data<=0.0031308)
    nonlinear_data[part1] = linear_data[part1] * 12.92
    part2 = ~part1
    nonlinear_data[part2] = 1.055 * linear_data[part2] ** (1/2.4) - 0.055
    return nonlinear_data

def srgb_inverse_gamma(nonlinear_data):
    """The per-channel transform from nonlinear sRGB data to linear sRGB data.

    The conversion formula is::

      C_linear = C / 12.92,                     if C <= 0.04045
                 ((C + 0.055) / 1.055) ^ 2.4,   otherwise

    | Accessed from: http://www.color.org/chardata/rgb/srgb.pdf.
    | Accessed on: Apr 24, 2015.
    """
    linear_data = np.empty(nonlinear_data.shape)
    part1 = (nonlinear_data<=0.04045)
    linear_data[part1] = nonlinear_data[part1] / 12.92
    part2 = ~part1
    linear_data[part2] = ((nonlinear_data[part2] + 0.055) / 1.055) ** 2.4
    return linear_data

# Chromaticity coordinates (xy) for primaries of Adobe RGB (1998) color space.
# Accessed from: http://www.adobe.com/digitalimag/pdfs/AdobeRGB1998.pdf.
# Accessed on: Nov 30, 2014.
adobe_red_xy = np.array([0.6400, 0.3300])
adobe_green_xy = np.array([0.2100, 0.7100])
adobe_blue_xy = np.array([0.1500, 0.0600])
adobe_white_xy = np.array([0.3127, 0.3290])

# The color transform matrix from (normalized) CIE-XYZ space to linear Adobe RGB
# (1998) space.
# Accessed from: http://www.adobe.com/digitalimag/pdfs/AdobeRGB1998.pdf.
# Accessed on: Nov 30, 2014.
xyz_to_adobe_matrix = np.array([
    [  2.04159, -0.56501, -0.34473],
    [ -0.96924,  1.87597,  0.04156],
    [  0.01344, -0.11836,  1.01517]
])
adobe_to_xyz_matrix = np.array([
    [ 0.57667, 0.18556, 0.18823],
    [ 0.29734, 0.62736, 0.07529],
    [ 0.02703, 0.07069, 0.99134]
])

# The color transform matrix from ICC-PCS (ICC Profile Connection Space) to
# linear Adobe RGB (1998) space.
# Accessed from: http://www.adobe.com/digitalimag/pdfs/AdobeRGB1998.pdf.
# Accessed on: Nov 30, 2014.
icc_pcs_to_adobe_matrix = np.array([
    [  1.96253, -0.61068, -0.34137],
    [ -0.97876,  1.91615,  0.03342],
    [  0.02869, -0.14067,  1.34926]
])
adobe_to_icc_pcs_matrix = np.array([
    [ 0.60974, 0.20528, 0.14919],
    [ 0.31111, 0.62567, 0.06322],
    [ 0.01947, 0.06087, 0.74457]
])

def adobe_abs_to_norm_xyz(abs_xyz):
    """Converting the absolute XYZ to Adobe normalized XYZ.

    The conversion formula is::

        Xn = (Xa - Xk) / (Xw - Xk) * Xw / Yw
        Yn = (Ya - Yk) / (Yw - Yk)
        Zn = (Za - Zk) / (Zw - Zk) * Zw / Yw

    where `(Xa, Ya, Za)` is absolute XYZ value, `(Xk, Yk, Zk) = (0.5282, 0.5557,
    0.6052)` is the reference display black point, `(Xw, Yw, Zw) = (152.07,
    160.00, 174.25)` is the reference display white point, and `(Xn, Yn, Zn)` is
    the normalized XYZ value.

    Parameters
    ----------
    abs_xyz: ndarray
        Absolute XYZ values, of the following format:
          * 1D ndarray of length 3.
          * 2D ndarray of size (3,N).

    Returns
    -------
    An `ndarray` of the same format as input.

    References
    ----------
    | Accessed from: http://www.adobe.com/digitalimag/pdfs/AdobeRGB1998.pdf.
    | Accessed on: Nov 30, 2014.

    """
    # Set the parameters.
    xk,yk,zk = 0.5282, 0.5557, 0.6052
    xw,yw,zw = 152.07, 160.00, 174.25
    # Load input.
    if len(abs_xyz.shape) == 1:
        xa,ya,za = abs_xyz[0], abs_xyz[1], abs_xyz[2]
    elif len(abs_xyz.shape) == 2:
        assert (abs_xyz.shape[0] == 3)
        xa,ya,za = abs_xyz[0,:], abs_xyz[1,:], abs_xyz[2,:]
    else:
        raise Exception("Unsupported `abs_xyz` format.")
    # Do the conversion.
    xn = (xa - xk) / (xw - xk) * xw / yw
    yn = (ya - yk) / (yw - yk)
    zn = (za - zk) / (zw - zk) * zw / yw
    # Organize output as the same format as input.
    if len(abs_xyz.shape) == 1:
        return np.array([xn,yn,zn])
    elif len(abs_xyz.shape) == 2:
        return np.concatenate([xn,yn,zn])

def adobe_gamma(linear_data):
    """The per-channel, nonlinear transfer function used in Adobe RGB (1998)
    color space.

    .. math::
        C= ( C_{linear} ) ^ {1 / 2.19921875}

    The value `2.19921875` is obtained from `(2 + 51/256)`.

    Accessed from: http://www.adobe.com/digitalimag/pdfs/AdobeRGB1998.pdf.
    Accessed on: Nov 30, 2014.
    """
    return np.power(linear_data, 1. / 2.19921875)

# CIE-D65's XYZ tristimulus values, normalized by relative luminance.
# Accessed from: http://en.wikipedia.org/wiki/Illuminant_D65
# Accessed on: Nov 30, 2014.
d65_xyz = np.array([95.047, 100.00, 108.883])

def read_cvrl_csv(csv_filename, empty_val = 0.0):
    """Read a csv file downloaded from cvrl.org.

    Some of the entries in the csv are empty, and will be filled with
    'empty_val'. If reading linear data, 'empty_val' should be set as 0.0, and
    if reading log data, it should be set as -np.inf.

    Returns
    -------
    A `ndarray` of size `Nx2` or `Nx4`.
        The first column is wavelength in unit of nm, and following columns the
        corresponding functions with respect to wavelength.

    """
    with open(csv_filename, 'r') as f:
        lines = f.readlines()
        cmfs = [[float(v or empty_val) for v in l.strip().split(',')]
                for l in lines]
        return np.array(cmfs)

def adjust_wl(fw_in, wl_in, wl_out):
    """Adjust the wavelength list of the input function(s)."""
    # For multiple functions, do the adjustment on each one of them and
    # concatenate results together.
    if len(fw_in.shape) > 1:
        fw_out_list = []
        for k in xrange(fw_in.shape[0]):
            fw_out_list.append(adjust_wl(fw_in[k], wl_in, wl_out))
        return np.vstack(fw_out_list)
    # Do the adjustment for a single wavelength function.
    assert (wl_in[1] - wl_in[0]) == (wl_out[1] - wl_out[0])
    dw = wl_in[1] - wl_in[0]
    left_index = int(np.round((wl_out[0] - wl_in[0]) / dw))
    right_index = int(np.round((wl_out[-1] - wl_in[-1]) / dw))
    # The result contains three parts (two pads and a center).
    left_pad = np.zeros(0)
    center = fw_in
    right_pad = np.zeros(0)
    # Compute left pad.
    if left_index < 0:
        left_pad = np.zeros(-left_index)
    else:
        center = center[left_index:]
    # Compute right pad.
    if right_index >= 0:
        right_pad = np.zeros(right_index)
    else:
        center = center[:right_pad]
    return np.concatenate((left_pad, center, right_pad))

def load_fw(name, wl=None):
    """Load function of wavelength.

    Parameters
    ----------
    name: str
        A string for the name of function, currently support:
          * "xyz-cmfs": CIE-XYZ color matching functions.
          * "d65-spd": CIE-D65 spectral power distribution.
    wl: ndarray
        Wavelength list, optional.

    Returns
    -------
    fw : ndarray
        The loaded function(s) of wavelength.
    wl : ndarray
        It will be the same as input `wl` if it is not `None`, or will be loaded
        together with `fw` as input data.
    """
    # Load the 'fw' and corresponding 'load_wl'.
    if name == "xyz-cmfs":
        csv_data = read_cvrl_csv(_data_path + "/cvrl/ciexyz31_1.csv")
        load_wl = csv_data[:,0]
        fw = csv_data[:, 1:].T
    elif name == "d65-spd":
        csv_data = read_cvrl_csv(_data_path + "/cvrl/Illuminantd65.csv")
        load_wl = csv_data[:,0]
        fw = csv_data[:,1]
    else:
        raise Exception("Unknown name '%s' for `load_fw`." % name)
    # Adjust the 'wl' if provided as input.
    if wl is not None:
        return (adjust_wl(fw, load_wl, wl), wl)
    else:
        return (fw, load_wl)

def get_blackbody_spd(temperature, wl):
    """Get blackbody radiation spectral power distribution."""
    # Setup constants.
    h = 6.6260695729e-34    # Planck constant.
    c = 299792458           # Speed of light.
    k = 1.380648813e-23     # Boltzmann constant.
    # Compute SPD by Planck's law.
    wl = wl * 1e-9
    spd = 2*h*(c**2) / np.power(wl,5) / (np.exp(h*c/wl/k/temperature) - 1)
    # Normalize the spd such that it sums to 1.
    return spd / np.sum(spd)
