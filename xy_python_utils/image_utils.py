#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Dec 11, 2014.

"""Some utility functions to handle images."""

import math
import numpy as np
import PIL.Image
from PIL.Image import ROTATE_180, ROTATE_90, ROTATE_270, FLIP_TOP_BOTTOM, FLIP_LEFT_RIGHT
import skimage.transform

def imcast(img, dtype, color_space="default"):
    """Cast the input image to a given data type.

    Parameters
    ----------
    img: ndarray
        The input image.

    dtype: np.dtype
        The type that output image to be cast into.

    color_space: string, optional
        The color space of the input image, which affects the casting operation.

    Returns
    -------
    The output image that is cast into `dtype`.

    Notes
    -----
    * For `color_space=="default"`, we perform a linear scaling with following
      range conventions:

      * `np.uint8`: `[0, 255]`;
      * `np.uint16`: `[0, 65535]`;
      * `np.float32` and `np.float64`: `[0.0, 1.0]`.

      For example, if the input `img` is of `np.uint8` type and the expected
      `dtype` is `np.float32`, then the output will be
      `np.asarray(img / 255., np.float32)`.

    * For `color_space=="CIE-L*a*b*"`, the "normal" value ranges are
      `0 <= L <= 100, -127 <= a, b <= 127`, and we perform the following cast:

      * `np.uint8`: `L <- L * 255 / 100,  a <- a + 128,  b <- b + 128`;
      * `np.uint16`: currently not supported;
      * `np.float32` and `np.float64`: left as is.

    """
    if img.dtype == dtype:
        return img
    if color_space == "default":
        if dtype == np.uint8:
            if img.dtype == np.uint16:
                return np.asarray(img / 257, np.uint8)
            elif img.dtype == np.float32 or img.dtype == np.float64:
                return np.asarray(img * 255., np.uint8)
        elif dtype == np.uint16:
            if img.dtype == np.uint8:
                return np.asarray(img, np.uint16) * 257
            elif img.dtype == np.float32 or img.dtype == np.float64:
                return np.asarray(img * 65535., np.uint16)
        elif dtype == np.float32 or dtype == np.float64:
            if img.dtype == np.uint8:
                return np.asarray(img, dtype) / 255.
            elif img.dtype == np.uint16:
                return np.asarray(img, dtype) / 65535.
            elif img.dtype == np.float32 or img.dtype == np.float64:
                return np.asarray(img, dtype)
    elif color_space == "CIE-L*a*b*":
        if dtype == np.uint8:
            if img.dtype == np.float32 or img.dtype == np.float64:
                dst = np.empty(img.shape, np.uint8)
                dst[:,:,0] = img[:,:,0] * 255. / 100.
                dst[:,:,1] = img[:,:,1] + 128.
                dst[:,:,2] = img[:,:,2] + 128.
                return dst
        elif dtype == np.float32 or dtype == np.float64:
            if img.dtype == np.uint8:
                dst = np.empty(img.shape, dtype)
                dst[:,:,0] = np.asarray(img[:,:,0], dtype) / 255. * 100.
                dst[:,:,1] = np.asarray(img[:,:,1], dtype) - 128.
                dst[:,:,2] = np.asarray(img[:,:,2], dtype) - 128.
                return dst
    raise Exception(
        "Unexpected conversion from '%s' to '%s' with '%s' color space" % \
        (img.dtype, dtype, color_space))

def imread(filename, dtype=np.uint8, color_space="default"):
    """Read the image followed by an :py:func:`imcast`."""
    img = PIL.Image.open(filename)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if hasattr(img, "_getexif"):
        try:
            exif = img._getexif() or {}
        except IOError:
            exif = {}
        orientation = exif.get(0x0112)
        if orientation:
            # see http://park2.wakwak.com/~tsuruzoh/Computer/Digicams/exif-e.html
            # for explanation of the magical constants
            # or see http://jpegclub.org/exif_orientation.html for a nice visual explanation
            # also, rotations are counter-clockwise in PIL
            orientation = int(orientation)
            rotation = [None, None, ROTATE_180, None, ROTATE_270, ROTATE_270, ROTATE_90, ROTATE_90]
            flip = [None, FLIP_LEFT_RIGHT, None, FLIP_TOP_BOTTOM, FLIP_LEFT_RIGHT, None,
                    FLIP_LEFT_RIGHT, None]
            orientation0 = orientation - 1 # it's 1-indexed per the EXIF spec
            if 0 <= orientation0 < len(rotation):
                if rotation[orientation0] is not None:
                    img = img.transpose(rotation[orientation0])
                if flip[orientation0] is not None:
                    img = img.transpose(flip[orientation0])
    return imcast(np.array(img), dtype, color_space)

def imwrite(filename, img, dtype=np.uint8, color_space="default"):
    """Perform an :py:func:`imcast` before writing to the output file."""
    import scipy.misc
    return scipy.misc.imsave(filename, imcast(img, dtype, color_space))

def imresize(img, size):
    """Resize the input image.

    Parameters
    ----------
    img: ndarray
        The input image to be resized.

    size: a scalar for `scale` or a 2-tuple for `(num_rows, num_cols)`
        One of the `num_rows` or `num_cols` can be -1, which will be inferred
        such that the output image has the same aspect ratio as the input.

    Returns
    -------
    The resized image.

    """
    if hasattr(size, "__len__"):
        num_rows, num_cols = size
        assert (num_rows > 0) or (num_cols > 0)
        if num_rows < 0:
            num_rows = num_cols * img.shape[0] / img.shape[1]
        if num_cols < 0:
            num_cols = num_rows * img.shape[1] / img.shape[0]
    else:
        num_rows = int(round(img.shape[0] * size))
        num_cols = int(round(img.shape[1] * size))
    return skimage.transform.resize(img, (num_rows, num_cols))

def create_icon_mosaic(icons, icon_shape=None,
                       border_size=1, border_color=None, empty_color=None,
                       mosaic_shape=None, mosaic_dtype=np.float):
    """Create a mosaic of image icons.

    Parameters
    ----------
    icons: a list of `ndarray`s
        A list of icons to be put together for mosaic. Currently we require all
        icons to be multi-channel images of the same size.

    icon_shape: 3-tuple, optional
        The shape of icons in the output mosaic as `(num_rows, num_cols, num_channels)`.
        If  not specified, use the shape of first image in `icons`.

    border_size: int, optional
        The size of border.

    border_color: 3-tuple, optional
        The color of border, black if not specified.

    empty_color: 3-tuple, optional
        The color for empty cells, black if not specified.

    mosaic_shape: 2-tuple, optional
        The shape of output mosaic as `(num_icons_per_row,
        num_icons_per_col)`. If not specified, try to make a square mosaic
        according to number of icons.

    mosaic_dtype: dtype
        The data type of output mosaic.

    Returns
    -------
    The created mosaic image.

    """
    # Set default parameters.
    num_icons = len(icons)
    assert num_icons > 0
    if icon_shape is None:
        icon_shape = icons[0].shape
    assert len(icon_shape) == 3
    num_channels = icon_shape[2]
    if border_color is None:
        border_color = np.zeros(num_channels)
    if empty_color is None:
        empty_color = np.zeros(num_channels)
    if mosaic_shape is None:
        num_cols = int(math.ceil(math.sqrt(num_icons)))
        num_rows = int(math.ceil(float(num_icons) / num_cols))
        mosaic_shape = (num_rows, num_cols)
    mosaic_image_shape = (
        mosaic_shape[0] * icon_shape[0] + (mosaic_shape[0]-1) * border_size,
        mosaic_shape[1] * icon_shape[1] + (mosaic_shape[1]-1) * border_size,
        icon_shape[2])
    # Create mosaic image and fill with border color.
    mosaic_image = np.empty(mosaic_image_shape, dtype=mosaic_dtype)
    for c in xrange(mosaic_image.shape[2]):
        mosaic_image[:,:,c] = border_color[c]
    # Fill in the input icons.
    for idx in xrange(num_icons):
        i = idx / mosaic_shape[1]
        j = idx % mosaic_shape[1]
        iStart = i * (icon_shape[0] + border_size)
        jStart = j * (icon_shape[1] + border_size)
        mosaic_image[iStart:iStart+icon_shape[0],
                     jStart:jStart+icon_shape[1],:] = icons[idx]
    # Fill the empty icons with empty colors.
    for idx in xrange(num_icons, mosaic_shape[0]*mosaic_shape[1]):
        i = idx / mosaic_shape[1]
        j = idx % mosaic_shape[1]
        iStart = i * (icon_shape[0] + border_size)
        jStart = j * (icon_shape[1] + border_size)
        for c in xrange(mosaic_image.shape[2]):
            mosaic_image[iStart:iStart+icon_shape[0],
                         jStart:jStart+icon_shape[1],c] = empty_color[c]
    return mosaic_image

def image_size_from_file(filename):
    """Read the image size from a file.

    This function only loads but the image header (rather than the whole
    rasterized data) in order to determine its dimension.

    Parameters
    ----------
    filename: string
        The input image file.

    Returns
    -------
    The 2-tuple for image size `(num_rows, num_cols)`.

    """
    with PIL.Image.open(filename) as img:
        width, height = img.size
    return height, width
