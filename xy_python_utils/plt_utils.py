#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Jun 02, 2017.

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from matplotlib.cbook import iterable

def histb(x, bins=None, **kargs):
    """Same as `plt.hist`, but clips out-of-boundary data points to the terminal bins."""
    if iterable(bins):
        x = np.copy(x)
        x_min = (bins[0] + bins[1]) / 2
        x_max = (bins[-2] + bins[-1]) / 2
        x[x < x_min] = x_min
        x[x > x_max] = x_max
    plt.hist(x, bins, **kargs)


def minorticks_on(ax=None, which="both"):
    """Same as ax.minorticks_on, but takes a `which` parameter specifying `x`, `y` or `both`."""
    if ax == None:
        ax = plt.gca()
    if which == "both":
        axes = [ax.xaxis, ax.yaxis]
    elif which == "x":
        axes = [ax.xaxis]
    elif which == "y":
        axes = [ax.yaxis]
    for ax in axes:
        scale = ax.get_scale()
        if scale == 'log':
            s = ax._scale
            ax.set_minor_locator(mticker.LogLocator(s.base, s.subs))
        elif scale == 'symlog':
            s = ax._scale
            ax.set_minor_locator(
                mticker.SymmetricalLogLocator(s._transform, s.subs))
        else:
            ax.set_minor_locator(mticker.AutoMinorLocator())
