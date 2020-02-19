#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Mar 05, 2015.

"""Some general utility classes and functions."""

import math

class Range():
    """A range of numbers from `start` (inclusive) to `end` (exclusive) with a
    given `step`. This class is similar to the `range` built-in in python3, but
    also supports floating point parameters.

    Note the rounding effect when using floating point parameters. The suggested
    way is to pad an `epsilon` at the stop point::

        Range(1.5, 1.8001, 0.3)   # 1.8 will be included.
        Range(1.5, 1.7999, 0.3)   # 1.5 will be excluded.
        Range(1.5, 1.8, 0.3)      # 1.8 should be excluded, but might not be
                                  # because of rounding effect. Avoid this.

    """
    def __init__(self, start, stop=None, step=None):
        """Range(stop): the range of integers [0, 1, 2, ..., stop-1].

        Range(start, stop[, step]): the range of numbers
            [start, start+step, start+2*step, ..., start+len*step]
        such that
                start + step * (len-1) < stop      if step > 0
                start + step * (len-1) > stop      if step < 0
        `step` defaults to 1.
        """
        if not stop:
            self.stop = start
            self.start = 0
            self.step = 1
        else:
            if not step:
                step = 1
            self.start = start
            self.stop = stop
            self.step = step

    def __len__(self):
        """The number of elements in range."""
        return int(math.ceil(float(self.stop-self.start) / self.step))

    def __iter__(self):
        """Create an iterator."""
        return RangeIterator(self, 0)

    def __repr__(self):
        return "Range(start=%s, stop=%s, step=%s)" % \
            (self.start, self.stop, self.step)

class RangeIterator():
    def __init__(self, range_, index):
        self.range_ = range_
        self.max_size = len(range_)
        self.index = index

    def __iter__(self):
        return self

    def next(self):
        if (self.index >= self.max_size):
            raise StopIteration()
        else:
            index = self.index
            self.index += 1
            return self.range_.start + self.range_.step * index
