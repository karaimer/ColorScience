#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Mar 05, 2015.

import unittest

from utils import *

class RangeTest(unittest.TestCase):
    def test_len(self):
        self.assertEqual(len(Range(100)), 100)
        self.assertEqual(len(Range(3, 100)), 97)
        self.assertEqual(len(Range(3, 8, 1)), 5)
        self.assertEqual(len(Range(8, 0, -1)), 8)
        self.assertEqual(len(Range(1, 10, 2)), 5)
        self.assertEqual(len(Range(10, 1, -2)), 5)
        self.assertEqual(len(Range(0.0, 1.49, 0.1)), 15)
        self.assertEqual(len(Range(0.0, 1.51, 0.1)), 16)

    def test_iteration(self):
        for i, x in enumerate(Range(20)):
            self.assertEqual(x, i)
        for i, x in enumerate(Range(10, 20)):
            self.assertEqual(x, 10 + i)
        for i, x in enumerate(Range(1.0, -5.51, -0.1)):
            self.assertEqual(x, 1.0 - i*0.1)
        r = Range(-10, 0.001, 0.1)
        self.assertEqual(len(r), len(list(r)))

if __name__ == "__main__":
    unittest.main()
