#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Jan 07, 2015.

import os
import filecmp
import numpy as np
import scipy.misc
import unittest

from image_utils import *
from os_utils import rm_rf
from unittest_utils import check_near

test_images_dir = os.path.dirname(os.path.realpath(__file__)) + \
                  "/test_data/images"

testorig_jpg = test_images_dir + "/testorig.jpg"
testorig_png = test_images_dir + "/testorig.png"

class ImageUtilsTest(unittest.TestCase):
    def test_imcast(self):
        tol = 1e-6
        uint8 = scipy.misc.imread(testorig_jpg)
        float32 = imcast(uint8, np.float32)
        check_near(np.asarray(uint8, np.float32) / 255., float32, tol)
        self.assertEqual(np.sum(uint8 != imcast(float32, np.uint8)), 0)

        lab = imcast(uint8, np.float32, "CIE-L*a*b*")
        check_near(np.asarray(uint8[:,:,0], np.float32),
                   lab[:,:,0] * 255. / 100.,
                   tol)
        check_near(np.asarray(uint8[:,:,1], np.float32),
                   lab[:,:,1] + 128.,
                   tol)
        check_near(np.asarray(uint8[:,:,2], np.float32),
                   lab[:,:,2] + 128.,
                   tol)
        self.assertEqual(
            np.sum(uint8 != imcast(lab, np.uint8, "CIE-L*a*b*")), 0)

    def test_imread(self):
        tol = 1e-8
        img = scipy.misc.imread(testorig_jpg)
        float32 = imread(testorig_jpg, np.float32)
        check_near(imcast(img, np.float32), float32, tol)
        uint8 = imread(testorig_jpg, np.uint8)
        self.assertEqual(np.sum(uint8 != img), 0)

    def test_imwrite(self):
        img = imread(testorig_png, np.float32)
        imwrite("test_imwrite.png", img, np.uint8)
        self.assertTrue(filecmp.cmp(testorig_png, "test_imwrite.png"))
        rm_rf("test_imwrite.png")

    def test_imresize(self):
        img = scipy.misc.imread(testorig_jpg)
        img2 = imresize(img, 0.5)
        self.assertEqual(img2.shape[0], round(img.shape[0]*0.5))
        self.assertEqual(img2.shape[1], round(img.shape[1]*0.5))
        self.assertEqual(img2.shape[2], img.shape[2])

    def test_create_icon_mosaic(self):
        tol = 1e-6
        icon = scipy.misc.imread(testorig_png)
        mosaic = create_icon_mosaic([icon] * 6)
        mosaic_ref = scipy.misc.imread(test_images_dir + "/testorig-mosaic.png")
        check_near(mosaic, mosaic_ref, tol)

    def test_image_size_from_file(self):
        imgSize = image_size_from_file(testorig_jpg)
        self.assertEqual(imgSize[0], 149)
        self.assertEqual(imgSize[1], 227)

if __name__ == "__main__":
    unittest.main()
