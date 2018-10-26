#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0.0
@author: zheng guang
@contact: zg.zhu@daocloud.io
@time: 2018/10/26 下午10:10
"""
from image_util import image_to_base64, show_image
from face_detect import get_face_image, get_face_list, face_match
import pytest
import base64
import numpy as np
import cv2 as cv


def test_show_image():
    face_list = get_face_list(image_to_base64(u'../images/demo_image.jpg'))

    img = cv.imdecode(np.fromstring(base64.b64decode(image_to_base64(u'../images/demo_image.jpg')), np.uint8),
                      cv.IMREAD_ANYCOLOR)
    img = get_face_image(img, face_list)
    cv.imshow("who", img)
    cv.waitKey()
    cv.destroyAllWindows()


def test_show_image():
    result = face_match(image_to_base64(u'../images/demo_image.jpg'), image_to_base64(u'../images/demo_image.jpg'))
