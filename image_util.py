#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0.0
@author: zheng guang
@contact: zg.zhu@daocloud.io
@time: 2018/10/26 下午10:00
#  docs:
# OpenCV入门: https://zhuanlan.zhihu.com/p/24425116
"""
from PIL import Image
import cv2
from io import BytesIO
import base64
import pathlib


def to_str(str_or_bytes):
    if type(str_or_bytes) == bytes:
        return str_or_bytes.decode('utf-8')
    return str_or_bytes


def image_to_base64(file):
    with open(file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return to_str(encoded_string)


def show_image(file_or_base64):
    file_or_base64 = to_str(file_or_base64)
    try:
        img = Image.open(BytesIO(base64.b64decode(file_or_base64)))
    except:
        img = Image.open(file_or_base64)
    img.show()
