#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0.0
@author: zheng guang
@contact: zg.zhu@daocloud.io
@time: 2018/10/27 上午2:52
"""
import os

# 修改该值,申请：http://ai.baidu.com/docs#/Begin/top
FACE_DETECT_CLIENT_ID = os.getenv('FACE_DETECT_CLIENT_ID', 'changeme')
FACE_DETECT_CLIENT_SECRET = os.getenv('FACE_DETECT_CLIENT_SECRET', 'changeme')

# 目标人脸列表，用于匹配名称，规则：目录下，存放名称.图片格式，如:alan.jpg,caiyilin.jpeg
FACE_MATCH_DIR = os.getenv('FACE_DIR', './images')
