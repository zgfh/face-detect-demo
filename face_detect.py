#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0.0
@author: zheng guang
@contact: zg.zhu@daocloud.io
@time: 2018/10/26 下午9:26
"""

import requests
import cv2 as cv
import numpy as np
import base64
from image_util import image_to_base64
from config import FACE_DETECT_CLIENT_ID, FACE_DETECT_CLIENT_SECRET, FACE_MATCH_DIR
import os

last_face_check_result = None


# client_id 为官网获取的AK， client_secret 为官网获取的SK
def _access_token():
    result = requests.get(
        'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={client_id}&client_secret={client_secret}'.format(
            client_id=FACE_DETECT_CLIENT_ID, client_secret=FACE_DETECT_CLIENT_SECRET),
        headers={'Content-Type': 'application/json; charset=UTF-8'}).json()
    if 'access_token' not in result:
        print(result)
        raise Exception('get access_token error')
    return result['access_token']


def _absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


'''
人脸的类型:
LIVE表示生活照：通常为手机、相机拍摄的人像图片、或从网络获取的人像图片等
IDCARD表示身份证芯片照：二代身份证内置芯片中的人像照片
WATERMARK表示带水印证件照：一般为带水印的小图，如公安网小图
CERT表示证件照片：如拍摄的身份证、工卡、护照、学生证等证件图片
默认LIVE
'''


def get_face_list(base64_image):
    if type(base64_image) == bytes:
        base64_image = base64_image.decode('utf-8')
    result = requests.post(
        'https://aip.baidubce.com/rest/2.0/face/v3/detect?access_token={}'.format(_access_token()),
        json={'image_type': 'BASE64',  # 图片类型: BASE64/URL/FACE_TOKEN
              'image': base64_image,
              # 图片信息(总数据大小应小于10M)，图片上传方式根据image_type来判断
              'face_field': 'age,beauty,expression,face_shape,gender,glasses,landmark,race,quality,face_type',
              'max_face_num': 10,  # 最多处理人脸的数目，默认值为1，仅检测图片中面积最大的那个人脸；最大值10，检测图片中面积最大的几张人脸。
              'face_type': 'LIVE'}).json()  # 人脸的类型LIVE/IDCARD/WATERMARK/CERT
    if not (result['error_code'] == 0 and 'result' in result and result['result']['face_num'] >= 1):
        print(result)
    else:
        for face_data in result['result']['face_list']:
            face_data['face_match'] = '__'
            for image_file in _absoluteFilePaths(FACE_MATCH_DIR):
                face_match_result = face_match(face_data['face_token'], image_to_base64(image_file))
                if face_match_result and 'score' in face_match_result['result'] and face_match_result['result'][
                    'score'] > 75:
                    face_data['face_match'] = os.path.splitext(os.path.basename(image_file))[0]
    return result


def face_match(face_token, target_base64_image):
    if type(target_base64_image) == bytes:
        target_base64_image = target_base64_image.decode('utf-8')
    result = requests.post(
        'https://aip.baidubce.com/rest/2.0/face/v3/match?access_token={}'.format(_access_token()),
        json=[{'image_type': 'FACE_TOKEN',  # 图片类型: BASE64/URL/FACE_TOKEN
               'image': face_token,
               # 图片信息(总数据大小应小于10M)，图片上传方式根据image_type来判断
               'face_type': 'LIVE',
               'quality_control': 'NONE',
               'liveness_control': 'NONE'},
              {'image_type': 'BASE64',  # 图片类型: BASE64/URL/FACE_TOKEN
               'image': target_base64_image,
               # 图片信息(总数据大小应小于10M)，图片上传方式根据image_type来判断
               'face_type': 'LIVE',
               'quality_control': 'NONE',
               'liveness_control': 'NONE'},
              ]).json()  # 人脸的类型LIVE/IDCARD/WATERMARK/CERT
    if not (result['error_code'] == 0):
        print(result)

    print(result)
    return result


def get_face_image(img, face_list, face_rectangle=True, landmark72=True, color=(0, 255, 0), point_color=(0, 0, 255)):
    if not (face_list['error_code'] == 0 and 'result' in face_list and face_list['result']['face_num'] >= 1):
        return img
    for face_data in face_list['result']['face_list']:
        if face_rectangle:
            # 增加人脸矩形框
            face_rectangle_pts = face_data['location']
            face_rectangle_pts = [(int(face_rectangle_pts['left']), int(face_rectangle_pts['top'])),
                                  (int(face_rectangle_pts['left']),
                                   int(face_rectangle_pts['top'] + face_rectangle_pts['height'])),
                                  (int(face_rectangle_pts['left'] + face_rectangle_pts['width']),
                                   int(face_rectangle_pts['top'] + face_rectangle_pts['height'])),
                                  (int(face_rectangle_pts['left'] + face_rectangle_pts['width']),
                                   int(face_rectangle_pts['top']))]

            cv.polylines(img, [np.array(face_rectangle_pts).reshape((-1, 1, 2))], True, color)
            # 加粗4个点
            for pt in face_rectangle_pts:
                cv.circle(img, (pt[0], pt[1]), 1, point_color, -1)

        pts = np.array([[point['x'], point['y']] for point in face_data['landmark72']],
                       np.int32)
        if landmark72:
            '''
            # 显示 landmark72
            "landmark72": [ 
                            {
                                "x": 115.86531066895,
                                "y": 170.0546875
                            }，
                            ...
                        ]
            '''

            # 下巴
            cv.polylines(img, [pts[0:12 + 1].reshape((-1, 1, 2))], False, color)
            # 左眼
            cv.polylines(img, [pts[13:20 + 1].reshape((-1, 1, 2))], True, color)
            # 左眉
            cv.polylines(img, [pts[22:29 + 1].reshape((-1, 1, 2))], True, color)

            # 右眼
            cv.polylines(img, [pts[30:37 + 1].reshape((-1, 1, 2))], True, color)
            # 右眉
            cv.polylines(img, [pts[39:46 + 1].reshape((-1, 1, 2))], True, color)

            # 鼻子
            cv.polylines(img, [pts[47:56 + 1].reshape((-1, 1, 2))], True, color)
            cv.polylines(img, [(pts[51:52 + 1] + pts[57:57 + 1]).reshape((-1, 1, 2))], True, color)

            # 嘴
            cv.polylines(img, [pts[58:65 + 1].reshape((-1, 1, 2))], True, color)
            cv.polylines(img, [pts[66:71 + 1].reshape((-1, 1, 2))], True, color)

            # 加粗72个点
            for pt in pts:
                cv.circle(img, (pt[0], pt[1]), 1, point_color, -1)

            cv.putText(img, u'name:{}'.format(face_data['face_match']), (pts[1][0], pts[1][1] + 0),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 1, cv.LINE_AA)
            cv.putText(img, u'sex:{}'.format(face_data['gender']['type']), (pts[1][0], pts[1][1] + 25),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 1, cv.LINE_AA)
            cv.putText(img, 'age:{}'.format(face_data['age']), (pts[1][0], pts[1][1] + 50),
                       cv.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (255, 255, 255), 1, cv.LINE_AA)

    return img
