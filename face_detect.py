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
from config import FACE_DETECT_CLIENT_ID, FACE_DETECT_CLIENT_SECRET


# client_id 为官网获取的AK， client_secret 为官网获取的SK
def _access_token():
    access_token = requests.get(
        'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={client_id}&client_secret={client_secret}'.format(
            client_id=FACE_DETECT_CLIENT_ID, client_secret=FACE_DETECT_CLIENT_SECRET),
        headers={'Content-Type': 'application/json; charset=UTF-8'}).json()['access_token']
    return access_token


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
    return result


def get_face_image(base64_image, face_rectangle=True, landmark72=True, color=(0, 255, 0), point_color=(0, 0, 255)):
    if type(base64_image) == bytes:
        base64_image = base64_image.decode('utf-8')
    face_list = get_face_list(base64_image)
    if not (face_list['error_code'] == 0 and 'result' in face_list and face_list['result']['face_num'] >= 1):
        print(face_list)
        return base64_image
    # {'left': 141.997818, 'top': 124.7159348, 'width': 114, 'height': 110, 'rotation': -12}
    img = cv.imdecode(np.fromstring(base64.b64decode(base64_image), np.uint8), cv.IMREAD_ANYCOLOR)  # 读取图像
    if face_rectangle:
        # 增加人脸矩形框
        face_rectangle_pts = face_list['result']['face_list'][0]['location']
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
        pts = np.array([[point['x'], point['y']] for point in face_list['result']['face_list'][0]['landmark72']],
                       np.int32)

        # 下巴
        cv.polylines(img, [pts[0:12 + 1].reshape((-1, 1, 2))], False, color)
        # 左眼
        cv.polylines(img, [pts[13:21 + 1].reshape((-1, 1, 2))], True, color)
        # 左眉
        cv.polylines(img, [pts[22:29 + 1].reshape((-1, 1, 2))], True, color)

        # 右眼
        cv.polylines(img, [pts[30:38 + 1].reshape((-1, 1, 2))], True, color)
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

        retval, buffer = cv.imencode('.jpg', img)
        jpg_as_text = base64.b64encode(buffer)
        return jpg_as_text

        cv.imshow("who", img)
        cv.waitKey()
        cv.destroyAllWindows()
