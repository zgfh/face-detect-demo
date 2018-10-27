#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0.0
@author: zheng guang
@contact: zg.zhu@daocloud.io
@time: 2018/10/26 下午10:54
# 参考： https://www.cnblogs.com/warcraft/p/8145894.html
"""

# 打开摄像头
import cv2
import numpy
import matplotlib.pyplot as plot
from face_detect import get_face_image, get_face_list
import base64
import numpy as np
import time
from threading import Thread

# 摄像头对象
cap = cv2.VideoCapture(0)
# 显示
last_run_time = 0
last_face_result = None


def face_detect_job(img):
    global last_face_result
    print('check face start:', time.time())
    retval, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer)
    last_face_result = get_face_list(jpg_as_text)
    print('check face finish', time.time())


while (1):
    ret, img = cap.read()
    keycode = cv2.waitKey(1)

    # ' ':保存图片
    if keycode == ord(' '):
        imgName = "i_" + str(time.time())
        cv2.imwrite("./img/" + imgName + ".jpeg", img)
        print("save image {} ok".format("img/" + imgName + ".jpeg"))
    # 'q':退出
    if keycode == ord('q'):
        break

    if time.time() - last_run_time > 1:
        last_run_time = time.time()
        poll_thread = Thread(target=face_detect_job, args=(img,))
        poll_thread.daemon = False
        poll_thread.start()
        poll_thread.join()
        last_run_time = time.time()

    if last_face_result:
        img = get_face_image(img, last_face_result, face_rectangle=False)

    cv2.imshow("capture", img)

cap.release()
cv2.destroyAllWindows()

if __name__ == '__main__':
    pass
