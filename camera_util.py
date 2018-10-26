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
from face_detect import get_face_image
import base64
import numpy as np

# 摄像头对象
cap = cv2.VideoCapture(0)
# 显示
i = 0
while (1):
    ret, frame = cap.read()
    retval, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer)
    face_image = get_face_image(jpg_as_text,face_rectangle=False)
    img = cv2.imdecode(np.fromstring(base64.b64decode(face_image), np.uint8), cv2.IMREAD_ANYCOLOR)
    cv2.imshow("capture", img)
    # ' ':保存图片

    if cv2.waitKey(1) & 0xFF == ord(' '):
        i = i + 1
        imgName = "i_" + str(i)
        cv2.imwrite("./img/" + imgName + ".jpeg", frame)
        print("save image {} ok".format("img/" + imgName + ".jpeg"))
    # 'q':退出
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()

if __name__ == '__main__':
    pass
