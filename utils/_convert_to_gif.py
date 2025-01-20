# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

# This code references the source code of OpenMMLab projects, which are
# licensed under the Apache License, Version 2.0.

import os
import cv2
import imageio as iio


def main_from_file():
    # duration = 0.05
    # file_i = '/media/generalz/Storage/Active/LynxiWorks/20210817LynBIDL/visualizations/RGBGesture_show_1.avi'
    # file_o = '/media/generalz/Storage/Active/LynxiWorks/20210817LynBIDL/visualizations/rgbgesture.gif'
    duration = 0.08
    file_i = '/media/generalz/Storage/Active/LynxiWorks/20210817LynBIDL/visualizations/Lungnode_show.avi'
    file_o = '/media/generalz/Storage/Active/LynxiWorks/20210817LynBIDL/visualizations/lungnodule_cls.gif'
    vcap = cv2.VideoCapture(file_i)
    frames = []
    while True:
        flag, frame = vcap.read()
        if flag is not True:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        frames.append(frame)
    vcap.release()
    iio.mimsave(file_o, frames, duration=duration)


def main_from_fold():
    duration = 0.08
    fold_i = '/media/generalz/Storage/Active/LynxiWorks/20210817LynBIDL/visualizations/dvsmnist/'
    file_o = '/media/generalz/Storage/Active/LynxiWorks/20210817LynBIDL/visualizations/dvsmnist.gif'
    # duration = 0.8
    # fold_i = '/media/generalz/Storage/Active/LynxiWorks/20210817LynBIDL/visualizations/babiqa_jpg/'
    # file_o = '/media/generalz/Storage/Active/LynxiWorks/20210817LynBIDL/visualizations/babiqa.gif'
    frames = []
    for fn in os.listdir(fold_i):
        frame = cv2.imread(fold_i + fn)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        frames.append(frame)
    iio.mimsave(file_o, frames, duration=duration)


if __name__ == '__main__':
    # main_from_file()
    main_from_fold()
