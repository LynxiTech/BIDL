# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import numpy as np
import glob
import pandas as pd
import cv2
import time
import os
from cal import showPrecision, showSuccess
import sys
sys.path.append("../../../")
from lynadapter.lyn_sdk_model import ApuRun_Single

root = os.getcwd()
for _ in range(3):
    root = os.path.dirname(root)

NumOfNeRow = 30
NumOfNeCol = 56
# NumOfNeRow = 120
# NumOfNeCol = 160
beta = 1
const = 200
k = beta/const

current_path = os.path.dirname(os.path.abspath(__file__))


def show_dataset(dataset_path, label_path, pred_path):
    image_list_now = glob.glob(dataset_path + "/*.jpg")
    image_list_now.sort()
    label_now_df = pd.read_csv(label_path, header=None, delimiter=",")
    label_now = np.asarray(label_now_df)
    pred_now_df = pd.read_csv(pred_path, header=None, delimiter=",")
    pred_now = np.asarray(pred_now_df)

    for counter, image_now in enumerate(image_list_now):
        image_origin = cv2.imread(image_now)
        h, w, c = image_origin.shape

        x0 = label_now[counter, 0]
        y0 = label_now[counter, 1]
        x1 = label_now[counter, 0] + label_now[counter, 2]
        y1 = label_now[counter, 1] + label_now[counter, 3]

        x_p = pred_now[counter, 0]
        y_p = pred_now[counter, 1]
        w1 = pred_now[counter, 2]
        h1 = pred_now[counter, 3]

        xp1 = abs(2 * x_p - w1) / 2 * w / NumOfNeCol
        xp1 = round(xp1)
        yp1 = abs(2 * y_p - h1) / 2 * h / NumOfNeRow
        yp1 = round(yp1)
        xp2 = (2 * x_p + w1) / 2 * w / NumOfNeCol
        xp2 = round(xp2)
        yp2 = (2 * y_p + h1) / 2 * h / NumOfNeRow
        yp2 = round(yp2)

        cv2.rectangle(image_origin, (x0, y0), (x1, y1), (0, 0, 255), 3)
        cv2.rectangle(image_origin, (xp1, yp1), (xp2, yp2), (0, 255, 255), 3)
        i = counter + 1
        name = '%04d' % i
        dir_name = 'output'
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        filename = current_path + '/' + dir_name + '/' + name + '.jpg'
        cv2.imwrite(filename, image_origin)
        # cv2.imshow("origin", image_origin)
        # if (cv2.waitKey(30) & 0xFF) == ord('q'):
        #     break


def load_dataset(dataset_path, label_path):
    image_list_now = glob.glob(dataset_path + "/*.jpg")
    image_list_now.sort()
    label_now_df = pd.read_csv(label_path, header=None, delimiter=",")
    label_now = np.asarray(label_now_df)
    image_all_resize = np.zeros((len(image_list_now), NumOfNeRow, NumOfNeCol))  # , dtype='uint8'
    label_resize = np.zeros((len(image_list_now), 4))

    for counter, image_now in enumerate(image_list_now):
        image_origin = cv2.imread(image_now)
        if counter == 0:
            h, w, c = image_origin.shape
            image_all = np.zeros((len(image_list_now), h, w, c), dtype='uint8')
        # print(counter)

        image_all[counter, :] = image_origin
        img_gray = cv2.cvtColor(image_origin, cv2.COLOR_RGB2GRAY)
        image_all_resize[counter, :] = cv2.resize(img_gray, (NumOfNeCol, NumOfNeRow), interpolation=cv2.INTER_CUBIC)
        # image_buf = cv2.resize(img_gray, (100, 100), interpolation=cv2.INTER_CUBIC)

        x0 = label_now[counter, 0]
        y0 = label_now[counter, 1]
        x1 = label_now[counter, 0] + label_now[counter, 2]
        y1 = label_now[counter, 1] + label_now[counter, 3]

        x0_r = int((x0 * NumOfNeCol) / w)
        y0_r = int((y0 * NumOfNeRow) / h)
        x1_r = int((x1 * NumOfNeCol) / w)
        y1_r = int((y1 * NumOfNeRow) / h)

        label_resize[counter, 0] = (x0_r + x1_r) // 2  # x
        label_resize[counter, 1] = (y0_r + y1_r) // 2  # y
        label_resize[counter, 2] = abs(x1_r - x0_r)  # w
        label_resize[counter, 3] = abs(y1_r - y0_r)  # h

    print("load_finsh")
    return label_resize, image_all_resize, len(image_list_now)


def main():
    dataset_path = root + "/data/data_cann/Tiger1/img"
    label_path = root + "/data/data_cann/Tiger1/groundtruth_rect.txt"
    model_path = root + "/model_files/videotracking/cann"

    net_Iext_np = np.zeros((NumOfNeRow, NumOfNeCol)).astype(np.float32)

    for counter_row in range(-7, 1, 7):
        for counter_col in range(-7, 1, 7):
            net_Iext_np[counter_row, counter_col] = 30

    net_r_np = np.zeros((NumOfNeRow, NumOfNeCol)).astype(np.float32)

    net_in = np.concatenate((net_Iext_np.reshape(-1), net_r_np.reshape(-1))).astype(np.float32)

    label_resize, image_all_resize, image_length = load_dataset(dataset_path, label_path)

    arun = ApuRun_Single(0, model_path+'/Net_0/')
    net_U, net_recSum, net_r = arun.run(net_in, 1)

    pred_box = np.zeros((image_length, 4))
    t1 = time.time()
    for counter in range(image_length):
        if counter == 0:
            max_fire = (net_r == net_r.max()).nonzero()
            x_p = max_fire[1][0]
            y_p = max_fire[0][0]
            origin_box = label_resize[counter, :].astype('int64')
            x_1, y_1, w, h = origin_box[0], origin_box[1], origin_box[2], origin_box[3]
            pred_box[counter, 0], pred_box[counter, 1], pred_box[counter, 2], pred_box[counter, 3] = x_p, y_p, w, h
        elif counter:
            net_Iext = abs(image_all_resize[counter, :] - image_all_resize[counter-1, :]).astype(np.float32)
            net_in = np.concatenate((net_Iext.reshape(-1), net_r.reshape(-1))).astype(np.float32)

            net_U, net_recSum, net_r = arun.run(net_in, 1)

            max_fire = (net_r == net_r.max()).nonzero()
            x_p = max_fire[1][0]
            y_p = max_fire[0][0]
            origin_box = label_resize[counter, :].astype('int64')
            x_1, y_1, w, h = origin_box[0], origin_box[1], origin_box[2], origin_box[3]
            pred_box[counter, 0], pred_box[counter, 1], pred_box[counter, 2], pred_box[counter, 3] = x_p, y_p, w, h

    t2 = time.time()
    arun.apu_unload()
    frame_rate = image_length / (t2 - t1)
    print("frame_rate is %s" % frame_rate)

    # calculate precision plot and success plot
    showSuccess(pred_box, label_resize, "CANN", "r")
    # visualization
    pred_path = root + '/data/data_cann/Tiger1/pred_box.txt'
    pd.DataFrame(pred_box).to_csv(pred_path, header=False, index=False)
    show_dataset(dataset_path, label_path, pred_path)
    print("finish")


if __name__ == '__main__':
    main()
