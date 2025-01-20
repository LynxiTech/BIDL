"""
Functions to display events and boxes
Copyright: (c) 2019-2020 Prophesee
"""
# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

from __future__ import print_function

import os

import bbox_visualizer as bbv
import cv2
import numpy as np
from matplotlib import pyplot as plt

LABELMAP_GEN1 = ("car", "pedestrian")
LABELMAP_GEN4 = ('pedestrian', 'two wheeler', 'car', 'truck', 'bus', 'traffic sign', 'traffic light')
LABELMAP_GEN4_SHORT = ('pedestrian', 'two wheeler', 'car')
current_file_path = os.path.abspath(__file__)

root_dir = current_file_path
for _ in range(5):
    root_dir = os.path.dirname(root_dir)


def make_binary_histo(events, img=None, width=304, height=240):
    """
    simple display function that shows negative events as blacks dots and positive as white one
    on a gray background
    args :
        - events structured numpy array
        - img (numpy array, height x width x 3) optional array to paint event on.
        - width int
        - height int
    return:
        - img numpy array, height x width x 3)
    """
    if img is None:
        img = 127 * np.ones((height, width, 3), dtype=np.uint8)
    else:
        # if an array was already allocated just paint it grey
        img[...] = 127
    if events.size:
        assert events['x'].max() < width, "out of bound events: x = {}, w = {}".format(events['x'].max(), width)
        assert events['y'].max() < height, "out of bound events: y = {}, h = {}".format(events['y'].max(), height)

        img[events['y'], events['x'], :] = 255 * events['p'][:, None]
    return img


def draw_bboxes_bbv(img, boxes, save_id, labelmap=LABELMAP_GEN1, is_pred=False) -> np.ndarray:
    """
    draw bboxes in the image img
    """
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
    colors = [tuple(*item) for item in colors.tolist()]

    if labelmap == LABELMAP_GEN1:
        classid2colors = {
            0: (255, 255, 0),  # car -> yellow (rgb)
            1: (0, 0, 255),  # ped -> blue (rgb)
        }
        scale_multiplier = 4
    else:
        assert labelmap == LABELMAP_GEN4_SHORT
        classid2colors = {
            0: (0, 0, 255),  # ped -> blue (rgb)
            1: (0, 255, 255),  # 2-wheeler cyan (rgb)
            2: (255, 255, 0),  # car -> yellow (rgb)
        }
        scale_multiplier = 2

    add_score = True
    ht, wd, ch = img.shape
    dim_new_wh = (int(wd * scale_multiplier), int(ht * scale_multiplier))
    if scale_multiplier != 1:
        img = cv2.resize(img, dim_new_wh, interpolation=cv2.INTER_AREA)
    if boxes is not None:
        for i in range(boxes.shape[0]):
            pt1 = (int(boxes['x'][i]), int(boxes['y'][i]))
            size = (int(boxes['w'][i]), int(boxes['h'][i]))
            pt2 = (pt1[0] + size[0], pt1[1] + size[1])
            bbox = (pt1[0], pt1[1], pt2[0], pt2[1])
            bbox = tuple(x * scale_multiplier for x in bbox)

            score = boxes['class_confidence'][i]
            class_id = boxes['class_id'][i]
            class_name = labelmap[class_id % len(labelmap)]
            bbox_txt = class_name
            if add_score:
                bbox_txt += f' {score:.2f}'
            color_tuple_rgb = classid2colors[class_id]
            img = bbv.draw_rectangle(img, bbox, bbox_color=color_tuple_rgb)
            img = bbv.add_label(img, bbox_txt, bbox, text_bg_color=color_tuple_rgb, top=True)

    visualize_dir = os.path.join(root_dir, 'visualize')

    if not os.path.exists(visualize_dir):
        os.makedirs(visualize_dir)
        pred_dir = os.path.join(visualize_dir, 'pred')
        os.makedirs(pred_dir)

    save_id = '{:04d}'.format(save_id)
    if is_pred:
        save_path = root_dir + '/visualize/pred/' + str(save_id) + ".png"
    else:
        save_path = root_dir + '/visualize/label/' + str(save_id) + ".png"
    cv2.imwrite(save_path, img)
    return img


def draw_bboxes(img, boxes, labelmap=LABELMAP_GEN1) -> None:
    """
    draw bboxes in the image img
    """
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
    colors = [tuple(*item) for item in colors.tolist()]

    for i in range(boxes.shape[0]):
        pt1 = (int(boxes['x'][i]), int(boxes['y'][i]))
        size = (int(boxes['w'][i]), int(boxes['h'][i]))
        pt2 = (pt1[0] + size[0], pt1[1] + size[1])
        score = boxes['class_confidence'][i]
        class_id = boxes['class_id'][i]
        class_name = labelmap[class_id % len(labelmap)]
        color = colors[class_id * 60 % 255]
        center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.rectangle(img, pt1, pt2, color, 1)
        cv2.putText(img, class_name, (center[0], pt2[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        cv2.putText(img, str(score), (center[0], pt1[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)


def filter_predictions(yolox_preds_proph, img):
    image_shape = img.shape

    # Define the color values
    background_color = 127
    black_color = 0
    white_color = 255

    # Define the threshold values for black and white color counts and class confidence
    color_threshold = 20
    confidence_threshold = 0.8

    yolox_preds_proph_copy = np.copy(yolox_preds_proph)[0].tolist()

    # Iterate over each prediction in yolox_preds_proph_copy in reverse order
    for i in reversed(range(len(yolox_preds_proph_copy))):
        # Get the x, y, w, h values for the bounding box
        x, y, w, h = yolox_preds_proph_copy[i][1:5]

        # Calculate the coordinates of the top-left and bottom-right corner of the bounding box
        x_min = int(x)
        y_min = int(y)
        x_max = int(x + w)
        y_max = int(y + h)

        # Check if the bounding box is within the image boundaries
        if x_min >= 0 and y_min >= 0 and x_max < image_shape[1] and y_max < image_shape[0]:
            # Extract the region of interest from the image
            roi = img[y_min:y_max, x_min:x_max, 0]

            # Count the number of black and white pixels in the region of interest
            color_count = np.sum(roi == black_color) + np.sum(roi == white_color)

            # Check if color count is below the threshold and class confidence is below the threshold
            if color_count < color_threshold and yolox_preds_proph_copy[i][-1] < confidence_threshold:
                # Remove the prediction by setting all values to 0
                yolox_preds_proph_copy.pop(i)

    # Update yolox_preds_proph with the filtered predictions
    yolox_preds_proph[0] = np.array(yolox_preds_proph_copy, dtype=yolox_preds_proph[0].dtype)

    return yolox_preds_proph