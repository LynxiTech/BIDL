# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

from collections import OrderedDict

import lyngor as lyn
import torch
from torch import ops
import sys
sys.path.append("../../../")
from models.detection.recurrent_backbone.yolo_backbone import YoloDetN, YoloDetS, YoloDetX
import argparse
import os

root = os.getcwd()
for _ in range(3):
    root = os.path.dirname(root)    
ckpt_path = root + "/weight_files/dvsdetection/styolo/"


def parse_args():
    parser = argparse.ArgumentParser(description='specify model to compile only')
    parser.add_argument('--model', default='yolov5s', type=str,help='specify a model in yolov5n, yolov5s and yolov5x')
    args = parser.parse_args()
    return args


def load_library():
    # load libcustom_op.so
    library_path = "../../../lynadapter/custom_op_in_pytorch/build/libcustom_ops.so"
    ops.load_library(library_path)


def compile_backbone(yolo_dict="yolov5s"):
    target = 'apu'
    import lynadapter.custom_op_in_lyn.custom_op_my_lif

    load_library()
    
    if yolo_dict == "yolov5n":
        pt_model = YoloDetN()
        checkpoint_file = ckpt_path + 'epoch=001-step=394428-val_AP=0.41.ckpt'
    elif yolo_dict == "yolov5s":
        pt_model = YoloDetS()
        checkpoint_file = ckpt_path + 'epoch=001-step=394428-val_AP=0.44.ckpt'
    elif yolo_dict == "yolov5x":
        pt_model = YoloDetX()
        checkpoint_file = ckpt_path + 'epoch=000-step=197214-val_AP=0.46.ckpt'
    else:
        raise NameError
    
    out_path = root + "/model_files/dvsdetection/styolo/" + "st_" + yolo_dict + "_backbone"

    params = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    n = len(params['state_dict']) - 300
    temp_dict = dict(list(params['state_dict'].items())[:n])
    state_dict = OrderedDict((k[13:], v) for k, v in temp_dict.items())
    pt_model.load_state_dict(state_dict)

    inputs_dict = {'xi': (1, 20, 256, 320)}

    model = lyn.DLModel()
    model.load(pt_model.eval(), model_type="Pytorch", inputs_dict=inputs_dict)

    builder = lyn.Builder(target=target, is_map=True)

    out_path = builder.build(model.mod, model.params, opt_level=3, out_path=out_path,
                             run_batch=1, version=0)
                             
if __name__ == '__main__':
    args = parse_args()
    compile_backbone(yolo_dict=args.model)