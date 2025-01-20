# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

# This code references the source code of OpenMMLab projects, which are
# licensed under the Apache License, Version 2.0.
import argparse
import json
import os
import threading
from argparse import Namespace
from copy import deepcopy

import mmcv
import numpy as np
import torch
from flask import Flask, request, jsonify

from bidlcls.datasets import build_dataset
from bidlcls.models import build_classifier
from bidlcls.models.backbones.lynxi_backbones_lyngor import DualFlifp1Fc1BQSplit, SeqClif3Flif2DGSplit
from bidlcls.models.backbones.lynxi_resnetlif_lyngor import ResNetLifSplit

TORCH_LYNGOR_MAP = {
    'DualFlifp1Fc1BQ': DualFlifp1Fc1BQSplit,
    'SeqClif3Flif2DG': SeqClif3Flif2DGSplit,
    'ResNetLif': ResNetLifSplit
}


class RunDemo:

    def __init__(self, args):
        self.lock = threading.Lock()
        self.cc = args.gcc_linaro

        cfg = mmcv.Config.fromfile(args.config)
        assert os.path.exists(args.model) and len(os.listdir(args.model)) > 0
        self.base_path = '/'.join(os.path.abspath(args.model).split('/')[:-1])
        cfg.data.test.data_prefix = args.dataset

        cfg.model.pretrained = None
        cfg.data.test.test_mode = True

        cfg.data.samples_per_gpu = 1

        self.dataset = build_dataset(cfg.data.test)
        self.model = build_classifier(cfg.model)

        self.CLASSES = self.dataset.__class__.CLASSES  # XXX ??? not work for ``BabiQa`` but its ``WORD_IDX`` is okay
        self.ishape = self.dataset.__getitem__(0)['img'].shape

    def predict(self, data: dict, which: int = -1):
        with self.lock:
            model = deepcopy(self.model)
            model.backbone = TORCH_LYNGOR_MAP[self.model.backbone.__class__.__name__](
                self.model.backbone, True, self.cc, 1, self.base_path, self.ishape
            )
            model = model.cpu()
            model.eval()
            

            data['img'] = data['img'][None, ...]

            with torch.no_grad():
                result = model(return_loss=False, **data)
            scores = np.vstack(result)
            pred_score = np.max(scores, axis=1)[0]
            pred_label = np.argmax(scores, axis=1)[0]

            pred_class = self.CLASSES[int(pred_label)]
            lbl = self.dataset.data_infos[which]['gt_label'] if which >= 0 else -1
            cls = self.CLASSES[int(lbl)] if lbl >= 0 else '-1'

            return {
                'true_label': str(lbl),
                'true_class': cls,
                'pred_score': str(pred_score),
                'pred_label': str(pred_label),
                'pred_class': pred_class,
            }

    @staticmethod
    def is_integer_string(chars: str):
        try:
            int(chars)
            return True
        except:
            return False

    @staticmethod
    def is_text_content(chars: str):
        return len(chars.strip().split(' ')) > 1

    def load_input(self, task: str, input: str):
        # if task == 'bAbI-QA':
        #     if self.is_text_content(input):
        #         xmpl = self.dataset.wrap_text(input)
        #     elif self.is_integer_string(input):
        #         xmpl = self.dataset.__getitem__(int(input))
        #     else:
        #         raise NotImplemented
        # elif task == 'DVS-Gesture':
        #     if os.path.isfile(input):
        #         raise self.dataset.wrap_spike(input)
        #     elif self.is_integer_string(input):
        #         xmpl = self.dataset.__getitem__(int(input))
        #     else:
        #         raise NotImplemented
        # elif task == 'Jester' or task is None:  # XXX make-shift
        #     if os.path.isfile(input):
        #         xmpl = self.dataset.wrap_video(input)
        #     elif self.is_integer_string(input):
        #         xmpl = self.dataset.__getitem__(int(input))
        #     else:
        #         raise NotImplemented
        # else:
        #     raise NotImplemented
        xmpl = self.dataset.__getitem__(int(input))
        which = int(input) if self.is_integer_string(input) else -1
        return xmpl, which


class FlaskApp(Flask):

    def __init__(self, demo):
        super(FlaskApp, self).__init__(__name__)
        self.demo = demo
        self.r_hello = self.route('/', methods=['GET'])(self.hello_world)
        self.r_proc = self.route('/process', methods=['POST'])(self.process)

    def hello_world(self):
        return 'Hello World'

    def process(self):
        if request.method == "POST":
            msg = request.get_data()
            print('msg:', msg)
            req_info = json.loads(msg)

            # xmpl, which = self.demo.load_input(req_info.get('task'), req_info.get('input'))
            xmpl, which = self.demo.load_input(req_info.get('input'))
            rslt = self.demo.predict(xmpl, which)
            print(rslt)

            resp_info = {'result': rslt}
            return jsonify(resp_info)
        else:
            raise NotImplementedError


def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('-c', '--config', type=str,
                        default='configs/clif3flif2dg-b16x1-dvsgesture.py')
    parser.add_argument('-m', '--model', type=str,
                        default='work_dirs/SeqClif3Flif2DGSplit-1/')
    parser.add_argument('-d', '--dataset', type=str,
                        default='data/dvsgesture/')
    parser.add_argument('-g', '--gcc_linaro', type=str,
                        default='/home/lynchip/ProgramFiles/gcc-linaro-7.4.1-2019.02-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++')
    return parser.parse_args()


def main():
    os.chdir('../')
    is_testing = 1
    if is_testing:
        args = parse_args()
        host = '192.168.11.143'
    else:
        args = Namespace(config=os.environ["CONFIG_PATH"], model=os.environ["MODEL_PATH"],
                         dataset=os.environ["DATASET_DIR"])
        host = '0.0.0.0'
    demo = RunDemo(args)
    app = FlaskApp(demo)
    app.run(host, port=9080, threaded=True)


if __name__ == '__main__':
    main()

"""
python flaskdemos.py \
    -c "/home/lynchip/Active/lynbidl-mmlab-20220521/configs/clif3flif2dg-b16x1-dvsgesture.py" \
    -m "/home/lynchip/Active/lynbidl-mmlab-20220521/work_dirs/SeqClif3Flif2DGSplit-1/" \
    -d "/home/lynchip/Static/datasets/dvsgesture/"
python flaskdemos.py \
    -c "/home/lynchip/Active/lynbidl-mmlab-20220521/configs/flif1fc1bq-b32x1-babiqa_en20.py" \
    -m "/home/lynchip/Active/lynbidl-mmlab-20220521/work_dirs/DualFlifp1Fc1BQSplit-1/" \
    -d "/home/lynchip/Static/datasets/babiqa/en/"
python flaskdemos.py \
    -c "/home/lynchip/Active/lynbidl-mmlab-20220521/configs/resnetlif18-b16x4-jester-cos160e.py" \
    -m "/home/lynchip/Active/lynbidl-mmlab-20220521/work_dirs/ResNetLifSplit-1/" \
    -d "/home/lynchip/Static/datasets/20bn-jester-v1/"
"""
