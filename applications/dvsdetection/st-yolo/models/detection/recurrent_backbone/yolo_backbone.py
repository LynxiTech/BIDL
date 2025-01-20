# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

from typing import Dict, Optional, Tuple

import torch as th
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from data.utils.types import FeatureMap, BackboneFeatures, LstmState, LstmStates

import argparse
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from .common_apu import *
from models.experimental import *

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


# yolov5s
class YoloDetS(nn.Module):
    def __init__(self, in_channels=20):
        super().__init__()
        self.num_stages = 4
        self.conv1 = Conv(in_channels, 32, 6, 2, 1)
        self.conv2 = Conv_lif(32, 64, 3, 2, 1)

        self.C3_1 = C3(64, 64, 1)
        self.conv3 = Conv_lif(64, 128, 3, 2, 1)

        self.C3_2 = C3(128, 128, 2)
        self.conv4 = Conv(128, 256, 3, 2, 1)
        self.C3_3 = C3(256, 256, 3, lif=True)

        self.conv5 = Conv(256, 512, 3, 2, 1)
        self.C3_4 = C3(512, 512, 1)
        self.SPPF = SPPF(512, 512, lif=True)
        self._register_lyn_reset_hook()

    def _register_lyn_reset_hook(self):
        for child in self.modules():
            if isinstance(child, Lif2d):  # Lif, Lif1d, Conv2dLif, FcLif...
                assert hasattr(child, 'reset')
                child.register_forward_pre_hook(self.lyn_reset_hook)

    @staticmethod
    def lyn_reset_hook(m, xi: tuple):
        assert isinstance(xi, tuple) and len(xi) == 1
        xi = xi[0]
        if not hasattr(m, 'lyn_cnt'):
            setattr(m, 'lyn_cnt', 0)
        if m.lyn_cnt == 0:
            # print(m)
            m.reset(xi)
            m.lyn_cnt += 1
        else:
            m.lyn_cnt += 1

    def forward(self, x: th.Tensor) \
            -> Tuple[BackboneFeatures, LstmStates]:

        output: BackboneFeatures = list()

        x = self.conv1(x)
        x = self.conv2(x)
        output.append(x)

        x = self.C3_1(x)
        x = self.conv3(x)
        output.append(x)

        x = self.C3_2(x)
        x = self.conv4(x)
        x = self.C3_3(x)
        output.append(x)

        x = self.conv5(x)
        x = self.C3_4(x)
        x = self.SPPF(x)
        output.append(x)

        return output[0], output[1], output[2], output[3]

    def _reset_lyn_cnt(self):
        for child in self.modules():
            if hasattr(child, 'lyn_cnt'):
                child.lyn_cnt = 0


# yolov5x
class YoloDetX(nn.Module):
    def __init__(self, in_channels=20):
        super().__init__()
        self.num_stages = 4
        self.conv1 = Conv(in_channels, 80, 6, 2, 2)
        self.conv2 = Conv_lif(80, 160, 3, 2, 1)

        self.C3_1 = C3(160, 160, 4)
        self.conv3 = Conv_lif(160, 320, 3, 2, 1)

        self.C3_2 = C3(320, 320, 8)
        self.conv4 = Conv(320, 640, 3, 2, 1)
        self.C3_3 = C3(640, 640, 12, lif=True)

        self.conv5 = Conv(640, 1280, 3, 2, 1)
        self.C3_4 = C3(1280, 1280, 4)
        self.SPPF = SPPF(1280, 1280, lif=True)
        self._register_lyn_reset_hook()

    def _register_lyn_reset_hook(self):
        for child in self.modules():
            if isinstance(child, Lif2d):  # Lif, Lif1d, Conv2dLif, FcLif...
                assert hasattr(child, 'reset')
                child.register_forward_pre_hook(self.lyn_reset_hook)

    @staticmethod
    def lyn_reset_hook(m, xi: tuple):
        assert isinstance(xi, tuple) and len(xi) == 1
        xi = xi[0]
        if not hasattr(m, 'lyn_cnt'):
            setattr(m, 'lyn_cnt', 0)
        if m.lyn_cnt == 0:
            # print(m)
            m.reset(xi)
            m.lyn_cnt += 1
        else:
            m.lyn_cnt += 1

    def forward(self, x: th.Tensor) \
            -> Tuple[BackboneFeatures, LstmStates]:

        output: BackboneFeatures = list()

        x = self.conv1(x)
        x = self.conv2(x)
        output.append(x)

        x = self.C3_1(x)
        x = self.conv3(x)
        output.append(x)

        x = self.C3_2(x)
        x = self.conv4(x)
        x = self.C3_3(x)
        output.append(x)

        x = self.conv5(x)
        x = self.C3_4(x)
        x = self.SPPF(x)
        output.append(x)

        return output[0], output[1], output[2], output[3]


# yolov5x
class YoloDetN(nn.Module):
    def __init__(self, in_channels=20):
        super().__init__()
        self.num_stages = 4
        self.conv1 = Conv(in_channels, 16, 6, 2, 2)
        self.conv2 = Conv_lif(16, 32, 3, 2, 1)

        self.C3_1 = C3(32, 32, 1)
        self.conv3 = Conv_lif(32, 64, 3, 2, 1)

        self.C3_2 = C3(64, 64, 2)
        self.conv4 = Conv(64, 128, 3, 2, 1)
        self.C3_3 = C3(128, 128, 3, lif=True)

        self.conv5 = Conv(128, 256, 3, 2, 1)
        self.C3_4 = C3(256, 256, 1)
        self.SPPF = SPPF(256, 256, lif=True)

        self._register_lyn_reset_hook()

    def _register_lyn_reset_hook(self):
        for child in self.modules():
            if isinstance(child, Lif2d):  # Lif, Lif1d, Conv2dLif, FcLif...
                assert hasattr(child, 'reset')
                child.register_forward_pre_hook(self.lyn_reset_hook)

    @staticmethod
    def lyn_reset_hook(m, xi: tuple):
        assert isinstance(xi, tuple) and len(xi) == 1
        xi = xi[0]
        if not hasattr(m, 'lyn_cnt'):
            setattr(m, 'lyn_cnt', 0)
        if m.lyn_cnt == 0:
            # print(m)
            m.reset(xi)
            m.lyn_cnt += 1
        else:
            m.lyn_cnt += 1

    def forward(self, x: th.Tensor) \
            -> Tuple[BackboneFeatures, LstmStates]:

        output: BackboneFeatures = list()

        x = self.conv1(x)
        x = self.conv2(x)
        output.append(x)

        x = self.C3_1(x)
        x = self.conv3(x)
        output.append(x)

        x = self.C3_2(x)
        x = self.conv4(x)
        x = self.C3_3(x)
        output.append(x)
        
        x = self.conv5(x)
        x = self.C3_4(x)
        x = self.SPPF(x)
        output.append(x)

        return output[0], output[1], output[2], output[3]

    def _reset_lyn_cnt(self):
        for child in self.modules():
            if hasattr(child, 'lyn_cnt'):
                child.lyn_cnt = 0


YoloDet = YoloDetX
