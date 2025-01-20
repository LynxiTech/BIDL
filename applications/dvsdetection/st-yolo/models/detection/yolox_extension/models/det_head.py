# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

from typing import Dict, Optional, Tuple, Union

import torch as th
from omegaconf import DictConfig

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from .build import build_yolox_fpn_apu, build_yolox_head_apu


class Detection_head_N(th.nn.Module):
    def __init__(self,
                 model_cfg: DictConfig):
        super().__init__()
        backbone_cfg = model_cfg.backbone
        fpn_cfg = model_cfg.fpn
        head_cfg = model_cfg.head
        
        in_channels = (64, 128, 256)

        self.fpn = build_yolox_fpn_apu(fpn_cfg, in_channels=in_channels)

        strides = (8, 16, 32)
        self.yolox_head = build_yolox_head_apu(head_cfg, in_channels=in_channels, strides=strides)

    def forward(self, data_in):
        features4 = data_in[0:20480].reshape((1, 256, 8, 10))
        features3 = data_in[20480:20480 + 40960].reshape((1, 128, 16, 20))
        features2 = data_in[20480 + 40960:20480 + 40960 + 81920].reshape((1, 64, 32, 40))

        backbone_features = [features2, features3, features4]
        fpn_features = self.fpn(backbone_features)
        outputs = self.yolox_head(fpn_features)
        return outputs


class Detection_head_S(th.nn.Module):
    def __init__(self,
                 model_cfg: DictConfig):
        super().__init__()
        backbone_cfg = model_cfg.backbone
        fpn_cfg = model_cfg.fpn
        head_cfg = model_cfg.head
        
        in_channels = (128, 256, 512)

        self.fpn = build_yolox_fpn_apu(fpn_cfg, in_channels=in_channels)

        strides = (8, 16, 32)
        self.yolox_head = build_yolox_head_apu(head_cfg, in_channels=in_channels, strides=strides)

    def forward(self, data_in):
        features4 = data_in[0:40960].reshape((1, 512, 8, 10))
        features3 = data_in[40960:40960+81920].reshape((1, 256, 16, 20))
        features2 = data_in[40960+81920:40960+81920+163840].reshape((1, 128, 32, 40))


        backbone_features = [features2, features3, features4]
        fpn_features = self.fpn(backbone_features)
        outputs = self.yolox_head(fpn_features)
        return outputs
        

class Detection_head_X(th.nn.Module):
    def __init__(self,
                 model_cfg: DictConfig):
        super().__init__()
        backbone_cfg = model_cfg.backbone
        fpn_cfg = model_cfg.fpn
        head_cfg = model_cfg.head
        
        in_channels = (320, 640, 1280)

        self.fpn = build_yolox_fpn_apu(fpn_cfg, in_channels=in_channels)

        strides = (8, 16, 32)
        self.yolox_head = build_yolox_head_apu(head_cfg, in_channels=in_channels, strides=strides)

    def forward(self, data_in):
        features4 = data_in[0:102400].reshape((1, 1280, 8, 10))
        features3 = data_in[102400:102400+204800].reshape((1, 640, 16, 20))
        features2 = data_in[102400+204800:102400+204800+409600].reshape((1, 320, 32, 40))

        backbone_features = [features2, features3, features4]
        fpn_features = self.fpn(backbone_features)
        outputs = self.yolox_head(fpn_features)
        return outputs