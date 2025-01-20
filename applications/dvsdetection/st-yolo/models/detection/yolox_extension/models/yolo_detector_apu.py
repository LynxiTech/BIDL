# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

from typing import Dict, Optional, Tuple, Union

import torch as th
from omegaconf import DictConfig

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from ...recurrent_backbone.yolo_backbone import YoloDetN, YoloDetS, YoloDetX
from .build import build_yolox_fpn_apu, build_yolox_head_apu
from utils.timers import TimerDummy as CudaTimer
from data.utils.types import BackboneFeatures, LstmStates


class YoloV5Detector_N(th.nn.Module):
    def __init__(self,
                 model_cfg: DictConfig):
        super().__init__()
        backbone_cfg = model_cfg.backbone
        fpn_cfg = model_cfg.fpn
        head_cfg = model_cfg.head
        
        self.backbone = YoloDetN()
        in_channels = (64, 128, 256)

        self.fpn = build_yolox_fpn_apu(fpn_cfg, in_channels=in_channels)

        strides = (8, 16, 32)
        self.yolox_head = build_yolox_head_apu(head_cfg, in_channels=in_channels, strides=strides)

    def forward_backbone(self,
                         x: th.Tensor) -> \
            Tuple[BackboneFeatures, LstmStates]:
        with CudaTimer(device=x.device, timer_name="Backbone"):
            backbone_features = self.backbone(x)
        return backbone_features

    def forward_detect(self,
                       backbone_features: BackboneFeatures) -> \
            Tuple[th.Tensor, Union[Dict[str, th.Tensor], None]]:
        device = th.device('cpu')
        with CudaTimer(device=device, timer_name="FPN"):
            fpn_features = self.fpn(backbone_features)

        with CudaTimer(device=device, timer_name="HEAD"):
            outputs = self.yolox_head(fpn_features)
       
        return outputs

    def forward(self,
                x: th.Tensor) -> \
            Tuple[Union[th.Tensor, None], Union[Dict[str, th.Tensor], None], LstmStates]:
        backbone_features = self.forward_backbone(x)

        outputs = self.forward_detect(backbone_features=backbone_features)
        return outputs


class YoloV5Detector_S(th.nn.Module):
    def __init__(self,
                 model_cfg: DictConfig):
        super().__init__()
        backbone_cfg = model_cfg.backbone
        fpn_cfg = model_cfg.fpn
        head_cfg = model_cfg.head

        self.backbone = YoloDetS()
        in_channels = (128, 256, 512)

        self.fpn = build_yolox_fpn_apu(fpn_cfg, in_channels=in_channels)

        strides = (8, 16, 32)
        self.yolox_head = build_yolox_head_apu(head_cfg, in_channels=in_channels, strides=strides)

    def forward_backbone(self,
                         x: th.Tensor) -> \
            Tuple[BackboneFeatures, LstmStates]:
        with CudaTimer(device=x.device, timer_name="Backbone"):
            backbone_features = self.backbone(x)
        return backbone_features

    def forward_detect(self,
                       backbone_features: BackboneFeatures) -> \
            Tuple[th.Tensor, Union[Dict[str, th.Tensor], None]]:
        device = th.device('cpu')
        with CudaTimer(device=device, timer_name="FPN"):
            fpn_features = self.fpn(backbone_features)

        with CudaTimer(device=device, timer_name="HEAD"):
            outputs = self.yolox_head(fpn_features)
       
        return outputs

    def forward(self,
                x: th.Tensor) -> \
            Tuple[Union[th.Tensor, None], Union[Dict[str, th.Tensor], None], LstmStates]:
        backbone_features = self.forward_backbone(x)

        outputs = self.forward_detect(backbone_features=backbone_features)
        return outputs
        
class YoloV5Detector_X(th.nn.Module):
    def __init__(self,
                 model_cfg: DictConfig):
        super().__init__()
        backbone_cfg = model_cfg.backbone
        fpn_cfg = model_cfg.fpn
        head_cfg = model_cfg.head

        self.backbone = YoloDetX()
        in_channels = (320, 640, 1280)
        
        self.fpn = build_yolox_fpn_apu(fpn_cfg, in_channels=in_channels)

        strides = (8, 16, 32)
        self.yolox_head = build_yolox_head_apu(head_cfg, in_channels=in_channels, strides=strides)

    def forward_backbone(self,
                         x: th.Tensor) -> \
            Tuple[BackboneFeatures, LstmStates]:
        with CudaTimer(device=x.device, timer_name="Backbone"):
            backbone_features = self.backbone(x)
        return backbone_features

    def forward_detect(self,
                       backbone_features: BackboneFeatures) -> \
            Tuple[th.Tensor, Union[Dict[str, th.Tensor], None]]:
        device = th.device('cpu')
        with CudaTimer(device=device, timer_name="FPN"):
            fpn_features = self.fpn(backbone_features)

        with CudaTimer(device=device, timer_name="HEAD"):
            outputs = self.yolox_head(fpn_features)
       
        return outputs

    def forward(self,
                x: th.Tensor) -> \
            Tuple[Union[th.Tensor, None], Union[Dict[str, th.Tensor], None], LstmStates]:
        backbone_features = self.forward_backbone(x)

        outputs = self.forward_detect(backbone_features=backbone_features)
        return outputs