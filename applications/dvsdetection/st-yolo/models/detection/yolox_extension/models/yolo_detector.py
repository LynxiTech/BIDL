from typing import Dict, Optional, Tuple, Union

import torch as th
from omegaconf import DictConfig

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from ...recurrent_backbone.yolo import YoloDetN, YoloDetS, YoloDetX
from .build import build_yolox_fpn, build_yolox_head
from utils.timers import TimerDummy as CudaTimer
from thop import clever_format, profile
from data.utils.types import BackboneFeatures, LstmStates


class YoloV5Detector(th.nn.Module):
    def __init__(self,
                 model_cfg: DictConfig):
        super().__init__()
        backbone_cfg = model_cfg.backbone
        fpn_cfg = model_cfg.fpn
        head_cfg = model_cfg.head

        yolo_dict = backbone_cfg.yolo_dict
        if yolo_dict == "yolov5n":
            self.backbone = YoloDetN()
            in_channels = (64, 128, 256)
        elif yolo_dict == "yolov5s":
            self.backbone = YoloDetS()
            in_channels = (128, 256, 512)
        elif yolo_dict == "yolov5x":
            self.backbone = YoloDetX()
            in_channels = (320, 640, 1280)
        self.fpn = build_yolox_fpn(fpn_cfg, in_channels=in_channels)

        strides = (8, 16, 32)
        self.yolox_head = build_yolox_head(head_cfg, in_channels=in_channels, strides=strides)

    def forward_backbone(self,
                         x: th.Tensor,
                         previous_states: Optional[LstmStates] = None,
                         token_mask: Optional[th.Tensor] = None) -> \
            Tuple[BackboneFeatures, LstmStates]:
        with CudaTimer(device=x.device, timer_name="Backbone"):
            backbone_features, states = self.backbone(x, previous_states, token_mask)

        return backbone_features, states

    def forward_detect(self,
                       backbone_features: BackboneFeatures,
                       targets: Optional[th.Tensor] = None) -> \
            Tuple[th.Tensor, Union[Dict[str, th.Tensor], None]]:
        device = next(iter(backbone_features.values())).device
        with CudaTimer(device=device, timer_name="FPN"):
            fpn_features = self.fpn(backbone_features)
        if self.training:
            assert targets is not None
            with CudaTimer(device=device, timer_name="HEAD + Loss"):
                outputs, losses = self.yolox_head(fpn_features, targets)
            return outputs, losses
        with CudaTimer(device=device, timer_name="HEAD"):
            outputs, losses = self.yolox_head(fpn_features)
        assert losses is None
        return outputs, losses

    def forward(self,
                x: th.Tensor,
                previous_states: Optional[LstmStates] = None,
                retrieve_detections: bool = True,
                targets: Optional[th.Tensor] = None) -> \
            Tuple[Union[th.Tensor, None], Union[Dict[str, th.Tensor], None], LstmStates]:
        backbone_features, states = self.forward_backbone(x, previous_states)
        outputs, losses = None, None
        if not retrieve_detections:
            assert targets is None
            return outputs, losses, states
        outputs, losses = self.forward_detect(backbone_features=backbone_features, targets=targets)
        return outputs, losses, states
