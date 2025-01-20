# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import torch as pt
import torch.nn as nn
import sys
sys.path.append("../")
from layers.lif import Lif2d
from layers.time_distributed import TimeDistributed


class CompatOp: 

    @staticmethod
    def add(x1, x2): 
        assert type(x1) == type(x2)
        if isinstance(x1, (list, tuple)):
            x3 = [a + b for a, b in zip(x1, x2)]
        elif isinstance(x1, pt.Tensor):
            x3 = x1 + x2
        else:
            raise NotImplementedError
        return x3

    @staticmethod
    def sub(x1, x2):  
        assert type(x1) == type(x2)
        if isinstance(x1, (list, tuple)):
            x3 = [a - b for a, b in zip(x1, x2)]
        elif isinstance(x1, pt.Tensor):
            x3 = x1 - x2
        else:
            raise NotImplementedError
        return x3

    @staticmethod
    def mul(x1, x2):  
        assert type(x1) == type(x2)
        if isinstance(x1, (list, tuple)):
            x3 = [a * b for a, b in zip(x1, x2)]
        elif isinstance(x1, pt.Tensor):
            x3 = x1 * x2
        else:
            raise NotImplementedError
        return x3


class AvgPoolTimeVideo(nn.AvgPool1d):

    def __init__(self, down: int):
        super(AvgPoolTimeVideo, self).__init__(down)

    def forward(self, xis):
        """
        :param xis: (b,t,c,h,w)
        """
        b, t, c, h, w = xis.shape
        assert t % self.kernel_size[0] == 0
        x1s = xis.permute(0, 2, 3, 4, 1).reshape(-1, t)
        x2s = super(AvgPoolTimeVideo, self).forward(x1s)
        xos = x2s.reshape(b, c, h, w, -1).permute(0, 4, 1, 2, 3)
        return xos


class MaxPoolTimeVideo(nn.MaxPool1d):

    def __init__(self, down: int):
        super(MaxPoolTimeVideo, self).__init__(down)

    def forward(self, xis):
        """
        :param xis: (b,t,c,h,w)
        """
        b, t, c, h, w = xis.shape
        assert t % self.kernel_size == 0
        x1s = xis.permute(0, 2, 3, 4, 1).reshape(-1, t)
        x2s = super(MaxPoolTimeVideo, self).forward(x1s)
        xos = x2s.reshape(b, c, h, w, -1).permute(0, 4, 1, 2, 3)
        return xos


class GlobalAvgPoolVideo(nn.AdaptiveAvgPool3d):

    def __init__(self):
        super(GlobalAvgPoolVideo, self).__init__((1, 1, 1))

    def forward(self, xis):
        """
        :param xis: (b,t,c,h,w)
        """
        assert self.output_size == (1, 1, 1)
        x1s = xis.permute(0, 2, 1, 3, 4)
        x2s = super(GlobalAvgPoolVideo, self).forward(x1s)
        xos = x2s[:, :, 0, 0, 0]
        return xos


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
            inplanes, planes, stride=1, downsample=None, noise=0,mode='spike', soma_params='all_share', hidden_channels=None
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = TimeDistributed(nn.Sequential(nn.Conv2d(inplanes, planes, 3, stride, 1), nn.BatchNorm2d(planes)))
        self.lif1 = Lif2d(mode=mode, noise=noise, soma_params=soma_params, hidden_channel=hidden_channels[0],spike_func=None,use_inner_loop=True)
        self.conv2 = TimeDistributed(nn.Sequential(nn.Conv2d(planes, planes, 3, 1, 1), nn.BatchNorm2d(planes)))
        self.downsample = downsample
        self.lif2 = Lif2d(mode=mode, noise=noise, soma_params=soma_params, hidden_channel=hidden_channels[1],spike_func=None,use_inner_loop=True)

    def forward(self, x):
        identity = x
        out = self.lif1(self.conv1(x))
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = CompatOp.add(out, identity)  # ``ADD``
        return self.lif2(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
            inplanes, planes, stride=1, downsample=None, noise=0, mode='spike', soma_params='all_share', hidden_channels=None,
    ):
        super(Bottleneck, self).__init__()
        _mid_ = planes // self.expansion
        self.conv1 = TimeDistributed(nn.Sequential(nn.Conv2d(inplanes, _mid_, 1, 1, 0), nn.BatchNorm2d(_mid_)))
        self.lif1 = Lif2d(mode=mode, noise=noise, soma_params=soma_params, hidden_channel=hidden_channels[0],spike_func=None,use_inner_loop=True)
        self.conv2 = TimeDistributed(nn.Sequential(nn.Conv2d(_mid_, _mid_, 3, stride, 1), nn.BatchNorm2d(_mid_)))
        self.lif2 = Lif2d(mode=mode, noise=noise, soma_params=soma_params, hidden_channel=hidden_channels[1],spike_func=None,use_inner_loop=True)
        self.conv3 = TimeDistributed(nn.Sequential(nn.Conv2d(_mid_, planes, 1, 1, 0), nn.BatchNorm2d(planes)))
        self.downsample = downsample
        self.lif3 = Lif2d(mode=mode, noise=noise, soma_params=soma_params, hidden_channel=hidden_channels[2],spike_func=None,use_inner_loop=True)

    def forward(self, x):
        identity = x
        out = self.lif1(self.conv1(x))
        out = self.lif2(self.conv2(out))
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = CompatOp.add(out, identity)  # ``ADD``
        return self.lif3(out)



class ResNetLif(nn.Module):
    arch_settings = {
        10: (BasicBlock, (1, 1, 1, 1)),
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }
    time_pools = {
        'avg': AvgPoolTimeVideo,
        'max': MaxPoolTimeVideo
    }

    def __init__(self,
            depth, nclass, low_resolut=False,
            input_channels=3, stem_channels=64, base_channels=64,
            down_t=(4, 'max'),
            zero_init_residual=False,
            noise=1e-3,cmode='spike', soma_params='all_share',norm = None
    ):
        super(ResNetLif, self).__init__()

        if low_resolut:  # for resolution less than 64
            self.conv = TimeDistributed(nn.Sequential(nn.Conv2d(input_channels, stem_channels, 3, 1, 1),
                nn.BatchNorm2d(stem_channels)))
        else:
            self.conv = TimeDistributed(nn.Sequential(nn.Conv2d(input_channels, stem_channels, 7, 2, 3),
                nn.BatchNorm2d(stem_channels)))
        self.lif = Lif2d(mode=cmode, noise=noise, soma_params=soma_params, hidden_channel=stem_channels,spike_func=None,use_inner_loop=True)
        self.pool = nn.Sequential(self.time_pools[down_t[1]](down_t[0]) if down_t[0] > 1 else nn.Identity(),
            TimeDistributed(nn.MaxPool2d((3, 3), (2, 2), 1)))  # ``ceil_mode=True`` not needed with ``padding``
        assert depth in [10, 18, 34, 50, 101, 152]
        if depth == 50:
            hidden_channels = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]
        elif depth in [10, 18, 34]:
            hidden_channels = [[0, 0], [0, 0], [0, 0], [0, 0]]
        elif depth in [50, 101, 152]:
            hidden_channels = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        block, layers = self.arch_settings[depth]
        ci, co = stem_channels, base_channels * block.expansion
        self.layer1 = self._make_layer(block, ci, co, layers[0], 1, noise,mode=cmode, soma_params=soma_params, hidden_channels=hidden_channels[0])
        ci, co = co, co * 2
        self.layer2 = self._make_layer(block, ci, co, layers[1], 2, noise,mode=cmode, soma_params=soma_params, hidden_channels=hidden_channels[1])
        ci, co = co, co * 2
        self.layer3 = self._make_layer(block, ci, co, layers[2], 2, noise,mode=cmode, soma_params=soma_params, hidden_channels=hidden_channels[2])
        ci, co = co, co * 2
        self.layer4 = self._make_layer(block, ci, co, layers[3], 2, noise,mode=cmode, soma_params=soma_params, hidden_channels=hidden_channels[3])

        self.gap = GlobalAvgPoolVideo()
        self.fc = nn.Linear(co, nclass)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
        #     elif isinstance(m, nn.BatchNorm3d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

        if zero_init_residual:
            self.zero_init_blocks()

    def zero_init_blocks(self):
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.conv3.module[1].weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.conv2.module[1].weight, 0)

    @staticmethod
    def _make_layer(block, ci, co, blocks, stride, noise, mode='spike', soma_params='all_share', hidden_channels=None):
        downsample = None
        if stride != 1 or ci != co:
            downsample = TimeDistributed(nn.Sequential(nn.AvgPool2d(stride, ceil_mode=True),
                nn.Conv2d(ci, co, 1, 1, 0), nn.BatchNorm2d(co)))
            # downsample = nn.Sequential(TimeDistributed(nn.AvgPool2d(stride, ceil_mode=True)),
            #     Conv2dLif(ci, co, 1, 1, 0, feed_back=False, norm_state=True, mode='spike', noise=noise))
        layers = [
            block(ci, co, stride, downsample=downsample, noise=noise, mode=mode, soma_params=soma_params, hidden_channels=hidden_channels)
        ]
        for i in range(1, blocks):
            layers.append(block(co, co, noise=noise, mode=mode, soma_params=soma_params, hidden_channels=hidden_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: (b,t,c,h,w)
        """
        x = self.lif(self.conv(x))
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gap(x)
        x = self.fc(x)
        return x
